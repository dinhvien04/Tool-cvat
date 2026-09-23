"""Synchronize canonical root modules (app, core, config) into serverless build contexts.

Root is canonical development source. Serverless requires local copies for Docker build context.
This script ensures zero drift between root and serverless copies using target-scoped manifests:
- COMMON_CORE: modules needed by all functions
- WEEK2_CORE: modules needed only by Pose17 and VF50
- THREE_DETECTOR_CORE: modules needed by Rectangle/Polygon/Polyline
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

# Modules and config files common to ALL serverless functions
COMMON_CORE: Tuple[str, ...] = (
    "core/__init__.py",
    "core/cvat_safety.py",
    "core/geometry.py",
    "core/line_geometry.py",
    "core/quality_gate.py",
    "core/taxonomy.py",
    "core/vision_contract.py",
    "app/__init__.py",
    "app/client.py",
    "app/config.py",
    "app/cvat_sync.py",
    "app/feedback.py",
    "app/image_ops.py",
    "app/parser.py",
    "config/labels.yaml",
    "config/cvat_labels.json",
    "config/label_semantics.yaml",
    "config/label_geometry.yaml",
)

# Modules and config files needed ONLY by Week 2 detectors (Pose17 and VF50)
WEEK2_CORE: Tuple[str, ...] = (
    "core/week2_schema.py",
    "core/pose_face_schema.py",
    "core/skeleton_contract.py",
    "config/week2_pose17.yaml",
    "config/week2_vf50.yaml",
)

# Modules and config files needed ONLY by the active Three Detectors (Rectangle, Polygon, Polyline)
THREE_DETECTOR_CORE: Tuple[str, ...] = (
    "app/models.py",
    "app/pipeline.py",
    "app/retrieval.py",
    "app/service.py",
    "config/learning.yaml",
)

# Active serverless functions
SERVERLESS_TARGETS: Tuple[str, ...] = (
    "serverless/ninerouter-rectangle-mask/nuclio",
    "serverless/ninerouter-polygon-mask/nuclio",
    "serverless/ninerouter-polyline/nuclio",
    "serverless/ninerouter-human-pose-17/nuclio",
    "serverless/ninerouter-face-vf50/nuclio",
)

TARGET_MANIFESTS: Dict[str, Tuple[str, ...]] = {
    "serverless/ninerouter-rectangle-mask/nuclio": COMMON_CORE + THREE_DETECTOR_CORE,
    "serverless/ninerouter-polygon-mask/nuclio": COMMON_CORE + THREE_DETECTOR_CORE,
    "serverless/ninerouter-polyline/nuclio": COMMON_CORE + THREE_DETECTOR_CORE,
    "serverless/ninerouter-human-pose-17/nuclio": COMMON_CORE + WEEK2_CORE,
    "serverless/ninerouter-face-vf50/nuclio": COMMON_CORE + WEEK2_CORE,
}

TARGET_GROUPS: Dict[str, Tuple[str, ...]] = {
    "all": SERVERLESS_TARGETS,
    "three": (
        "serverless/ninerouter-rectangle-mask/nuclio",
        "serverless/ninerouter-polygon-mask/nuclio",
        "serverless/ninerouter-polyline/nuclio",
    ),
    "week2": (
        "serverless/ninerouter-human-pose-17/nuclio",
        "serverless/ninerouter-face-vf50/nuclio",
    ),
    "rectangle-mask": ("serverless/ninerouter-rectangle-mask/nuclio",),
    "polygon-mask": ("serverless/ninerouter-polygon-mask/nuclio",),
    "polyline": ("serverless/ninerouter-polyline/nuclio",),
    "human-pose-17": ("serverless/ninerouter-human-pose-17/nuclio",),
    "face-vf50": ("serverless/ninerouter-face-vf50/nuclio",),
}


def get_target_manifest(rel_target: str) -> Tuple[str, ...]:
    """Return the exact manifest tuple of repo-relative file paths for the given target directory."""
    norm_target = rel_target.replace("\\", "/").rstrip("/")
    if norm_target in TARGET_MANIFESTS:
        return TARGET_MANIFESTS[norm_target]
    if "human-pose-17" in norm_target or "face-vf50" in norm_target:
        return COMMON_CORE + WEEK2_CORE
    return COMMON_CORE + THREE_DETECTOR_CORE


def check_drift(repo_root: Path, target_dir: Path) -> List[Tuple[str, str]]:
    """Check for drift between repo root modules and a serverless target directory.

    Checks:
    - Missing files required by target manifest
    - Content mismatches between root and target files
    - Orphan / forbidden files present in target directory that are not in target manifest

    Returns:
        List of tuples (file_path, diff_type).
    """
    drifts: List[Tuple[str, str]] = []
    try:
        rel_target = target_dir.relative_to(repo_root).as_posix()
    except ValueError:
        rel_target = target_dir.as_posix()

    expected_files = set(get_target_manifest(rel_target))

    # 1. Check all expected files in manifest
    for rel_file in sorted(expected_files):
        root_file = repo_root / rel_file
        target_file = target_dir / rel_file

        if not root_file.exists():
            continue
        if not target_file.exists():
            drifts.append((rel_file, "missing_in_target"))
            continue

        try:
            r_norm = root_file.read_bytes().replace(b"\r\n", b"\n")
            t_norm = target_file.read_bytes().replace(b"\r\n", b"\n")
            if r_norm != t_norm:
                drifts.append((rel_file, "content_mismatch"))
        except Exception:
            drifts.append((rel_file, "content_mismatch"))

    # 2. Check for orphan / unneeded files in target directories (app, core, config)
    for sub in ("app", "core", "config"):
        target_sub = target_dir / sub
        if not target_sub.exists():
            continue
        for p in target_sub.rglob("*"):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                continue
            rel_file = p.relative_to(target_dir).as_posix()
            if rel_file not in expected_files:
                drifts.append((rel_file, "orphan_in_target"))

    return drifts


def sync_modules(repo_root: Path, target_dir: Path) -> int:
    """Copy only scoped manifest files into target serverless directory and prune unneeded files."""
    try:
        rel_target = target_dir.relative_to(repo_root).as_posix()
    except ValueError:
        rel_target = target_dir.as_posix()

    expected_files = set(get_target_manifest(rel_target))
    synced = 0

    # 1. Copy/update expected manifest files from repo root
    for rel_file in expected_files:
        src = repo_root / rel_file
        dst = target_dir / rel_file
        if not src.exists():
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        # Avoid touching file if already byte-for-byte identical
        if dst.exists():
            try:
                src_bytes = src.read_bytes().replace(b"\r\n", b"\n")
                dst_bytes = dst.read_bytes().replace(b"\r\n", b"\n")
                if src_bytes == dst_bytes:
                    synced += 1
                    continue
            except Exception:
                pass

        shutil.copy2(src, dst)
        synced += 1

    # 2. Prune orphan / unneeded files in target_dir under app, core, config
    for sub in ("app", "core", "config"):
        sub_dir = target_dir / sub
        if not sub_dir.exists():
            continue
        for p in list(sub_dir.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                continue
            rel_file = p.relative_to(target_dir).as_posix()
            if rel_file not in expected_files:
                p.unlink(missing_ok=True)

        # Remove empty directories
        for p in sorted(sub_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                try:
                    p.rmdir()
                except OSError:
                    pass

    return synced


def smoke_test_target(target_dir: Path) -> Tuple[bool, str]:
    """Run an isolated import smoke test in the build context of target_dir."""
    cmd = [
        sys.executable,
        "-c",
        (
            f"import sys; "
            f"sys.path.insert(0, r'{target_dir}'); "
            f"import model_handler; "
            f"import main; "
            f"print('SMOKE_TEST_OK')"
        ),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode == 0 and "SMOKE_TEST_OK" in proc.stdout:
            return True, ""
        return False, proc.stderr or proc.stdout
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync canonical modules to serverless directories")
    parser.add_argument("--check", action="store_true", help="Check for drift without modifying files")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=list(TARGET_GROUPS.keys()),
        help=(
            "Target group or detector to sync/check (default: 'all'). "
            "Use 'week2' for Pose17 and VF50 only, 'three' for Rectangle/Polygon/Polyline only."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run isolated import smoke tests for selected targets",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    selected_targets = TARGET_GROUPS.get(args.target, SERVERLESS_TARGETS)

    if args.smoke_test:
        print(f"Running import smoke tests for targets ('{args.target}')...")
        failed = False
        for rel_target in selected_targets:
            target_path = repo_root / rel_target
            if not target_path.exists():
                print(f"  [SKIP] {rel_target} (directory not found)")
                continue
            ok, err = smoke_test_target(target_path)
            if ok:
                print(f"  [PASS] {rel_target}")
            else:
                print(f"  [FAIL] {rel_target}: {err.strip()}")
                failed = True
        return 1 if failed else 0

    all_drifts = []
    for rel_target in selected_targets:
        target_path = repo_root / rel_target
        if not target_path.exists():
            continue

        drifts = check_drift(repo_root, target_path)
        if drifts:
            for d in drifts:
                all_drifts.append((rel_target, d[0], d[1]))

    if args.check:
        if all_drifts:
            print(f"[DRIFT DETECTED] Found {len(all_drifts)} differences between root and serverless copies:")
            for target, fpath, reason in all_drifts:
                print(f"  - {target}: {fpath} ({reason})")
            return 1
        print(f"[OK] Serverless modules ({args.target}) are in perfect sync with canonical root.")
        return 0

    # Sync
    total_synced = 0
    for rel_target in selected_targets:
        target_path = repo_root / rel_target
        if not target_path.exists():
            continue
        count = sync_modules(repo_root, target_path)
        total_synced += count
        print(f"Synced {count} files to {rel_target}")

    print(f"[COMPLETE] Synchronized canonical modules to '{args.target}' serverless targets ({total_synced} copies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
