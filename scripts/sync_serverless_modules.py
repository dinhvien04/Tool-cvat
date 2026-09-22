"""Synchronize canonical root modules (app, core, config) into serverless build contexts.

Root is canonical development source. Serverless requires local copies for Docker build context.
This script ensures zero drift between root and serverless copies.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

MODULES_TO_SYNC = ("app", "core", "config")
# Only the three active user-facing detectors are synchronized.
# Historical serverless folders remain in git for reference but are no longer deploy/sync targets.
SERVERLESS_TARGETS = (
    "serverless/ninerouter-rectangle-mask/nuclio",
    "serverless/ninerouter-polygon-mask/nuclio",
    "serverless/ninerouter-polyline/nuclio",
    "serverless/ninerouter-human-pose-17/nuclio",
    "serverless/ninerouter-face-vf50/nuclio",
)


def ignore_patterns(path: str, names: List[str]) -> List[str]:
    """Filter out caches and temporary files."""
    ignored = []
    for name in names:
        if name == "__pycache__" or name.endswith(".pyc") or name.endswith(".pyo"):
            ignored.append(name)
    return ignored


def check_drift(repo_root: Path, target_dir: Path) -> List[Tuple[str, str]]:
    """Check for drift between repo root modules and a serverless target directory.

    Returns:
        List of tuples (file_path, diff_type).
    """
    drifts: List[Tuple[str, str]] = []

    for mod in MODULES_TO_SYNC:
        root_mod = repo_root / mod
        target_mod = target_dir / mod

        if not root_mod.exists():
            continue
        if not target_mod.exists():
            drifts.append((str(target_mod), "missing_directory"))
            continue

        # Check all files in root_mod
        for p in root_mod.rglob("*"):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                continue
            rel_path = p.relative_to(root_mod)
            target_file = target_mod / rel_path

            if not target_file.exists():
                drifts.append((f"{mod}/{rel_path}", "missing_in_target"))
            elif not filecmp.cmp(p, target_file, shallow=False):
                # Fallback to normalized content comparison to prevent spurious CRLF/LF drift on Windows
                try:
                    p_norm = p.read_bytes().replace(b"\r\n", b"\n")
                    t_norm = target_file.read_bytes().replace(b"\r\n", b"\n")
                    if p_norm != t_norm:
                        drifts.append((f"{mod}/{rel_path}", "content_mismatch"))
                except Exception:
                    drifts.append((f"{mod}/{rel_path}", "content_mismatch"))

        # Check for orphan files in target_mod
        for p in target_mod.rglob("*"):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                continue
            rel_path = p.relative_to(target_mod)
            root_file = root_mod / rel_path
            if not root_file.exists():
                drifts.append((f"{mod}/{rel_path}", "orphan_in_target"))

    return drifts


def sync_modules(repo_root: Path, target_dir: Path) -> int:
    """Copy canonical root modules into target serverless directory."""
    synced = 0
    for mod in MODULES_TO_SYNC:
        src = repo_root / mod
        dst = target_dir / mod
        if not src.exists():
            continue

        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore_patterns)
        synced += 1
    return synced


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync canonical modules to serverless directories")
    parser.add_argument("--check", action="store_true", help="Check for drift without modifying files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    all_drifts = []
    for rel_target in SERVERLESS_TARGETS:
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
        print("[OK] All serverless modules are in perfect sync with canonical root.")
        return 0

    # Sync
    total_synced = 0
    for rel_target in SERVERLESS_TARGETS:
        target_path = repo_root / rel_target
        if not target_path.exists():
            continue
        count = sync_modules(repo_root, target_path)
        total_synced += count
        print(f"Synced {count} modules to {rel_target}")

    print(f"[COMPLETE] Synchronized canonical modules across serverless targets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
