"""Test to detect drift between canonical root modules and serverless build copies.

Ensures that root app/, core/, and config/ directories remain identical
with their deployment copies using target-scoped manifests:
- COMMON_CORE: modules needed by all functions
- WEEK2_CORE: modules needed only by Pose17 and VF50
- THREE_DETECTOR_CORE: modules needed by Rectangle/Polygon/Polyline
"""

from __future__ import annotations

from pathlib import Path
import pytest

from scripts.sync_serverless_modules import (
    COMMON_CORE,
    SERVERLESS_TARGETS,
    TARGET_GROUPS,
    TARGET_MANIFESTS,
    THREE_DETECTOR_CORE,
    WEEK2_CORE,
    check_drift,
    get_target_manifest,
    smoke_test_target,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

THREE_TARGETS = (
    "serverless/ninerouter-rectangle-mask/nuclio",
    "serverless/ninerouter-polygon-mask/nuclio",
    "serverless/ninerouter-polyline/nuclio",
)

WEEK2_TARGETS = (
    "serverless/ninerouter-human-pose-17/nuclio",
    "serverless/ninerouter-face-vf50/nuclio",
)


def test_no_drift_between_root_and_active_serverless_targets():
    """Verify that all active serverless targets have zero drift from canonical root."""
    for rel_target in SERVERLESS_TARGETS:
        target = REPO_ROOT / rel_target
        assert target.exists(), f"Target directory {target} does not exist"
        drifts = check_drift(REPO_ROOT, target)
        assert not drifts, f"Found drift between root and {rel_target}: {drifts}"


def test_three_detectors_exclude_week2_core():
    """Verify that Week 2 specific files are NOT copied to Rectangle, Polygon, or Polyline."""
    for rel_target in THREE_TARGETS:
        target_dir = REPO_ROOT / rel_target
        for rel_week2_file in WEEK2_CORE:
            forbidden_path = target_dir / rel_week2_file
            assert not forbidden_path.exists(), (
                f"Unwanted Week 2 file '{rel_week2_file}' found in three-detector target: {rel_target}"
            )


def test_week2_detectors_include_week2_core():
    """Verify that Week 2 detectors contain all required WEEK2_CORE modules and configs."""
    for rel_target in WEEK2_TARGETS:
        target_dir = REPO_ROOT / rel_target
        for rel_week2_file in WEEK2_CORE:
            expected_path = target_dir / rel_week2_file
            assert expected_path.exists(), (
                f"Required Week 2 file '{rel_week2_file}' missing in week2 target: {rel_target}"
            )


def test_target_manifest_scoping():
    """Verify manifest definitions for each target group."""
    for rel_target in THREE_TARGETS:
        manifest = get_target_manifest(rel_target)
        for w2 in WEEK2_CORE:
            assert w2 not in manifest, f"{w2} should not be in manifest for {rel_target}"
        for common in COMMON_CORE:
            assert common in manifest, f"{common} must be in manifest for {rel_target}"
        for three in THREE_DETECTOR_CORE:
            assert three in manifest, f"{three} must be in manifest for {rel_target}"

    for rel_target in WEEK2_TARGETS:
        manifest = get_target_manifest(rel_target)
        for w2 in WEEK2_CORE:
            assert w2 in manifest, f"{w2} must be in manifest for {rel_target}"
        for common in COMMON_CORE:
            assert common in manifest, f"{common} must be in manifest for {rel_target}"
        for three in THREE_DETECTOR_CORE:
            assert three not in manifest, f"{three} should not be in manifest for {rel_target}"


def test_drift_detection_catches_unwanted_files():
    """Verify that check_drift detects orphan/unneeded files added to a target."""
    rect_target = REPO_ROOT / "serverless/ninerouter-rectangle-mask/nuclio"
    dummy_orphan = rect_target / "core" / "unwanted_dummy_file.py"
    try:
        dummy_orphan.write_text("# dummy orphan", encoding="utf-8")
        assert dummy_orphan.is_file()
        drifts = check_drift(REPO_ROOT, rect_target)
        orphan_drifts = [d for d in drifts if "unwanted_dummy_file.py" in d[0] and d[1] == "orphan_in_target"]
        assert len(orphan_drifts) == 1, f"Expected orphan drift detection, got: {drifts}"
    finally:
        dummy_orphan.unlink(missing_ok=True)


@pytest.mark.parametrize("rel_target", SERVERLESS_TARGETS)
def test_isolated_import_smoke_tests(rel_target: str):
    """Verify that each serverless function imports cleanly in its isolated build context."""
    target_dir = REPO_ROOT / rel_target
    assert target_dir.exists(), f"Target directory {target_dir} does not exist"
    ok, err = smoke_test_target(target_dir)
    assert ok, f"Import smoke test failed for {rel_target}: {err}"
