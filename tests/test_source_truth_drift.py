"""Test to detect drift between canonical root modules and serverless build copies.

Ensures that root app/, core/, and config/ directories remain identical
with their deployment copies in serverless/ninerouter-vision-31/nuclio/.
"""

from __future__ import annotations

from pathlib import Path

from scripts.sync_serverless_modules import SERVERLESS_TARGETS, check_drift


def test_no_drift_between_root_and_active_serverless_targets():
    """Verify that all active serverless targets have zero drift from canonical root."""
    repo_root = Path(__file__).resolve().parent.parent
    for rel_target in SERVERLESS_TARGETS:
        target = repo_root / rel_target
        assert target.exists(), f"Target directory {target} does not exist"
        drifts = check_drift(repo_root, target)
        assert not drifts, f"Found drift between root and {rel_target}: {drifts}"

