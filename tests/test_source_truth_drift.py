"""Test to detect drift between canonical root modules and serverless build copies.

Ensures that root app/, core/, and config/ directories remain identical
with their deployment copies in serverless/ninerouter-vision-31/nuclio/.
"""

from __future__ import annotations

from pathlib import Path

from scripts.sync_serverless_modules import check_drift


def test_no_drift_between_root_and_ninerouter_vision_31():
    """Verify that serverless/ninerouter-vision-31/nuclio has zero drift from root."""
    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "serverless" / "ninerouter-vision-31" / "nuclio"
    assert target.exists(), f"Target directory {target} does not exist"

    drifts = check_drift(repo_root, target)
    assert not drifts, f"Found drift between root and serverless copies: {drifts}"
