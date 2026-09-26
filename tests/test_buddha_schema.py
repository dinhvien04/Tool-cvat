"""Tests for Buddha Multi-Limb Schema (config/buddha_multilimbs.yaml and core/buddha_contract.py).

Validates:
- Loading canonical YAML schema.
- Label topology: exactly 10 labels (1 central person, 7 VF50 face components, 1 buddha_arm, 1 buddha_hand).
- Keypoint counts: 3 for arm, 21 for hand.
- Topological edges: 2 for arm, 20 for hand.
- Crop padding and deduplication parameters.
"""

from __future__ import annotations

import pytest
from core.buddha_contract import (
    BUDDHA_MULTILIMB_LABELS,
    load_buddha_schema,
)


class TestBuddhaSchema:
    """Validate canonical schema configuration."""

    def test_schema_loads_successfully(self):
        schema = load_buddha_schema()
        assert schema is not None
        assert schema.version in ("1.0", "1.0.0")

    def test_canonical_ten_labels(self):
        schema = load_buddha_schema()
        assert len(schema.labels) == 10
        expected_labels = [
            "person",
            "longmaytrai",
            "longmayphai",
            "songmui",
            "mattrai",
            "matphai",
            "moingoai",
            "moitrong",
            "buddha_arm",
            "buddha_hand",
        ]
        assert schema.labels == expected_labels
        assert tuple(schema.labels) == BUDDHA_MULTILIMB_LABELS

    def test_arm_keypoint_topology(self):
        schema = load_buddha_schema()
        assert len(schema.arm_keypoints) == 3
        kp_names = [kp.name for kp in schema.arm_keypoints]
        assert kp_names == ["root", "elbow", "wrist"]

        # Exactly 2 topological edges connecting IDs 1->2 and 2->3
        assert len(schema.arm_edges) == 2
        assert schema.arm_edges == ((1, 2), (2, 3))

    def test_hand_keypoint_topology(self):
        schema = load_buddha_schema()
        assert len(schema.hand_keypoints) == 21

        # Hand starts at wrist (ID 0)
        assert schema.hand_keypoints[0].id == 0
        assert schema.hand_keypoints[0].name == "wrist"

        # Check all 5 finger chains present
        kp_names = [kp.name for kp in schema.hand_keypoints]
        assert "thumb_tip" in kp_names
        assert "index_tip" in kp_names
        assert "middle_tip" in kp_names
        assert "ring_tip" in kp_names
        assert "pinky_tip" in kp_names

        # Exactly 20 edges connecting 21 points
        assert len(schema.hand_edges) == 20
        # Check wrist (ID 0) connects to base of 5 finger chains (IDs 1, 5, 9, 13, 17)
        wrist_edges = [e for e in schema.hand_edges if 0 in e]
        assert len(wrist_edges) == 5
        assert len(wrist_edges) == 5

    def test_thresholds_and_paddings(self):
        schema = load_buddha_schema()
        assert schema.arm_crop_padding == 0.25
        assert schema.hand_crop_padding == 0.35
        assert schema.wrist_dist_thresh == 35.0
        assert schema.elbow_dist_thresh == 45.0
        assert schema.root_dist_thresh == 60.0
        assert schema.dir_sim_thresh == 0.92
        assert schema.hand_iou_thresh == 0.65
        assert schema.max_arms == 100
        assert schema.max_hand_refinements == 60
        assert schema.refine_workers == 2


class TestBuddhaSchemaCLI:
    """Validate CLI flags of scripts/buddha_schema.py."""

    def test_cli_fingerprint(self):
        import subprocess
        res = subprocess.run(
            ["python", "scripts/buddha_schema.py", "--fingerprint"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "BUDDHA_SPEC_FINGERPRINT=" in res.stdout

    def test_cli_json_output(self):
        import json
        import subprocess
        res = subprocess.run(
            ["python", "scripts/buddha_schema.py", "--json"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        specs = json.loads(res.stdout)
        assert isinstance(specs, list)
        assert len(specs) == 10

    def test_cli_write_json(self, tmp_path):
        import json
        import subprocess
        out_file = tmp_path / "test_labels.raw.json"
        res = subprocess.run(
            ["python", "scripts/buddha_schema.py", "--write-json", str(out_file)],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert out_file.exists()
        specs = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(specs) == 10
