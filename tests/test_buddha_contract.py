"""Tests for Buddha Multi-Limb Contract and Spec Generation (core/buddha_contract.py).

Validates:
- Global, arm, and hand prompt templates.
- CVAT skeleton specs for buddha_arm (3 keypoints) and buddha_hand (21 keypoints).
- SVG validity, node ID sequences, and edge connections.
- Complete 10-label CVAT full specification integrity.
"""

from __future__ import annotations

import re
import pytest
from core.buddha_contract import (
    build_buddha_arm_refine_prompt,
    build_buddha_global_prompt,
    build_buddha_hand21_prompt,
    build_cvat_buddha_arm_spec,
    build_cvat_buddha_hand_spec,
    build_cvat_buddha_multilimbs_full_spec,
)
from core.pose_face_schema import validate_svg_node_ids


class TestBuddhaPrompts:
    """Validate prompt templates contain required topological guidance."""

    def test_global_prompt_content(self):
        prompt = build_buddha_global_prompt()
        assert "central_body" in prompt
        assert "face_roi" in prompt
        assert "torso_center" in prompt
        assert "arms" in prompt
        assert "root" in prompt
        assert "wrist" in prompt

    def test_arm_refine_prompt_content(self):
        prompt = build_buddha_arm_refine_prompt()
        assert "root" in prompt
        assert "elbow" in prompt
        assert "wrist" in prompt
        assert "[x, y, visibility]" in prompt

    def test_hand_refine_prompt_content(self):
        prompt = build_buddha_hand21_prompt()
        assert "21 2D hand landmark coordinates" in prompt
        assert "wrist" in prompt
        assert "thumb_tip" in prompt
        assert "index_tip" in prompt
        assert "pinky_tip" in prompt


class TestBuddhaCvatSpecs:
    """Validate CVAT skeleton specification builders."""

    def test_buddha_arm_spec(self):
        spec = build_cvat_buddha_arm_spec(label_id=9)
        assert spec["id"] == 9
        assert spec["name"] == "buddha_arm"
        assert spec["type"] == "skeleton"

        sublabels = spec["sublabels"]
        assert len(sublabels) == 3
        sub_names = [s["name"] for s in sublabels]
        assert sub_names == ["root", "elbow", "wrist"]
        for sub in sublabels:
            assert sub["type"] == "points"

        svg = spec["svg"]
        assert svg.startswith("<svg")
        # Validate node IDs 1..3
        is_valid, errs = validate_svg_node_ids(svg, expected_node_ids={1, 2, 3})
        assert is_valid, f"Arm SVG invalid: {errs}"

        # 3 circles and 2 lines
        circles = re.findall(r"<circle\b", svg)
        lines = re.findall(r"<line\b", svg)
        assert len(circles) == 3
        assert len(lines) == 2

    def test_buddha_hand_spec(self):
        spec = build_cvat_buddha_hand_spec(label_id=10)
        assert spec["id"] == 10
        assert spec["name"] == "buddha_hand"
        assert spec["type"] == "skeleton"

        sublabels = spec["sublabels"]
        assert len(sublabels) == 21
        assert sublabels[0]["name"] == "wrist"
        for sub in sublabels:
            assert sub["type"] == "points"

        svg = spec["svg"]
        assert svg.startswith("<svg")
        # Validate node IDs 1..21
        is_valid, errs = validate_svg_node_ids(svg, expected_node_ids=set(range(1, 22)))
        assert is_valid, f"Hand SVG invalid: {errs}"

        # 21 circles and 20 lines
        circles = re.findall(r"<circle\b", svg)
        lines = re.findall(r"<line\b", svg)
        assert len(circles) == 21
        assert len(lines) == 20

    def test_full_buddha_multilimbs_spec(self):
        specs = build_cvat_buddha_multilimbs_full_spec()
        assert len(specs) == 10

        ids = [s["id"] for s in specs]
        assert ids == list(range(1, 11)), "IDs must be sequential from 1 to 10"

        names = [s["name"] for s in specs]
        assert names[0] == "person"
        assert names[1:8] == [
            "longmaytrai",
            "longmayphai",
            "songmui",
            "mattrai",
            "matphai",
            "moingoai",
            "moitrong",
        ]
        assert names[8] == "buddha_arm"
        assert names[9] == "buddha_hand"

        for s in specs:
            assert s["type"] == "skeleton"
            assert "sublabels" in s
            assert len(s["sublabels"]) > 0
            assert "svg" in s
            assert s["svg"].startswith("<svg")
