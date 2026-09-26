"""Tests for Buddha Multi-Limb Contract and Spec Generation (core/buddha_contract.py).

Validates:
- Global, arm, and hand prompt templates.
- CVAT skeleton specs for buddha_arm (3 keypoints) and buddha_hand (21 keypoints).
- SVG validity, node ID sequences, and edge connections.
- Complete 10-label CVAT full specification integrity.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core.buddha_contract import (
    BuddhaArmInstance,
    BuddhaArmKeypoint,
    BuddhaHand21Instance,
    BuddhaHand21Keypoint,
    build_buddha_arm_refine_prompt,
    build_buddha_global_prompt,
    build_buddha_hand21_prompt,
    build_cvat_buddha_arm_spec,
    build_cvat_buddha_hand_spec,
    build_cvat_buddha_multilimbs_full_spec,
    compute_buddha_spec_fingerprint,
    validate_shapes_against_cvat_spec,
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

    def test_artwork_statue_mandate_in_prompts(self):
        """Verify all prompts instruct the model to detect depicted anatomy on statues and artwork."""
        global_prompt = build_buddha_global_prompt()
        assert "DOMAIN MANDATE" in global_prompt
        assert "statue" in global_prompt.lower()
        assert "sculpture" in global_prompt.lower()
        assert "artwork" in global_prompt.lower()
        assert "NOT a real-human-only detector" in global_prompt

        arm_prompt = build_buddha_arm_refine_prompt()
        assert "statue" in arm_prompt.lower()
        assert "sculpture" in arm_prompt.lower()
        assert "artwork" in arm_prompt.lower()

        hand_prompt = build_buddha_hand21_prompt()
        assert "statue" in hand_prompt.lower()
        assert "sculpture" in hand_prompt.lower()
        assert "mudra" in hand_prompt.lower()

    def test_anti_hallucination_rules_in_prompts(self):
        """Verify strict prohibitions against halo rays, lotus petals, and ornaments."""
        global_prompt = build_buddha_global_prompt()
        assert "ANTI-HALLUCINATION" in global_prompt
        assert "CROWNS" in global_prompt
        assert "LOTUS PETALS" in global_prompt
        assert "HALO RAYS" in global_prompt
        assert "ROBE FOLDS" in global_prompt


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


class TestBuddhaFingerprint:
    """Validate deterministic schema fingerprint."""

    def test_fingerprint_deterministic(self):
        fp1 = compute_buddha_spec_fingerprint()
        fp2 = compute_buddha_spec_fingerprint()
        assert len(fp1) == 64
        assert fp1 == fp2
        assert all(c in "0123456789abcdef" for c in fp1)

    def test_fingerprint_changes_on_modification(self):
        specs = build_cvat_buddha_multilimbs_full_spec()
        fp_orig = compute_buddha_spec_fingerprint(specs)

        modified = [s.copy() for s in specs]
        modified[0]["name"] = "altered_person"
        fp_mod = compute_buddha_spec_fingerprint(modified)

        assert fp_orig != fp_mod


class TestBuddhaSpecValidation:
    """Validate ModelHandler output shape compliance against CVAT spec."""

    def test_validate_valid_shapes(self):
        shapes = [
            {
                "type": "skeleton",
                "label": "person",
                "elements": [
                    {"type": "points", "label": str(i), "points": [500.0, 500.0], "occluded": False, "outside": False}
                    for i in range(1, 18)
                ],
            },
            {
                "type": "skeleton",
                "label": "buddha_arm",
                "elements": [
                    {"type": "points", "label": "root", "points": [400.0, 300.0]},
                    {"type": "points", "label": "elbow", "points": [350.0, 250.0]},
                    {"type": "points", "label": "wrist", "points": [300.0, 200.0]},
                ],
            },
            {
                "type": "skeleton",
                "label": "buddha_hand",
                "elements": [
                    {"type": "points", "label": "wrist", "points": [300.0, 200.0]},
                    {"type": "points", "label": "thumb_cmc", "points": [305.0, 195.0]},
                ],
            },
        ]
        is_valid, errors = validate_shapes_against_cvat_spec(shapes)
        assert is_valid, f"Expected valid, got: {errors}"
        assert len(errors) == 0

    def test_validate_non_skeleton_rejected(self):
        shapes = [{"type": "polygon", "label": "person", "points": [10, 10, 20, 20]}]
        is_valid, errors = validate_shapes_against_cvat_spec(shapes)
        assert not is_valid
        assert any("must have type 'skeleton'" in e for e in errors)

    def test_validate_unknown_parent_rejected(self):
        shapes = [{"type": "skeleton", "label": "unknown_body_part", "elements": []}]
        is_valid, errors = validate_shapes_against_cvat_spec(shapes)
        assert not is_valid
        assert any("invalid parent label" in e for e in errors)

    def test_validate_semantic_sublabel_rejected_for_person(self):
        """Reproduce Task 22 bug: 'person' elements with semantic names fail spec validation."""
        shapes = [
            {
                "type": "skeleton",
                "label": "person",
                "elements": [
                    {"type": "points", "label": "nose", "points": [500.0, 500.0]}
                ],
            }
        ]
        is_valid, errors = validate_shapes_against_cvat_spec(shapes)
        assert not is_valid
        assert any("unknown sublabel 'nose'" in e for e in errors)

    def test_validate_invalid_points(self):
        shapes = [
            {
                "type": "skeleton",
                "label": "buddha_arm",
                "elements": [
                    {"type": "points", "label": "root", "points": [float("nan"), 300.0]}
                ],
            }
        ]
        is_valid, errors = validate_shapes_against_cvat_spec(shapes)
        assert not is_valid
        assert any("NaN" in e for e in errors)


class TestBuddhaDrift:
    """Validate zero-drift between canonical spec, function.yaml, and raw JSON."""

    def test_function_yaml_drift(self):
        yaml_path = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-buddha-multilimbs" / "nuclio" / "function.yaml"
        assert yaml_path.exists(), f"Missing {yaml_path}"

        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        spec_raw = content["metadata"]["annotations"]["spec"]
        yaml_specs = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw

        canonical_specs = build_cvat_buddha_multilimbs_full_spec()

        assert len(yaml_specs) == len(canonical_specs)
        for y_lbl, c_lbl in zip(yaml_specs, canonical_specs):
            assert y_lbl["name"] == c_lbl["name"]
            assert y_lbl["type"] == c_lbl["type"]
            assert len(y_lbl["sublabels"]) == len(c_lbl["sublabels"])
            y_subnames = [s["name"] for s in y_lbl["sublabels"]]
            c_subnames = [s["name"] for s in c_lbl["sublabels"]]
            assert y_subnames == c_subnames

    def test_raw_json_drift(self):
        json_path = Path(__file__).resolve().parent.parent / "config" / "buddha_cvat_labels.raw.json"
        assert json_path.exists(), f"Missing {json_path}"

        raw_specs = json.loads(json_path.read_text(encoding="utf-8"))
        canonical_specs = build_cvat_buddha_multilimbs_full_spec()

        assert raw_specs == canonical_specs


class TestTask22MismatchReproduction:
    """Validate recursive inspector detects Task 22 divergence and generates mapping plan."""

    def test_nested_task_mismatch_detected(self):
        from scripts.buddha_schema import inspect_task_compatibility

        canonical_specs = build_cvat_buddha_multilimbs_full_spec()
        comp = inspect_task_compatibility(task_id=22, specs=canonical_specs)

        if comp.get("error") and "connect" in comp["error"].lower():
            pytest.skip("CVAT container or API not accessible in this environment")

        assert comp.get("compatible") is False
        sublabel_issues = comp.get("sublabel_issues", [])
        assert len(sublabel_issues) >= 1

        person_issue = next((iss for iss in sublabel_issues if iss["parent"] == "person"), None)
        assert person_issue is not None
        assert "nose" in person_issue["task_sublabels"]
        assert "1" in person_issue["model_sublabels"]
