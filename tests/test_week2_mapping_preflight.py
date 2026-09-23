"""Tests for CVAT 2.75.1 Skeleton Mapping Preflight Validator (scripts/week2_mapping_preflight.py).

Validates:
1. Pose17 sublabel mapping:
   - Case A: Semantic sublabels ("nose", "right_eye", ..., "right_ankle").
   - Case B: Numeric sublabels ("1", "2", ..., "17").
   - Case C: Uppercase/normalized aliases ("NOSE", "RIGHT_EYE", "R Eye", etc.).
2. Completeness validation:
   - Pose17: Strictly require 17/17 matched sublabels; <=16 rejected as INCOMPLETE_POSE_MAPPING.
   - VF50: Strictly require 7/7 component skeletons and 50/50 matched sublabels;
     <7 components or <50 sublabels rejected as INCOMPLETE_VF50_MAPPING.
   - Legacy monolithic "face" skeleton (5-point or 50-point) cleanly diagnosed
     and rejected as INCOMPATIBLE_FULL_VF50.
3. CVAT 2.75.1 backend simulation (validate_cvat_mapping):
   - Missing or empty "sublabels" triggers CRITICAL CVAT 2.75.1 ERROR.
   - Target sublabel mismatch and duplicate target detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.pose_face_schema import POSE17_KEYPOINTS
from core.skeleton_contract import VF50_COMPONENT_CONFIG, VF50_COMPONENT_NAMES
from scripts.week2_mapping_preflight import (
    POSE17_KEYPOINT_SPECS,
    build_compatible_mapping_payload,
    get_model_spec,
    normalize_sublabel_name,
    validate_cvat_mapping,
)


def _make_pose17_task_labels(sublabel_names: List[str], label_name: str = "person") -> List[Dict[str, Any]]:
    """Helper to construct CVAT task labels with a skeleton."""
    return [
        {
            "id": 100,
            "name": label_name,
            "type": "skeleton",
            "attributes": [],
            "sublabels": [
                {"id": idx + 1, "name": name, "type": "points", "attributes": []}
                for idx, name in enumerate(sublabel_names)
            ],
        }
    ]


def _make_canonical_vf50_task_labels() -> List[Dict[str, Any]]:
    """Helper to construct all 7 canonical VF50 component skeletons (50 sublabels)."""
    labels = []
    lid = 1
    for cname in VF50_COMPONENT_NAMES:
        cfg = VF50_COMPONENT_CONFIG[cname]
        subs = []
        for sid in range(cfg["start"], cfg["end"] + 1):
            subs.append({"id": sid + 1, "name": str(sid), "type": "points", "attributes": []})
        labels.append({
            "id": lid,
            "name": cname,
            "type": "skeleton",
            "attributes": [],
            "sublabels": subs,
        })
        lid += 1
    return labels


class TestPose17SublabelMapping:
    """Validate Pose17 sublabel mapping across Case A, Case B, and Case C."""

    def test_case_a_semantic_sublabels(self):
        """Case A: exact semantic COCO sublabels ('nose', 'right_eye', ..., 'right_ankle')."""
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-human-pose-17", task_labels)

        assert payload is not None, f"Expected valid payload, got None. Diagnostics: {diagnostics}"
        mapping = payload["mapping"]
        assert "person" in mapping
        person_map = mapping["person"]
        assert person_map["name"] == "person"
        assert "sublabels" in person_map
        assert len(person_map["sublabels"]) == 17

        for kp in POSE17_KEYPOINTS:
            assert kp in person_map["sublabels"]
            assert person_map["sublabels"][kp]["name"] == kp

        assert any("Case A" in d for d in diagnostics)

        # Validate against CVAT 2.75.1 backend rules
        model_spec = get_model_spec("ninerouter-human-pose-17")
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert is_valid, f"CVAT validation errors: {errors}"
        assert len(errors) == 0

    def test_case_b_numeric_sublabels(self):
        """Case B: numeric sublabels ('1', '2', ..., '17')."""
        task_labels = _make_pose17_task_labels([str(i) for i in range(1, 18)])
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-human-pose-17", task_labels)

        assert payload is not None, f"Expected valid payload, got None. Diagnostics: {diagnostics}"
        mapping = payload["mapping"]
        assert "person" in mapping
        person_map = mapping["person"]
        assert "sublabels" in person_map
        assert len(person_map["sublabels"]) == 17

        for idx, coco_name, _ in POSE17_KEYPOINT_SPECS:
            assert coco_name in person_map["sublabels"]
            assert person_map["sublabels"][coco_name]["name"] == str(idx)

        assert any("Case B" in d for d in diagnostics)

        # Validate against CVAT 2.75.1 backend rules
        model_spec = get_model_spec("ninerouter-human-pose-17")
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert is_valid, f"CVAT validation errors: {errors}"
        assert len(errors) == 0

    def test_case_c_uppercase_and_normalized_aliases(self):
        """Case C: uppercase/normalized aliases ('NOSE', 'R Eye', 'L_Ear', etc.)."""
        aliases = [
            "NOSE",
            "R Eye",
            "left_eye",
            "r_ear",
            "L_EAR",
            "Right_Shoulder",
            "left-shoulder",
            "R_Elbow",
            "LEFT_ELBOW",
            "right-wrist",
            "L Wrist",
            "RIGHT_HIP",
            "Left_Hip",
            "r_knee",
            "LEFT_KNEE",
            "right_ankle",
            "L_ANKLE",
        ]
        assert len(aliases) == 17
        task_labels = _make_pose17_task_labels(aliases)
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-human-pose-17", task_labels)

        assert payload is not None, f"Expected valid payload, got None. Diagnostics: {diagnostics}"
        mapping = payload["mapping"]
        assert "person" in mapping
        person_map = mapping["person"]
        assert "sublabels" in person_map
        assert len(person_map["sublabels"]) == 17

        assert person_map["sublabels"]["nose"]["name"] == "NOSE"
        assert person_map["sublabels"]["right_eye"]["name"] == "R Eye"
        assert person_map["sublabels"]["right_ear"]["name"] == "r_ear"
        assert person_map["sublabels"]["left_ear"]["name"] == "L_EAR"
        assert person_map["sublabels"]["left_shoulder"]["name"] == "left-shoulder"
        assert person_map["sublabels"]["left_wrist"]["name"] == "L Wrist"

        assert any("Case C" in d for d in diagnostics)

        # Validate against CVAT 2.75.1 backend rules
        model_spec = get_model_spec("ninerouter-human-pose-17")
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert is_valid, f"CVAT validation errors: {errors}"
        assert len(errors) == 0

    def test_case_c_r_eye_and_l_eye_normalization(self):
        """Case C: explicitly tests 'R Eye' and 'L Eye' normalization with dot and space variants."""
        aliases = [
            "NOSE",
            "R Eye",
            "L Eye",
            "r.ear",
            "l.ear",
            "Right_Shoulder",
            "left-shoulder",
            "R_Elbow",
            "LEFT_ELBOW",
            "right-wrist",
            "L Wrist",
            "RIGHT_HIP",
            "Left_Hip",
            "r_knee",
            "LEFT_KNEE",
            "right_ankle",
            "L_ANKLE",
        ]
        task_labels = _make_pose17_task_labels(aliases)
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-human-pose-17", task_labels)
        assert payload is not None
        person_map = payload["mapping"]["person"]
        assert person_map["sublabels"]["right_eye"]["name"] == "R Eye"
        assert person_map["sublabels"]["left_eye"]["name"] == "L Eye"
        assert person_map["sublabels"]["right_ear"]["name"] == "r.ear"
        assert person_map["sublabels"]["left_ear"]["name"] == "l.ear"


class TestPose17CompletenessValidation:
    """Validate strict 17/17 completeness gate for Pose17."""

    def test_pose17_16_of_17_rejected(self):
        """16/17 sublabels must be rejected with INCOMPLETE_POSE_MAPPING."""
        # Omit 'left_ankle'
        sublabels_16 = list(POSE17_KEYPOINTS[:-1])
        assert len(sublabels_16) == 16
        task_labels = _make_pose17_task_labels(sublabels_16)

        payload, diagnostics = build_compatible_mapping_payload("ninerouter-human-pose-17", task_labels)
        assert payload is None, "Should reject incomplete 16/17 mapping"
        assert any("[INCOMPLETE_POSE_MAPPING]" in d for d in diagnostics)
        assert any("16/17 matched" in d for d in diagnostics)
        assert any("left_ankle" in d for d in diagnostics)

    def test_pose17_backend_validation_rejects_incomplete_mapping(self):
        """CVAT backend validator rejects if mapping contains < 17 sublabels."""
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))
        model_spec = get_model_spec("ninerouter-human-pose-17")

        # Manually create partial mapping with 16 sublabels
        partial_mapping = {
            "person": {
                "name": "person",
                "attributes": {},
                "sublabels": {kp: {"name": kp, "attributes": {}} for kp in POSE17_KEYPOINTS[:-1]},
            }
        }
        is_valid, errors = validate_cvat_mapping(partial_mapping, model_spec, task_labels)
        assert not is_valid
        assert any("[INCOMPLETE_POSE_MAPPING]" in e for e in errors)
        assert any("left_ankle" in e for e in errors)


class TestVF50CompletenessValidation:
    """Validate strict 7/7 components and 50/50 sublabels for VF50."""

    def test_vf50_canonical_success(self):
        """Full 7 components and 50 sublabels should succeed."""
        task_labels = _make_canonical_vf50_task_labels()
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-face-vf50", task_labels)

        assert payload is not None, f"Expected valid payload, got None. Diagnostics: {diagnostics}"
        mapping = payload["mapping"]
        assert len(mapping) == 7
        total_subs = sum(len(m["sublabels"]) for m in mapping.values())
        assert total_subs == 50

        # Validate against CVAT 2.75.1 backend rules
        model_spec = get_model_spec("ninerouter-face-vf50")
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert is_valid, f"CVAT validation errors: {errors}"
        assert len(errors) == 0

    def test_vf50_6_of_7_components_rejected(self):
        """6/7 components must be rejected with INCOMPLETE_VF50_MAPPING."""
        task_labels = _make_canonical_vf50_task_labels()
        # Remove component 'moitrong' (42..49)
        task_labels_6 = [l for l in task_labels if l["name"] != "moitrong"]
        assert len(task_labels_6) == 6

        payload, diagnostics = build_compatible_mapping_payload("ninerouter-face-vf50", task_labels_6)
        assert payload is None, "Should reject incomplete 6/7 component mapping"
        assert any("[INCOMPLETE_VF50_MAPPING]" in d for d in diagnostics)
        assert any("6/7" in d for d in diagnostics)
        assert any("moitrong" in d for d in diagnostics)

    def test_vf50_49_of_50_sublabels_rejected(self):
        """7/7 components but 49/50 sublabels must be rejected with INCOMPLETE_VF50_MAPPING."""
        task_labels = _make_canonical_vf50_task_labels()
        # Drop point '49' from 'moitrong'
        for l in task_labels:
            if l["name"] == "moitrong":
                l["sublabels"] = [s for s in l["sublabels"] if s["name"] != "49"]

        payload, diagnostics = build_compatible_mapping_payload("ninerouter-face-vf50", task_labels)
        assert payload is None, "Should reject incomplete 49/50 sublabel mapping"
        assert any("[INCOMPLETE_VF50_MAPPING]" in d for d in diagnostics)
        assert any("49/50" in d for d in diagnostics)
        assert any("moitrong" in d for d in diagnostics)

    def test_vf50_backend_validation_rejects_missing_component(self):
        """CVAT backend validator rejects if a component skeleton is missing from mapping."""
        task_labels = _make_canonical_vf50_task_labels()
        model_spec = get_model_spec("ninerouter-face-vf50")

        # Create mapping missing 'moitrong'
        mapping_6 = {}
        for cname in VF50_COMPONENT_NAMES[:-1]:
            cfg = VF50_COMPONENT_CONFIG[cname]
            mapping_6[cname] = {
                "name": cname,
                "attributes": {},
                "sublabels": {str(i): {"name": str(i), "attributes": {}} for i in range(cfg["start"], cfg["end"] + 1)},
            }

        is_valid, errors = validate_cvat_mapping(mapping_6, model_spec, task_labels)
        assert not is_valid
        assert any("[INCOMPLETE_VF50_MAPPING]" in e for e in errors)
        assert any("moitrong" in e for e in errors)


class TestMonolithicFaceRejection:
    """Validate rejection of legacy monolithic face skeletons."""

    def test_5_point_monolithic_face_rejected(self):
        """Legacy 5-point face skeleton must be cleanly diagnosed as INCOMPATIBLE_FULL_VF50."""
        task_labels = [
            {
                "id": 1,
                "name": "face",
                "type": "skeleton",
                "attributes": [],
                "sublabels": [
                    {"id": 1, "name": "right_eye", "type": "points", "attributes": []},
                    {"id": 2, "name": "left_eye", "type": "points", "attributes": []},
                    {"id": 3, "name": "nose", "type": "points", "attributes": []},
                    {"id": 4, "name": "right_mouth", "type": "points", "attributes": []},
                    {"id": 5, "name": "left_mouth", "type": "points", "attributes": []},
                ],
            }
        ]
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-face-vf50", task_labels)
        assert payload is None
        assert any("[INCOMPATIBLE_FULL_VF50]" in d for d in diagnostics)
        assert any("5 sublabels" in d for d in diagnostics)
        assert any("Recommended action" in d for d in diagnostics)

    def test_50_point_monolithic_face_rejected(self):
        """50-point single monolithic skeleton 'face' must be rejected as INCOMPATIBLE_FULL_VF50."""
        task_labels = [
            {
                "id": 1,
                "name": "face",
                "type": "skeleton",
                "attributes": [],
                "sublabels": [
                    {"id": i + 1, "name": str(i), "type": "points", "attributes": []}
                    for i in range(50)
                ],
            }
        ]
        payload, diagnostics = build_compatible_mapping_payload("ninerouter-face-vf50", task_labels)
        assert payload is None
        assert any("[INCOMPATIBLE_FULL_VF50]" in d for d in diagnostics)
        assert any("50 sublabels" in d for d in diagnostics)
        assert any("cannot merge multiple model skeletons" in d for d in diagnostics)

    def test_backend_validation_rejects_face_mapping_for_vf50(self):
        """Backend validator flags monolithic face mapping as INCOMPATIBLE_FULL_VF50."""
        task_labels = [
            {
                "id": 1,
                "name": "face",
                "type": "skeleton",
                "attributes": [],
                "sublabels": [{"id": 1, "name": "0", "type": "points", "attributes": []}],
            }
        ]
        model_spec = get_model_spec("ninerouter-face-vf50")
        mapping = {"face": {"name": "face", "attributes": {}, "sublabels": {"0": {"name": "0", "attributes": {}}}}}
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert not is_valid
        assert any("[INCOMPATIBLE_FULL_VF50]" in e for e in errors)


class TestCVAT2751ContractSimulation:
    """Validate backend validation of CVAT 2.75.1 skeleton contracts."""

    def test_missing_sublabels_key_raises_critical_error(self):
        """Missing 'sublabels' key triggers CRITICAL CVAT 2.75.1 ERROR."""
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))
        model_spec = get_model_spec("ninerouter-human-pose-17")

        invalid_mapping = {
            "person": {
                "name": "person",
                "attributes": {},
                # 'sublabels' key intentionally omitted
            }
        }
        is_valid, errors = validate_cvat_mapping(invalid_mapping, model_spec, task_labels)
        assert not is_valid
        assert any("[CRITICAL CVAT 2.75.1 ERROR]" in e for e in errors)
        assert any('missing "sublabels" key' in e for e in errors)

    def test_empty_sublabels_raises_critical_error(self):
        """Empty 'sublabels' dict triggers CRITICAL CVAT 2.75.1 ERROR."""
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))
        model_spec = get_model_spec("ninerouter-human-pose-17")

        invalid_mapping = {
            "person": {
                "name": "person",
                "attributes": {},
                "sublabels": {},
            }
        }
        is_valid, errors = validate_cvat_mapping(invalid_mapping, model_spec, task_labels)
        assert not is_valid
        assert any("[CRITICAL CVAT 2.75.1 ERROR]" in e for e in errors)
        assert any("empty sublabels mapping" in e for e in errors)

    def test_duplicate_target_sublabel_detected(self):
        """Mapping two model keypoints to the same task sublabel must be detected."""
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))
        model_spec = get_model_spec("ninerouter-human-pose-17")

        sublabels_map = {kp: {"name": kp, "attributes": {}} for kp in POSE17_KEYPOINTS}
        # Force duplicate: left_eye also maps to nose
        sublabels_map["left_eye"] = {"name": "nose", "attributes": {}}

        mapping = {"person": {"name": "person", "attributes": {}, "sublabels": sublabels_map}}
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert not is_valid
        assert any("Duplicate mapping" in e for e in errors)

    def test_incompatible_label_type_rejected(self):
        """Mapping a skeleton model label to a non-skeleton task label is rejected."""
        model_spec = get_model_spec("ninerouter-human-pose-17")
        task_labels = [
            {
                "id": 1,
                "name": "person",
                "type": "rectangle",
                "attributes": [],
                "sublabels": [],
            }
        ]
        mapping = {"person": {"name": "person", "attributes": {}}}
        is_valid, errors = validate_cvat_mapping(mapping, model_spec, task_labels)
        assert not is_valid
        assert any("[INCOMPATIBLE_TYPE]" in e for e in errors)

    def test_malformed_non_dict_mapping_items_handled_safely(self):
        """Non-dict mapping item or non-dict sublabel item produces validation errors without crashing."""
        model_spec = get_model_spec("ninerouter-human-pose-17")
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))

        # Case 1: non-dict top-level mapping item
        mapping_str = {"person": "person"}
        is_valid, errors = validate_cvat_mapping(mapping_str, model_spec, task_labels)
        assert not is_valid
        assert any("must be a dictionary" in e for e in errors)

        # Case 2: non-dict sublabel item
        mapping_sub_str = {
            "person": {
                "name": "person",
                "attributes": {},
                "sublabels": {"nose": "nose"},
            }
        }
        is_valid, errors = validate_cvat_mapping(mapping_sub_str, model_spec, task_labels)
        assert not is_valid
        assert any("must be a dictionary" in e for e in errors)


class TestPreflightCLIExecution:
    """Validate scripts/week2_mapping_preflight.py CLI behavior with --labels-file."""

    def test_cli_pose17_success(self, tmp_path: Path):
        """CLI generates valid Pose17 mapping and exits with 0."""
        import subprocess
        import sys

        labels_file = tmp_path / "task_labels_pose17.json"
        out_mapping = tmp_path / "output_mapping.json"
        task_labels = _make_pose17_task_labels(list(POSE17_KEYPOINTS))
        labels_file.write_text(json.dumps(task_labels), encoding="utf-8")

        res = subprocess.run(
            [
                sys.executable,
                "scripts/week2_mapping_preflight.py",
                "--labels-file",
                str(labels_file),
                "--function",
                "pose17",
                "--output-mapping",
                str(out_mapping),
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI stderr: {res.stderr}\nCLI stdout: {res.stdout}"
        assert "[SUCCESS] Generated compliant CVAT 2.75.1 mapping payload." in res.stdout
        assert out_mapping.exists()
        saved = json.loads(out_mapping.read_text(encoding="utf-8"))
        assert "person" in saved["mapping"]
        assert len(saved["mapping"]["person"]["sublabels"]) == 17

    def test_cli_vf50_success(self, tmp_path: Path):
        """CLI generates valid VF-50 mapping and exits with 0."""
        import subprocess
        import sys

        labels_file = tmp_path / "task_labels_vf50.json"
        task_labels = _make_canonical_vf50_task_labels()
        labels_file.write_text(json.dumps(task_labels), encoding="utf-8")

        res = subprocess.run(
            [
                sys.executable,
                "scripts/week2_mapping_preflight.py",
                "--labels-file",
                str(labels_file),
                "--function",
                "vf50",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"CLI stderr: {res.stderr}\nCLI stdout: {res.stdout}"
        assert "Successfully mapped all 7/7 VF-50 component skeletons (50/50 sublabels)" in res.stdout
        assert "[SUCCESS] Generated compliant CVAT 2.75.1 mapping payload." in res.stdout

    def test_cli_monolithic_face_rejected(self, tmp_path: Path):
        """CLI cleanly diagnoses and rejects monolithic face with exit code 1."""
        import subprocess
        import sys

        labels_file = tmp_path / "task_labels_mono.json"
        task_labels = [{"id": 1, "name": "face", "type": "skeleton", "attributes": [], "sublabels": [{"id": 1, "name": "0", "type": "points", "attributes": []}]}]
        labels_file.write_text(json.dumps(task_labels), encoding="utf-8")

        res = subprocess.run(
            [
                sys.executable,
                "scripts/week2_mapping_preflight.py",
                "--labels-file",
                str(labels_file),
                "--function",
                "vf50",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 1
        assert "[INCOMPATIBLE_FULL_VF50]" in res.stdout

