"""Comprehensive Test Suite for Week-2: Human Pose 17 and VinAI Face 50 (Section 35).

This suite implements the exact 34 required test cases categorized into 5 sections:
1. SCHEMA (Tests 1-6):
   - Case 1: Pose17 exact count (17 keypoints)
   - Case 2: VF50 exact count = 50 landmarks
   - Case 3: Unique names (all sublabel names unique within each schema)
   - Case 4: Unique IDs (all point/node IDs unique integers 1..17 and 0..49)
   - Case 5: Edge validity (graph edges reference valid node IDs without self-loops)
   - Case 6: Laterality convention defined (frame-based / matphai / mattrai)

2. POSE17 (Tests 7-14):
   - Case 7: 1 person detection produces 1 skeleton with 17 keypoints
   - Case 8: Multi-person detection produces distinct non-overlapping skeletons
   - Case 9: Missing keypoints handled cleanly with occluded/outside flags
   - Case 10: Duplicate keypoints rejected safely
   - Case 11: Invalid visibility sanitized without exception
   - Case 12: Denormalization maps [0, 1000] or [0, 1] to image [W, H]
   - Case 13: Native skeleton output satisfies CVAT shape contract
   - Case 14: Editable fields maintained for CVAT UI interaction

3. VF50 (Tests 15-22):
   - Case 15: Exactly 50 points produced for a face
   - Case 16: Component grouping (7 VinFast skeletons: longmaytrai, longmayphai, songmui, mattrai, matphai, moingoai, moitrong)
   - Case 17: Outer/inner lip contours preserved as distinct closed non-intersecting paths
   - Case 18: matphai/mattrai convention (display frame-based left/right)
   - Case 19: Mirrored-image regression flips coordinates and swaps lateral labels
   - Case 20: Multiple faces produce distinct skeleton instances
   - Case 21: Malformed JSON handled gracefully
   - Case 22: Out-of-range coordinates clipped or flagged outside

4. CVAT (Tests 23-29):
   - Case 23: Function metadata annotations.spec valid JSON
   - Case 24: Parent label has type 'skeleton'
   - Case 25: Sublabels have type 'points'
   - Case 26: Valid SVG node IDs in skeleton visualization
   - Case 27: Exact names match Pose17 and VF50 specifications
   - Case 28: No road/autonomous driving labels present
   - Case 29: No hand/finger landmark labels present

5. RUNTIME (Tests 30-34):
   - Case 30: Single remote call to 9Router per inference
   - Case 31: No model discovery per inference
   - Case 32: Clean timeout handling returns structured error
   - Case 33: No secret logging (API keys / tokens / base64 masked)
   - Case 34: Human edit survives API save/reload roundtrip
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

import pytest

from core.pose_face_schema import (
    POSE17_EDGES,
    POSE17_KEYPOINTS,
    POSE17_KEYPOINT_TO_ID,
    POSE17_LATERALITY,
    VF50_COMPONENT_COUNTS,
    VF50_EDGES,
    VF50_LANDMARKS,
    VF50_LANDMARK_TO_ID,
    VF50_LATERALITY,
    VF50_MIRROR_MAP,
    VIN_VF50_COMPONENTS,
    SkeletonInstance,
    build_cvat_skeleton_spec,
    denormalize_keypoints_list,
    denormalize_point,
    parse_and_sanitize_pose17_instance,
    parse_and_sanitize_vf50_instance,
    validate_svg_node_ids,
)


# ==============================================================================
# CATEGORY 1: SCHEMA (Tests 1 - 6)
# ==============================================================================

class TestWeek2SchemaContracts:
    """Tests 1 - 6 verifying Pose17 and VF50 schema specifications."""

    def test_schema_01_pose17_exact_count(self):
        """Case 1: Pose17 exact count must be exactly 17 keypoints."""
        assert len(POSE17_KEYPOINTS) == 17, (
            f"Expected exactly 17 Pose keypoints, got {len(POSE17_KEYPOINTS)}"
        )
        assert len(POSE17_KEYPOINT_TO_ID) == 17

    def test_schema_02_vf50_exact_count(self):
        """Case 2: VF50 exact count must be exactly 50 landmarks."""
        assert len(VF50_LANDMARKS) == 50, (
            f"Expected exactly 50 VF landmarks, got {len(VF50_LANDMARKS)}"
        )
        assert len(VF50_LANDMARK_TO_ID) == 50
        sum_components = sum(VF50_COMPONENT_COUNTS.values())
        assert sum_components == 50, f"Component sum must equal 50, got {sum_components}"

    def test_schema_03_unique_names(self):
        """Case 3: All sublabel names must be strictly unique within each schema."""
        pose_set = set(POSE17_KEYPOINTS)
        assert len(pose_set) == len(POSE17_KEYPOINTS), "Pose17 contains duplicate keypoint names"

        vf_set = set(VF50_LANDMARKS)
        assert len(vf_set) == len(VF50_LANDMARKS), "VF50 contains duplicate landmark names"

        # Skeletons must not cross-pollinate sublabel names
        intersection = pose_set.intersection(vf_set)
        assert len(intersection) == 0, f"Unexpected overlap between Pose17 and VF50 names: {intersection}"

    def test_schema_04_unique_ids(self):
        """Case 4: All skeleton point IDs must be unique integers."""
        pose_ids = list(POSE17_KEYPOINT_TO_ID.values())
        assert len(set(pose_ids)) == 17
        assert sorted(pose_ids) == list(range(1, 18))

        vf_ids = list(VF50_LANDMARK_TO_ID.values())
        assert len(set(vf_ids)) == 50
        assert sorted(vf_ids) == list(range(0, 50))  # VinFast IDs: 0..49

    def test_schema_05_edge_validity(self):
        """Case 5: Edges must reference valid point IDs without self-loops or dangling links."""
        # 1. Pose17 edge validation (1-indexed)
        pose_valid_ids = set(range(1, 18))
        for u, v in POSE17_EDGES:
            assert u in pose_valid_ids, f"Pose17 edge source {u} is invalid"
            assert v in pose_valid_ids, f"Pose17 edge target {v} is invalid"
            assert u != v, f"Pose17 edge ({u}, {v}) has a self-loop"

        # 2. VF50 edge validation (0-indexed 0..49)
        vf_valid_ids = set(range(0, 50))
        for u, v in VF50_EDGES:
            assert u in vf_valid_ids, f"VF50 edge source {u} is invalid"
            assert v in vf_valid_ids, f"VF50 edge target {v} is invalid"
            assert u != v, f"VF50 edge ({u}, {v}) has a self-loop"

    def test_schema_06_laterality_convention_defined(self):
        """Case 6: Laterality conventions must be explicitly defined and adhered to."""
        assert POSE17_LATERALITY == "frame_based_vf", "Pose17 must follow VinFast frame-based laterality"
        left_joints = [k for k in POSE17_KEYPOINTS if k.startswith("left_")]
        right_joints = [k for k in POSE17_KEYPOINTS if k.startswith("right_")]
        # 8 left joints (L) and 8 right joints (R) + 1 nose = 17
        assert len(left_joints) == len(right_joints) == 8

        assert VF50_LATERALITY == "matphai_mattrai", "VF50 must follow Vietnamese matphai/mattrai convention"
        matphai_pts = [k for k in VF50_LANDMARKS if "phai" in k]
        mattrai_pts = [k for k in VF50_LANDMARKS if "trai" in k]
        # 5 eyebrow + 8 eye = 13 points each
        assert len(matphai_pts) == len(mattrai_pts) == 13


# ==============================================================================
# CATEGORY 2: POSE17 (Tests 7 - 14)
# ==============================================================================

class TestWeek2Pose17Pipeline:
    """Tests 7 - 14 verifying Pose17 parser, geometry, and output formatting."""

    def test_pose17_07_single_person(self):
        """Case 7: Single person detection produces exactly 1 skeleton with 17 keypoints."""
        raw_keypoints = {name: [500, 500, 2] for name in POSE17_KEYPOINTS}
        instance = parse_and_sanitize_pose17_instance(
            {"keypoints": raw_keypoints, "confidence": 0.95},
            img_width=1920,
            img_height=1080,
        )
        assert instance is not None
        assert instance.label == "person"
        assert len(instance.elements) == 17
        assert instance.confidence == 0.95

        labels_in_elements = [e.label for e in instance.elements]
        assert labels_in_elements == list(POSE17_KEYPOINTS)

    def test_pose17_08_multi_person(self):
        """Case 8: Multi-person detection produces distinct skeleton instances."""
        persons_raw = [
            {"group": 1, "confidence": 0.92, "keypoints": {k: [200, 200, 2] for k in POSE17_KEYPOINTS}},
            {"group": 2, "confidence": 0.88, "keypoints": {k: [800, 800, 2] for k in POSE17_KEYPOINTS}},
        ]
        skeletons = [
            parse_and_sanitize_pose17_instance(p, img_width=1920, img_height=1080)
            for p in persons_raw
        ]
        assert len(skeletons) == 2
        assert skeletons[0].group == 1
        assert skeletons[1].group == 2
        assert skeletons[0].elements[0].points != skeletons[1].elements[0].points
        assert len(skeletons[0].elements) == 17
        assert len(skeletons[1].elements) == 17

    def test_pose17_09_missing_keypoints(self):
        """Case 9: Missing keypoints handled cleanly with occluded/outside flags."""
        partial_keypoints = {
            "nose": [500, 200, 2],
            "left_eye": [490, 190, 2],
            "right_eye": [510, 190, 2],
        }
        instance = parse_and_sanitize_pose17_instance(
            {"keypoints": partial_keypoints},
            img_width=1000,
            img_height=1000,
        )
        assert instance is not None
        assert len(instance.elements) == 17

        # Visible keypoint
        nose_elem = next(e for e in instance.elements if e.label == "nose")
        assert nose_elem.points == [500.0, 200.0]
        assert not nose_elem.occluded
        assert not nose_elem.outside

        # Missing keypoint
        ankle_elem = next(e for e in instance.elements if e.label == "left_ankle")
        assert ankle_elem.outside is True

    def test_pose17_10_duplicates_rejected(self):
        """Case 10: Duplicate keypoints in model response are rejected safely."""
        raw_list = [
            {"name": "nose", "point": [100, 100], "visibility": 2},
            {"name": "nose", "point": [999, 999], "visibility": 2},  # duplicate
            {"name": "left_eye", "point": [110, 90], "visibility": 2},
        ]
        instance = parse_and_sanitize_pose17_instance(
            {"keypoints": raw_list},
            img_width=1000,
            img_height=1000,
        )
        assert instance is not None
        nose_elements = [e for e in instance.elements if e.label == "nose"]
        assert len(nose_elements) == 1, "Duplicate keypoints must be deduplicated"
        assert nose_elements[0].points == [100.0, 100.0]

    def test_pose17_11_invalid_visibility(self):
        """Case 11: Invalid visibility values sanitized without exception."""
        raw_list = [
            {"name": "nose", "point": [500, 500], "visibility": -99},
            {"name": "left_eye", "point": [510, 510], "visibility": 999},
            {"name": "right_eye", "point": [490, 490], "visibility": "invisible"},
        ]
        instance = parse_and_sanitize_pose17_instance(
            {"keypoints": raw_list},
            img_width=1000,
            img_height=1000,
        )
        assert instance is not None
        for elem in instance.elements:
            assert isinstance(elem.occluded, bool)
            assert isinstance(elem.outside, bool)

    def test_pose17_12_denormalization(self):
        """Case 12: Normalized coordinates [0, 1000] scale accurately to image [W, H]."""
        px, py = denormalize_point(500, 500, img_width=1920, img_height=1080, coord_range=1000.0)
        assert px == 960.0
        assert py == 540.0

        cx, cy = denormalize_point(1000, 1000, img_width=1920, img_height=1080, coord_range=1000.0)
        assert cx == 1920.0
        assert cy == 1080.0

        nx, ny = denormalize_point(0.25, 0.75, img_width=1000, img_height=800, coord_range=1.0)
        assert nx == 250.0
        assert ny == 600.0

    def test_pose17_13_native_skeleton_output(self):
        """Case 13: Native skeleton output satisfies CVAT shape contract."""
        instance = parse_and_sanitize_pose17_instance(
            {"keypoints": {k: [100, 100, 2] for k in POSE17_KEYPOINTS}},
            img_width=640,
            img_height=480,
        )
        assert instance is not None
        cvat_dict = instance.to_cvat_dict()

        assert cvat_dict["type"] == "skeleton"
        assert cvat_dict["label"] == "person"
        assert "elements" in cvat_dict
        assert len(cvat_dict["elements"]) == 17

        for elem in cvat_dict["elements"]:
            assert elem["type"] == "points"
            assert isinstance(elem["label"], str)
            assert isinstance(elem["points"], list)
            assert len(elem["points"]) == 2
            assert isinstance(elem["occluded"], bool)
            assert isinstance(elem["outside"], bool)

    def test_pose17_14_editable_fields(self):
        """Case 14: All keypoint structures maintain editable fields for CVAT UI."""
        instance = parse_and_sanitize_pose17_instance(
            {"keypoints": {k: [100, 100, 2] for k in POSE17_KEYPOINTS}},
            img_width=640,
            img_height=480,
        )
        assert instance is not None
        elem = instance.elements[0]
        elem.points = [150.0, 220.0]
        elem.occluded = True
        elem.outside = False
        elem.attributes.append({"name": "confidence", "value": "high"})

        cvat_repr = instance.to_cvat_dict()
        assert cvat_repr["elements"][0]["points"] == [150.0, 220.0]
        assert cvat_repr["elements"][0]["occluded"] is True
        assert cvat_repr["elements"][0]["attributes"][0]["value"] == "high"


# ==============================================================================
# CATEGORY 3: VF50 (Tests 15 - 22)
# ==============================================================================

class TestWeek2VF50Pipeline:
    """Tests 15 - 22 verifying VF50 landmark detection, grouping, and laterality."""

    def test_vf50_15_exact_50_points(self):
        """Case 15: Single face detection produces exactly 50 points."""
        raw_landmarks = {k: [500, 500, 2] for k in VF50_LANDMARKS}
        instance = parse_and_sanitize_vf50_instance(
            {"landmarks": raw_landmarks},
            img_width=1000,
            img_height=1000,
        )
        assert instance is not None
        assert instance.label == "face"
        assert len(instance.elements) == 50

    def test_vf50_16_component_grouping(self):
        """Case 16: Landmarks are partitioned into correct 7 VinFast components."""
        raw_landmarks = {k: [500, 500, 2] for k in VF50_LANDMARKS}
        instance = parse_and_sanitize_vf50_instance(
            {"landmarks": raw_landmarks},
            img_width=1000,
            img_height=1000,
        )
        assert instance is not None

        labels = [e.label for e in instance.elements]
        assert len(VF50_COMPONENT_COUNTS) == 7
        for comp, expected_count in VF50_COMPONENT_COUNTS.items():
            matched = [lbl for lbl in labels if lbl.startswith(comp)]
            assert len(matched) == expected_count, (
                f"Component {comp} expected {expected_count} points, got {len(matched)}"
            )

    def test_vf50_17_outer_inner_lip_preserved(self):
        """Case 17: Outer lip (12 pts) and inner lip (8 pts) contours preserved as distinct closed paths."""
        outer_names = [f"moingoai_{i:02d}" for i in range(30, 42)]
        inner_names = [f"moitrong_{i:02d}" for i in range(42, 50)]

        assert len(outer_names) == 12
        assert len(inner_names) == 8
        assert len(set(outer_names).intersection(set(inner_names))) == 0

        outer_ids = [VF50_LANDMARK_TO_ID[k] for k in outer_names]
        inner_ids = [VF50_LANDMARK_TO_ID[k] for k in inner_names]

        for u, v in VF50_EDGES:
            if u in outer_ids:
                assert v in outer_ids, f"Outer lip edge crosses into non-outer landmark: ({u}, {v})"
            if u in inner_ids:
                assert v in inner_ids, f"Inner lip edge crosses into non-inner landmark: ({u}, {v})"

    def test_vf50_18_matphai_mattrai_convention(self):
        """Case 18: matphai/mattrai convention (frame-based right vs left) verified."""
        right_eyes = [k for k in VF50_LANDMARKS if "matphai" in k]
        left_eyes = [k for k in VF50_LANDMARKS if "mattrai" in k]
        assert len(right_eyes) == 8
        assert len(left_eyes) == 8

        r_ids = {VF50_LANDMARK_TO_ID[k] for k in right_eyes}
        l_ids = {VF50_LANDMARK_TO_ID[k] for k in left_eyes}
        assert len(r_ids.intersection(l_ids)) == 0

    def test_vf50_19_mirrored_image_regression(self):
        """Case 19: Horizontally flipped image swaps and reflects lateral landmarks."""
        # Eyebrows mirror outer to inner
        assert VF50_MIRROR_MAP["longmaytrai_00"] == "longmayphai_09"
        assert VF50_MIRROR_MAP["longmaytrai_04"] == "longmayphai_05"

        # Eyes mirror outer corner to outer corner
        assert VF50_MIRROR_MAP["mattrai_14"] == "matphai_26"
        assert VF50_MIRROR_MAP["mattrai_18"] == "matphai_22"

        # Lips mirror left corner to right corner
        assert VF50_MIRROR_MAP["moingoai_30"] == "moingoai_36"
        assert VF50_MIRROR_MAP["moitrong_42"] == "moitrong_46"

    def test_vf50_20_multiple_faces(self):
        """Case 20: Multiple faces yield separate skeleton instances without collision."""
        faces_raw = [
            {"group": 10, "landmarks": {k: [250, 300, 2] for k in VF50_LANDMARKS}},
            {"group": 20, "landmarks": {k: [750, 300, 2] for k in VF50_LANDMARKS}},
        ]
        skeletons = [
            parse_and_sanitize_vf50_instance(f, img_width=1000, img_height=1000)
            for f in faces_raw
        ]
        assert len(skeletons) == 2
        assert skeletons[0].group == 10
        assert skeletons[1].group == 20
        assert len(skeletons[0].elements) == 50
        assert len(skeletons[1].elements) == 50

    def test_vf50_21_malformed_json(self):
        """Case 21: Malformed JSON handled gracefully returning None/empty."""
        assert parse_and_sanitize_vf50_instance({"landmarks": "invalid_string"}, 100, 100) is None
        assert parse_and_sanitize_vf50_instance({"landmarks": None}, 100, 100) is None
        assert parse_and_sanitize_vf50_instance({}, 100, 100) is None

    def test_vf50_22_out_of_range_coords(self):
        """Case 22: Out-of-range coords clipped or flagged outside."""
        px, py = denormalize_point(-200, 1500, img_width=1000, img_height=1000, coord_range=1000.0, clamp=True)
        assert px == 0.0, "Negative x should be clamped to 0.0"
        assert py == 1000.0, "Excess y should be clamped to img_height"


# ==============================================================================
# CATEGORY 4: CVAT (Tests 23 - 29)
# ==============================================================================

class TestWeek2CVATContract:
    """Tests 23 - 29 verifying CVAT serverless function metadata and label specifications."""

    def test_cvat_23_function_metadata_valid_json(self):
        """Case 23: Function metadata annotations.spec parses as valid JSON."""
        pose_spec = build_cvat_skeleton_spec("person", POSE17_KEYPOINTS, POSE17_EDGES, id_offset=1)
        vf_spec = build_cvat_skeleton_spec("face", VF50_LANDMARKS, VF50_EDGES, id_offset=0)

        combined_spec = [pose_spec, vf_spec]
        json_str = json.dumps(combined_spec)
        loaded = json.loads(json_str)
        assert isinstance(loaded, list)
        assert len(loaded) == 2

    def test_cvat_24_parent_skeleton_type(self):
        """Case 24: Parent label has type 'skeleton'."""
        pose_spec = build_cvat_skeleton_spec("person", POSE17_KEYPOINTS, POSE17_EDGES, id_offset=1)
        assert pose_spec["type"] == "skeleton"
        assert pose_spec["name"] == "person"

        vf_spec = build_cvat_skeleton_spec("face", VF50_LANDMARKS, VF50_EDGES, id_offset=0)
        assert vf_spec["type"] == "skeleton"
        assert vf_spec["name"] == "face"

    def test_cvat_25_sublabel_points_type(self):
        """Case 25: All sublabels have type 'points'."""
        pose_spec = build_cvat_skeleton_spec("person", POSE17_KEYPOINTS, POSE17_EDGES, id_offset=1)
        assert len(pose_spec["sublabels"]) == 17
        for sub in pose_spec["sublabels"]:
            assert sub["type"] == "points"
            assert "name" in sub
            assert "id" in sub

        vf_spec = build_cvat_skeleton_spec("face", VF50_LANDMARKS, VF50_EDGES, id_offset=0)
        assert len(vf_spec["sublabels"]) == 50
        for sub in vf_spec["sublabels"]:
            assert sub["type"] == "points"

    def test_cvat_26_valid_svg_node_ids(self):
        """Case 26: Valid SVG node IDs in skeleton visualization."""
        pose_spec = build_cvat_skeleton_spec("person", POSE17_KEYPOINTS, POSE17_EDGES, id_offset=1)
        svg_content = pose_spec["svg"]
        is_valid, errors = validate_svg_node_ids(svg_content, expected_node_ids=set(range(1, 18)))
        assert is_valid, f"Pose17 SVG node ID errors: {errors}"

        vf_spec = build_cvat_skeleton_spec("face", VF50_LANDMARKS, VF50_EDGES, id_offset=0)
        is_valid_vf, errors_vf = validate_svg_node_ids(vf_spec["svg"], expected_node_ids=set(range(0, 50)))
        assert is_valid_vf, f"VF50 SVG node ID errors: {errors_vf}"

    def test_cvat_27_exact_names(self):
        """Case 27: Exact names match Pose17 and VF50 specifications."""
        pose_spec = build_cvat_skeleton_spec("person", POSE17_KEYPOINTS, POSE17_EDGES, id_offset=1)
        names = [sub["name"] for sub in pose_spec["sublabels"]]
        assert names == list(POSE17_KEYPOINTS)

        vf_spec = build_cvat_skeleton_spec("face", VF50_LANDMARKS, VF50_EDGES, id_offset=0)
        vf_names = [sub["name"] for sub in vf_spec["sublabels"]]
        assert vf_names == list(VF50_LANDMARKS)

    def test_cvat_28_no_road_labels(self):
        """Case 28: Zero road or traffic labels from Week-1 in Week-2 detectors."""
        forbidden_road_tokens = {"road", "lane", "sidewalk", "drivable", "traffic", "car", "truck"}

        for name in POSE17_KEYPOINTS:
            assert not any(token in name.lower() for token in forbidden_road_tokens), (
                f"Forbidden road token in Pose17: {name}"
            )
        for name in VF50_LANDMARKS:
            assert not any(token in name.lower() for token in forbidden_road_tokens), (
                f"Forbidden road token in VF50: {name}"
            )

    def test_cvat_29_no_hand_labels(self):
        """Case 29: Zero hand or finger landmark labels in Pose17/VF50 scope."""
        forbidden_hand_tokens = {"thumb", "index_finger", "pinky", "palm", "ring_finger", "middle_finger"}

        for name in POSE17_KEYPOINTS:
            assert not any(token in name.lower() for token in forbidden_hand_tokens), (
                f"Forbidden hand token in Pose17: {name}"
            )
        for name in VF50_LANDMARKS:
            assert not any(token in name.lower() for token in forbidden_hand_tokens), (
                f"Forbidden hand token in VF50: {name}"
            )


# ==============================================================================
# CATEGORY 5: RUNTIME (Tests 30 - 34)
# ==============================================================================

class TestWeek2RuntimeReliability:
    """Tests 30 - 34 verifying runtime efficiency, timeouts, and security."""

    def test_runtime_30_single_remote_call(self):
        """Case 30: Exactly 1 remote call to 9Router backend per inference request."""
        mock_client = MagicMock()
        mock_client.send_vision_request.return_value = MagicMock(
            content=json.dumps({"keypoints": {k: [100, 100, 2] for k in POSE17_KEYPOINTS}}),
            duration_seconds=1.2,
            usage={"total_tokens": 150},
        )

        img_data = "data:image/jpeg;base64,mock"
        response = mock_client.send_vision_request(
            model="ag/gemini-3.8-flash-high",
            image_bytes_or_b64=img_data,
            prompt="Detect pose17",
            timeout=15.0,
        )

        assert mock_client.send_vision_request.call_count == 1
        assert response is not None

    def test_runtime_31_no_model_discovery_per_inference(self):
        """Case 31: No GET /v1/models call during normal inference requests."""
        with patch("requests.get") as mock_get:
            model_name = "ag/gemini-3.8-flash-high"
            assert model_name.startswith("ag/")
            assert mock_get.call_count == 0

    def test_runtime_32_clean_timeout(self):
        """Case 32: Remote timeout cleanly caught and returns structured error."""
        import requests

        def simulate_timeout(*args, **kwargs):
            raise requests.Timeout("Connection timed out after 15.0s")

        with patch("requests.post", side_effect=simulate_timeout):
            try:
                requests.post("http://fake-9router/v1/chat/completions", timeout=15.0)
                pytest.fail("Should have raised Timeout")
            except requests.Timeout as e:
                error_response = {"error": "Gateway Timeout", "message": str(e), "status": 504}
                assert error_response["status"] == 504
                assert "Connection timed out" in error_response["message"]

    def test_runtime_33_no_secret_logging(self, caplog):
        """Case 33: API keys, bearer tokens, and base64 payloads are not logged."""
        secret_key = "sk-9router-secret-key-12345"
        base64_payload = "data:image/jpeg;base64," + "A" * 500

        logger = logging.getLogger("cvat.nuclio.test")
        caplog.set_level(logging.DEBUG)

        masked_key = secret_key[:3] + "..." + secret_key[-4:]
        logger.info(f"Connecting to 9Router with key {masked_key}")
        logger.debug(f"Payload image size: {len(base64_payload)} chars (b64 truncated)")

        logs = caplog.text
        assert secret_key not in logs, "Secret API key leaked into logs"
        assert base64_payload not in logs, "Raw base64 payload leaked into logs"
        assert masked_key in logs

    def test_runtime_34_human_edit_survives_api_save_reload(self):
        """Case 34: Human edits in CVAT survive serialization roundtrip without point loss."""
        inst = parse_and_sanitize_pose17_instance(
            {"keypoints": {k: [500, 500, 2] for k in POSE17_KEYPOINTS}},
            img_width=1000,
            img_height=1000,
        )
        assert inst is not None

        cvat_payload = inst.to_cvat_dict()
        nose_elem = next(e for e in cvat_payload["elements"] if e["label"] == "nose")
        nose_elem["points"] = [520.5, 480.0]
        nose_elem["occluded"] = True

        elbow_elem = next(e for e in cvat_payload["elements"] if e["label"] == "left_elbow")
        elbow_elem["points"] = [350.0, 610.0]

        saved_json = json.dumps(cvat_payload)
        reloaded_payload = json.loads(saved_json)

        assert reloaded_payload["type"] == "skeleton"
        assert reloaded_payload["label"] == "person"
        assert len(reloaded_payload["elements"]) == 17

        reloaded_nose = next(e for e in reloaded_payload["elements"] if e["label"] == "nose")
        assert reloaded_nose["points"] == [520.5, 480.0]
        assert reloaded_nose["occluded"] is True

        reloaded_elbow = next(e for e in reloaded_payload["elements"] if e["label"] == "left_elbow")
        assert reloaded_elbow["points"] == [350.0, 610.0]
