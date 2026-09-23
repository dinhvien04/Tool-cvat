"""Unit tests for Human Pose 17 skeleton contract, CVAT generation, and quality gate."""

from __future__ import annotations

import json
import pytest

from core.skeleton_contract import (
    DEFAULT_PARENT_LABEL,
    KEYPOINT_COUNT,
    KEYPOINT_INDEX_MAP,
    POSE17_KEYPOINTS,
    POSE17_KEYPOINTS_SET,
    POSE17_SKELETON_EDGES,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUTSIDE,
    VISIBILITY_VISIBLE,
    PersonPose17,
    Pose17QualityReport,
    PoseKeypoint,
    assess_pose17_quality,
    build_cvat_pose17_spec,
    build_pose17_prompt,
    denormalize_keypoint,
    map_cvat_to_visibility,
    map_visibility_to_cvat,
    merge_refined_keypoint,
    normalize_keypoint,
    parse_pose17_response,
    poses_to_cvat_skeletons,
    reproject_crop_point,
)


class TestKeypointTopology:
    """Verify canonical COCO 17-keypoint topology and ordering."""

    def test_keypoint_count_and_uniqueness(self):
        assert len(POSE17_KEYPOINTS) == 17
        assert KEYPOINT_COUNT == 17
        assert len(POSE17_KEYPOINTS_SET) == 17
        assert len(set(POSE17_KEYPOINTS)) == 17

    def test_authoritative_vinfast_pose17_ordering(self):
        expected_vinfast_order = [
            "nose",            # 1
            "right_eye",       # 2 (R Eye - even)
            "left_eye",        # 3 (L Eye - odd)
            "right_ear",       # 4
            "left_ear",        # 5
            "right_shoulder",  # 6
            "left_shoulder",   # 7
            "right_elbow",     # 8
            "left_elbow",      # 9
            "right_wrist",     # 10
            "left_wrist",      # 11
            "right_hip",       # 12
            "left_hip",        # 13
            "right_knee",      # 14
            "left_knee",       # 15
            "right_ankle",     # 16
            "left_ankle",      # 17
        ]
        assert list(POSE17_KEYPOINTS) == expected_vinfast_order
        for idx, name in enumerate(expected_vinfast_order):
            assert KEYPOINT_INDEX_MAP[name] == idx

    def test_skeleton_edges_connectivity(self):
        # 18 edges per config/week2_pose17.yaml (includes ear-to-shoulder)
        assert len(POSE17_SKELETON_EDGES) == 18
        for p1, p2 in POSE17_SKELETON_EDGES:
            assert p1 in POSE17_KEYPOINTS_SET
            assert p2 in POSE17_KEYPOINTS_SET
            assert p1 != p2


class TestVisibilityMapping:
    """Section 8: Visibility and Occlusion mapping tests."""

    def test_numeric_visibility_mapping(self):
        # 0 -> outside=True, occluded=False
        assert map_visibility_to_cvat(0) == (True, False)
        assert map_visibility_to_cvat(0.0) == (True, False)

        # 1 -> outside=False, occluded=True
        assert map_visibility_to_cvat(1) == (False, True)
        assert map_visibility_to_cvat(1.0) == (False, True)

        # 2 -> outside=False, occluded=False
        assert map_visibility_to_cvat(2) == (False, False)
        assert map_visibility_to_cvat(2.0) == (False, False)

    def test_string_and_bool_visibility_mapping(self):
        assert map_visibility_to_cvat("outside") == (True, False)
        assert map_visibility_to_cvat("absent") == (True, False)
        assert map_visibility_to_cvat("occluded") == (False, True)
        assert map_visibility_to_cvat("visible") == (False, False)
        assert map_visibility_to_cvat(True) == (False, False)
        assert map_visibility_to_cvat(False) == (True, False)

    def test_roundtrip_mapping(self):
        # outside=True, occluded=False -> 0
        assert map_cvat_to_visibility(True, False) == VISIBILITY_OUTSIDE
        # outside=False, occluded=True -> 1
        assert map_cvat_to_visibility(False, True) == VISIBILITY_OCCLUDED
        # outside=False, occluded=False -> 2
        assert map_cvat_to_visibility(False, False) == VISIBILITY_VISIBLE


class TestCoordinateTransforms:
    """Test normalized 0..1000 to pixel coordinate conversion and clamping."""

    def test_denormalize_exact_center(self):
        # 500 in [0, 1000] -> exactly half of width and height
        px, py = denormalize_keypoint(500, 500, width=1920, height=1080)
        assert px == 960.0
        assert py == 540.0

    def test_denormalize_corners(self):
        # (0, 0) -> (0.0, 0.0)
        assert denormalize_keypoint(0, 0, width=800, height=600) == (0.0, 0.0)
        # (1000, 1000) -> (800.0, 600.0)
        assert denormalize_keypoint(1000, 1000, width=800, height=600) == (800.0, 600.0)

    def test_denormalize_clamping(self):
        # Out-of-bounds coordinates clamp safely
        px, py = denormalize_keypoint(-50, 1200, width=1000, height=1000, clamp=True)
        assert px == 0.0
        assert py == 1000.0

    def test_denormalize_invalid_dimensions(self):
        with pytest.raises(ValueError):
            denormalize_keypoint(100, 100, width=0, height=100)
        with pytest.raises(ValueError):
            denormalize_keypoint(100, 100, width=100, height=-10)

    def test_normalize_keypoint(self):
        nx, ny = normalize_keypoint(480, 270, width=960, height=540)
        assert nx == 500.0
        assert ny == 500.0


class TestNativeCVATSkeletonFormatting:
    """Section 6: Native CVAT skeleton formatting."""

    def test_single_person_cvat_skeleton(self):
        kps = {
            "nose": PoseKeypoint(name="nose", x=500, y=200, visibility=2, confidence=0.98),
            "left_shoulder": PoseKeypoint(name="left_shoulder", x=550, y=300, visibility=1, confidence=0.90),
            "right_ankle": PoseKeypoint(name="right_ankle", x=450, y=900, visibility=0, confidence=0.10),
        }
        person = PersonPose17(id=1, label="person", confidence=0.95, keypoints=kps)
        cvat_dict = person.to_cvat_skeleton(width=1000, height=1000)

        assert cvat_dict["label"] == "person"
        assert cvat_dict["type"] == "skeleton"
        assert cvat_dict["confidence"] == 0.95
        assert isinstance(cvat_dict["elements"], list)
        # Skeletons MUST contain all 17 elements
        assert len(cvat_dict["elements"]) == 17

        # Check nose element
        nose_elem = next(el for el in cvat_dict["elements"] if el["label"] == "nose")
        assert nose_elem["type"] == "points"
        assert nose_elem["points"] == [500.0, 200.0]
        assert nose_elem["outside"] is False
        assert nose_elem["occluded"] is False
        assert nose_elem["confidence"] == 0.98

        # Check left_shoulder element (occluded)
        ls_elem = next(el for el in cvat_dict["elements"] if el["label"] == "left_shoulder")
        assert ls_elem["points"] == [550.0, 300.0]
        assert ls_elem["outside"] is False
        assert ls_elem["occluded"] is True
        assert ls_elem["confidence"] == 0.90

        # Check right_ankle element (outside)
        ra_elem = next(el for el in cvat_dict["elements"] if el["label"] == "right_ankle")
        assert ra_elem["outside"] is True
        assert ra_elem["occluded"] is False

        # Elements order must strictly match canonical POSE17_KEYPOINTS
        element_labels = [el["label"] for el in cvat_dict["elements"]]
        assert element_labels == list(POSE17_KEYPOINTS)


class TestMultiPersonIsolation:
    """Section 12: Multi-person support and zero cross-contamination."""

    def test_zero_people(self):
        res = poses_to_cvat_skeletons([], width=1920, height=1080)
        assert res == []

    def test_single_person(self):
        kps = {name: PoseKeypoint(name=name, x=500, y=500 + i * 20, visibility=2) for i, name in enumerate(POSE17_KEYPOINTS)}
        person = PersonPose17(id=1, keypoints=kps)
        res = poses_to_cvat_skeletons([person], width=1000, height=1000)
        assert len(res) == 1
        assert res[0]["type"] == "skeleton"

    def test_n_people_distinct_and_isolated(self):
        # Create 3 distinct people at different locations
        people = []
        for p_idx in range(3):
            kps = {}
            base_x = 200 + p_idx * 300
            for k_idx, name in enumerate(POSE17_KEYPOINTS):
                kps[name] = PoseKeypoint(
                    name=name,
                    x=base_x,
                    y=100 + k_idx * 40,
                    visibility=2,
                    confidence=0.90 + p_idx * 0.02,
                )
            people.append(PersonPose17(id=p_idx + 1, keypoints=kps))

        res = poses_to_cvat_skeletons(people, width=1000, height=1000)
        assert len(res) == 3

        # Verify each person retains its own coordinates without bleeding
        for p_idx, skel in enumerate(res):
            expected_x = 200.0 + p_idx * 300.0
            nose_el = next(el for el in skel["elements"] if el["label"] == "nose")
            assert nose_el["points"][0] == expected_x

        # Modifying an element in person 1 must NOT affect person 2
        res[0]["elements"][0]["points"][0] = 999.0
        assert res[1]["elements"][0]["points"][0] == 500.0


class TestRemoteModelContractAndParser:
    """Section 9: Remote model contract & parser robustness."""

    def test_parse_json_with_markdown_fences(self):
        raw = """```json
        {
          "people": [
            {
              "id": 1,
              "label": "person",
              "confidence": 0.95,
              "keypoints": [
                {"name": "nose", "point": [500, 200], "visibility": 2, "confidence": 0.99},
                {"name": "left_eye", "point": [510, 195], "visibility": 2, "confidence": 0.95}
              ]
            }
          ]
        }
        ```"""
        people = parse_pose17_response(raw)
        assert len(people) == 1
        p = people[0]
        assert p.id == 1
        assert p.get_keypoint("nose").x == 500.0
        assert p.get_keypoint("nose").y == 200.0
        assert p.get_keypoint("left_eye").x == 510.0
        # Missing keypoints are auto-populated as outside
        assert p.get_keypoint("right_ankle").is_outside is True

    def test_parse_ordered_tuples_list(self):
        # 17 keypoint tuples [x, y, visibility] in VinFast order
        # Index 15 = right_ankle (100 + 15*10 = 250), Index 16 = left_ankle (100 + 16*10 = 260)
        points_list = [[100 + i * 10, 200 + i * 20, 2] for i in range(17)]
        payload = [{"id": 1, "keypoints": points_list}]
        people = parse_pose17_response(payload)
        assert len(people) == 1
        p = people[0]
        assert p.get_keypoint("nose").x == 100.0
        assert p.get_keypoint("right_ankle").x == 250.0
        assert p.get_keypoint("left_ankle").x == 260.0

    def test_empty_and_corrupt_response_handling(self):
        assert parse_pose17_response("") == []
        assert parse_pose17_response("{}") == []
        assert parse_pose17_response('{"people": []}') == []
        assert parse_pose17_response("INVALID NON-JSON TEXT") == []


class TestDeterministicQualityGate:
    """Section 16: Deterministic quality gate and corrupt pose rejection."""

    def test_healthy_pose_accepted(self):
        kps = {}
        # Anatomically reasonable upright pose
        coords = {
            "nose": (500, 150),
            "left_eye": (510, 140),
            "right_eye": (490, 140),
            "left_ear": (525, 145),
            "right_ear": (475, 145),
            "left_shoulder": (550, 220),
            "right_shoulder": (450, 220),
            "left_elbow": (580, 320),
            "right_elbow": (420, 320),
            "left_wrist": (600, 420),
            "right_wrist": (400, 420),
            "left_hip": (540, 500),
            "right_hip": (460, 500),
            "left_knee": (550, 700),
            "right_knee": (450, 700),
            "left_ankle": (560, 900),
            "right_ankle": (440, 900),
        }
        for name, (x, y) in coords.items():
            kps[name] = PoseKeypoint(name=name, x=x, y=y, visibility=2, confidence=0.90)

        person = PersonPose17(id=1, keypoints=kps)
        report = assess_pose17_quality(person)
        assert report.is_valid is True
        assert report.score >= 0.85

    def test_too_few_keypoints_rejected(self):
        kps = {
            "nose": PoseKeypoint(name="nose", x=500, y=500, visibility=2),
            "left_eye": PoseKeypoint(name="left_eye", x=510, y=500, visibility=2),
        }
        person = PersonPose17(id=1, keypoints=kps)
        report = assess_pose17_quality(person, min_visible_keypoints=4)
        assert report.is_valid is False
        assert any("too_few_keypoints" in r for r in report.reasons)

    def test_degenerate_collapse_rejected(self):
        # All keypoints collapsed onto (500, 500)
        kps = {name: PoseKeypoint(name=name, x=500, y=500, visibility=2) for name in POSE17_KEYPOINTS}
        person = PersonPose17(id=1, keypoints=kps)
        report = assess_pose17_quality(person)
        assert report.is_valid is False
        assert any("degenerate_collapse" in r for r in report.reasons)

    def test_excessive_bone_jump_penalized(self):
        kps = {
            "nose": PoseKeypoint(name="nose", x=500, y=100, visibility=2),
            "left_shoulder": PoseKeypoint(name="left_shoulder", x=500, y=200, visibility=2),
            "right_shoulder": PoseKeypoint(name="right_shoulder", x=400, y=200, visibility=2),
            # Left elbow teleports across the entire image to (500, 999) - jump > 800 normalized units
            "left_elbow": PoseKeypoint(name="left_elbow", x=500, y=999, visibility=2),
            "left_wrist": PoseKeypoint(name="left_wrist", x=500, y=1000, visibility=2),
            "left_hip": PoseKeypoint(name="left_hip", x=500, y=500, visibility=2),
            "right_hip": PoseKeypoint(name="right_hip", x=400, y=500, visibility=2),
        }
        person = PersonPose17(id=1, keypoints=kps)
        report = assess_pose17_quality(person)
        assert any("excessive_bone_length" in r for r in report.reasons)


class TestEdgeCasesAndHardening:
    """Additional hardening and edge case tests."""

    def test_edge_case_missing_keypoints_populated(self):
        # Only nose provided
        person = PersonPose17(id=1, keypoints={"nose": PoseKeypoint(name="nose", x=500, y=200, visibility=2)})
        cvat_dict = person.to_cvat_skeleton(width=1000, height=1000)
        assert len(cvat_dict["elements"]) == 17
        missing_elem = next(el for el in cvat_dict["elements"] if el["label"] == "right_wrist")
        assert missing_elem["outside"] is True
        assert missing_elem["occluded"] is False
        assert missing_elem["confidence"] == 0.0

    def test_edge_case_clamping_negative_and_overflow(self):
        kp = PoseKeypoint(name="nose", x=-20, y=1050, visibility=2, confidence=1.5)
        assert kp.x == 0.0
        assert kp.y == 1000.0
        assert kp.confidence == 1.0

    def test_edge_case_arm_proportion_anomaly(self):
        # Shoulder at (500, 200), elbow at (500, 220) [length=20], wrist at (500, 600) [length=380] -> ratio = 20/380 = 0.052 < 0.15
        kps = {
            "nose": PoseKeypoint(name="nose", x=500, y=100, visibility=2),
            "left_shoulder": PoseKeypoint(name="left_shoulder", x=500, y=200, visibility=2),
            "right_shoulder": PoseKeypoint(name="right_shoulder", x=400, y=200, visibility=2),
            "left_elbow": PoseKeypoint(name="left_elbow", x=500, y=220, visibility=2),
            "left_wrist": PoseKeypoint(name="left_wrist", x=500, y=600, visibility=2),
            "left_hip": PoseKeypoint(name="left_hip", x=500, y=500, visibility=2),
            "right_hip": PoseKeypoint(name="right_hip", x=400, y=500, visibility=2),
        }
        person = PersonPose17(id=1, keypoints=kps)
        report = assess_pose17_quality(person)
        assert any("implausible_left_arm_ratio" in r for r in report.reasons)

    def test_edge_case_non_square_aspect_ratios(self):
        # 1920x1080
        px, py = denormalize_keypoint(250, 750, width=1920, height=1080)
        assert px == round(0.25 * 1920, 2)
        assert py == round(0.75 * 1080, 2)

    def test_edge_case_dict_keypoints_input(self):
        raw = {
            "people": [
                {
                    "id": 1,
                    "keypoints": {
                        "nose": {"point": [500, 200], "visibility": 2, "confidence": 0.98},
                        "left_eye": {"x": 510, "y": 190, "outside": False, "occluded": False},
                    },
                }
            ]
        }
        people = parse_pose17_response(raw)
        assert len(people) == 1
        p = people[0]
        assert p.get_keypoint("nose").is_visible is True
        assert p.get_keypoint("left_eye").is_visible is True


class TestCVATSpecHelper:
    """Verify CVAT function.yaml spec generation."""

    def test_build_cvat_pose17_spec(self):
        spec = build_cvat_pose17_spec("person")
        assert spec["name"] == "person"
        assert spec["type"] == "skeleton"
        assert len(spec["sublabels"]) == 17
        for sub in spec["sublabels"]:
            assert sub["type"] == "points"
            assert sub["name"] in POSE17_KEYPOINTS_SET

        # Mandatory SVG field for CVAT lambda_manager
        assert "svg" in spec
        svg = spec["svg"]
        assert svg.count("<circle") == 17
        assert svg.count("<line") == 18  # 18 edges per authoritative YAML
        for nid in range(1, 18):
            assert f'data-node-id="{nid}"' in svg

    def test_build_cvat_pose17_spec_numbered_sublabels(self):
        numbered = [str(i) for i in range(1, 18)]
        spec = build_cvat_pose17_spec("body", sublabel_names=numbered)
        assert spec["name"] == "body"
        assert [s["name"] for s in spec["sublabels"]] == numbered
        for s in numbered:
            assert f'data-label-name="{s}"' in spec["svg"]


class TestVinFastPose17Convention:
    """Verify VinFast HumanPose-17 convention (output_pose17_guideline.txt)."""

    def test_vinfast_mapping_roundtrip(self):
        from core.skeleton_contract import (
            COCO_NAME_TO_VINFAST_ID,
            VINFAST_ID_TO_COCO_NAME,
            VINFAST_POSE17_INDEX_TO_NAME,
            VINFAST_POSE17_KEYPOINTS,
        )

        assert len(VINFAST_POSE17_KEYPOINTS) == 17
        assert len(COCO_NAME_TO_VINFAST_ID) == 17
        assert len(VINFAST_ID_TO_COCO_NAME) == 17

        # Guideline Sec 2.1: Even numbers are Right, Odd numbers are Left
        even_ids = [2, 4, 6, 8, 10, 12, 14, 16]
        for eid in even_ids:
            name = VINFAST_POSE17_INDEX_TO_NAME[eid]
            assert name.startswith("R "), f"Expected {eid} to be Right joint, got {name}"

        odd_ids = [3, 5, 7, 9, 11, 13, 15, 17]
        for oid in odd_ids:
            name = VINFAST_POSE17_INDEX_TO_NAME[oid]
            assert name.startswith("L "), f"Expected {oid} to be Left joint, got {name}"

    def test_to_cvat_skeleton_vinfast_convention(self):
        # Create person with distinctive nose and right shoulder
        kps = {
            "nose": PoseKeypoint(name="nose", x=500, y=200, visibility=2),
            "right_shoulder": PoseKeypoint(name="right_shoulder", x=400, y=300, visibility=2),
            "left_shoulder": PoseKeypoint(name="left_shoulder", x=600, y=300, visibility=2),
        }
        person = PersonPose17(id=1, keypoints=kps)
        skel = person.to_cvat_skeleton(width=1000, height=1000, convention="vinfast")

        # Elements must be numbered "1".."17"
        labels = [el["label"] for el in skel["elements"]]
        assert labels == [str(i) for i in range(1, 18)]

        # Label "1" is nose (at 500, 200)
        elem_1 = skel["elements"][0]
        assert elem_1["label"] == "1"
        assert elem_1["points"] == [500.0, 200.0]

        # Label "6" is R Shoulder (COCO right_shoulder at 400, 300)
        elem_6 = skel["elements"][5]
        assert elem_6["label"] == "6"
        assert elem_6["points"] == [400.0, 300.0]

        # Label "7" is L Shoulder (COCO left_shoulder at 600, 300)
        elem_7 = skel["elements"][6]
        assert elem_7["label"] == "7"
        assert elem_7["points"] == [600.0, 300.0]


class TestCornerDumpSanitization:
    """Verify machine prediction corner dump detection and sanitization."""

    def test_detect_and_sanitize_corner_dump(self):
        from core.skeleton_contract import sanitize_corner_dump_keypoints

        # Machine dumps right ankle and left ankle at top-left [10, 10]
        kps = {
            "nose": PoseKeypoint(name="nose", x=500, y=200, visibility=2),
            "left_shoulder": PoseKeypoint(name="left_shoulder", x=550, y=300, visibility=2),
            "right_shoulder": PoseKeypoint(name="right_shoulder", x=450, y=300, visibility=2),
            "left_hip": PoseKeypoint(name="left_hip", x=540, y=500, visibility=2),
            "right_hip": PoseKeypoint(name="right_hip", x=460, y=500, visibility=2),
            "right_ankle": PoseKeypoint(name="right_ankle", x=12.0, y=8.0, visibility=2),
            "left_ankle": PoseKeypoint(name="left_ankle", x=15.0, y=10.0, visibility=2),
        }
        person = PersonPose17(id=1, keypoints=kps)

        # Quality gate detects corner dump cluster
        rep = assess_pose17_quality(person)
        assert any("corner_dump_cluster" in r for r in rep.reasons)

        # Sanitize
        sanitized = sanitize_corner_dump_keypoints(person)
        assert set(sanitized) == {"right_ankle", "left_ankle"}
        assert person.keypoints["right_ankle"].is_outside is True
        assert person.keypoints["left_ankle"].is_outside is True
        assert person.keypoints["right_ankle"].confidence == 0.0


class TestMergeRefinedKeypoint:
    """Verify Pass 2 person crop refinement merge policy and visibility normalization."""

    def test_case_a_retains_pass1_out_of_crop_limb(self):
        # Pass 1 right_ankle at [600, 950, 2] is outside crop [100, 200, 800, 700]
        crop_bbox = [100.0, 200.0, 800.0, 700.0]
        pass1_pt = [600.0, 950.0, 2]
        pass2_pt = [0.0, 0.0, 0]  # Pass 2 says outside / cannot label because limb cut off

        merged = merge_refined_keypoint("right_ankle", pass1_pt, pass2_pt, crop_bbox)
        assert merged is not None
        assert merged == [600.0, 950.0, 2]

    def test_case_b_accepts_pass2_occluded_point(self):
        # Pass 2 refines an occluded joint (e.g. wrist behind steering wheel)
        crop_bbox = [100.0, 200.0, 600.0, 700.0]
        pass1_pt = [300.0, 450.0, 2]
        pass2_pt = [200.0, 500.0, 1]  # crop-relative: x=200, y=500, vis=1 (occluded)

        merged = merge_refined_keypoint("left_wrist", pass1_pt, pass2_pt, crop_bbox)
        assert merged is not None
        # x_full = 200 + 0.2 * 500 = 300.0, y_full = 100 + 0.5 * 500 = 350.0
        assert merged[0] == pytest.approx(300.0, abs=0.5)
        assert merged[1] == pytest.approx(350.0, abs=0.5)
        assert merged[2] == VISIBILITY_OCCLUDED

    def test_case_c_discards_pass1_hallucination_inside_crop(self):
        # Pass 1 hallucinated right_wrist at [650, 300, 2] INSIDE crop bounds [100, 200, 600, 700]
        crop_bbox = [100.0, 200.0, 600.0, 700.0]
        pass1_pt = [650.0, 300.0, 2]
        pass2_pt = [0.0, 0.0, 0]  # Pass 2 inspected crop and determined wrist is absent/outside

        merged = merge_refined_keypoint("right_wrist", pass1_pt, pass2_pt, crop_bbox)
        assert merged is not None
        # Must NOT restore Pass 1 visible detection!
        assert merged[2] == VISIBILITY_OUTSIDE

    def test_case_d_falls_back_when_pass2_omitted(self):
        # Pass 2 response omitted keypoint entirely
        crop_bbox = [100.0, 200.0, 600.0, 700.0]
        pass1_pt = [450.0, 180.0, 2]
        pass2_pt = None

        merged = merge_refined_keypoint("nose", pass1_pt, pass2_pt, crop_bbox)
        assert merged is not None
        assert merged == [450.0, 180.0, 2]

    def test_happy_path_pass2_visible(self):
        # Pass 2 detected clear visible keypoint
        crop_bbox = [100.0, 200.0, 600.0, 700.0]
        pass1_pt = [450.0, 180.0, 2]
        pass2_pt = [500.0, 200.0, 2]

        merged = merge_refined_keypoint("nose", pass1_pt, pass2_pt, crop_bbox)
        assert merged is not None
        # x_full = 200 + 0.5 * 500 = 450.0, y_full = 100 + 0.2 * 500 = 200.0
        assert merged[0] == pytest.approx(450.0, abs=0.5)
        assert merged[1] == pytest.approx(200.0, abs=0.5)
        assert merged[2] == VISIBILITY_VISIBLE

    def test_visibility_normalization_variants(self):
        crop_bbox = [100.0, 200.0, 600.0, 700.0]
        pass1_pt = [450.0, 180.0, 2]

        # String aliases
        m_vis = merge_refined_keypoint("nose", pass1_pt, [500, 200, "visible"], crop_bbox)
        assert m_vis is not None and m_vis[2] == 2

        m_occ = merge_refined_keypoint("nose", pass1_pt, [500, 200, "occluded"], crop_bbox)
        assert m_occ is not None and m_occ[2] == 1

        m_out = merge_refined_keypoint("nose", [10, 10, 2], [500, 200, "outside"], crop_bbox)
        assert m_out is not None and m_out[2] == 2  # Case A: [10, 10] outside crop, retains pass1

        m_part = merge_refined_keypoint("nose", pass1_pt, [500, 200, "partial"], crop_bbox)
        assert m_part is not None and m_part[2] == 1

        # Numeric and boolean variations
        m_flt = merge_refined_keypoint("nose", pass1_pt, [500, 200, 1.0], crop_bbox)
        assert m_flt is not None and m_flt[2] == 1

        m_bool = merge_refined_keypoint("nose", pass1_pt, [500, 200, True], crop_bbox)
        assert m_bool is not None and m_bool[2] == 2

        m_neg = merge_refined_keypoint("nose", [450, 300, 2], [500, 200, -1], crop_bbox)
        assert m_neg is not None and m_neg[2] == 0  # inside crop, -1 is outside, Case C: vis=0

    def test_both_none_returns_none(self):
        crop_bbox = [100.0, 200.0, 600.0, 700.0]
        assert merge_refined_keypoint("nose", None, None, crop_bbox) is None


