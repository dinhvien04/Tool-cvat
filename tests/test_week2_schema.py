"""Tests for Canonical Week-2 Schema Loader (core/week2_schema.py).

Validates:
- Pose17 schema integrity (17 keypoints, 18 edges including ear-to-shoulder).
- VF50 schema integrity (50 points, 47 edges across 7 components).
- Laterality conventions and COCO name mappings.
- Deterministic SHA-256 fingerprinting.
- CVAT visibility contract round-tripping.
- Build SHA resolution.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from core.week2_schema import (
    LATERALITY_VIEWER,
    POSE_CVAT_SPEC_HASH,
    POSE_SCHEMA_HASH,
    VF50_CVAT_SPEC_HASH,
    VF50_SCHEMA_HASH,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUTSIDE,
    VISIBILITY_VISIBLE,
    compute_spec_fingerprint,
    get_build_sha,
    get_canonical_cvat_spec,
    get_canonical_cvat_spec_hash,
    get_canonical_schema_hash,
    load_pose17,
    load_vf50,
    map_cvat_to_visibility,
    map_visibility_to_cvat,
    pose17_coco_keypoints,
    pose17_edges_0indexed,
    pose17_edges_as_coco_names,
    vf50_component_names,
    vf50_component_point_counts,
    vf50_component_point_ranges,
)


class TestPose17CanonicalSchema:
    """Validate Pose17 schema loaded from config/week2_pose17.yaml."""

    def test_pose17_point_and_edge_counts(self):
        pose = load_pose17()
        assert len(pose.keypoints) == 17
        assert len(pose.edges) == 18, "Authoritative YAML spec defines exactly 18 edges including ear-to-shoulder"
        assert pose.parent_label == "person"
        assert pose.laterality_convention == LATERALITY_VIEWER

    def test_pose17_keypoint_attributes(self):
        pose = load_pose17()
        assert len(pose.keypoint_names) == 17
        assert pose.keypoint_names == frozenset(str(i) for i in range(1, 18))

        # Check even/odd viewer laterality distribution
        assert len(pose.right_keypoint_ids) == 8  # 2, 4, 6, 8, 10, 12, 14, 16
        assert len(pose.left_keypoint_ids) == 8   # 3, 5, 7, 9, 11, 13, 15, 17
        assert pose.id_to_keypoint[1].side == "center"

    def test_pose17_ear_to_shoulder_edges_present(self):
        pose = load_pose17()
        # 1-based edge tuples: (4, 6) right_ear -> right_shoulder, (5, 7) left_ear -> left_shoulder
        assert (4, 6) in pose.edges
        assert (5, 7) in pose.edges

        # Named edges
        assert ("right_ear", "right_shoulder") in pose.edges_as_coco_names
        assert ("left_ear", "left_shoulder") in pose.edges_as_coco_names

    def test_pose17_convenience_accessors(self):
        coco_kps = pose17_coco_keypoints()
        assert len(coco_kps) == 17
        assert coco_kps[0] == "nose"
        assert coco_kps[1] == "right_eye"
        assert coco_kps[2] == "left_eye"

        edges_0idx = pose17_edges_0indexed()
        assert len(edges_0idx) == 18
        # (4, 6) -> (3, 5)
        assert (3, 5) in edges_0idx
        assert (4, 6) in edges_0idx

        coco_edges = pose17_edges_as_coco_names()
        assert len(coco_edges) == 18


class TestVF50CanonicalSchema:
    """Validate VF50 schema loaded from config/week2_vf50.yaml."""

    def test_vf50_total_counts(self):
        vf = load_vf50()
        assert vf.total_points == 50
        assert len(vf.all_keypoints) == 50
        assert vf.total_edges == 47
        assert len(vf.all_edges) == 47
        assert len(vf.components) == 7

    def test_vf50_canonical_component_names_and_ranges(self):
        expected_names = (
            "longmaytrai",
            "longmayphai",
            "songmui",
            "mattrai",
            "matphai",
            "moingoai",
            "moitrong",
        )
        vf = load_vf50()
        assert vf.component_names == expected_names
        assert vf50_component_names() == expected_names

        ranges = vf50_component_point_ranges()
        assert ranges["longmaytrai"] == (0, 4)
        assert ranges["longmayphai"] == (5, 9)
        assert ranges["songmui"] == (10, 13)
        assert ranges["mattrai"] == (14, 21)
        assert ranges["matphai"] == (22, 29)
        assert ranges["moingoai"] == (30, 41)
        assert ranges["moitrong"] == (42, 49)

        counts = vf50_component_point_counts()
        assert counts["longmaytrai"] == 5
        assert counts["longmayphai"] == 5
        assert counts["songmui"] == 4
        assert counts["mattrai"] == 8
        assert counts["matphai"] == 8
        assert counts["moingoai"] == 12
        assert counts["moitrong"] == 8
        assert sum(counts.values()) == 50

    def test_vf50_point_to_component_mapping(self):
        vf = load_vf50()
        assert vf.point_to_component[0] == "longmaytrai"
        assert vf.point_to_component[4] == "longmaytrai"
        assert vf.point_to_component[5] == "longmayphai"
        assert vf.point_to_component[10] == "songmui"
        assert vf.point_to_component[14] == "mattrai"
        assert vf.point_to_component[22] == "matphai"
        assert vf.point_to_component[30] == "moingoai"
        assert vf.point_to_component[42] == "moitrong"
        assert vf.point_to_component[49] == "moitrong"


class TestVisibilityContract:
    """Validate CVAT visibility contract mappings."""

    def test_map_visibility_to_cvat(self):
        # 0: outside=True, occluded=False
        assert map_visibility_to_cvat(VISIBILITY_OUTSIDE) == (True, False)
        # 1: outside=False, occluded=True
        assert map_visibility_to_cvat(VISIBILITY_OCCLUDED) == (False, True)
        # 2: outside=False, occluded=False
        assert map_visibility_to_cvat(VISIBILITY_VISIBLE) == (False, False)

    def test_map_cvat_to_visibility(self):
        assert map_cvat_to_visibility(outside=True, occluded=False) == VISIBILITY_OUTSIDE
        assert map_cvat_to_visibility(outside=False, occluded=True) == VISIBILITY_OCCLUDED
        assert map_cvat_to_visibility(outside=False, occluded=False) == VISIBILITY_VISIBLE

    def test_visibility_round_trip(self):
        for vis in (VISIBILITY_OUTSIDE, VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE):
            outside, occluded = map_visibility_to_cvat(vis)
            assert map_cvat_to_visibility(outside=outside, occluded=occluded) == vis


class TestSpecFingerprintingAndBuildSha:
    """Validate SHA-256 fingerprinting and build SHA env retrieval."""

    def test_fingerprint_deterministic(self):
        data = {"b": 2, "a": [1, 2, 3]}
        fp1 = compute_spec_fingerprint(data)
        fp2 = compute_spec_fingerprint({"a": [1, 2, 3], "b": 2})
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_pose17_and_vf50_have_fingerprints(self):
        pose = load_pose17()
        vf = load_vf50()
        assert isinstance(pose.spec_fingerprint, str)
        assert len(pose.spec_fingerprint) == 64
        assert isinstance(vf.spec_fingerprint, str)
        assert len(vf.spec_fingerprint) == 64
        assert pose.spec_fingerprint != vf.spec_fingerprint

    def test_get_build_sha_returns_env_var(self):
        with mock.patch.dict(os.environ, {"TOOL_CVAT_BUILD_SHA": "abc123def456"}):
            assert get_build_sha() == "abc123def456"

        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_build_sha() is None

    def test_schema_hash_vs_cvat_spec_hash_separation(self):
        """Verify semantic schema hash is strictly separated from CVAT spec hash."""
        assert len(POSE_SCHEMA_HASH) == 64
        assert len(VF50_SCHEMA_HASH) == 64
        assert len(POSE_CVAT_SPEC_HASH) == 64
        assert len(VF50_CVAT_SPEC_HASH) == 64

        # Semantic schema fingerprints the abstract anatomy definition
        # CVAT spec fingerprints the concrete CVAT label dictionary list with SVG
        assert POSE_SCHEMA_HASH != POSE_CVAT_SPEC_HASH
        assert VF50_SCHEMA_HASH != VF50_CVAT_SPEC_HASH
        assert POSE_SCHEMA_HASH != VF50_SCHEMA_HASH
        assert POSE_CVAT_SPEC_HASH != VF50_CVAT_SPEC_HASH

    def test_canonical_spec_getters_and_hashes(self):
        """Verify getters return valid specs and matching fingerprints."""
        pose_spec = get_canonical_cvat_spec("pose17")
        assert isinstance(pose_spec, list)
        assert len(pose_spec) == 1
        assert pose_spec[0]["name"] == "person"
        assert len(pose_spec[0]["sublabels"]) == 17
        assert get_canonical_cvat_spec_hash("pose17") == POSE_CVAT_SPEC_HASH
        assert get_canonical_schema_hash("pose17") == POSE_SCHEMA_HASH

        vf50_spec = get_canonical_cvat_spec("vf50")
        assert isinstance(vf50_spec, list)
        assert len(vf50_spec) == 7
        assert get_canonical_cvat_spec_hash("vf50") == VF50_CVAT_SPEC_HASH
        assert get_canonical_schema_hash("vf50") == VF50_SCHEMA_HASH

    def test_canonical_getters_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown detector key"):
            get_canonical_cvat_spec("unknown")

        with pytest.raises(ValueError, match="Unknown detector key"):
            get_canonical_schema_hash("unknown")

        with pytest.raises(ValueError, match="Unknown detector key"):
            get_canonical_cvat_spec_hash("unknown")


class TestSingleSourceOfTruthContractConsistency:
    """Strict repository-level verification that downstream modules derive directly from SSOT schemas."""

    def test_skeleton_contract_matches_yaml_ssot(self):
        import core.skeleton_contract as sc
        from core.week2_schema import (
            vf50_all_edges,
            vf50_component_configs,
            vf50_edges_by_component,
            vf50_point_to_component,
        )

        pose = load_pose17()
        vf = load_vf50()

        # Pose17 SSOT consistency
        assert sc.POSE17_KEYPOINTS == pose17_coco_keypoints()
        assert sc.KEYPOINT_COUNT == 17
        assert sc.POSE17_SKELETON_EDGES == pose17_edges_as_coco_names()
        assert sc.DEFAULT_PARENT_LABEL == pose.parent_label
        assert len(sc.POSE17_SKELETON_EDGES) == 18

        # VF50 SSOT consistency
        assert sc.VF50_COMPONENT_NAMES == vf.component_names
        assert sc.VF50_POINTS_COUNT == 50
        assert sc.VF50_COMPONENT_CONFIG == vf50_component_configs()
        assert sc.VF50_ALL_EDGES == vf50_all_edges()
        assert sc.VF50_POINT_TO_COMPONENT == dict(vf.point_to_component)
        assert sc.VF50_EDGES_BY_COMPONENT == vf50_edges_by_component()

    def test_pose_face_schema_matches_yaml_ssot(self):
        import core.pose_face_schema as pfs
        from core.week2_schema import (
            vf50_all_edges,
            vf50_landmarks,
            vf50_mirror_map,
        )

        pose = load_pose17()
        vf = load_vf50()

        assert pfs.POSE17_KEYPOINTS == pose17_coco_keypoints()
        assert pfs.POSE17_EDGES == pose.edges
        assert len(pfs.POSE17_EDGES) == 18
        assert pfs.VF50_LANDMARKS == vf50_landmarks()
        assert len(pfs.VF50_LANDMARKS) == 50
        assert pfs.VF50_EDGES == vf50_all_edges()
        assert pfs.VF50_MIRROR_MAP == vf50_mirror_map()

    def test_quality_gate_matches_yaml_ssot(self):
        import core.quality_gate as qg
        from core.week2_schema import (
            pose17_articulated_paired_keypoints,
            pose17_paired_keypoints,
            pose17_rigid_paired_keypoints,
            vf50_component_point_counts,
            vf50_eyelid_opposing_pairs,
        )

        vf = load_vf50()

        assert qg.POSE17_KEYPOINTS == pose17_coco_keypoints()
        assert qg.POSE17_SKELETON_EDGES == pose17_edges_as_coco_names()
        assert len(qg.POSE17_SKELETON_EDGES) == 18
        assert qg.POSE17_PAIRED_KEYPOINTS == pose17_paired_keypoints()
        assert qg.POSE17_RIGID_PAIRED_KEYPOINTS == pose17_rigid_paired_keypoints()
        assert qg.POSE17_ARTICULATED_PAIRED_KEYPOINTS == pose17_articulated_paired_keypoints()
        assert qg.VF50_COMPONENT_NAMES == vf.component_names
        assert qg.VF50_COMPONENT_COUNTS == vf50_component_point_counts()
        assert qg.VF50_EXPECTED_TOTAL_POINTS == 50
        assert qg.VF50_EYELID_OPPOSING_PAIRS == vf50_eyelid_opposing_pairs()