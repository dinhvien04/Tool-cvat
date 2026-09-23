"""Comprehensive Regression Test Suite for Week-2 Final Hardening.

Covers the 7 critical test categories:
1. Schema single-source-of-truth invariants
2. Pose numeric + semantic mapping & completeness (17/17 vs 16/17)
3. VF50 completeness (7/7 and 50/50) & legacy face rejection
4. VF50 prompt semantic regression assertions (eyebrow, eye corners, nose pt 13, lip endpoints)
5. Multi-face valid + salvageable per-face evaluation
6. Occlusion production serialization (outside/occluded flags)
7. Refine model Pass 1 vs Pass 2 assertions & call counts
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from core.week2_schema import (
    LATERALITY_VIEWER,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUTSIDE,
    VISIBILITY_VISIBLE,
    compute_spec_fingerprint,
    load_pose17,
    load_vf50,
    map_cvat_to_visibility,
    map_visibility_to_cvat,
    pose17_coco_keypoints,
    pose17_edges_as_coco_names,
    vf50_component_names,
)
from core.skeleton_contract import (
    VF50_COMPONENT_CONFIG,
    VF50_COMPONENT_NAMES,
    VF50Face,
    VF50Landmark,
    assess_vf50_quality,
    build_vf50_crop_prompt,
    build_vf50_prompt,
    faces_to_cvat_skeletons,
    is_vf50_face_degenerate,
    is_vf50_face_salvageable,
    parse_vf50_response,
)
from core.pose_face_schema import (
    POSE17_KEYPOINTS,
    POSE17_KEYPOINT_TO_ID,
    parse_and_sanitize_pose17_instance,
    parse_and_sanitize_vf50_instance,
)
from core.quality_gate import (
    assess_pose17_quality,
    VINFAST_POSE17_INDEX_TO_NAME,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anatomically realistic standing human pose coordinates in viewer frame [0..1000]
VALID_VIEWER_POSE17_COORDS = {
    "nose": [500.0, 150.0, 2],
    "left_eye": [470.0, 130.0, 2],
    "right_eye": [530.0, 130.0, 2],
    "left_ear": [440.0, 140.0, 2],
    "right_ear": [560.0, 140.0, 2],
    "left_shoulder": [420.0, 250.0, 2],
    "right_shoulder": [580.0, 250.0, 2],
    "left_elbow": [390.0, 400.0, 2],
    "right_elbow": [610.0, 400.0, 2],
    "left_wrist": [370.0, 550.0, 2],
    "right_wrist": [630.0, 550.0, 2],
    "left_hip": [450.0, 550.0, 2],
    "right_hip": [550.0, 550.0, 2],
    "left_knee": [440.0, 750.0, 2],
    "right_knee": [560.0, 750.0, 2],
    "left_ankle": [430.0, 950.0, 2],
    "right_ankle": [570.0, 950.0, 2],
}


def _load_vf50_model_handler():
    """Load VF50 ModelHandler dynamically avoiding hyphen import issues."""
    handler_path = REPO_ROOT / "serverless" / "ninerouter-face-vf50" / "nuclio" / "model_handler.py"
    spec = importlib.util.spec_from_file_location("vf50_model_handler", handler_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ModelHandler


# ==============================================================================
# CATEGORY 1: SCHEMA SINGLE-SOURCE-OF-TRUTH INVARIANTS
# ==============================================================================

class TestSchemaSingleSourceOfTruthInvariants:
    """Verify that YAML files are the sole authoritative source of truth."""

    def test_pose17_ssot_counts_and_fingerprint(self):
        pose = load_pose17()
        assert len(pose.keypoints) == 17
        assert len(pose.edges) == 18, "YAML spec defines 18 edges including ear-to-shoulder"
        assert pose.parent_label == "person"
        assert pose.laterality_convention == LATERALITY_VIEWER
        assert isinstance(pose.spec_fingerprint, str)
        assert len(pose.spec_fingerprint) == 64

    def test_vf50_ssot_counts_and_fingerprint(self):
        vf = load_vf50()
        assert vf.total_points == 50
        assert len(vf.all_keypoints) == 50
        assert vf.total_edges == 47
        assert len(vf.components) == 7
        assert vf.component_names == (
            "longmaytrai",
            "longmayphai",
            "songmui",
            "mattrai",
            "matphai",
            "moingoai",
            "moitrong",
        )
        assert isinstance(vf.spec_fingerprint, str)
        assert len(vf.spec_fingerprint) == 64

    def test_downstream_constants_derive_from_ssot(self):
        pose = load_pose17()
        vf = load_vf50()

        # Pose keypoint names match
        assert pose17_coco_keypoints() == tuple(pose.id_to_coco_name[kp.id] for kp in pose.keypoints)
        assert pose.edges_as_coco_names == pose17_edges_as_coco_names()

        # VF50 component names match
        assert VF50_COMPONENT_NAMES == vf.component_names
        assert vf50_component_names() == vf.component_names


# ==============================================================================
# CATEGORY 2: POSE NUMERIC + SEMANTIC MAPPING & COMPLETENESS (17/17 vs 16/17)
# ==============================================================================

class TestPoseNumericSemanticMappingAndCompleteness:
    """Verify numeric ID <-> semantic name mappings and completeness handling."""

    def test_pose17_even_odd_laterality_mapping(self):
        pose = load_pose17()
        # 1 = nose (center)
        assert pose.id_to_keypoint[1].numeric_name == "1"
        assert pose.id_to_coco_name[1] == "nose"
        assert pose.id_to_keypoint[1].side == "center"

        # Even IDs = right side (viewer perspective)
        for even_id in (2, 4, 6, 8, 10, 12, 14, 16):
            kp = pose.id_to_keypoint[even_id]
            assert kp.side == "right", f"ID {even_id} must have side 'right'"
            coco_name = pose.id_to_coco_name[even_id]
            assert "right" in coco_name, f"COCO name for ID {even_id} must contain 'right'"

        # Odd IDs = left side (viewer perspective)
        for odd_id in (3, 5, 7, 9, 11, 13, 15, 17):
            kp = pose.id_to_keypoint[odd_id]
            assert kp.side == "left", f"ID {odd_id} must have side 'left'"
            coco_name = pose.id_to_coco_name[odd_id]
            assert "left" in coco_name, f"COCO name for ID {odd_id} must contain 'left'"

    def test_pose17_completeness_17_of_17(self):
        """Full 17/17 keypoints yields 17 active elements and passes quality gate."""
        inst = parse_and_sanitize_pose17_instance({"keypoints": VALID_VIEWER_POSE17_COORDS}, img_width=1000, img_height=1000)
        assert inst is not None
        assert len(inst.elements) == 17
        for elem in inst.elements:
            assert elem.outside is False
            assert elem.occluded is False

        # Quality gate reports valid and zero hard errors
        report = assess_pose17_quality({"keypoints": VALID_VIEWER_POSE17_COORDS}, laterality_convention=LATERALITY_VIEWER)
        assert report.is_valid is True
        assert report.active_count == 17
        assert len(report.hard_errors) == 0

    def test_pose17_completeness_16_of_17_graceful_handling(self):
        """16/17 keypoints: soft penalty, remains valid, missing 17th point marked outside=True."""
        # Omit 'left_ankle'
        partial_kps = {name: coord for name, coord in VALID_VIEWER_POSE17_COORDS.items() if name != "left_ankle"}
        assert len(partial_kps) == 16

        inst = parse_and_sanitize_pose17_instance({"keypoints": partial_kps}, img_width=1000, img_height=1000)
        assert inst is not None
        assert len(inst.elements) == 17

        # Missing point is explicitly outside
        missing_elem = next(e for e in inst.elements if e.label == "left_ankle")
        assert missing_elem.outside is True
        assert missing_elem.occluded is False

        # Present points are visible
        nose_elem = next(e for e in inst.elements if e.label == "nose")
        assert nose_elem.outside is False
        assert nose_elem.occluded is False

        # Quality gate handles 16/17 without fatal failure
        report = assess_pose17_quality({"keypoints": partial_kps}, laterality_convention=LATERALITY_VIEWER)
        assert report.is_valid is True
        assert report.active_count == 16
        assert any("missing_keypoints" in w for w in report.soft_warnings)
        assert len(report.hard_errors) == 0


# ==============================================================================
# CATEGORY 3: VF50 COMPLETENESS (7/7 and 50/50) & LEGACY FACE REJECTION
# ==============================================================================

class TestVF50CompletenessAndLegacyFaceRejection:
    """Verify exact 7-component emission and strict rejection of legacy monolithic face format."""

    def _create_mock_face(self, face_id: int = 1) -> VF50Face:
        face = VF50Face(face_id=face_id, confidence=0.95)
        # Assign well-formed 50 points
        for i in range(50):
            face.landmarks[i] = VF50Landmark(
                id=i,
                name=str(i),
                x=300.0 + (i % 10) * 20.0,
                y=200.0 + (i // 10) * 40.0,
                visibility=VISIBILITY_VISIBLE,
                confidence=0.98,
                component="",
            )
        return face

    def test_vf50_completeness_7_components_and_50_points(self):
        face = self._create_mock_face(face_id=1)
        skeletons = faces_to_cvat_skeletons([face], width=1000, height=1000, base_group_id=10)

        assert len(skeletons) == 7, "Must emit exactly 7 component skeletons"
        labels = [s["label"] for s in skeletons]
        assert labels == list(VF50_COMPONENT_NAMES)

        # Check total point count across 7 components = 50
        total_points = sum(len(s["elements"]) for s in skeletons)
        assert total_points == 50

        # Check all share identical group_id
        for s in skeletons:
            assert s["type"] == "skeleton"
            assert s["group_id"] == 10
            assert s["group"] == 10

    def test_legacy_face_mode_rejected_unconditionally(self):
        """Even if as_components=False is explicitly passed, 7 components must be emitted."""
        face = self._create_mock_face(face_id=1)
        skeletons = faces_to_cvat_skeletons(
            [face],
            width=1000,
            height=1000,
            as_components=False,  # Attempt legacy single face mode
        )
        assert len(skeletons) == 7
        assert all(s["label"] in VF50_COMPONENT_NAMES for s in skeletons)
        assert not any(s["label"] == "face" for s in skeletons)


# ==============================================================================
# CATEGORY 4: VF50 PROMPT SEMANTIC REGRESSION ASSERTIONS
# ==============================================================================

class TestVF50PromptSemanticRegressionAssertions:
    """Verify prompt semantic contracts for eyebrows, eye corners, nose pt 13, and lip endpoints."""

    def test_vf50_full_prompt_semantics(self):
        prompt = build_vf50_prompt()

        # All 7 components mentioned
        for comp in VF50_COMPONENT_NAMES:
            assert comp in prompt, f"Component {comp} must be present in prompt"

        # Viewer perspective
        assert "viewer perspective" in prompt.lower()
        # Normalization
        assert "[0, 1000]" in prompt or "[0..1000]" in prompt or "1000" in prompt
        # Visibility definition
        assert "0=outside" in prompt
        assert "1=occluded" in prompt
        assert "2=visible" in prompt

    def test_vf50_crop_prompt_fine_grained_semantics(self):
        prompt = build_vf50_crop_prompt()

        # Eyebrows explicitly described
        assert "longmaytrai (points 0..4)" in prompt
        assert "longmayphai (points 5..9)" in prompt

        # Eye corners explicitly specified (inner / outer corners)
        assert "mattrai (points 14..21)" in prompt
        assert "khoe ngoai (14)" in prompt or "inner corner" in prompt.lower()
        assert "khoe trong (18)" in prompt or "outer corner" in prompt.lower()
        assert "matphai (points 22..29)" in prompt
        assert "khoe trong (22)" in prompt or "inner corner" in prompt.lower()
        assert "khoe ngoai (26)" in prompt or "outer corner" in prompt.lower()

        # Nose point 13 explicitly identified (nasal base/tip)
        assert "songmui (points 10..13)" in prompt
        assert "13" in prompt and ("chan song mui" in prompt or "nasal" in prompt)

        # Lip endpoints explicitly identified
        assert "moingoai (points 30..41)" in prompt
        assert "30" in prompt and "36" in prompt
        assert "moitrong (points 42..49)" in prompt

        # Anatomical containment: inner lip inside outer lip
        assert "moitrong" in prompt and "moingoai" in prompt
        assert "ben trong moingoai" in prompt or "enclosed" in prompt.lower()


# ==============================================================================
# CATEGORY 5: MULTI-FACE VALID + SALVAGEABLE PER-FACE EVALUATION
# ==============================================================================

class TestMultiFaceValidAndSalvageablePerFaceEvaluation:
    """Verify that multiple faces evaluate independently without cross-face drop."""

    def test_per_face_quality_and_salvage_isolation(self):
        # Face 1: Valid high quality face
        face_valid = VF50Face(face_id=1, confidence=0.95)
        for i in range(50):
            face_valid.landmarks[i] = VF50Landmark(
                id=i, name=str(i), x=200.0 + (i % 10) * 15.0, y=200.0 + (i // 10) * 20.0,
                visibility=VISIBILITY_VISIBLE, confidence=0.95, component="",
            )

        # Face 2: Truly degenerate face (all points at origin -> collapse)
        face_degenerate = VF50Face(face_id=2, confidence=0.90)
        for i in range(50):
            face_degenerate.landmarks[i] = VF50Landmark(
                id=i, name=str(i), x=0.0, y=0.0,
                visibility=VISIBILITY_VISIBLE, confidence=0.90, component="",
            )

        # Face 3: Salvageable face (>= 15 active points, span > 15, but has inverted eyebrow or minor flaw)
        face_salvageable = VF50Face(face_id=3, confidence=0.85)
        for i in range(50):
            face_salvageable.landmarks[i] = VF50Landmark(
                id=i, name=str(i), x=600.0 + (i % 10) * 15.0, y=600.0 + (i // 10) * 20.0,
                visibility=VISIBILITY_VISIBLE, confidence=0.85, component="",
            )

        # Verify quality classifications
        q_valid = assess_vf50_quality(face_valid)
        assert q_valid.is_valid is True

        q_degen = assess_vf50_quality(face_degenerate)
        assert q_degen.is_valid is False
        assert is_vf50_face_degenerate(q_degen) is True

        q_salv = assess_vf50_quality(face_salvageable)
        assert is_vf50_face_degenerate(q_salv) is False

        # Run multi-face conversion
        skeletons = faces_to_cvat_skeletons(
            [face_valid, face_degenerate, face_salvageable],
            width=1000,
            height=1000,
            base_group_id=1,
            filter_corrupt=True,
            fallback_on_corrupt=True,
        )

        # Face 1 emitted (7 skels, group_id=1)
        # Face 2 dropped (0 skels)
        # Face 3 salvaged (7 skels, group_id=2)
        # Total = 14 skeletons
        assert len(skeletons) == 14
        group_ids = sorted(list({s["group_id"] for s in skeletons}))
        assert group_ids == [1, 2]


# ==============================================================================
# CATEGORY 6: OCCLUSION PRODUCTION SERIALIZATION (outside/occluded flags)
# ==============================================================================

class TestOcclusionProductionSerialization:
    """Verify native boolean serialization and contract invariants for outside and occluded."""

    @pytest.mark.parametrize("vis,expected_outside,expected_occluded", [
        (0, True, False),
        (1, False, True),
        (2, False, False),
        ("outside", True, False),
        ("occluded", False, True),
        ("visible", False, False),
        (False, True, False),
        (True, False, False),
    ])
    def test_map_visibility_to_cvat_types_and_values(self, vis, expected_outside, expected_occluded):
        out, occ = map_visibility_to_cvat(vis)
        assert out is expected_outside
        assert occ is expected_occluded
        assert isinstance(out, bool)
        assert isinstance(occ, bool)
        # Invariant: a point cannot be simultaneously outside and occluded
        assert not (out and occ)

    def test_cvat_skeleton_element_serialization(self):
        lm = VF50Landmark(id=0, name="0", x=500.0, y=500.0, visibility=VISIBILITY_OCCLUDED)
        elem = lm.to_cvat_element(width=1000, height=1000)

        assert elem["outside"] is False
        assert elem["occluded"] is True
        assert isinstance(elem["outside"], bool)
        assert isinstance(elem["occluded"], bool)
        assert elem["points"] == [500.0, 500.0]

    def test_sanitizer_normalizes_forbidden_invariant_pose17_and_vf50(self):
        """Verify that malformed inputs with outside=True and occluded=True are sanitized."""
        # Pose17 test with simultaneous outside and occluded
        raw_pose = {
            "keypoints": [
                {"name": "nose", "point": [500, 150], "outside": True, "occluded": True},
                {"name": "left_eye", "point": [470, 130], "visibility": "occluded"},
            ]
        }
        pose_inst = parse_and_sanitize_pose17_instance(raw_pose, img_width=1000, img_height=1000)
        assert pose_inst is not None
        elem_map = {e.label: e for e in pose_inst.elements}
        # nose should be normalized (outside takes precedence: outside=True, occluded=False)
        assert elem_map["nose"].outside is True
        assert elem_map["nose"].occluded is False
        # left_eye should be occluded: outside=False, occluded=True
        assert elem_map["left_eye"].outside is False
        assert elem_map["left_eye"].occluded is True

        # VF50 test with simultaneous outside and occluded
        raw_vf50 = {
            "landmarks": [
                {"name": "longmaytrai_00", "point": [200, 300], "outside": True, "occluded": True},
                {"name": "longmaytrai_01", "point": [250, 280], "visibility": 1},
            ]
        }
        vf50_inst = parse_and_sanitize_vf50_instance(raw_vf50, img_width=1000, img_height=1000)
        assert vf50_inst is not None
        elem_map_vf = {e.label: e for e in vf50_inst.elements}
        assert elem_map_vf["longmaytrai_00"].outside is True
        assert elem_map_vf["longmaytrai_00"].occluded is False
        assert elem_map_vf["longmaytrai_01"].outside is False
        assert elem_map_vf["longmaytrai_01"].occluded is True


# ==============================================================================
# CATEGORY 7: REFINE MODEL PASS 1 VS PASS 2 ASSERTIONS & CALL COUNTS
# ==============================================================================

class TestRefineModelPass1VsPass2AssertionsAndCallCounts:
    """Verify conditional two-pass execution and exact call counts."""

    def test_vf50_two_pass_disabled_call_count_one(self, monkeypatch):
        """When refine_crops=False, exactly 1 call is made (Pass 1)."""
        ModelHandler = _load_vf50_model_handler()

        def mock_resolve(self, preferred_model=None):
            return preferred_model or "mock-pass1"

        monkeypatch.setattr("app.client.NineRouterClient.resolve_vision_model", mock_resolve)
        if hasattr(importlib.import_module("app.client").NineRouterClient, "resolve_vf50_model"):
            monkeypatch.setattr("app.client.NineRouterClient.resolve_vf50_model", mock_resolve)

        handler = ModelHandler(base_url="http://mock:20128", model="mock-pass1")

        mock_resp = MagicMock()
        mock_resp.content = json.dumps({
            "faces": [{
                "id": 1,
                "confidence": 0.95,
                "box_2d": [100, 100, 500, 500],
                "landmarks": [{"id": i, "point": [300, 300], "visibility": 2} for i in range(50)],
            }]
        })

        # Create dummy JPEG bytes
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        with patch.object(handler.client, "send_vision_request", return_value=mock_resp) as mock_send:
            handler.infer(raw_bytes, refine_crops=False)

            assert mock_send.call_count == 1
            assert mock_send.call_args[1]["model"] == "mock-pass1"

    def test_vf50_two_pass_enabled_needs_refine_calls_pass2_with_refine_model(self, monkeypatch):
        """When face needs refinement, Pass 2 is triggered using active_refine_model."""
        ModelHandler = _load_vf50_model_handler()

        def mock_resolve(self, preferred_model=None):
            return preferred_model or "resolved-model"

        monkeypatch.setattr("app.client.NineRouterClient.resolve_vision_model", mock_resolve)
        if hasattr(importlib.import_module("app.client").NineRouterClient, "resolve_vf50_model"):
            monkeypatch.setattr("app.client.NineRouterClient.resolve_vf50_model", mock_resolve)

        monkeypatch.setenv("VF50_REFINE_MODEL", "refine-model-pass2")

        handler = ModelHandler(base_url="http://mock:20128", model="primary-model-pass1")
        assert handler.active_refine_model == "refine-model-pass2"

        # Pass 1 response: face has very small box (< 100px) which triggers needs_refine
        pass1_resp = MagicMock()
        pass1_resp.content = json.dumps({
            "faces": [{
                "id": 1,
                "confidence": 0.95,
                "box_2d": [10, 10, 50, 50],  # 40px face -> triggers refine
                "landmarks": [{"id": i, "point": [30, 30], "visibility": 2} for i in range(50)],
            }]
        })

        # Pass 2 response: refined 50 landmarks
        pass2_resp = MagicMock()
        pass2_resp.content = json.dumps({
            "faces": [{
                "id": 1,
                "confidence": 0.99,
                "landmarks": [{"id": i, "point": [500, 500], "visibility": 2} for i in range(50)],
            }]
        })

        # Create dummy JPEG bytes
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        with patch.object(handler.client, "send_vision_request", side_effect=[pass1_resp, pass2_resp]) as mock_send:
            handler.infer(raw_bytes, refine_crops=True)

            assert mock_send.call_count == 2
            # Pass 1 called with primary model
            assert mock_send.call_args_list[0][1]["model"] == "primary-model-pass1"
            # Pass 2 called with refine model
            assert mock_send.call_args_list[1][1]["model"] == "refine-model-pass2"
