"""Comprehensive Unit and Integration Test Suite for VF-50 Face Landmark Pipeline.

Tests Cover Mandate:
- Section 10 & 11: Face detection & crop coordinate mapping (zero off-by-one, no axis inversion).
- Section 10: Parent vs Component skeleton architecture (7 components totaling 50 points, 47 edges).
- Section 12: Multi-face isolation (0 faces -> [], 1 face -> 7 shapes, N faces -> distinct group_id, no mixing).
- Section 17 & 18: Quality gate, semantic distinctions (eyelid boundaries, outer/inner lip containment, nose centrality, closed mouth/eyes).
- Response parsing & prompt/spec building.
"""

from __future__ import annotations

import json
import math
import pytest
from typing import Any, Dict, List

from core.skeleton_contract import (
    VF50_ALL_EDGES,
    VF50_ANCHOR_POINTS,
    VF50_COMPONENT_CONFIG,
    VF50_COMPONENT_NAMES,
    VF50_EDGES_BY_COMPONENT,
    VF50_LATERALITY,
    VF50_POINT_NAMES,
    VF50_POINT_TO_COMPONENT,
    VF50_POINTS_COUNT,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUTSIDE,
    VISIBILITY_VISIBLE,
    VF50Face,
    VF50Landmark,
    VF50QualityReport,
    assess_vf50_quality,
    build_cvat_vf50_spec,
    build_vf50_prompt,
    crop_to_image_coords,
    expand_face_bbox,
    faces_to_cvat_skeletons,
    image_to_crop_coords,
    parse_vf50_response,
)


# ==============================================================================
# Helper to build a canonical synthetic frontal face in normalized 0..1000 space
# ==============================================================================

def create_canonical_frontal_face(face_id: int = 1, confidence: float = 0.98) -> VF50Face:
    """Create a topologically valid frontal face for testing."""
    lms: Dict[int, VF50Landmark] = {}

    # longmaytrai (0..4): left eyebrow (viewer perspective: left side of image, x small)
    # y ~ 200, x ranges 250..400
    for idx, x in enumerate([260.0, 290.0, 330.0, 370.0, 410.0]):
        lms[idx] = VF50Landmark(id=idx, name=str(idx), x=x, y=200.0 - 10.0 * math.sin(idx * 0.7), visibility=2)

    # longmayphai (5..9): right eyebrow (viewer perspective: right side, x larger)
    # y ~ 200, x ranges 590..740
    for idx, x in enumerate([590.0, 630.0, 670.0, 710.0, 740.0], start=5):
        lms[idx] = VF50Landmark(id=idx, name=str(idx), x=x, y=200.0 - 10.0 * math.sin((idx - 5) * 0.7), visibility=2)

    # songmui (10..13): nose bridge
    # x ~ 500, y descends from 250 down to 450
    for idx, y in enumerate([250.0, 310.0, 370.0, 430.0], start=10):
        lms[idx] = VF50Landmark(id=idx, name=str(idx), x=500.0, y=y, visibility=2)

    # mattrai (14..21): left eye (closed 8-gon)
    # Center ~(340, 280), width ~100, height ~40
    # 14: outer corner (leftmost: x=290, y=280)
    # 15, 16, 17: upper eyelid (y ~ 260)
    # 18: inner corner (rightmost: x=390, y=280)
    # 19, 20, 21: lower eyelid (y ~ 300)
    mattrai_coords = [
        (14, 290.0, 280.0),  # outer corner
        (15, 315.0, 265.0),  # upper 1/4 out
        (16, 340.0, 260.0),  # upper center (peak)
        (17, 365.0, 265.0),  # upper 1/4 in
        (18, 390.0, 280.0),  # inner corner
        (19, 365.0, 295.0),  # lower 1/4 in
        (20, 340.0, 300.0),  # lower center (trough)
        (21, 315.0, 295.0),  # lower 1/4 out
    ]
    for pt_id, x, y in mattrai_coords:
        lms[pt_id] = VF50Landmark(id=pt_id, name=str(pt_id), x=x, y=y, visibility=2)

    # matphai (22..29): right eye (closed 8-gon)
    # Center ~(660, 280), width ~100, height ~40
    # 22: inner corner (leftmost: x=610, y=280)
    # 23, 24, 25: upper eyelid (y ~ 260)
    # 26: outer corner (rightmost: x=710, y=280)
    # 27, 28, 29: lower eyelid (y ~ 300)
    matphai_coords = [
        (22, 610.0, 280.0),  # inner corner
        (23, 635.0, 265.0),  # upper 1/4 in
        (24, 660.0, 260.0),  # upper center (peak)
        (25, 685.0, 265.0),  # upper 1/4 out
        (26, 710.0, 280.0),  # outer corner
        (27, 685.0, 295.0),  # lower 1/4 out
        (28, 660.0, 300.0),  # lower center (trough)
        (29, 635.0, 295.0),  # lower 1/4 in
    ]
    for pt_id, x, y in matphai_coords:
        lms[pt_id] = VF50Landmark(id=pt_id, name=str(pt_id), x=x, y=y, visibility=2)

    # moingoai (30..41): outer lip (12-gon)
    # Center ~(500, 600), width ~160, height ~70
    # 30: left corner (420, 600)
    # 31..35: upper lip
    # 36: right corner (580, 600)
    # 37..41: lower lip
    moingoai_coords = [
        (30, 420.0, 600.0),
        (31, 450.0, 580.0),
        (32, 480.0, 575.0),
        (33, 500.0, 570.0),  # upper center
        (34, 520.0, 575.0),
        (35, 550.0, 580.0),
        (36, 580.0, 600.0),
        (37, 550.0, 625.0),
        (38, 525.0, 635.0),
        (39, 500.0, 640.0),  # lower center
        (40, 475.0, 635.0),
        (41, 450.0, 625.0),
    ]
    for pt_id, x, y in moingoai_coords:
        lms[pt_id] = VF50Landmark(id=pt_id, name=str(pt_id), x=x, y=y, visibility=2)

    # moitrong (42..49): inner lip (8-gon) strictly contained inside moingoai
    # Center ~(500, 600), width ~100, height ~30
    moitrong_coords = [
        (42, 445.0, 600.0),
        (43, 470.0, 590.0),
        (44, 500.0, 588.0),  # upper inner
        (45, 530.0, 590.0),
        (46, 555.0, 600.0),
        (47, 530.0, 610.0),
        (48, 500.0, 612.0),  # lower inner
        (49, 470.0, 610.0),
    ]
    for pt_id, x, y in moitrong_coords:
        lms[pt_id] = VF50Landmark(id=pt_id, name=str(pt_id), x=x, y=y, visibility=2)

    return VF50Face(
        face_id=face_id,
        box_2d=[150, 220, 700, 780],
        confidence=confidence,
        landmarks=lms,
    )


# ==============================================================================
# SECTION 10: TOPOLOGY & SCHEMA
# ==============================================================================

class TestVF50Topology:
    """Verify authoritative 7-skeleton structure, point counts, and edge counts."""

    def test_component_counts_and_names(self):
        assert len(VF50_COMPONENT_NAMES) == 7
        assert VF50_COMPONENT_NAMES == (
            "longmaytrai",
            "longmayphai",
            "songmui",
            "mattrai",
            "matphai",
            "moingoai",
            "moitrong",
        )
        assert VF50_POINTS_COUNT == 50
        assert len(VF50_POINT_NAMES) == 50
        assert len(VF50_POINT_TO_COMPONENT) == 50

    def test_point_ranges(self):
        assert VF50_COMPONENT_CONFIG["longmaytrai"]["count"] == 5
        assert VF50_COMPONENT_CONFIG["longmayphai"]["count"] == 5
        assert VF50_COMPONENT_CONFIG["songmui"]["count"] == 4
        assert VF50_COMPONENT_CONFIG["mattrai"]["count"] == 8
        assert VF50_COMPONENT_CONFIG["matphai"]["count"] == 8
        assert VF50_COMPONENT_CONFIG["moingoai"]["count"] == 12
        assert VF50_COMPONENT_CONFIG["moitrong"]["count"] == 8

        total = sum(cfg["count"] for cfg in VF50_COMPONENT_CONFIG.values())
        assert total == 50

    def test_edge_counts(self):
        assert len(VF50_EDGES_BY_COMPONENT["longmaytrai"]) == 4
        assert len(VF50_EDGES_BY_COMPONENT["longmayphai"]) == 4
        assert len(VF50_EDGES_BY_COMPONENT["songmui"]) == 3
        assert len(VF50_EDGES_BY_COMPONENT["mattrai"]) == 8
        assert len(VF50_EDGES_BY_COMPONENT["matphai"]) == 8
        assert len(VF50_EDGES_BY_COMPONENT["moingoai"]) == 12
        assert len(VF50_EDGES_BY_COMPONENT["moitrong"]) == 8
        assert len(VF50_ALL_EDGES) == 47

    def test_anchor_points_count(self):
        assert len(VF50_ANCHOR_POINTS) == 12
        assert VF50_ANCHOR_POINTS == (0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36)


# ==============================================================================
# SECTION 10 & 11: CROP STRATEGY & COORDINATE TRANSFORMS
# ==============================================================================

class TestCropCoordinateTransforms:
    """Verify face crop to full image coordinate transformations with zero off-by-one."""

    def test_expand_face_bbox(self):
        box = [200, 300, 600, 700]  # h=400, w=400
        expanded = expand_face_bbox(box, margin=0.10)  # pad 40 on each side
        assert expanded == [160, 260, 640, 740]

    def test_expand_face_bbox_clamped_to_image_boundaries(self):
        box = [50, 20, 950, 980]
        expanded = expand_face_bbox(box, margin=0.15)
        assert expanded[0] == 0  # clamped to 0
        assert expanded[1] == 0
        assert expanded[2] == 1000  # clamped to 1000
        assert expanded[3] == 1000

    def test_crop_to_image_mapping_zero_off_by_one(self):
        # Image 1280 x 720
        # Crop occupies normalized [200, 300, 600, 700]
        crop_box = [200, 300, 600, 700]
        w, h = 1280, 720

        # Top-left of crop (0, 0) in crop space -> (300, 200) in image normalized space
        # px_x = 300 / 1000 * 1280 = 384.0
        # px_y = 200 / 1000 * 720 = 144.0
        px, py = crop_to_image_coords(0.0, 0.0, crop_box, orig_w=w, orig_h=h)
        assert px == pytest.approx(384.0, abs=0.01)
        assert py == pytest.approx(144.0, abs=0.01)

        # Bottom-right of crop (1000, 1000) in crop space -> (700, 600) in normalized space
        # px_x = 700 / 1000 * 1280 = 896.0
        # px_y = 600 / 1000 * 720 = 432.0
        px, py = crop_to_image_coords(1000.0, 1000.0, crop_box, orig_w=w, orig_h=h)
        assert px == pytest.approx(896.0, abs=0.01)
        assert py == pytest.approx(432.0, abs=0.01)

        # Center of crop (500, 500) -> (500, 400) in normalized space
        # px_x = 500 / 1000 * 1280 = 640.0
        # px_y = 400 / 1000 * 720 = 288.0
        px, py = crop_to_image_coords(500.0, 500.0, crop_box, orig_w=w, orig_h=h)
        assert px == pytest.approx(640.0, abs=0.01)
        assert py == pytest.approx(288.0, abs=0.01)

    def test_full_image_crop_is_identity(self):
        full_crop = [0, 0, 1000, 1000]
        w, h = 1920, 1080
        px, py = crop_to_image_coords(500.0, 500.0, full_crop, orig_w=w, orig_h=h)
        assert px == 960.0
        assert py == 540.0

    def test_roundtrip_crop_mapping(self):
        crop_box = [150, 250, 650, 750]
        orig_norm_x, orig_norm_y = 500.0, 400.0
        crop_x, crop_y = image_to_crop_coords(orig_norm_x, orig_norm_y, crop_box)

        # Map back to full image
        px, py = crop_to_image_coords(crop_x, crop_y, crop_box, orig_w=1000, orig_h=1000)
        assert px == pytest.approx(orig_norm_x, abs=0.2)
        assert py == pytest.approx(orig_norm_y, abs=0.2)


# ==============================================================================
# SECTION 10 & 12: CVAT SKELETON GENERATION & MULTI-FACE ISOLATION
# ==============================================================================

class TestCvatSkeletonGeneration:
    """Verify CVAT shape emission: 7 component skeletons, shared group_id, multi-face."""

    def test_single_face_emits_7_component_skeletons(self):
        face = create_canonical_frontal_face(face_id=1)
        shapes = face.to_cvat_component_skeletons(width=1280, height=720, group_id=42)

        assert len(shapes) == 7
        labels = [s["label"] for s in shapes]
        assert labels == list(VF50_COMPONENT_NAMES)

        # All 7 shapes must share the same group_id
        for s in shapes:
            assert s["type"] == "skeleton"
            assert s["group_id"] == 42
            assert isinstance(s["elements"], list)

        # Check element counts per component
        assert len(shapes[0]["elements"]) == 5   # longmaytrai
        assert len(shapes[1]["elements"]) == 5   # longmayphai
        assert len(shapes[2]["elements"]) == 4   # songmui
        assert len(shapes[3]["elements"]) == 8   # mattrai
        assert len(shapes[4]["elements"]) == 8   # matphai
        assert len(shapes[5]["elements"]) == 12  # moingoai
        assert len(shapes[6]["elements"]) == 8   # moitrong

        # Sublabels must match string numbers "0".."49"
        sublabel_names = [elem["label"] for s in shapes for elem in s["elements"]]
        assert sublabel_names == [str(i) for i in range(50)]

    def test_single_face_unified_parent_mode(self):
        face = create_canonical_frontal_face(face_id=1)
        shape = face.to_cvat_single_skeleton(width=1280, height=720, label="face", group_id=10)

        assert shape["type"] == "skeleton"
        assert shape["label"] == "face"
        assert shape["group_id"] == 10
        assert len(shape["elements"]) == 50

    def test_zero_faces_emits_empty_list(self):
        shapes = faces_to_cvat_skeletons([], width=1280, height=720)
        assert shapes == []

    def test_multi_face_isolation_and_distinct_groups(self):
        f1 = create_canonical_frontal_face(face_id=1)
        f2 = create_canonical_frontal_face(face_id=2)
        # Shift f2 landmarks horizontally
        for lm in f2.landmarks.values():
            lm.x = min(1000.0, lm.x + 50.0)

        shapes = faces_to_cvat_skeletons([f1, f2], width=1280, height=720, base_group_id=1)
        assert len(shapes) == 14  # 2 faces * 7 skeletons

        # First 7 shapes must have group_id=1
        for s in shapes[:7]:
            assert s["group_id"] == 1

        # Next 7 shapes must have group_id=2
        for s in shapes[7:]:
            assert s["group_id"] == 2

        # Skeletons of Face 1 and Face 2 must not mix elements
        f1_px_0 = shapes[0]["elements"][0]["points"][0]
        f2_px_0 = shapes[7]["elements"][0]["points"][0]
        assert f1_px_0 != f2_px_0


# ==============================================================================
# SECTION 17 & 18: DETERMINISTIC QUALITY GATE
# ==============================================================================

class TestVF50QualityGate:
    """Verify geometric sanity checks, eyelid boundary containment, and topological rules."""

    def test_canonical_face_passes_quality_gate(self):
        face = create_canonical_frontal_face()
        report = assess_vf50_quality(face, width=1280, height=720)
        assert report.is_valid is True
        assert report.score >= 0.85
        assert report.anomalies == []

    def test_degenerate_collapse_rejection(self):
        face = create_canonical_frontal_face()
        # Collapse all points to single point (500, 500)
        for lm in face.landmarks.values():
            lm.x = 500.0
            lm.y = 500.0

        report = assess_vf50_quality(face)
        assert report.is_valid is False
        assert any("degenerate_collapse" in r for r in report.reasons)

    def test_insufficient_active_landmarks(self):
        face = create_canonical_frontal_face()
        # Mark 35 points as outside
        for i in range(35):
            face.landmarks[i].visibility = VISIBILITY_OUTSIDE

        report = assess_vf50_quality(face, min_active_landmarks=20)
        assert report.is_valid is False
        assert any("too_few_landmarks" in r for r in report.reasons)

    def test_eyelid_inversion_detected(self):
        face = create_canonical_frontal_face()
        # Invert left upper and lower eyelids: pull upper eyelid (pt 16) below lower eyelid (pt 20)
        face.landmarks[16].y = 350.0  # Lower than pt 20 (y=300.0)
        face.landmarks[20].y = 250.0  # Higher than pt 16

        report = assess_vf50_quality(face)
        assert any("left_eyelid_inversion" in a for a in report.anomalies)

    def test_closed_eyes_does_not_fail_quality_gate(self):
        """Guideline Section 6.2: Squinting or closed eyes allow upper and lower eyelids to coincide."""
        face = create_canonical_frontal_face()
        # Set upper and lower eyelids along the seam y=280
        for pt_id in [15, 16, 17, 19, 20, 21]:
            face.landmarks[pt_id].y = 280.0

        report = assess_vf50_quality(face)
        # Should not trigger eyelid inversion
        assert not any("left_eyelid_inversion" in a for a in report.anomalies)
        assert report.is_valid is True

    def test_self_intersecting_eye_contour_x_crossing_detected(self):
        """Guideline Section 3.3 & 9: Eye contour must not cross into an X-shape."""
        face = create_canonical_frontal_face()
        # Swap outer and inner upper points to create crossing edges
        pt15_orig = (face.landmarks[15].x, face.landmarks[15].y)
        face.landmarks[15].x, face.landmarks[15].y = face.landmarks[17].x, face.landmarks[17].y
        face.landmarks[17].x, face.landmarks[17].y = pt15_orig

        report = assess_vf50_quality(face)
        assert any("left_eye_self_intersection" in a for a in report.anomalies)

    def test_inner_lip_protrusion_outside_outer_lip_detected(self):
        """Guideline Section 3.4: moitrong must lie strictly inside moingoai."""
        face = create_canonical_frontal_face()
        # Shift inner lip corner 42 far to the left past outer lip corner 30 (x=420)
        face.landmarks[42].x = 380.0

        report = assess_vf50_quality(face)
        assert any("inner_lip_protrudes_left" in a for a in report.anomalies)

    def test_closed_mouth_condition_does_not_fail(self):
        """Guideline Section 6.4: Closed mouth (80% of dataset) has inner lips on contact seam."""
        face = create_canonical_frontal_face()
        # Close mouth: inner upper and lower lip points coincide at y=600
        for pt_id in range(42, 50):
            face.landmarks[pt_id].y = 600.0

        report = assess_vf50_quality(face)
        assert report.is_valid is True
        assert not any("inner_lip_exceeds_outer_area" in a for a in report.anomalies)

    def test_inverted_laterality_detected(self):
        face = create_canonical_frontal_face()
        # Swap left eye and right eye coordinates
        for i in range(8):
            l_id = 14 + i
            r_id = 22 + i
            face.landmarks[l_id].x, face.landmarks[r_id].x = face.landmarks[r_id].x, face.landmarks[l_id].x

        report = assess_vf50_quality(face)
        assert any("inverted_laterality" in a for a in report.anomalies)


# ==============================================================================
# SECTION 12: RESPONSE PARSER & MULTI-SCHEMA SUPPORT
# ==============================================================================

class TestVF50Parser:
    """Verify parsing multiple model output payload formats."""

    def test_parse_nested_faces_and_landmarks(self):
        payload = {
            "faces": [
                {
                    "id": 1,
                    "confidence": 0.95,
                    "landmarks": [
                        {"id": i, "point": [200.0 + i * 10, 300.0], "visibility": 2}
                        for i in range(50)
                    ],
                }
            ]
        }
        faces = parse_vf50_response(payload)
        assert len(faces) == 1
        assert len(faces[0].landmarks) == 50
        assert faces[0].landmarks[0].x == 200.0
        assert faces[0].landmarks[49].x == 690.0

    def test_parse_nested_components_format(self):
        payload = {
            "faces": [
                {
                    "id": 1,
                    "components": {
                        "longmaytrai": [[100, 200], [110, 205], [120, 210], [130, 215], [140, 220]],
                        "songmui": [[500, 250], [500, 300], [500, 350], [500, 400]],
                    },
                }
            ]
        }
        faces = parse_vf50_response(payload)
        assert len(faces) == 1
        assert faces[0].landmarks[0].x == 100.0
        assert faces[0].landmarks[4].x == 140.0
        assert faces[0].landmarks[10].x == 500.0
        assert faces[0].landmarks[13].y == 400.0

    def test_parse_with_crop_box_remapping(self):
        # Crop is [200, 300, 600, 700] (normalized)
        crop_box = [200, 300, 600, 700]
        # In crop space, landmark 0 is at (0, 0)
        payload = {
            "faces": [
                {
                    "id": 1,
                    "landmarks": [
                        {"id": 0, "point": [0.0, 0.0], "visibility": 2},
                        {"id": 1, "point": [1000.0, 1000.0], "visibility": 2},
                    ],
                }
            ]
        }
        faces = parse_vf50_response(payload, crop_box=crop_box)
        assert len(faces) == 1
        # In full image space, (0, 0) in crop should map to (crop_xmin=300, crop_ymin=200)
        assert faces[0].landmarks[0].x == pytest.approx(300.0, abs=0.01)
        assert faces[0].landmarks[0].y == pytest.approx(200.0, abs=0.01)
        # (1000, 1000) in crop should map to (crop_xmax=700, crop_ymax=600)
        assert faces[0].landmarks[1].x == pytest.approx(700.0, abs=0.01)
        assert faces[0].landmarks[1].y == pytest.approx(600.0, abs=0.01)

    def test_parse_markdown_wrapped_json(self):
        raw_text = """
        Here is the landmark detection:
        ```json
        {
            "faces": [
                {"id": 1, "landmarks": [{"id": 0, "point": [400, 500]}]}
            ]
        }
        ```
        """
        faces = parse_vf50_response(raw_text)
        assert len(faces) == 1
        assert faces[0].landmarks[0].x == 400.0
        assert faces[0].landmarks[0].y == 500.0


# ==============================================================================
# SECTION 10: PROMPT & CVAT SPEC BUILDERS
# ==============================================================================

class TestVF50PromptAndSpec:
    """Verify strict prompt and CVAT function.yaml spec generation."""

    def test_prompt_builder_contents(self):
        prompt = build_vf50_prompt()
        assert "longmaytrai" in prompt
        assert "moingoai" in prompt
        assert "moitrong" in prompt
        assert "50 landmarks" in prompt
        assert "0 = outside" in prompt or "0=outside" in prompt

    def test_cvat_spec_structure(self):
        specs = build_cvat_vf50_spec()
        assert len(specs) == 7
        total_sublabels = 0
        for spec in specs:
            assert spec["type"] == "skeleton"
            assert spec["attributes"] == []
            assert isinstance(spec["sublabels"], list)
            assert "svg" in spec
            assert "<circle" in spec["svg"]
            assert "<line" in spec["svg"]
            total_sublabels += len(spec["sublabels"])
        assert total_sublabels == 50

    def test_function_yaml_validates(self):
        from pathlib import Path
        from scripts.validate_function_spec import validate_function_yaml
        yaml_path = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml"
        is_valid, errs = validate_function_yaml(yaml_path)
        assert is_valid is True, f"function.yaml validation failed: {errs}"


# ==============================================================================
# SECTION 10: MODEL HANDLER INTEGRATION
# ==============================================================================

class TestVF50ModelHandler:
    """Verify ModelHandler infer lifecycle and 7-component output formatting."""

    @pytest.fixture
    def mock_handler(self, monkeypatch):
        import importlib.util
        from pathlib import Path

        def mock_resolve(self, requested_model):
            return "mock-gemini-vision"

        monkeypatch.setattr("app.client.NineRouterClient.resolve_vision_model", mock_resolve)

        handler_path = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-face-vf50" / "nuclio" / "model_handler.py"
        spec = importlib.util.spec_from_file_location("vf50_model_handler", handler_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        handler = mod.ModelHandler(base_url="http://mock:20128", api_key="test-key", model="mock-gemini-vision")
        return handler

    def test_handler_emits_7_components_for_single_face(self, mock_handler, monkeypatch):
        import io
        from PIL import Image
        from app.client import VisionResponse

        # Build mock canonical face JSON
        mock_face = {
            "id": 1,
            "confidence": 0.98,
            "landmarks": [
                {"id": i, "point": [400 + (i % 10) * 10, 300 + (i // 10) * 10], "visibility": 2, "confidence": 0.95}
                for i in range(50)
            ],
        }
        mock_resp = VisionResponse(
            content=json.dumps({"faces": [mock_face]}),
            raw_response={},
            duration_seconds=0.2,
            model="mock-gemini-vision",
            status_code=200,
        )
        monkeypatch.setattr(mock_handler.client, "send_vision_request", lambda **kwargs: mock_resp)

        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        shapes = mock_handler.infer(img_bytes, threshold=0.5)
        # Exactly 7 component skeletons
        assert len(shapes) == 7
        labels = [s["label"] for s in shapes]
        assert labels == list(VF50_COMPONENT_NAMES)
        for s in shapes:
            assert s["type"] == "skeleton"
            assert s["group"] == 1
            for el in s["elements"]:
                assert el["type"] == "points"
                assert len(el["points"]) == 2
                assert el["label"].isdigit()

    def test_handler_multi_face_groups(self, mock_handler, monkeypatch):
        import io
        from PIL import Image
        from app.client import VisionResponse

        face1 = create_canonical_frontal_face(face_id=1)
        face2 = create_canonical_frontal_face(face_id=2)
        mock_face1 = {
            "id": 1,
            "confidence": 0.98,
            "landmarks": [
                {"id": lm.id, "point": [lm.x, lm.y], "visibility": lm.visibility, "confidence": lm.confidence}
                for lm in face1.landmarks.values()
            ],
        }
        mock_face2 = {
            "id": 2,
            "confidence": 0.95,
            "landmarks": [
                {"id": lm.id, "point": [lm.x, lm.y], "visibility": lm.visibility, "confidence": lm.confidence}
                for lm in face2.landmarks.values()
            ],
        }
        mock_resp = VisionResponse(
            content=json.dumps({"faces": [mock_face1, mock_face2]}),
            raw_response={},
            duration_seconds=0.2,
            model="mock-gemini-vision",
            status_code=200,
        )
        monkeypatch.setattr(mock_handler.client, "send_vision_request", lambda **kwargs: mock_resp)

        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        shapes = mock_handler.infer(buf.getvalue(), threshold=0.5)
        # 2 faces * 7 components = 14 skeletons
        assert len(shapes) == 14
        groups_face1 = {s["group"] for s in shapes[:7]}
        groups_face2 = {s["group"] for s in shapes[7:]}
        assert groups_face1 == {1}
        assert groups_face2 == {2}

    def test_handler_confidence_threshold_filtering(self, mock_handler, monkeypatch):
        import io
        from PIL import Image
        from app.client import VisionResponse

        mock_face_low_conf = {
            "id": 1,
            "confidence": 0.3,
            "landmarks": [{"id": i, "point": [200, 300], "visibility": 2} for i in range(50)],
        }
        mock_resp = VisionResponse(
            content=json.dumps({"faces": [mock_face_low_conf]}),
            raw_response={},
            duration_seconds=0.2,
            model="mock-gemini-vision",
            status_code=200,
        )
        monkeypatch.setattr(mock_handler.client, "send_vision_request", lambda **kwargs: mock_resp)

        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        # Threshold 0.5 filters out 0.3 face -> returns empty list
        shapes = mock_handler.infer(buf.getvalue(), threshold=0.5)
        assert shapes == []

