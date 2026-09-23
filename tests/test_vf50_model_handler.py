"""Targeted unit tests for 9Router VF-50 Face ModelHandler & Two-Pass Refinement Loop."""

from __future__ import annotations

import importlib.util
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.skeleton_contract import (
    VF50_COMPONENT_NAMES,
    VF50_POINTS_COUNT,
    VF50Face,
    VF50Landmark,
    assess_vf50_quality,
)
from core.week2_schema import load_vf50

REPO_ROOT = Path(__file__).resolve().parent.parent
VF50_HANDLER_PATH = REPO_ROOT / "serverless" / "ninerouter-face-vf50" / "nuclio" / "model_handler.py"


def _load_vf50_model_handler():
    spec = importlib.util.spec_from_file_location("vf50_model_handler", VF50_HANDLER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vf50_mod = _load_vf50_model_handler()
ModelHandler = vf50_mod.ModelHandler
build_vf50_prompt = vf50_mod.build_vf50_prompt
build_vf50_crop_prompt = vf50_mod.build_vf50_crop_prompt


def _create_test_image(width: int = 640, height: int = 480) -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_canonical_vf50_face_dict(
    face_id: int = 1,
    confidence: float = 0.98,
    box_2d: list[int] | None = None,
) -> Dict[str, Any]:
    """Create raw dictionary conforming to 9Router format for a topologically valid frontal face."""
    if box_2d is None:
        box_2d = [180, 240, 720, 760]

    landmarks: List[Dict[str, Any]] = []

    # 0..4 longmaytrai (viewer left eyebrow: x 260..410, y ~ 200)
    for idx, x in enumerate([260.0, 290.0, 330.0, 370.0, 410.0]):
        landmarks.append({"id": idx, "point": [x, 200.0 - 10.0 * math.sin(idx * 0.7)], "visibility": 2, "confidence": 0.99})

    # 5..9 longmayphai (viewer right eyebrow: x 590..740, y ~ 200)
    for idx, x in enumerate([590.0, 630.0, 670.0, 710.0, 740.0], start=5):
        landmarks.append({"id": idx, "point": [x, 200.0 - 10.0 * math.sin((idx - 5) * 0.7)], "visibility": 2, "confidence": 0.99})

    # 10..13 songmui (nose bridge: x ~ 500, y descends 250 -> 430)
    for idx, y in enumerate([250.0, 310.0, 370.0, 430.0], start=10):
        landmarks.append({"id": idx, "point": [500.0, y], "visibility": 2, "confidence": 0.99})

    # 14..21 mattrai (left eye loop)
    mattrai_coords = [
        (14, 290.0, 280.0), (15, 315.0, 265.0), (16, 340.0, 260.0), (17, 365.0, 265.0),
        (18, 390.0, 280.0), (19, 365.0, 295.0), (20, 340.0, 300.0), (21, 315.0, 295.0),
    ]
    for pt_id, x, y in mattrai_coords:
        landmarks.append({"id": pt_id, "point": [x, y], "visibility": 2, "confidence": 0.99})

    # 22..29 matphai (right eye loop)
    matphai_coords = [
        (22, 610.0, 280.0), (23, 635.0, 265.0), (24, 660.0, 260.0), (25, 685.0, 265.0),
        (26, 710.0, 280.0), (27, 685.0, 295.0), (28, 660.0, 300.0), (29, 635.0, 295.0),
    ]
    for pt_id, x, y in matphai_coords:
        landmarks.append({"id": pt_id, "point": [x, y], "visibility": 2, "confidence": 0.99})

    # 30..41 moingoai (outer lips)
    moingoai_coords = [
        (30, 420.0, 580.0), (31, 450.0, 555.0), (32, 480.0, 550.0), (33, 500.0, 555.0),
        (34, 520.0, 550.0), (35, 550.0, 555.0), (36, 580.0, 580.0), (37, 550.0, 615.0),
        (38, 520.0, 630.0), (39, 500.0, 632.0), (40, 480.0, 630.0), (41, 450.0, 615.0),
    ]
    for pt_id, x, y in moingoai_coords:
        landmarks.append({"id": pt_id, "point": [x, y], "visibility": 2, "confidence": 0.99})

    # 42..49 moitrong (inner lips)
    moitrong_coords = [
        (42, 440.0, 580.0), (43, 475.0, 570.0), (44, 500.0, 572.0), (45, 525.0, 570.0),
        (46, 560.0, 580.0), (47, 525.0, 595.0), (48, 500.0, 598.0), (49, 475.0, 595.0),
    ]
    for pt_id, x, y in moitrong_coords:
        landmarks.append({"id": pt_id, "point": [x, y], "visibility": 2, "confidence": 0.99})

    return {
        "id": face_id,
        "confidence": confidence,
        "box_2d": box_2d,
        "landmarks": landmarks,
    }


class TestVF50PromptsAndOcclusionAudit:
    """Audit VF-50 prompt contracts, canonical component descriptions, and occlusion instructions."""

    def test_global_and_crop_prompts_share_canonical_descriptions(self):
        """Mandate: Both global prompt and crop prompt share the exact same canonical landmark descriptions generated from schema data."""
        schema = load_vf50()
        global_p = build_vf50_prompt()
        crop_p = build_vf50_crop_prompt()

        for idx, comp in enumerate(schema.components, start=1):
            assert comp.name in global_p, f"Component {comp.name} missing from global prompt"
            assert comp.name in crop_p, f"Component {comp.name} missing from crop prompt"
            # Verify exact point range string e.g. "points 0..4"
            range_str = f"points {comp.start_id}..{comp.end_id}"
            assert range_str in global_p, f"Range {range_str} missing from global prompt"
            assert range_str in crop_p, f"Range {range_str} missing from crop prompt"
            # If component has canonical description in YAML, verify it appears in both
            if getattr(comp, "description", None):
                assert comp.description in global_p
                assert comp.description in crop_p

    def test_occlusion_instructions_comprehensive_audit(self):
        """Mandate: Physical occlusion (hair, hand, mask, glasses) vs optical blur/shadows/low light (visibility=2, Confidence != Occlusion!)."""
        for prompt_name, prompt in [("global", build_vf50_prompt()), ("crop", build_vf50_crop_prompt())]:
            assert "Physical Occlusion vs Optical Blur" in prompt, f"Occlusion header missing in {prompt_name}"
            assert "Confidence != Occlusion!" in prompt, f"Confidence != Occlusion missing in {prompt_name}"

            # Physical occlusion items (visibility = 1)
            assert "visibility = 1" in prompt
            assert "hair" in prompt.lower()
            assert "eyebrow" in prompt.lower()
            assert "hand" in prompt.lower() or "fingers" in prompt.lower()
            assert "mask" in prompt.lower()
            assert "glasses" in prompt.lower() or "sunglasses" in prompt.lower()

            # Optical degradation items (visibility = 2)
            assert "visibility = 2" in prompt
            assert "motion blur" in prompt.lower()
            assert "shadow" in prompt.lower()
            assert "sensor noise" in prompt.lower() or "defocus" in prompt.lower() or "low light" in prompt.lower()

            # Outside frame (visibility = 0)
            assert "visibility = 0" in prompt
            assert "outside" in prompt.lower()

    def test_topological_constraints_in_both_prompts(self):
        """Verify key anatomical constraints: viewer perspective, eyebrow ordering, eyelid vertical ordering, lips containment."""
        for prompt_name, prompt in [("global", build_vf50_prompt()), ("crop", build_vf50_crop_prompt())]:
            assert "longmaytrai" in prompt
            assert "moitrong" in prompt
            assert "moingoai" in prompt
            assert "upper eyelid" in prompt.lower()
            assert "viewer perspective" in prompt.lower() or "trai" in prompt

    def test_no_ambiguous_visibility_choice_and_inner_lip_ordering_in_vf50_prompts(self):
        """Mandate: No ambiguous visibility choice and explicit moitrong (42..49) ordering in VF50 prompts."""
        for prompt_name, prompt in [("global", build_vf50_prompt()), ("crop", build_vf50_crop_prompt())]:
            assert "visibility = 1 or 0" not in prompt, f"Ambiguous rule found in {prompt_name}"
            assert "1 (occluded) or 0" not in prompt, f"Ambiguous rule found in {prompt_name}"
            assert "Physical occlusion within frame & inferable -> visibility = 1" in prompt, f"Missing strict occlusion rule in {prompt_name}"
            assert "visibility = 0 is ONLY for points outside image frame" in prompt, f"Missing strict visibility 0 rule in {prompt_name}"
            # Verify explicit moitrong 42..49 ordering
            assert "moitrong starts at inner left corner (42)" in prompt, f"Missing moitrong ordering in {prompt_name}"
            assert "closes back to 42" in prompt, f"Missing moitrong closed loop in {prompt_name}"


class TestVF50ModelResolutionAndInit:
    """Verify Pass 1 (VF50_MODEL) and Pass 2 (VF50_REFINE_MODEL) dynamic resolution and initialization."""

    @patch.object(vf50_mod, "NineRouterClient")
    def test_model_resolution_pass1_and_pass2(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(
            base_url="http://mock:20128",
            api_key="secret-key-12345",
            model="ag/gemini-3.8-flash-low",
            refine_model="ag/gemini-3.8-flash-high",
            max_refine_faces=5,
        )

        assert handler.active_model == "ag/gemini-3.8-flash-low"
        assert handler.active_refine_model == "ag/gemini-3.8-flash-high"
        assert handler.max_refine_faces == 5
        assert "secret-key-12345" not in repr(handler)

    @patch.object(vf50_mod, "NineRouterClient")
    def test_refine_model_fallback_to_active_model_when_unset(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(
            base_url="http://mock:20128",
            api_key="dummy",
            model="ag/gemini-3.8-flash-low",
        )

        assert handler.active_model == "ag/gemini-3.8-flash-low"
        assert handler.active_refine_model == "ag/gemini-3.8-flash-low"
        assert handler.max_refine_faces == 3  # Default 3

    @patch.object(vf50_mod, "NineRouterClient")
    def test_refine_model_fallback_on_resolution_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.side_effect = Exception("Model not found")
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(
            base_url="http://mock:20128",
            api_key="dummy",
            model="ag/gemini-3.8-flash-low",
            refine_model="nonexistent-model",
        )

        # Refine model falls back to active_model gracefully
        assert handler.active_model == "ag/gemini-3.8-flash-low"
        assert handler.active_refine_model == "ag/gemini-3.8-flash-low"


class TestVF50InferenceAndCropRefinement:
    """Verify Pass 1 & Pass 2 inference flow, conditional refinement, coordinate remapping, and caps."""

    @patch.object(vf50_mod, "NineRouterClient")
    def test_infer_happy_path_skips_refinement_for_high_quality_face(self, mock_client_cls):
        """Mandate: Refinement is conditional, not called on every clean detection."""
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        face_dict = _create_canonical_vf50_face_dict()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"faces": [face_dict]})
        mock_client.send_vision_request.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)

        # High-quality canonical face -> send_vision_request called exactly ONCE (Pass 1 only)
        assert mock_client.send_vision_request.call_count == 1
        # 7 component skeletons returned for 1 face
        assert len(shapes) == 7
        assert {s["label"] for s in shapes} == set(VF50_COMPONENT_NAMES)

    @patch.object(vf50_mod, "NineRouterClient")
    def test_refinement_triggers_on_low_quality_and_remaps_coordinates(self, mock_client_cls):
        """Mandate: Low quality triggers Pass 2 refinement; coordinates remapped back to full image space."""
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"

        # Pass 1: Face with severe eyelid inversion anomaly
        imperfect_face = _create_canonical_vf50_face_dict()
        # Invert left eyelid: lower eyelid point 20 placed above upper eyelid point 16
        imperfect_face["landmarks"][16]["point"][1] = 310.0
        imperfect_face["landmarks"][20]["point"][1] = 250.0

        # Pass 2: Refined crop response (coordinates relative to crop [0..1000])
        # In crop patch, face is centered, eyelids fixed
        crop_face = _create_canonical_vf50_face_dict(
            box_2d=[100, 100, 900, 900]
        )

        mock_resp_1 = MagicMock()
        mock_resp_1.content = json.dumps({"faces": [imperfect_face]})

        mock_resp_2 = MagicMock()
        mock_resp_2.content = json.dumps({"faces": [crop_face]})

        mock_client.send_vision_request.side_effect = [mock_resp_1, mock_resp_2]
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(
            base_url="http://mock:20128",
            api_key="dummy",
            refine_model="ag/gemini-3.8-flash-high",
        )
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)

        # Both Pass 1 and Pass 2 executed
        assert mock_client.send_vision_request.call_count == 2

        # Verify Pass 2 called with refine model
        pass2_call = mock_client.send_vision_request.call_args_list[1]
        assert pass2_call.kwargs["model"] == "ag/gemini-3.8-flash-high"

        # 7 component skeletons returned
        assert len(shapes) == 7

    @patch.object(vf50_mod, "NineRouterClient")
    def test_refinement_triggers_on_small_face_bbox(self, mock_client_cls):
        """Mandate: Face bounding box < 100px triggers Pass 2 crop refinement."""
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # Small face: bbox width & height are 50 in 1000-space -> 32px x 24px in 640x480 image (< 100px)
        small_face = _create_canonical_vf50_face_dict(box_2d=[100, 100, 150, 150])
        refined_face = _create_canonical_vf50_face_dict(box_2d=[100, 100, 900, 900])

        mock_resp_1 = MagicMock()
        mock_resp_1.content = json.dumps({"faces": [small_face]})
        mock_resp_2 = MagicMock()
        mock_resp_2.content = json.dumps({"faces": [refined_face]})

        mock_client.send_vision_request.side_effect = [mock_resp_1, mock_resp_2]
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)
        assert mock_client.send_vision_request.call_count == 2
        assert len(shapes) == 7

    @patch.object(vf50_mod, "NineRouterClient")
    def test_max_refine_faces_cap_enforced_in_crowd_scene(self, mock_client_cls):
        """Mandate: Refinement cap limits the number of crop refinements in multi-face scenes."""
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # 6 faces, all with eyelid anomalies needing refinement
        faces = []
        for i in range(6):
            f = _create_canonical_vf50_face_dict(face_id=i + 1, box_2d=[100, i * 150, 400, (i + 1) * 150])
            f["landmarks"][16]["point"][1] = 310.0
            f["landmarks"][20]["point"][1] = 250.0
            faces.append(f)

        refined_single = _create_canonical_vf50_face_dict(box_2d=[100, 100, 900, 900])

        mock_resp_1 = MagicMock()
        mock_resp_1.content = json.dumps({"faces": faces})

        # Up to 3 refined responses
        mock_resps = [mock_resp_1] + [MagicMock(content=json.dumps({"faces": [refined_single]})) for _ in range(3)]
        mock_client.send_vision_request.side_effect = mock_resps
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(
            base_url="http://mock:20128",
            api_key="dummy",
            max_refine_faces=3,  # Cap at 3
        )
        img_bytes = _create_test_image(1280, 720)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)

        # Exactly 1 global request + 3 crop refinement requests = 4 total calls
        assert mock_client.send_vision_request.call_count == 4
        # All 6 faces returned (each with 7 components = 42 shapes)
        assert len(shapes) == 6 * 7
        group_ids = {s.get("group_id") for s in shapes if "group_id" in s}
        assert len(group_ids) == 6

    @patch.object(vf50_mod, "NineRouterClient")
    def test_initial_face_retained_if_refinement_yields_lower_quality(self, mock_client_cls):
        """Mandate: Superior candidate is chosen by comparing quality score against initial face."""
        mock_client = MagicMock()
        mock_client.resolve_vf50_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # Small face that triggers refine
        initial_face = _create_canonical_vf50_face_dict(box_2d=[100, 100, 150, 150])

        # Refined crop returns a badly distorted face (lip protrusion, eyelid inversion)
        bad_refined_face = _create_canonical_vf50_face_dict(box_2d=[100, 100, 900, 900])
        bad_refined_face["landmarks"][16]["point"][1] = 310.0
        bad_refined_face["landmarks"][20]["point"][1] = 250.0
        bad_refined_face["landmarks"][42]["point"][0] = 350.0  # inner lip protrusion past outer lip

        mock_resp_1 = MagicMock()
        mock_resp_1.content = json.dumps({"faces": [initial_face]})
        mock_resp_2 = MagicMock()
        mock_resp_2.content = json.dumps({"faces": [bad_refined_face]})

        mock_client.send_vision_request.side_effect = [mock_resp_1, mock_resp_2]
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)
        assert mock_client.send_vision_request.call_count == 2
        # Initial face is safely preserved
        assert len(shapes) == 7
