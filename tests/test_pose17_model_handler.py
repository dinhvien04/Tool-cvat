"""Targeted unit tests for 9Router Human Pose 17 ModelHandler & Refinement Loop."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.pose_face_schema import POSE17_KEYPOINTS
from core.quality_gate import LATERALITY_VIEWER, assess_pose17_quality
from core.week2_schema import load_pose17

REPO_ROOT = Path(__file__).resolve().parent.parent
POSE17_HANDLER_PATH = REPO_ROOT / "serverless" / "ninerouter-human-pose-17" / "nuclio" / "model_handler.py"

def _load_pose17_model_handler():
    spec = importlib.util.spec_from_file_location("pose17_model_handler", POSE17_HANDLER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pose17_mod = _load_pose17_model_handler()
ModelHandler = pose17_mod.ModelHandler
build_pose17_prompt = pose17_mod.build_pose17_prompt
build_pose17_crop_prompt = pose17_mod.build_pose17_crop_prompt
derive_person_bbox = pose17_mod.derive_person_bbox


def _create_test_image(width: int = 640, height: int = 480) -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_canonical_pose17_person() -> Dict[str, Any]:
    """Canonical upright front-facing person in [0..1000] normalized space."""
    return {
        "id": 1,
        "label": "person",
        "confidence": 0.95,
        "box_2d": [100, 300, 900, 700],
        "keypoints": {
            "nose": [500, 150, 2],
            "right_eye": [530, 140, 2],    # Viewer right (x > 500)
            "left_eye": [470, 140, 2],     # Viewer left (x < 500)
            "right_ear": [560, 150, 2],
            "left_ear": [440, 150, 2],
            "right_shoulder": [600, 250, 2],
            "left_shoulder": [400, 250, 2],
            "right_elbow": [630, 400, 2],
            "left_elbow": [370, 400, 2],
            "right_wrist": [650, 520, 2],
            "left_wrist": [350, 520, 2],
            "right_hip": [560, 550, 2],
            "left_hip": [440, 550, 2],
            "right_knee": [570, 720, 2],
            "left_knee": [430, 720, 2],
            "right_ankle": [580, 880, 2],
            "left_ankle": [420, 880, 2],
        },
    }


class TestPose17PromptsAndBbox:
    def test_pose17_prompt_contains_viewer_laterality_and_kinematic_constraints(self):
        prompt = build_pose17_prompt(POSE17_KEYPOINTS)
        assert "shoulder -> elbow -> wrist" in prompt
        assert "VIEWER'S RIGHT" in prompt
        assert "VIEWER'S LEFT" in prompt
        assert "box_2d" in prompt
        assert "cabin" in prompt.lower() or "floating" in prompt.lower()
        assert "VinFast Week-2 HumanPose-17 topology" in prompt
        assert "COCO" not in prompt, "Prompt should use VinFast Week-2 HumanPose-17 topology rather than conflicting COCO terminology"

    def test_pose17_crop_prompt_structure(self):
        crop_prompt = build_pose17_crop_prompt(POSE17_KEYPOINTS)
        assert "High-resolution close-up person crop refinement" in crop_prompt
        assert "shoulder -> elbow -> wrist" in crop_prompt
        assert "VinFast Week-2 HumanPose-17 topology" in crop_prompt
        assert "COCO" not in crop_prompt, "Crop prompt should use VinFast Week-2 HumanPose-17 topology rather than conflicting COCO terminology"

    def test_occlusion_prompt_distinguishes_physical_from_optical_blur(self):
        prompt = build_pose17_prompt()
        assert "Physical Occlusion vs Optical Blur" in prompt
        assert "Confidence != Occlusion!" in prompt
        assert "steering wheel" in prompt.lower()
        assert "optical blur" in prompt.lower()
        assert "motion blur" in prompt.lower()
        assert "visibility = 1" in prompt
        assert "visibility = 2" in prompt
        assert "visibility = 0" in prompt

        crop_prompt = build_pose17_crop_prompt()
        assert "Confidence != Occlusion!" in crop_prompt
        assert "physically occluded" in crop_prompt.lower()
        assert "optical blur is not occlusion" in crop_prompt.lower()

    def test_prompt_keypoints_derived_from_canonical_schema_metadata(self):
        schema = load_pose17()
        prompt = build_pose17_prompt()
        # Ensure all 17 canonical keypoint names appear in the prompt
        for kp in schema.keypoints:
            coco_name = schema.id_to_coco_name[kp.id]
            assert f"*{coco_name}*" in prompt or f"{coco_name}:" in prompt or f'"{coco_name}"' in prompt

        # Ensure right and left keypoints are properly listed per schema
        right_names = [schema.id_to_coco_name[i] for i in sorted(schema.right_keypoint_ids)]
        left_names = [schema.id_to_coco_name[i] for i in sorted(schema.left_keypoint_ids)]
        for r_name in right_names:
            assert r_name in prompt
        for l_name in left_names:
            assert l_name in prompt

    def test_derive_person_bbox_from_box_2d(self):
        person = {"box_2d": [150, 200, 850, 800]}
        bbox = derive_person_bbox(person)
        assert bbox == [150, 200, 850, 800]

    def test_derive_person_bbox_from_keypoints(self):
        person = {
            "keypoints": {
                "nose": [500, 200, 2],
                "left_ankle": [400, 800, 2],
                "right_ankle": [600, 800, 2],
            }
        }
        bbox = derive_person_bbox(person, pad_ratio=0.10)
        assert bbox is not None
        # ys in [200, 800], h=600, pad_y=60 -> [140, ..., 860, ...]
        assert bbox[0] == 140
        assert bbox[2] == 860


class TestPose17ModelHandlerInference:
    @patch.object(pose17_mod, "NineRouterClient")
    def test_infer_happy_path(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_pose17_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        person = _create_canonical_pose17_person()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"people": [person]})
        mock_client.send_vision_request.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=False)
        assert len(shapes) == 1
        shape = shapes[0]
        assert shape["type"] == "skeleton"
        assert shape["label"] == "person"
        assert len(shape["elements"]) == 17

    @patch.object(pose17_mod, "NineRouterClient")
    def test_two_pass_refinement_triggers_on_suspect_floating_limb(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_pose17_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # Pass 1: Corrupt pose with floating wrist high in cabin (y=10, causing excessive bone length)
        corrupt_person = _create_canonical_pose17_person()
        corrupt_person["keypoints"]["right_wrist"] = [980, 10, 2]  # Floating far away in top right
        corrupt_person["keypoints"]["right_elbow"] = [370, 550, 2]

        # Pass 2: Refined crop gives anatomically sound wrist
        refined_person = _create_canonical_pose17_person()

        mock_resp_1 = MagicMock()
        mock_resp_1.content = json.dumps({"people": [corrupt_person]})

        mock_resp_2 = MagicMock()
        mock_resp_2.content = json.dumps({"people": [refined_person]})

        mock_client.send_vision_request.side_effect = [mock_resp_1, mock_resp_2]
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)
        # Should have called send_vision_request twice (Pass 1 global + Pass 2 crop refinement)
        assert mock_client.send_vision_request.call_count == 2
        assert len(shapes) == 1

    @patch.object(pose17_mod, "NineRouterClient")
    def test_discards_degenerate_collapse(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_pose17_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # All keypoints collapsed to a 1x1 point
        collapsed_person = {
            "id": 1,
            "confidence": 0.95,
            "keypoints": {k: [500, 500, 2] for k in POSE17_KEYPOINTS},
        }

        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"people": [collapsed_person]})
        mock_client.send_vision_request.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=False)
        # Truly collapsed skeleton must be safely discarded
        assert len(shapes) == 0

    @patch.object(pose17_mod, "NineRouterClient")
    def test_refinement_is_conditional_and_skipped_for_high_quality_pose(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_pose17_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # Canonical high-quality pose
        person = _create_canonical_pose17_person()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"people": [person]})
        mock_client.send_vision_request.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
        img_bytes = _create_test_image(640, 480)

        # refine_crops is enabled, but pose is high quality -> must NOT call refine
        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)
        assert mock_client.send_vision_request.call_count == 1
        assert len(shapes) == 1

    @patch.object(pose17_mod, "NineRouterClient")
    def test_pass1_and_pass2_use_distinct_models(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_pose17_model.return_value = "custom-pass1-model"
        mock_client.resolve_vision_model.side_effect = lambda m: f"resolved-{m}"

        corrupt_person = _create_canonical_pose17_person()
        corrupt_person["keypoints"]["right_wrist"] = [990, 10, 2]  # Floating in cabin roof
        refined_person = _create_canonical_pose17_person()

        mock_resp_1 = MagicMock()
        mock_resp_1.content = json.dumps({"people": [corrupt_person]})
        mock_resp_2 = MagicMock()
        mock_resp_2.content = json.dumps({"people": [refined_person]})
        mock_client.send_vision_request.side_effect = [mock_resp_1, mock_resp_2]
        mock_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"POSE17_REFINE_MODEL": "custom-pass2-model"}):
            handler = ModelHandler(base_url="http://mock:20128", api_key="dummy", model="custom-pass1-model")
            assert handler.active_model == "custom-pass1-model"
            assert handler.active_refine_model == "resolved-custom-pass2-model"

            img_bytes = _create_test_image(640, 480)
            shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)
            assert mock_client.send_vision_request.call_count == 2
            # Pass 1 call used active_model
            assert mock_client.send_vision_request.call_args_list[0].kwargs["model"] == "custom-pass1-model"
            # Pass 2 call used active_refine_model
            assert mock_client.send_vision_request.call_args_list[1].kwargs["model"] == "resolved-custom-pass2-model"
            assert len(shapes) == 1

    def test_floating_limb_in_cabin_detected_by_quality_gate(self):
        # Detached wrist floating high in cabin roof
        person = _create_canonical_pose17_person()
        person["keypoints"]["right_wrist"] = [950, 20, 2]
        report = assess_pose17_quality(person, laterality_convention=LATERALITY_VIEWER)
        assert len(report.suspect_bones) > 0
        assert "right_elbow-right_wrist" in report.suspect_bones
        assert report.needs_refine is True

    @patch.object(pose17_mod, "NineRouterClient")
    def test_crowd_scene_refinement_is_capped_at_max_limit(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.resolve_pose17_model.return_value = "ag/gemini-3.8-flash-low"
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-low"

        # 5 people needing refinement
        people = []
        for i in range(5):
            p = _create_canonical_pose17_person()
            p["keypoints"]["right_wrist"] = [950 + i * 5, 20 + i * 5, 2]
            people.append(p)

        mock_resp_main = MagicMock()
        mock_resp_main.content = json.dumps({"people": people})

        mock_resp_crop = MagicMock()
        mock_resp_crop.content = json.dumps({"people": [_create_canonical_pose17_person()]})

        mock_client.send_vision_request.side_effect = [
            mock_resp_main,
            mock_resp_crop,
            mock_resp_crop,
            mock_resp_crop,
            mock_resp_crop,
            mock_resp_crop,
        ]
        mock_client_cls.return_value = mock_client

        # Handler with max_refine_crops capped at 2
        handler = ModelHandler(
            base_url="http://mock:20128",
            api_key="dummy",
            max_refine_crops=2,
        )
        img_bytes = _create_test_image(640, 480)
        shapes = handler.infer(img_bytes, threshold=0.5, refine_crops=True)

        # 1 pass-1 call + exactly 2 crop refinement calls = 3 total calls (capped from 5)
        assert mock_client.send_vision_request.call_count == 3

    def test_empty_image_raises_value_error(self):
        with patch.object(pose17_mod, "NineRouterClient"):
            handler = ModelHandler(base_url="http://mock:20128", api_key="dummy")
            with pytest.raises(ValueError, match="Empty image payload"):
                handler.infer(b"")
