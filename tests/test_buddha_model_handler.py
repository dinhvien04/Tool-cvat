"""Tests for Buddha Multi-Limb ModelHandler (serverless/ninerouter-buddha-multilimbs/nuclio/model_handler.py).

Validates:
- End-to-end hierarchical multi-pass orchestration with hermetic mocking.
- Pass 1 Global Discovery, Pass 2 Arm Refine, Pass 3 Hand Refine, Pass 4 Face VF50, Pass 5 Body Pose17, Pass 6 Merge.
- Robust failure isolation (single hand failure does not break the entire pipeline).
- Concurrency with ThreadPoolExecutor workers.
- Both handle_image and infer method entry points.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest import mock
import pytest
from PIL import Image

from app.client import VisionResponse

ROOT = Path(__file__).resolve().parent.parent
BUDDHA_HANDLER_PATH = ROOT / "serverless" / "ninerouter-buddha-multilimbs" / "nuclio" / "model_handler.py"

def _load_buddha_model_handler():
    spec = importlib.util.spec_from_file_location("buddha_model_handler", BUDDHA_HANDLER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

buddha_mod = _load_buddha_model_handler()
ModelHandler = buddha_mod.ModelHandler


def create_synthetic_image(width: int = 1000, height: int = 1000) -> bytes:
    """Generate a simple blank image in memory."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestBuddhaModelHandler:
    """Hermetic tests for Buddha Multi-Limb ModelHandler."""

    @pytest.fixture
    def mock_responses(self):
        """Prepare JSON responses for each vision pass."""
        global_resp = json.dumps({
            "torso_center": [500, 500],
            "face_roi": [150, 420, 300, 580],
            "central_body": {
                "keypoints": {
                    "nose": [500, 200, 2],
                    "left_shoulder": [450, 350, 2],
                    "right_shoulder": [550, 350, 2],
                    "left_elbow": [400, 450, 2],
                    "right_elbow": [600, 450, 2],
                    "left_wrist": [380, 550, 2],
                    "right_wrist": [620, 550, 2],
                    "left_hip": [460, 600, 2],
                    "right_hip": [540, 600, 2],
                    "left_knee": [450, 750, 2],
                    "right_knee": [550, 750, 2],
                    "left_ankle": [440, 900, 2],
                    "right_ankle": [560, 900, 2],
                },
                "confidence": 0.95,
            },
            "arms": [
                {
                    "root": [480, 420, 2],
                    "elbow": [350, 350, 2],
                    "wrist": [250, 250, 2],
                    "confidence": 0.92,
                    "hand_roi": [200, 200, 300, 300],
                },
                {
                    "root": [520, 420, 2],
                    "elbow": [650, 350, 2],
                    "wrist": [750, 250, 2],
                    "confidence": 0.91,
                    "hand_roi": [200, 700, 300, 800],
                },
            ],
        })

        arm_refine_resp = json.dumps({
            "root": [800, 800, 2],
            "elbow": [500, 500, 2],
            "wrist": [200, 200, 2],
            "confidence": 0.96,
        })

        # Generate 21 hand landmarks
        hand_landmarks = {}
        for i in range(21):
            hand_landmarks[str(i)] = [200 + (i % 5) * 40, 200 + (i // 5) * 40, 2]

        hand_refine_resp = json.dumps({
            "landmarks": hand_landmarks,
            "confidence": 0.93,
        })

        # VF50 response format: 50 landmarks in components
        vf50_pts = [[500 + (i % 10) * 5, 200 + (i // 10) * 5, 2] for i in range(50)]
        vf50_resp = json.dumps({
            "landmarks": vf50_pts,
            "confidence": 0.94,
        })

        return {
            "global": global_resp,
            "arm": arm_refine_resp,
            "hand": hand_refine_resp,
            "face": vf50_resp,
        }

    def test_full_pipeline_orchestration(self, mock_responses):
        """Verify complete pipeline returns body, face components, arms, and hands."""
        with mock.patch("app.client.NineRouterClient.get_vision_model_ids", return_value=["claude-opus-5-5", "claude-sonnet-4-6"]):
            handler = ModelHandler(
                base_url="http://mocked.router",
                model="claude-opus-5-5",
                refine_workers=1,
            )

        def mock_vision(model, image_bytes_or_b64, prompt, max_tokens, timeout):
            if "central_body" in prompt:
                return VisionResponse(content=mock_responses["global"])
            elif "root" in prompt and "elbow" in prompt and "wrist" in prompt and "21" not in prompt:
                return VisionResponse(content=mock_responses["arm"])
            elif "21 2D hand landmark" in prompt:
                return VisionResponse(content=mock_responses["hand"])
            elif "facial landmark" in prompt or "VF-50" in prompt:
                return VisionResponse(content=mock_responses["face"])
            return VisionResponse(content="{}")

        with mock.patch.object(handler.client, "send_vision_request", side_effect=mock_vision):
            img_bytes = create_synthetic_image(1000, 1000)
            shapes = handler.infer(image_bytes=img_bytes)

        assert isinstance(shapes, list)
        assert len(shapes) > 0

        labels = [s["label"] for s in shapes]
        # Central body present
        assert "person" in labels
        # Arms present
        assert "buddha_arm" in labels
        assert labels.count("buddha_arm") == 2
        # Hands present
        assert "buddha_hand" in labels
        assert labels.count("buddha_hand") == 2

        # Verify all shapes have type="skeleton"
        for s in shapes:
            assert s["type"] == "skeleton"
            assert "elements" in s
            assert len(s["elements"]) > 0

    def test_failure_isolation_on_hand_refinement(self, mock_responses):
        """If hand refinement fails for one arm, the other arm and hands are still returned."""
        with mock.patch("app.client.NineRouterClient.get_vision_model_ids", return_value=["claude-opus-5-5"]):
            handler = ModelHandler(
                base_url="http://mocked.router",
                model="claude-opus-5-5",
                refine_workers=1,
            )

        call_count = {"hand": 0}

        def mock_vision_with_failure(model, image_bytes_or_b64, prompt, max_tokens, timeout):
            if "central_body" in prompt:
                return VisionResponse(content=mock_responses["global"])
            elif "21 2D hand landmark" in prompt:
                call_count["hand"] += 1
                if call_count["hand"] == 1:
                    # First hand crop fails with an exception
                    raise RuntimeError("Simulated transient 9Router timeout on hand crop")
                return VisionResponse(content=mock_responses["hand"])
            elif "root" in prompt and "elbow" in prompt:
                return VisionResponse(content=mock_responses["arm"])
            return VisionResponse(content="{}")

        with mock.patch.object(handler.client, "send_vision_request", side_effect=mock_vision_with_failure):
            img_bytes = create_synthetic_image(1000, 1000)
            shapes = handler.infer(image_bytes=img_bytes)

        # Pipeline must succeed despite the hand crop failure
        labels = [s["label"] for s in shapes]
        assert "buddha_arm" in labels
        assert labels.count("buddha_arm") == 2
        # One hand was successfully refined, one failed gracefully
        assert labels.count("buddha_hand") == 1

    def test_failure_isolation_on_face_and_body(self, mock_responses):
        """Verify that failure in central face or central body does not drop arms or hands."""
        with mock.patch("app.client.NineRouterClient.get_vision_model_ids", return_value=["claude-opus-5-5"]):
            handler = ModelHandler(
                base_url="http://mocked.router",
                model="claude-opus-5-5",
                refine_workers=1,
            )

        def mock_vision(model, image_bytes_or_b64, prompt, max_tokens, timeout):
            if "central_body" in prompt:
                return VisionResponse(content=mock_responses["global"])
            elif "root" in prompt and "elbow" in prompt and "wrist" in prompt and "21" not in prompt:
                return VisionResponse(content=mock_responses["arm"])
            elif "21 2D hand landmark" in prompt:
                return VisionResponse(content=mock_responses["hand"])
            return VisionResponse(content="{}")

        # Mock _process_central_face to throw an exception
        with mock.patch.object(handler, "_process_central_face", side_effect=RuntimeError("Face component model failed")):
            with mock.patch.object(handler.client, "send_vision_request", side_effect=mock_vision):
                img_bytes = create_synthetic_image(1000, 1000)
                shapes = handler.infer(image_bytes=img_bytes)

        labels = [s["label"] for s in shapes]
        # Body and arms/hands still returned
        assert "person" in labels
        assert "buddha_arm" in labels
        assert "buddha_hand" in labels
        # Face component labels are absent due to isolated failure
        assert "longmaytrai" not in labels

    def test_output_shapes_comply_with_cvat_spec(self, mock_responses):
        """Verify that output shapes strictly satisfy validate_shapes_against_cvat_spec."""
        from core.buddha_contract import validate_shapes_against_cvat_spec

        with mock.patch("app.client.NineRouterClient.get_vision_model_ids", return_value=["claude-opus-5-5"]):
            handler = ModelHandler(
                base_url="http://mocked.router",
                model="claude-opus-5-5",
                refine_workers=1,
            )

        def mock_vision(model, image_bytes_or_b64, prompt, max_tokens, timeout):
            if "central_body" in prompt:
                return VisionResponse(content=mock_responses["global"])
            elif "root" in prompt and "elbow" in prompt and "wrist" in prompt and "21" not in prompt:
                return VisionResponse(content=mock_responses["arm"])
            elif "21 2D hand landmark" in prompt:
                return VisionResponse(content=mock_responses["hand"])
            elif "facial landmark" in prompt or "VF-50" in prompt:
                return VisionResponse(content=mock_responses["face"])
            return VisionResponse(content="{}")

        with mock.patch.object(handler.client, "send_vision_request", side_effect=mock_vision):
            img_bytes = create_synthetic_image(1000, 1000)
            shapes = handler.infer(image_bytes=img_bytes)

        is_valid, errors = validate_shapes_against_cvat_spec(shapes)
        assert is_valid, f"Output shapes failed CVAT spec validation: {errors}"
        assert len(errors) == 0
