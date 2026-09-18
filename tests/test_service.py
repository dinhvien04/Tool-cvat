"""Tests for app.service module (in-memory annotation service)."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.parser import VisionParseError
from app.service import AnnotationResult, annotate_image


@pytest.fixture
def sample_pil_image():
    return Image.new("RGB", (800, 600), color="blue")


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NineRouterClient)
    client.get_health.return_value = {"ok": True}
    return client


def test_annotate_image_with_bytes(sample_image_bytes, mock_client):
    """Verify annotate_image accepts raw image bytes directly."""
    fake_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {"label": "car", "box_2d": [100, 200, 400, 600], "confidence": 0.95},
                {"label": "pedestrian", "box_2d": [50, 50, 150, 100], "confidence": 0.88},
            ]
        }),
        raw_response={},
        duration_seconds=0.75,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_image_bytes,
        client=mock_client,
        model="ag/gemini-3.8-flash-high",
    )

    assert isinstance(result, AnnotationResult)
    assert result.original_dimensions == (800, 600)
    assert result.model_used == "ag/gemini-3.8-flash-high"
    assert len(result.shapes) == 2

    # Check CVAT rectangle format
    car_shape = result.shapes[0]
    assert car_shape["label"] == "car"
    assert car_shape["type"] == "rectangle"
    assert car_shape["confidence"] == "0.95"
    assert len(car_shape["points"]) == 4

    # Points are [xtl, ytl, xbr, ybr] in original pixel coordinates
    # box_2d was [ymin, xmin, ymax, xmax] normalized [100, 200, 400, 600] / 1000
    # xmin = 200/1000 * 800 = 160.0
    # ymin = 100/1000 * 600 = 60.0
    # xmax = 600/1000 * 800 = 480.0
    # ymax = 400/1000 * 600 = 240.0
    assert car_shape["points"] == [160.0, 60.0, 480.0, 240.0]


def test_annotate_image_with_pil_and_path(tmp_path, sample_pil_image, mock_client):
    """Verify annotate_image accepts PIL Image and file path inputs."""
    fake_resp = VisionResponse(
        content='{"objects": [{"label": "bus", "box_2d": [200, 200, 500, 500]}]}',
        raw_response={},
        duration_seconds=0.5,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    # 1. PIL Image
    res_pil = annotate_image(image_source=sample_pil_image, client=mock_client)
    assert len(res_pil.shapes) == 1
    assert res_pil.shapes[0]["label"] == "bus"

    # 2. File Path
    img_path = tmp_path / "test.jpg"
    sample_pil_image.save(img_path)
    res_path = annotate_image(image_source=img_path, client=mock_client)
    assert len(res_path.shapes) == 1
    assert res_path.shapes[0]["label"] == "bus"


def test_annotate_image_threshold_filtering(sample_image_bytes, mock_client):
    """Verify detection filtering based on confidence threshold."""
    fake_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {"label": "car", "box_2d": [100, 100, 200, 200], "confidence": 0.95},
                {"label": "rider", "box_2d": [200, 200, 300, 300], "confidence": 0.40},
                {"label": "bicycle", "box_2d": [300, 300, 400, 400], "confidence": 0.65},
            ]
        }),
        raw_response={},
        duration_seconds=0.5,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    # Threshold 0.5 should keep car (0.95) and bicycle (0.65), dropping rider (0.40)
    result = annotate_image(
        image_source=sample_image_bytes,
        client=mock_client,
        threshold=0.5,
    )
    assert len(result.shapes) == 2
    labels = [s["label"] for s in result.shapes]
    assert "car" in labels
    assert "bicycle" in labels
    assert "rider" not in labels

    # Threshold 0.9 should keep only car (0.95)
    result_high = annotate_image(
        image_source=sample_image_bytes,
        client=mock_client,
        threshold=0.9,
    )
    assert len(result_high.shapes) == 1
    assert result_high.shapes[0]["label"] == "car"


def test_annotate_image_fallback_confidence(sample_image_bytes, mock_client):
    """Verify fallback confidence policy when model does not report confidence."""
    fake_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {"label": "car", "box_2d": [100, 100, 200, 200]},  # omitted confidence
            ]
        }),
        raw_response={},
        duration_seconds=0.4,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    # With explicit fallback 1.0 (if user deliberately configured it)
    res = annotate_image(sample_image_bytes, mock_client, fallback_confidence=1.0)
    assert res.shapes[0]["confidence"] == "1.0"

    # With custom fallback 0.8
    res_custom = annotate_image(sample_image_bytes, mock_client, fallback_confidence=0.8)
    assert res_custom.shapes[0]["confidence"] == "0.8"


def test_annotate_image_omitted_confidence_never_fake_one(sample_image_bytes, mock_client):
    """Regression test: missing confidence is NEVER represented as fake 1.0 (100% certainty)."""
    fake_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {"label": "car", "box_2d": [100, 100, 200, 200]},  # confidence omitted by VLM
            ]
        }),
        raw_response={},
        duration_seconds=0.4,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    # 1. When threshold == 0.0 and fallback_confidence is None (default):
    # Shape must be kept, but confidence key MUST be omitted (never "1.0")
    res_zero_threshold = annotate_image(
        sample_image_bytes,
        mock_client,
        threshold=0.0,
        fallback_confidence=None,
    )
    assert len(res_zero_threshold.shapes) == 1
    assert "confidence" not in res_zero_threshold.shapes[0]
    assert res_zero_threshold.shapes[0]["label"] == "car"

    # 2. When threshold > 0.0 and fallback_confidence is None:
    # Unrated detections cannot satisfy the threshold and MUST be filtered out
    res_filtered = annotate_image(
        sample_image_bytes,
        mock_client,
        threshold=0.5,
        fallback_confidence=None,
    )
    assert len(res_filtered.shapes) == 0


def test_annotate_image_dynamic_model_resolution(sample_image_bytes, mock_client):
    """Verify annotate_image dynamically resolves model when not specified."""
    mock_client.resolve_vision_model.return_value = "dynamic/resolved-vision-model"
    fake_resp = VisionResponse(
        content=json.dumps({
            "objects": [{"label": "bus", "box_2d": [50, 50, 200, 200], "confidence": 0.9}]
        }),
        raw_response={},
        duration_seconds=0.3,
        model="dynamic/resolved-vision-model",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    # Omit model argument
    res = annotate_image(sample_image_bytes, mock_client, model=None)

    mock_client.resolve_vision_model.assert_called_once()
    assert res.model_used == "dynamic/resolved-vision-model"
    assert len(res.shapes) == 1


def test_annotate_image_strict_mode_error(sample_image_bytes, mock_client):
    """Verify strict mode raises VisionParseError on disallowed label."""
    fake_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {"label": "unsupported_alien_vehicle", "box_2d": [100, 100, 200, 200]},
            ]
        }),
        raw_response={},
        duration_seconds=0.3,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    # Non-strict mode skips the invalid label
    res_lenient = annotate_image(sample_image_bytes, mock_client, strict=False)
    assert len(res_lenient.shapes) == 0

    # Strict mode raises
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        annotate_image(sample_image_bytes, mock_client, strict=True)


def test_annotate_image_full_31_mode_propagation(sample_image_bytes, mock_client):
    """Verify annotate_image with mode='full_31' passes mode='full_31' to client.send_vision_request."""
    from core.vision_contract import MODE_FULL_31

    fake_resp = VisionResponse(
        content=json.dumps({"objects": [], "regions": [], "lanes": []}),
        raw_response={},
        duration_seconds=0.2,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request.return_value = fake_resp

    annotate_image(
        image_source=sample_image_bytes,
        client=mock_client,
        mode=MODE_FULL_31,
    )

    mock_client.send_vision_request.assert_called_once()
    _, kwargs = mock_client.send_vision_request.call_args
    assert kwargs.get("mode") == MODE_FULL_31


def test_annotate_image_payload_regression_system_prompt_full_31(sample_image_bytes):
    """Regression test: active_mode=MODE_FULL_31 causes client payload to contain SYSTEM_PROMPT_FULL_31, NOT SYSTEM_PROMPT."""
    from core.vision_contract import (
        MODE_BOX,
        MODE_FULL_31,
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_FULL_31,
    )

    real_client = NineRouterClient(base_url="http://127.0.0.1:20128")
    captured_payloads = []

    def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"objects": [], "regions": [], "lanes": []}'
                    }
                }
            ],
            "model": "ag/gemini-3.8-flash-high",
        }
        return resp

    with patch.object(real_client.session, "post", side_effect=mock_post):
        # 1. When mode=MODE_FULL_31: payload must contain SYSTEM_PROMPT_FULL_31, NOT SYSTEM_PROMPT
        annotate_image(
            image_source=sample_image_bytes,
            client=real_client,
            model="ag/gemini-3.8-flash-high",
            mode=MODE_FULL_31,
        )

        assert len(captured_payloads) == 1
        full31_payload = captured_payloads[0]
        sys_msg_full31 = full31_payload["messages"][0]["content"]

        assert sys_msg_full31 == SYSTEM_PROMPT_FULL_31
        assert sys_msg_full31 != SYSTEM_PROMPT
        assert "Policy A — Foreground Instances (14 classes)" in sys_msg_full31
        assert "Policy B — Semantic Regions (10 classes)" in sys_msg_full31
        assert "Policy C — Lane Demarcations & Crosswalks (7 classes)" in sys_msg_full31

        # 2. When mode=MODE_BOX: payload must contain standard bbox SYSTEM_PROMPT
        annotate_image(
            image_source=sample_image_bytes,
            client=real_client,
            model="ag/gemini-3.8-flash-high",
            mode=MODE_BOX,
        )

        assert len(captured_payloads) == 2
        box_payload = captured_payloads[1]
        sys_msg_box = box_payload["messages"][0]["content"]

        assert sys_msg_box == SYSTEM_PROMPT
        assert sys_msg_box != SYSTEM_PROMPT_FULL_31

