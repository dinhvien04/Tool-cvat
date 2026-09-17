"""Tests for Nuclio serverless handler and ModelHandler."""

import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import sys

# Dynamically load serverless modules to avoid collision with root main.py
serverless_dir = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-vision" / "nuclio"
if str(serverless_dir) not in sys.path:
    sys.path.insert(0, str(serverless_dir))

spec_mh = importlib.util.spec_from_file_location("model_handler", str(serverless_dir / "model_handler.py"))
mh_module = importlib.util.module_from_spec(spec_mh)
sys.modules["model_handler"] = mh_module
spec_mh.loader.exec_module(mh_module)
ModelHandler = mh_module.ModelHandler
load_spec_labels_from_function_yaml = mh_module.load_spec_labels_from_function_yaml

spec_main = importlib.util.spec_from_file_location("nuclio_main", str(serverless_dir / "main.py"))
nuclio_main = importlib.util.module_from_spec(spec_main)
sys.modules["nuclio_main"] = nuclio_main
spec_main.loader.exec_module(nuclio_main)
handler = nuclio_main.handler
init_context = nuclio_main.init_context


class MockResponse:
    """Mock Nuclio HTTP Response."""
    def __init__(self, body, headers=None, content_type="application/json", status_code=200):
        self.body = body
        self.headers = headers or {}
        self.content_type = content_type
        self.status_code = status_code


class MockContext:
    """Mock Nuclio Context."""
    def __init__(self):
        self.logger = MagicMock()
        self.user_data = SimpleNamespace()
        self.Response = MockResponse


class MockEvent:
    """Mock Nuclio Event."""
    def __init__(self, body):
        self.body = body


@pytest.fixture
def dummy_image_b64():
    img = Image.new("RGB", (400, 300), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_init_context():
    """Verify init_context initializes ModelHandler onto context.user_data."""
    ctx = MockContext()
    with patch("model_handler.NineRouterClient"):
        init_context(ctx)
    assert hasattr(ctx.user_data, "model_handler")
    assert isinstance(ctx.user_data.model_handler, ModelHandler)


def test_load_spec_labels_from_function_yaml():
    """Verify extracting the 13 rectangle labels from function.yaml."""
    yaml_path = serverless_dir / "function.yaml"
    labels = load_spec_labels_from_function_yaml(yaml_path)
    assert len(labels) == 13
    assert "car" in labels
    assert "pedestrian" in labels
    assert "traffic light" in labels
    assert "person" in labels


def test_handler_success(dummy_image_b64):
    """Verify handler processes valid CVAT request returning rectangle shapes."""
    ctx = MockContext()
    mock_model_handler = MagicMock()
    mock_model_handler.infer.return_value = [
        {
            "confidence": "0.92",
            "label": "car",
            "points": [10.0, 20.0, 150.0, 200.0],
            "type": "rectangle",
        }
    ]
    ctx.user_data.model_handler = mock_model_handler

    event = MockEvent(body=json.dumps({"image": dummy_image_b64, "threshold": 0.6}))
    resp = handler(ctx, event)

    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    shapes = json.loads(resp.body)
    assert len(shapes) == 1
    assert shapes[0]["label"] == "car"
    assert shapes[0]["confidence"] == "0.92"
    assert shapes[0]["type"] == "rectangle"
    assert shapes[0]["points"] == [10.0, 20.0, 150.0, 200.0]

    mock_model_handler.infer.assert_called_once()
    call_args = mock_model_handler.infer.call_args
    assert call_args.kwargs["threshold"] == 0.6


def test_handler_data_url_prefix_stripped(dummy_image_b64):
    """Verify handler correctly strips 'data:image/jpeg;base64,' prefix if passed."""
    ctx = MockContext()
    mock_model_handler = MagicMock()
    mock_model_handler.infer.return_value = []
    ctx.user_data.model_handler = mock_model_handler

    data_url = f"data:image/jpeg;base64,{dummy_image_b64}"
    event = MockEvent(body={"image": data_url, "threshold": 0.5})
    resp = handler(ctx, event)

    assert resp.status_code == 200
    mock_model_handler.infer.assert_called_once()
    passed_bytes = mock_model_handler.infer.call_args.kwargs["image_bytes"]
    assert len(passed_bytes) > 0


def test_handler_invalid_json():
    """Verify handler returns 400 when body is malformed JSON."""
    ctx = MockContext()
    event = MockEvent(body="NOT_JSON_BODY")
    resp = handler(ctx, event)

    assert resp.status_code == 400
    data = json.loads(resp.body)
    assert "error" in data


def test_handler_missing_image():
    """Verify handler returns 400 when image field is omitted."""
    ctx = MockContext()
    event = MockEvent(body={"threshold": 0.5})
    resp = handler(ctx, event)

    assert resp.status_code == 400
    data = json.loads(resp.body)
    assert "Missing mandatory 'image' field" in data["error"]


def test_handler_invalid_base64():
    """Verify handler returns 400 when base64 is malformed."""
    ctx = MockContext()
    event = MockEvent(body={"image": "!!!not-valid-base64!!!"})
    resp = handler(ctx, event)

    assert resp.status_code == 400
    data = json.loads(resp.body)
    assert "Failed to decode Base64" in data["error"]


def test_handler_upstream_inference_failure(dummy_image_b64):
    """Verify handler returns 502 when 9Router inference fails."""
    ctx = MockContext()
    mock_model_handler = MagicMock()
    mock_model_handler.infer.side_effect = RuntimeError("Connection timed out to 9Router")
    ctx.user_data.model_handler = mock_model_handler

    event = MockEvent(body={"image": dummy_image_b64})
    resp = handler(ctx, event)

    assert resp.status_code == 502
    data = json.loads(resp.body)
    assert "Inference failed" in data["error"]
