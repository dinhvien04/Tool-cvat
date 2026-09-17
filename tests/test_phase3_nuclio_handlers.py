"""Tests for Phase 3 Nuclio serverless handlers (Mask and Box+Mask)."""

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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = REPO_ROOT / "serverless" / "ninerouter-vision-mask" / "nuclio"
BOX_MASK_DIR = REPO_ROOT / "serverless" / "ninerouter-vision-box-mask" / "nuclio"
BOX_DIR = REPO_ROOT / "serverless" / "ninerouter-vision" / "nuclio"


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
def sample_image_b64():
    img = Image.new("RGB", (320, 240), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------
# Tests for ninerouter-vision-mask
# ---------------------------------------------------------------------

def test_mask_function_yaml_contract():
    """Verify ninerouter-vision-mask function.yaml specifies type 'mask' on all labels."""
    yaml_path = MASK_DIR / "function.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    meta = data["metadata"]
    assert meta["name"] == "ninerouter-vision-mask"
    assert meta["namespace"] == "cvat"
    assert meta["annotations"]["name"] == "9Router Vision Mask (Instance Segmentation)"
    assert meta["annotations"]["type"] == "detector"

    spec_labels = json.loads(meta["annotations"]["spec"])
    assert len(spec_labels) == 13
    for lbl in spec_labels:
        assert lbl["type"] == "mask", f"Label {lbl['name']} must have type 'mask', got {lbl['type']}"

    assert data["spec"]["build"]["image"] == "cvat.custom.ninerouter.vision.mask"


def test_mask_model_handler_infer(sample_image_b64):
    """Verify Mask ModelHandler returns valid CVAT mask shape format."""
    mask_mh_mod = load_module_from_path("mask_model_handler", MASK_DIR / "model_handler.py")
    MaskModelHandler = mask_mh_mod.ModelHandler

    with patch.object(mask_mh_mod, "NineRouterClient") as MockClient, \
         patch.object(mask_mh_mod, "annotate_image") as mock_annotate:
        client_inst = MockClient.return_value
        client_inst.resolve_segmentation_model.return_value = "ag/gemini-3.8-flash-high"
        client_inst.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"

        # Mock annotation result with native CVAT mask shape
        mock_result = MagicMock()
        mock_result.original_dimensions = (320, 240)
        mock_result.api_duration_seconds = 0.45
        dummy_mask_shape = {
            "label": "car",
            "type": "mask",
            "confidence": "0.95",
            # Flattened crop (e.g. 4 pixels) + [xmin, ymin, xmax, ymax]
            "mask": [1, 1, 0, 1, 10, 20, 11, 21],
            "points": [10.0, 20.0, 11.0, 20.0, 11.0, 21.0, 10.0, 21.0],
        }
        mock_result.to_cvat_masks.return_value = [dummy_mask_shape]
        mock_annotate.return_value = mock_result

        handler_inst = MaskModelHandler()
        shapes = handler_inst.infer(image_bytes=base64.b64decode(sample_image_b64), threshold=0.5)

        assert len(shapes) == 1
        shape = shapes[0]
        assert shape["type"] == "mask"
        assert shape["label"] == "car"
        assert shape["confidence"] == "0.95"
        assert len(shape["mask"]) >= 5
        # Verify trailing 4 coordinates are bounding box
        xmin, ymin, xmax, ymax = shape["mask"][-4:]
        assert xmin == 10 and ymin == 20 and xmax == 11 and ymax == 21


def test_mask_main_handler_end_to_end(sample_image_b64):
    """Verify Mask main handler handles request, validates payload, and returns 200."""
    mask_main_mod = load_module_from_path("mask_main", MASK_DIR / "main.py")
    ctx = MockContext()

    mock_handler_inst = MagicMock()
    mock_handler_inst.infer.return_value = [
        {
            "label": "pedestrian",
            "type": "mask",
            "confidence": "0.89",
            "mask": [1, 0, 1, 1, 5, 10, 6, 11],
        }
    ]
    ctx.user_data.model_handler = mock_handler_inst

    event = MockEvent(body=json.dumps({"image": sample_image_b64, "threshold": 0.5}))
    resp = mask_main_mod.handler(ctx, event)

    assert resp.status_code == 200
    shapes = json.loads(resp.body)
    assert len(shapes) == 1
    assert shapes[0]["type"] == "mask"
    assert shapes[0]["label"] == "pedestrian"


# ---------------------------------------------------------------------
# Tests for ninerouter-vision-box-mask
# ---------------------------------------------------------------------

def test_box_mask_function_yaml_contract():
    """Verify ninerouter-vision-box-mask function.yaml specifies type 'any' on all labels."""
    yaml_path = BOX_MASK_DIR / "function.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    meta = data["metadata"]
    assert meta["name"] == "ninerouter-vision-box-mask"
    assert meta["namespace"] == "cvat"
    assert meta["annotations"]["name"] == "9Router Vision Box+Mask"
    assert meta["annotations"]["type"] == "detector"

    spec_labels = json.loads(meta["annotations"]["spec"])
    assert len(spec_labels) == 13
    for lbl in spec_labels:
        assert lbl["type"] == "any", f"Label {lbl['name']} must have type 'any', got {lbl['type']}"

    assert data["spec"]["build"]["image"] == "cvat.custom.ninerouter.vision.boxmask"


def test_box_mask_model_handler_infer(sample_image_b64):
    """Verify Box+Mask ModelHandler returns both rectangle and mask shapes with paired group_id."""
    boxmask_mh_mod = load_module_from_path("boxmask_model_handler", BOX_MASK_DIR / "model_handler.py")
    BoxMaskModelHandler = boxmask_mh_mod.ModelHandler

    with patch.object(boxmask_mh_mod, "NineRouterClient") as MockClient, \
         patch.object(boxmask_mh_mod, "annotate_image") as mock_annotate:
        client_inst = MockClient.return_value
        client_inst.resolve_segmentation_model.return_value = "ag/gemini-3.8-flash-high"
        client_inst.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"

        mock_result = MagicMock()
        mock_result.original_dimensions = (320, 240)
        mock_result.api_duration_seconds = 0.50
        mock_result.shapes = [
            {
                "label": "car",
                "type": "rectangle",
                "points": [10.0, 20.0, 100.0, 80.0],
                "confidence": "0.95",
                "group_id": 1,
            },
            {
                "label": "car",
                "type": "mask",
                "mask": [1, 1, 1, 1, 10, 20, 100, 80],
                "confidence": "0.95",
                "group_id": 1,
            }
        ]
        mock_annotate.return_value = mock_result

        handler_inst = BoxMaskModelHandler()
        shapes = handler_inst.infer(image_bytes=base64.b64decode(sample_image_b64), threshold=0.5)

        assert len(shapes) == 2
        rect_shape = next(s for s in shapes if s["type"] == "rectangle")
        mask_shape = next(s for s in shapes if s["type"] == "mask")

        assert rect_shape["group_id"] == mask_shape["group_id"] == 1
        assert rect_shape["label"] == mask_shape["label"] == "car"


def test_box_mask_main_handler_end_to_end(sample_image_b64):
    """Verify Box+Mask main handler returns 200 with both shapes."""
    boxmask_main_mod = load_module_from_path("boxmask_main", BOX_MASK_DIR / "main.py")
    ctx = MockContext()

    mock_handler_inst = MagicMock()
    mock_handler_inst.infer.return_value = [
        {
            "label": "truck",
            "type": "rectangle",
            "points": [50.0, 50.0, 200.0, 180.0],
            "confidence": "0.91",
            "group_id": 42,
        },
        {
            "label": "truck",
            "type": "mask",
            "mask": [1, 0, 1, 1, 50, 50, 200, 180],
            "confidence": "0.91",
            "group_id": 42,
        },
    ]
    ctx.user_data.model_handler = mock_handler_inst

    event = MockEvent(body=json.dumps({"image": sample_image_b64, "threshold": 0.5}))
    resp = boxmask_main_mod.handler(ctx, event)

    assert resp.status_code == 200
    shapes = json.loads(resp.body)
    assert len(shapes) == 2
    types = {s["type"] for s in shapes}
    assert types == {"rectangle", "mask"}
    assert shapes[0]["group_id"] == shapes[1]["group_id"] == 42


# ---------------------------------------------------------------------
# Error and Security Edge Cases
# ---------------------------------------------------------------------

def test_handler_oversized_payload_rejected():
    """Verify request bodies exceeding 32MB are rejected with 413 Payload Too Large."""
    mask_main_mod = load_module_from_path("mask_main_oversize", MASK_DIR / "main.py")
    ctx = MockContext()

    # Oversized raw string
    oversized_body = "x" * (33554432 + 10)
    event = MockEvent(body=oversized_body)
    resp = mask_main_mod.handler(ctx, event)
    assert resp.status_code == 413
    assert "exceeds maximum allowed size" in json.loads(resp.body)["error"]


def test_handler_api_key_masked_in_error_logs(sample_image_b64):
    """Verify secret API keys are masked if leaked into exception messages."""
    mask_main_mod = load_module_from_path("mask_main_sec", MASK_DIR / "main.py")
    ctx = MockContext()

    mock_handler_inst = MagicMock()
    secret_key = "sk-super-secret-key-12345"
    mock_handler_inst.api_key = secret_key
    mock_handler_inst.infer.side_effect = RuntimeError(f"Failed authenticating with {secret_key}")
    ctx.user_data.model_handler = mock_handler_inst

    event = MockEvent(body=json.dumps({"image": sample_image_b64}))
    resp = mask_main_mod.handler(ctx, event)

    assert resp.status_code == 502
    err = json.loads(resp.body)["error"]
    assert secret_key not in err
    assert "***" in err


# ---------------------------------------------------------------------
# Detection Mode Configuration & Override Tests
# ---------------------------------------------------------------------

def test_box_function_yaml_declares_detection_mode_box():
    """Verify ninerouter-vision function.yaml declares DETECTION_MODE 'box'."""
    yaml_path = BOX_DIR / "function.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    env_vars = {item["name"]: item["value"] for item in data["spec"]["env"]}
    assert env_vars.get("DETECTION_MODE") == "box"


def test_model_handler_detection_mode_env(sample_image_b64, monkeypatch):
    """Verify DETECTION_MODE env var configures default_mode in ModelHandler."""
    monkeypatch.setenv("DETECTION_MODE", "box_and_mask")
    mask_mh_mod = load_module_from_path("mask_mh_mode_env", MASK_DIR / "model_handler.py")
    with patch.object(mask_mh_mod, "NineRouterClient") as MockClient:
        MockClient.return_value.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        handler = mask_mh_mod.ModelHandler()
        assert handler.default_mode == "box_and_mask"


def test_model_handler_infer_mode_override(sample_image_b64):
    """Verify passing mode to infer() overrides default_mode and passes to annotate_image."""
    mask_mh_mod = load_module_from_path("mask_mh_override", MASK_DIR / "model_handler.py")
    with patch.object(mask_mh_mod, "NineRouterClient") as MockClient, \
         patch.object(mask_mh_mod, "annotate_image") as mock_annotate:
        MockClient.return_value.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        mock_result = MagicMock()
        mock_result.original_dimensions = (320, 240)
        mock_result.api_duration_seconds = 0.3
        mock_result.to_cvat_rectangles.return_value = [
            {"label": "car", "type": "rectangle", "points": [10, 20, 50, 60], "confidence": "0.9"}
        ]
        mock_annotate.return_value = mock_result

        handler = mask_mh_mod.ModelHandler()
        assert handler.default_mode == "mask"

        # Explicitly override with mode='box'
        shapes = handler.infer(
            image_bytes=base64.b64decode(sample_image_b64),
            threshold=0.5,
            mode="box",
        )
        assert len(shapes) == 1
        assert shapes[0]["type"] == "rectangle"

        # Ensure annotate_image was called with mode="box"
        mock_annotate.assert_called_once()
        _, kwargs = mock_annotate.call_args
        assert kwargs.get("mode") == "box"


def test_model_handler_invalid_mode_rejected(sample_image_b64):
    """Verify invalid detection mode raises ValueError."""
    mask_mh_mod = load_module_from_path("mask_mh_invalid", MASK_DIR / "model_handler.py")
    with patch.object(mask_mh_mod, "NineRouterClient") as MockClient:
        MockClient.return_value.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        handler = mask_mh_mod.ModelHandler()

        with pytest.raises(ValueError, match="Invalid detection mode"):
            handler.infer(
                image_bytes=base64.b64decode(sample_image_b64),
                mode="invalid_unsupported_mode",
            )


def test_main_handler_mode_payload_override(sample_image_b64):
    """Verify main handler extracts 'mode' from request JSON and forwards to infer()."""
    mask_main_mod = load_module_from_path("mask_main_override", MASK_DIR / "main.py")
    ctx = MockContext()
    mock_handler_inst = MagicMock()
    mock_handler_inst.infer.return_value = []
    ctx.user_data.model_handler = mock_handler_inst

    event = MockEvent(
        body=json.dumps({
            "image": sample_image_b64,
            "threshold": 0.6,
            "mode": "box_and_mask",
        })
    )
    resp = mask_main_mod.handler(ctx, event)
    assert resp.status_code == 200
    mock_handler_inst.infer.assert_called_once_with(
        image_bytes=base64.b64decode(sample_image_b64),
        threshold=0.6,
        mode="box_and_mask",
    )


def test_main_handler_invalid_mode_returns_400(sample_image_b64):
    """Verify main handler returns HTTP 400 Bad Request when invalid mode is supplied in payload."""
    mask_main_mod = load_module_from_path("mask_main_bad_mode", MASK_DIR / "main.py")
    ctx = MockContext()

    mock_handler_inst = MagicMock()
    mock_handler_inst.infer.side_effect = ValueError("Invalid detection mode 'unsupported_mode'. Must be one of: 'box', 'mask', 'box_and_mask'")
    ctx.user_data.model_handler = mock_handler_inst

    event = MockEvent(
        body=json.dumps({
            "image": sample_image_b64,
            "threshold": 0.5,
            "mode": "unsupported_mode",
        })
    )
    resp = mask_main_mod.handler(ctx, event)
    assert resp.status_code == 400
    err_body = json.loads(resp.body)
    assert "error" in err_body
    assert "Invalid detection mode" in err_body["error"]


def test_all_three_function_yamls_declare_detection_mode():
    """Verify all 3 function.yaml manifests declare their respective DETECTION_MODE in spec.env."""
    expectations = [
        (BOX_DIR / "function.yaml", "box"),
        (MASK_DIR / "function.yaml", "mask"),
        (BOX_MASK_DIR / "function.yaml", "box_and_mask"),
    ]
    for yaml_path, expected_mode in expectations:
        assert yaml_path.exists(), f"Missing {yaml_path}"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        env_vars = {item["name"]: item["value"] for item in data["spec"]["env"]}
        assert env_vars.get("DETECTION_MODE") == expected_mode, (
            f"{yaml_path.parent.parent.name} must have DETECTION_MODE='{expected_mode}', got {env_vars.get('DETECTION_MODE')}"
        )


def test_model_handler_constructor_mode_parameter():
    """Verify ModelHandler.__init__(mode=...) sets default_mode directly."""
    mask_mh_mod = load_module_from_path("mask_mh_ctor_mode", MASK_DIR / "model_handler.py")
    with patch.object(mask_mh_mod, "NineRouterClient") as MockClient:
        MockClient.return_value.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        handler = mask_mh_mod.ModelHandler(mode="box_and_mask")
        assert handler.default_mode == "box_and_mask"


def test_model_handler_invalid_env_mode_fallback(monkeypatch):
    """Verify invalid DETECTION_MODE in environment falls back to function default mode."""
    monkeypatch.setenv("DETECTION_MODE", "completely_bogus_mode")
    mask_mh_mod = load_module_from_path("mask_mh_bogus_env", MASK_DIR / "model_handler.py")
    with patch.object(mask_mh_mod, "NineRouterClient") as MockClient:
        MockClient.return_value.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        handler = mask_mh_mod.ModelHandler()
        # Should fall back to MODE_MASK for mask handler
        assert handler.default_mode == "mask"


@pytest.mark.parametrize(
    "handler_dir, default_mode",
    [
        (BOX_DIR, "box"),
        (MASK_DIR, "mask"),
        (BOX_MASK_DIR, "box_and_mask"),
    ],
)
def test_all_three_handlers_default_and_override(handler_dir: Path, default_mode: str, sample_image_b64):
    """Verify each handler defaults to its expected mode and can be overridden via infer(mode=...)."""
    mh_mod = load_module_from_path(f"mh_{handler_dir.parent.name}", handler_dir / "model_handler.py")
    with patch.object(mh_mod, "NineRouterClient") as MockClient, \
         patch.object(mh_mod, "annotate_image") as mock_annotate:
        MockClient.return_value.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        mock_result = MagicMock()
        mock_result.original_dimensions = (320, 240)
        mock_result.api_duration_seconds = 0.2
        mock_result.shapes = [{"label": "car", "type": "rectangle", "points": [0, 0, 10, 10], "group_id": 1}]
        mock_result.to_cvat_rectangles.return_value = [{"label": "car", "type": "rectangle", "points": [0, 0, 10, 10]}]
        mock_result.to_cvat_masks.return_value = [{"label": "car", "type": "mask", "mask": [1, 0, 0, 1, 1]}]
        mock_annotate.return_value = mock_result

        handler = mh_mod.ModelHandler()
        assert handler.default_mode == default_mode

        # Test default execution
        handler.infer(image_bytes=base64.b64decode(sample_image_b64))
        _, kwargs_def = mock_annotate.call_args
        assert kwargs_def.get("mode") == default_mode

        mock_annotate.reset_mock()

        # Test override with another mode
        override_target = "box" if default_mode != "box" else "mask"
        handler.infer(image_bytes=base64.b64decode(sample_image_b64), mode=override_target)
        _, kwargs_override = mock_annotate.call_args
        assert kwargs_override.get("mode") == override_target


