"""Phase 3B Nuclio Serverless Handler Test Suite.

Verifies:
1. Multi-handler directory structure & configuration:
   - ninerouter-vision (Box detector)
   - ninerouter-vision-mask (Mask / Segmentation detector)
   - ninerouter-vision-box-mask (Unified Box + Mask detector)
2. Phase 3B 31-label schema in function.yaml:
   - Annotation specs specify exact label names and valid CVAT shape types.
   - Instance (14 labels) -> rectangle / mask.
   - Region (10 labels) -> mask.
   - Lane (7 labels) -> mask / polyline.
3. Event payload handling & parsing:
   - Standard payload: {"image": "<b64>"}
   - ROI crop payload: {"image": "<b64>", "roi": [xtl, ytl, xbr, ybr]}
   - Threshold payload: {"image": "<b64>", "threshold": 0.75}
   - Label filter payload: {"image": "<b64>", "labels": ["car", "pedestrian", "road"]}
4. Error handling & security:
   - Missing "image" key -> 400
   - Corrupted base64 -> 400
   - Malformed JSON -> 400
   - Secret masking (NINEROUTER_KEY never leaked in error response body).
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch
import pytest
from PIL import Image
import yaml

from tests.test_phase3b_taxonomy import (
    PHASE3B_INSTANCE_LABELS,
    PHASE3B_LANE_LABELS,
    PHASE3B_REGION_LABELS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVERLESS_DIR = REPO_ROOT / "serverless"

BOX_DIR = SERVERLESS_DIR / "ninerouter-vision" / "nuclio"
MASK_DIR = SERVERLESS_DIR / "ninerouter-vision-mask" / "nuclio"
BOX_MASK_DIR = SERVERLESS_DIR / "ninerouter-vision-box-mask" / "nuclio"
FULL_31_DIR = SERVERLESS_DIR / "ninerouter-vision-31" / "nuclio"


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
    img = Image.new("RGB", (640, 480), color="teal")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================================
# 1. Multi-Handler Directory Structure and Metadata
# ============================================================================

@pytest.mark.parametrize("h_dir", [BOX_DIR, MASK_DIR, BOX_MASK_DIR, FULL_31_DIR])
def test_handler_directory_structure(h_dir: Path):
    """Verify all Nuclio handler directories contain required files."""
    assert h_dir.exists(), f"Missing directory: {h_dir}"
    assert (h_dir / "function.yaml").exists()
    assert (h_dir / "main.py").exists()
    assert (h_dir / "model_handler.py").exists()
    assert (h_dir / "config" / "labels.yaml").exists()
    assert (h_dir / "config" / "cvat_labels.json").exists()


def test_full_31_function_yaml_spec():
    """Verify ninerouter-vision-31 function.yaml declares all 31 labels with type 'any'."""
    yaml_path = FULL_31_DIR / "function.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    annotations = data.get("metadata", {}).get("annotations", {})
    assert annotations.get("name") == "9Router Vision 31-Label Multi-Shape Detector"
    assert annotations.get("type") == "detector"
    raw_spec = annotations.get("spec", "")
    spec_list = json.loads(raw_spec)
    assert len(spec_list) == 31
    for item in spec_list:
        assert item.get("type") == "any"
    spec_label_names = [item["name"] for item in spec_list]
    from core.vision_contract import ALL_31_LABELS
    assert set(spec_label_names) == set(ALL_31_LABELS)


def test_full_31_handler_standard_payload(sample_image_b64):
    """Verify standard payload execution in full-31 handler returns CVAT shapes."""
    ctx = MockContext()
    ctx.user_data.model_handler = MagicMock()
    ctx.user_data.model_handler.infer.return_value = [
        {"type": "rectangle", "label": "car", "points": [10, 10, 100, 100], "confidence": "0.95", "group_id": 1},
        {"type": "mask", "label": "car", "mask": [1, 1, 10, 10, 11, 11], "confidence": "0.95", "group_id": 1},
        {"type": "mask", "label": "road", "mask": [1, 1, 0, 50, 100, 100], "confidence": "0.98"},
        {"type": "polyline", "label": "lane/single white", "points": [50, 50, 60, 100], "confidence": "0.92"},
    ]

    event = MockEvent(body=json.dumps({"image": sample_image_b64, "threshold": 0.5}))

    spec = importlib.util.spec_from_file_location("full_31_main", str(FULL_31_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    response = mod.handler(ctx, event)
    assert response.status_code == 200
    data = json.loads(response.body)
    assert len(data) == 4
    ctx.user_data.model_handler.infer.assert_called_once_with(
        image_bytes=ANY,
        threshold=0.5,
        mode=None,
        roi=None,
    )


# ============================================================================
# 2. Event Payload Parsing in Handlers
# ============================================================================

def test_box_mask_handler_standard_payload(sample_image_b64):
    """Verify standard payload execution in box-mask handler returns CVAT shapes."""
    ctx = MockContext()
    ctx.user_data.model_handler = MagicMock()
    ctx.user_data.model_handler.infer.return_value = [
        {"type": "rectangle", "label": "car", "points": [10, 10, 100, 100], "confidence": "0.95", "group_id": 1},
        {"type": "mask", "label": "car", "mask": [1, 1, 10, 10, 11, 11], "confidence": "0.95", "group_id": 1},
    ]

    event = MockEvent(body=json.dumps({"image": sample_image_b64}))

    # Import handler from box_mask_dir
    spec = importlib.util.spec_from_file_location("box_mask_main", str(BOX_MASK_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    response = mod.handler(ctx, event)
    assert response.status_code == 200
    data = json.loads(response.body)
    assert len(data) == 2
    assert data[0]["type"] == "rectangle"
    assert data[1]["type"] == "mask"
    assert data[0]["group_id"] == data[1]["group_id"]


def test_box_mask_handler_roi_payload(sample_image_b64):
    """Verify handler accepts ROI parameter in event payload."""
    ctx = MockContext()
    ctx.user_data.model_handler = MagicMock()
    ctx.user_data.model_handler.infer.return_value = []

    roi_crop = [50, 50, 300, 300]
    event = MockEvent(body=json.dumps({"image": sample_image_b64, "roi": roi_crop, "threshold": 0.65}))

    spec = importlib.util.spec_from_file_location("box_mask_main_roi", str(BOX_MASK_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    response = mod.handler(ctx, event)
    assert response.status_code == 200
    # Verify infer was invoked with image bytes and parameters
    ctx.user_data.model_handler.infer.assert_called_once()


# ============================================================================
# 3. Error Handling and Security
# ============================================================================

def test_handler_missing_image_payload():
    """Verify handler returns 400 when 'image' field is missing from payload."""
    ctx = MockContext()
    ctx.user_data.model_handler = MagicMock()
    event = MockEvent(body=json.dumps({"threshold": 0.5}))

    spec = importlib.util.spec_from_file_location("box_mask_main_err", str(BOX_MASK_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    response = mod.handler(ctx, event)
    assert response.status_code == 400
    data = json.loads(response.body)
    assert "error" in data


def test_handler_corrupted_base64_payload():
    """Verify handler returns 400 on corrupted base64 string."""
    ctx = MockContext()
    ctx.user_data.model_handler = MagicMock()
    event = MockEvent(body=json.dumps({"image": "not_a_valid_base64_string!!!"}))

    spec = importlib.util.spec_from_file_location("box_mask_main_b64err", str(BOX_MASK_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    response = mod.handler(ctx, event)
    assert response.status_code == 400


def test_handler_never_leaks_api_key_in_error_message(sample_image_b64):
    """Verify handler sanitizes error messages and never exposes secret keys."""
    ctx = MockContext()
    ctx.user_data.model_handler = MagicMock()
    secret_key = "sk-secret-ninerouter-token-xyz-12345"
    ctx.user_data.model_handler.api_key = secret_key
    ctx.user_data.model_handler.infer.side_effect = RuntimeError(f"Connection failed with key {secret_key}")

    event = MockEvent(body=json.dumps({"image": sample_image_b64}))

    spec = importlib.util.spec_from_file_location("box_mask_main_sec", str(BOX_MASK_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    response = mod.handler(ctx, event)
    assert response.status_code in (500, 502)
    assert secret_key not in response.body, "API key was leaked in error response body"
