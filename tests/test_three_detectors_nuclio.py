"""Test Suite for the Exactly Three 9Router CVAT Detectors.

Verifies:
1. Directory structure and configuration for the 3 user-facing detectors:
   - 9Router Rectangle + Mask (ninerouter-rectangle-mask)
   - 9Router Polygon + Mask (ninerouter-polygon-mask)
   - 9Router Polyline (ninerouter-polyline)
2. Exact label taxonomy partition:
   - Detector 1: Exactly 14 foreground instance labels, type: "any"
   - Detector 2: Exactly 10 semantic region labels, type: "any"
   - Detector 3: Exactly 7 lane demarcation labels, type: "polyline"
3. Shape emission policies:
   - Detector 1: Emits paired rectangle + mask sharing identical group_id
   - Detector 2: Emits paired polygon + mask sharing identical group_id, strictly suppresses boxes
   - Detector 3: Emits polyline ONLY, no boxes, polygons, masks, or group_id
4. Edge cases, error handling, and secret masking.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import yaml

from core.taxonomy import BOX_MASK_LABELS, POLYGON_MASK_LABELS, POLYLINE_LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVERLESS_DIR = REPO_ROOT / "serverless"

RECT_MASK_DIR = SERVERLESS_DIR / "ninerouter-rectangle-mask" / "nuclio"
POLY_MASK_DIR = SERVERLESS_DIR / "ninerouter-polygon-mask" / "nuclio"
POLYLINE_DIR = SERVERLESS_DIR / "ninerouter-polyline" / "nuclio"


class MockResponse:
    def __init__(self, body, headers=None, content_type="application/json", status_code=200):
        self.body = body
        self.headers = headers or {}
        self.content_type = content_type
        self.status_code = status_code


class MockContext:
    def __init__(self):
        self.logger = MagicMock()
        self.user_data = SimpleNamespace()
        self.Response = MockResponse


class MockEvent:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}


def create_dummy_jpeg_base64(width=100, height=100) -> str:
    im = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestThreeDetectorsConfiguration:
    """Verify function.yaml specifications and partition across the three detectors."""

    def test_rectangle_mask_function_yaml(self):
        fn_yaml = RECT_MASK_DIR / "function.yaml"
        assert fn_yaml.exists()
        with open(fn_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["metadata"]["name"] == "ninerouter-rectangle-mask"
        assert data["metadata"]["annotations"]["name"] == "9Router Rectangle + Mask"
        assert data["metadata"]["annotations"]["type"] == "detector"

        spec = json.loads(data["metadata"]["annotations"]["spec"])
        assert len(spec) == 14
        spec_names = [item["name"] for item in spec]
        assert set(spec_names) == set(BOX_MASK_LABELS)
        for item in spec:
            assert item["type"] == "any", f"Label {item['name']} must have type 'any' for dual-shape"

        env_map = {e["name"]: e["value"] for e in data["spec"]["env"]}
        assert env_map["DETECTION_MODE"] == "rectangle_mask"

    def test_polygon_mask_function_yaml(self):
        fn_yaml = POLY_MASK_DIR / "function.yaml"
        assert fn_yaml.exists()
        with open(fn_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["metadata"]["name"] == "ninerouter-polygon-mask"
        assert data["metadata"]["annotations"]["name"] == "9Router Polygon + Mask"
        assert data["metadata"]["annotations"]["type"] == "detector"

        spec = json.loads(data["metadata"]["annotations"]["spec"])
        assert len(spec) == 10
        spec_names = [item["name"] for item in spec]
        assert set(spec_names) == set(POLYGON_MASK_LABELS)
        for item in spec:
            assert item["type"] == "any", f"Label {item['name']} must have type 'any' for dual-shape"

        env_map = {e["name"]: e["value"] for e in data["spec"]["env"]}
        assert env_map["DETECTION_MODE"] == "polygon_mask"

    def test_polyline_function_yaml(self):
        fn_yaml = POLYLINE_DIR / "function.yaml"
        assert fn_yaml.exists()
        with open(fn_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["metadata"]["name"] == "ninerouter-polyline"
        assert data["metadata"]["annotations"]["name"] == "9Router Polyline"
        assert data["metadata"]["annotations"]["type"] == "detector"

        spec = json.loads(data["metadata"]["annotations"]["spec"])
        assert len(spec) == 7
        spec_names = [item["name"] for item in spec]
        assert set(spec_names) == set(POLYLINE_LABELS)
        for item in spec:
            assert item["type"] == "polyline", f"Label {item['name']} must have type 'polyline'"

        env_map = {e["name"]: e["value"] for e in data["spec"]["env"]}
        assert env_map["DETECTION_MODE"] == "polyline"

    def test_partition_covers_31_labels_disjointly(self):
        fn1 = json.loads(yaml.safe_load(open(RECT_MASK_DIR / "function.yaml", encoding="utf-8"))["metadata"]["annotations"]["spec"])
        fn2 = json.loads(yaml.safe_load(open(POLY_MASK_DIR / "function.yaml", encoding="utf-8"))["metadata"]["annotations"]["spec"])
        fn3 = json.loads(yaml.safe_load(open(POLYLINE_DIR / "function.yaml", encoding="utf-8"))["metadata"]["annotations"]["spec"])

        set1 = {item["name"] for item in fn1}
        set2 = {item["name"] for item in fn2}
        set3 = {item["name"] for item in fn3}

        assert len(set1) == 14
        assert len(set2) == 10
        assert len(set3) == 7

        # Disjoint
        assert len(set1.intersection(set2)) == 0
        assert len(set1.intersection(set3)) == 0
        assert len(set2.intersection(set3)) == 0

        # Union is exactly 31
        union = set1.union(set2).union(set3)
        assert len(union) == 31


class TestThreeDetectorsHandlers:
    """Verify execution logic and policy output of the three handlers."""

    def test_rectangle_mask_handler_paired_output(self):
        mod = load_module_from_path("main_rect_mask", RECT_MASK_DIR / "main.py")
        ctx = MockContext()

        mock_shapes = [
            {"type": "rectangle", "label": "car", "points": [10.0, 20.0, 100.0, 150.0], "group_id": 1, "confidence": "0.95"},
            {"type": "mask", "label": "car", "points": [10.0, 20.0, 100.0, 150.0], "mask": [0, 0, 1, 1], "group_id": 1, "confidence": "0.95"},
        ]

        mock_handler = MagicMock()
        mock_handler.infer.return_value = mock_shapes
        ctx.user_data.model_handler = mock_handler

        b64 = create_dummy_jpeg_base64()
        event = MockEvent(body=json.dumps({"image": b64, "threshold": 0.5}))

        resp = mod.handler(ctx, event)
        assert resp.status_code == 200
        shapes = json.loads(resp.body)
        assert len(shapes) == 2
        assert shapes[0]["type"] == "rectangle"
        assert shapes[1]["type"] == "mask"
        assert shapes[0]["group_id"] == shapes[1]["group_id"] == 1

    def test_polygon_mask_handler_paired_output(self):
        mod = load_module_from_path("main_poly_mask", POLY_MASK_DIR / "main.py")
        ctx = MockContext()

        mock_shapes = [
            {"type": "polygon", "label": "road", "points": [0.0, 50.0, 100.0, 50.0, 100.0, 100.0, 0.0, 100.0], "group_id": 1, "confidence": "0.98"},
            {"type": "mask", "label": "road", "points": [0.0, 50.0, 100.0, 100.0], "mask": [1, 1], "group_id": 1, "confidence": "0.98"},
        ]

        mock_handler = MagicMock()
        mock_handler.infer.return_value = mock_shapes
        ctx.user_data.model_handler = mock_handler

        b64 = create_dummy_jpeg_base64()
        event = MockEvent(body=json.dumps({"image": b64, "threshold": 0.5}))

        resp = mod.handler(ctx, event)
        assert resp.status_code == 200
        shapes = json.loads(resp.body)
        assert len(shapes) == 2
        assert shapes[0]["type"] == "polygon"
        assert shapes[1]["type"] == "mask"
        assert shapes[0]["group_id"] == shapes[1]["group_id"] == 1
        # Bounding box must NOT be present
        for s in shapes:
            assert s["type"] != "rectangle"

    def test_polyline_handler_output(self):
        mod = load_module_from_path("main_polyline", POLYLINE_DIR / "main.py")
        ctx = MockContext()

        mock_shapes = [
            {"type": "polyline", "label": "lane/single white", "points": [10.0, 20.0, 50.0, 80.0], "confidence": "0.92"},
            {"type": "polyline", "label": "lane/crosswalk", "points": [5.0, 10.0, 45.0, 90.0], "confidence": "0.90"},
        ]

        mock_handler = MagicMock()
        mock_handler.infer.return_value = mock_shapes
        ctx.user_data.model_handler = mock_handler

        b64 = create_dummy_jpeg_base64()
        event = MockEvent(body=json.dumps({"image": b64, "threshold": 0.5}))

        resp = mod.handler(ctx, event)
        assert resp.status_code == 200
        shapes = json.loads(resp.body)
        assert len(shapes) == 2
        for s in shapes:
            assert s["type"] == "polyline"
            assert "group_id" not in s
            assert s["type"] not in ("rectangle", "polygon", "mask")

    def test_error_handling_missing_image(self):
        for dir_path in (RECT_MASK_DIR, POLY_MASK_DIR, POLYLINE_DIR):
            mod = load_module_from_path("main_err", dir_path / "main.py")
            ctx = MockContext()
            event = MockEvent(body=json.dumps({"threshold": 0.5}))
            resp = mod.handler(ctx, event)
            assert resp.status_code == 400
            assert "Missing mandatory 'image' field" in json.loads(resp.body)["error"]

    def test_secret_masking_in_error_responses(self):
        for dir_path in (RECT_MASK_DIR, POLY_MASK_DIR, POLYLINE_DIR):
            mod = load_module_from_path("main_sec", dir_path / "main.py")
            ctx = MockContext()
            mock_h = MagicMock()
            mock_h.api_key = "super_secret_key_9router_999"
            mock_h.infer.side_effect = RuntimeError("Failed connecting with super_secret_key_9router_999")
            ctx.user_data.model_handler = mock_h

            b64 = create_dummy_jpeg_base64()
            event = MockEvent(body=json.dumps({"image": b64}))
            resp = mod.handler(ctx, event)
            assert resp.status_code == 500
            err = json.loads(resp.body)["error"]
            assert "super_secret_key_9router_999" not in err
            assert "***" in err
