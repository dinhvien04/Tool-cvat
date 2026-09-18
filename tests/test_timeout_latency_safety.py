"""Comprehensive Automated Tests for Timeout Handling, Latency Safety, and Policy Integrity.

Directly verifies:
1. Model resolution is cached and does not hit /v1/models on every request.
2. Capability probe is never executed during normal annotation.
3. Feedback SQLite timeout/lock contention does not block inference past a small threshold
   (fail-open auxiliary behavior).
4. 9Router timeout is caught cleanly and returns a structured JSON error instead of crashing.
5. Connection errors return clean JSON.
6. Latency instrumentation logs safe metrics without leaking Base64, API keys, or secrets.
7. The 3 detectors maintain their exact 14/10/7 label splits and shape emission policies.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest
import requests
from PIL import Image
import yaml

from app.client import (
    _FULL_31_CAPABILITY_CACHE,
    _SEGMENTATION_CAPABILITY_CACHE,
    NineRouterAuthError,
    NineRouterClient,
    NineRouterConnectionError,
    NineRouterError,
    NineRouterRequestError,
    VisionResponse,
)
from app.config import (
    AppConfig,
    DEFAULT_NINEROUTER_TIMEOUT,
    mask_api_key,
)
from app.feedback import (
    FeedbackDatabase,
    compute_image_hash,
)
from app.pipeline import PipelineOptions
from app.service import AnnotationResult, annotate_image
from core.taxonomy import (
    BOX_MASK_LABELS,
    MASTER_31_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    Taxonomy,
    validate_cvat_output_shapes,
)
from core.vision_contract import (
    MODE_BOX,
    MODE_BOX_AND_MASK,
    MODE_FULL_31,
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVERLESS_DIR = REPO_ROOT / "serverless"
RECT_MASK_DIR = SERVERLESS_DIR / "ninerouter-rectangle-mask" / "nuclio"
POLY_MASK_DIR = SERVERLESS_DIR / "ninerouter-polygon-mask" / "nuclio"
POLYLINE_DIR = SERVERLESS_DIR / "ninerouter-polyline" / "nuclio"


# ==============================================================================
# Helper Classes & Utilities
# ==============================================================================

class MockNuclioResponse:
    def __init__(self, body: str, headers: Dict[str, str] = None, content_type: str = "application/json", status_code: int = 200):
        self.body = body
        self.headers = headers or {}
        self.content_type = content_type
        self.status_code = status_code


class MockNuclioContext:
    def __init__(self):
        self.logger = MagicMock()
        self.user_data = SimpleNamespace()
        self.Response = MockNuclioResponse


class MockNuclioEvent:
    def __init__(self, body: Any, headers: Dict[str, str] = None):
        self.body = body
        self.headers = headers or {}


def create_test_image_bytes(width: int = 200, height: int = 200) -> bytes:
    """Generate in-memory valid JPEG bytes."""
    im = Image.new("RGB", (width, height), color=(60, 120, 180))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def create_test_image_base64(width: int = 200, height: int = 200) -> str:
    """Generate Base64 encoded test image."""
    raw = create_test_image_bytes(width, height)
    return base64.b64encode(raw).decode("utf-8")


def load_detector_main(module_name: str, file_path: Path):
    """Dynamically load detector main.py module."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_detector_model_handler(module_name: str, file_path: Path):
    """Dynamically load detector model_handler.py module."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==============================================================================
# 1. Model Resolution Caching Tests
# ==============================================================================

class TestModelResolutionCaching:
    """Verify model resolution is cached and does NOT hit /v1/models on every request."""

    def test_model_handler_resolves_model_once_and_caches_for_all_subsequent_inferences(self):
        """ModelHandler resolves model at startup and reuses active_model for subsequent inferences."""
        mod = load_detector_model_handler("mh_rect_caching", RECT_MASK_DIR / "model_handler.py")

        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "data": [
                {"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}},
            ]
        }

        vision_completion_response = VisionResponse(
            content=json.dumps({
                "objects": [{
                    "label": "car",
                    "box_2d": [100, 100, 400, 400],
                    "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                }]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.35,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch("requests.Session.get", return_value=models_response) as mock_get, \
             patch("requests.Session.post") as mock_post:

            mock_post_resp = MagicMock()
            mock_post_resp.status_code = 200
            mock_post_resp.headers = {"content-type": "application/json"}
            mock_post_resp.json.return_value = {
                "model": "ag/gemini-3.8-flash-high",
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "objects": [{
                                "label": "car",
                                "confidence": 0.95,
                                "box_2d": [100, 100, 400, 400],
                                "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                            }]
                        })
                    }
                }]
            }
            mock_post.return_value = mock_post_resp

            # Step 1: Initialize ModelHandler -> triggers resolution once
            handler = mod.ModelHandler(base_url="http://127.0.0.1:20128", model="ag/gemini-3.8-flash-high")
            assert handler.active_model == "ag/gemini-3.8-flash-high"
            initial_get_calls = mock_get.call_count

            # Step 2: Run 5 sequential inference requests
            img_bytes = create_test_image_bytes()
            for _ in range(5):
                shapes = handler.infer(img_bytes)
                assert len(shapes) == 2  # rectangle + mask

            # Step 3: Verify GET /v1/models was NOT called during any of the infer() calls
            assert mock_get.call_count == initial_get_calls, (
                f"/v1/models was hit during inference! Expected {initial_get_calls} calls, got {mock_get.call_count}"
            )

    def test_annotate_image_with_specified_model_skips_discovery_endpoint(self):
        """annotate_image with explicit model never queries GET /v1/models."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        img_bytes = create_test_image_bytes()

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [{
                    "label": "car",
                    "box_2d": [100, 100, 400, 400],
                    "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                }]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.2,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client.session, "get") as mock_get, \
             patch.object(client, "send_vision_request", return_value=fake_resp) as mock_send:

            # Explicit model supplied
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_RECTANGLE_MASK,
                enable_feedback=False,
            )

            assert res.model_used == "ag/gemini-3.8-flash-high"
            assert mock_get.call_count == 0, "GET /v1/models must not be called when model is specified"
            assert mock_send.call_count == 1

    def test_segmentation_capability_cache_in_memory_avoids_re_probing(self):
        """Capability probing result is stored in memory and avoids duplicate probing."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        model_name = "ag/cached-test-model-seg"

        # Pre-seed capability cache
        _SEGMENTATION_CAPABILITY_CACHE[model_name] = True
        try:
            with patch.object(client, "send_vision_request") as mock_send, \
                 patch.object(client, "get_vision_model_ids") as mock_ids:

                is_capable, details = client.probe_segmentation_capability(model_name)
                assert is_capable is True
                assert "Cached segmentation capability" in details
                assert mock_send.call_count == 0
                assert mock_ids.call_count == 0
        finally:
            _SEGMENTATION_CAPABILITY_CACHE.pop(model_name, None)

    def test_full_31_capability_cache_in_memory_avoids_re_probing(self):
        """Full 31 capability probing is cached per (model, version) tuple."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        model_name = "ag/cached-test-model-31"
        cache_key = (model_name, "v1.0")

        _FULL_31_CAPABILITY_CACHE[cache_key] = True
        try:
            with patch.object(client, "send_vision_request") as mock_send, \
                 patch.object(client, "get_vision_model_ids") as mock_ids:

                is_capable, msg = client.probe_full_31_capability(model_name, prompt_version="v1.0")
                assert is_capable is True
                assert "Cached 3-policy capability" in msg
                assert mock_send.call_count == 0
                assert mock_ids.call_count == 0
        finally:
            _FULL_31_CAPABILITY_CACHE.pop(cache_key, None)


# ==============================================================================
# 2. Capability Probe Never Executed During Normal Annotation Tests
# ==============================================================================

class TestCapabilityProbeNeverExecutedDuringNormalAnnotation:
    """Verify capability probing is NEVER executed during standard image annotation."""

    @pytest.mark.parametrize("mode", [
        MODE_BOX,
        MODE_RECTANGLE_MASK,
        MODE_POLYGON_MASK,
        MODE_POLYLINE,
        MODE_FULL_31,
    ])
    def test_annotate_image_never_invokes_probe_methods(self, mode: str):
        """annotate_image across all modes must not call probe_segmentation_capability or probe_full_31_capability."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        img_bytes = create_test_image_bytes()

        fake_resp = VisionResponse(
            content=json.dumps({"objects": [], "regions": [], "lanes": []}),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "probe_segmentation_capability") as mock_probe_seg, \
             patch.object(client, "probe_full_31_capability") as mock_probe_31, \
             patch.object(client, "send_vision_request", return_value=fake_resp):

            annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=mode,
                enable_feedback=False,
            )

            mock_probe_seg.assert_not_called()
            mock_probe_31.assert_not_called()

    def test_all_three_nuclio_detectors_never_probe_during_inference(self):
        """Nuclio infer() on all 3 detectors must not execute capability probes."""
        detector_dirs = [
            ("mh_rect", RECT_MASK_DIR / "model_handler.py"),
            ("mh_poly", POLY_MASK_DIR / "model_handler.py"),
            ("mh_line", POLYLINE_DIR / "model_handler.py"),
        ]

        img_bytes = create_test_image_bytes()

        for mod_name, file_path in detector_dirs:
            mod = load_detector_model_handler(mod_name, file_path)

            with patch("requests.Session.get") as mock_get, \
                 patch("requests.Session.post") as mock_post:

                mock_get_resp = MagicMock()
                mock_get_resp.status_code = 200
                mock_get_resp.json.return_value = {
                    "data": [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}]
                }
                mock_get.return_value = mock_get_resp

                mock_post_resp = MagicMock()
                mock_post_resp.status_code = 200
                mock_post_resp.headers = {"content-type": "application/json"}
                mock_post_resp.json.return_value = {
                    "model": "ag/gemini-3.8-flash-high",
                    "choices": [{"message": {"content": json.dumps({"objects": [], "regions": [], "lanes": []})}}],
                }
                mock_post.return_value = mock_post_resp

                handler = mod.ModelHandler(base_url="http://127.0.0.1:20128", model="ag/gemini-3.8-flash-high")

                with patch.object(handler.client, "probe_segmentation_capability") as spy_seg, \
                     patch.object(handler.client, "probe_full_31_capability") as spy_31:

                    handler.infer(img_bytes)

                    spy_seg.assert_not_called()
                    spy_31.assert_not_called()

    def test_resolve_segmentation_model_defaults_probe_to_false(self):
        """resolve_segmentation_model defaults probe=False and does not call probe_segmentation_capability."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")

        with patch.object(client, "get_vision_model_ids", return_value=["ag/gemini-3.8-flash-high"]), \
             patch.object(client, "probe_segmentation_capability") as mock_probe:

            # Calling without probe parameter defaults probe=False
            resolved = client.resolve_segmentation_model()
            assert resolved == "ag/gemini-3.8-flash-high"
            mock_probe.assert_not_called()


# ==============================================================================
# 3. Feedback SQLite Timeout & Lock Contention (Fail-Open) Tests
# ==============================================================================

class TestFeedbackSQLiteTimeoutAndLockContentionFailOpen:
    """Verify SQLite contention or timeout never blocks or crashes annotation inference."""

    def test_feedback_sqlite_lock_contention_retrieval_fails_open(self):
        """If feedback SQLite retrieval raises database is locked, inference succeeds with warning."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        img_bytes = create_test_image_bytes()

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [{
                    "label": "car",
                    "box_2d": [50, 50, 200, 200],
                    "mask": [[50, 50], [200, 50], [200, 200], [50, 200]],
                }]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.25,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp), \
             patch("app.retrieval.CorrectionRetrievalEngine.retrieve", side_effect=sqlite3.OperationalError("database is locked")):

            # Run annotate_image with feedback enabled
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_RECTANGLE_MASK,
                enable_feedback=True,
            )

            # Inference must NOT crash and must return valid shapes
            assert len(res.shapes) == 2  # paired rectangle + mask
            assert any("feedback_retrieval_warning" in w for w in res.warnings)
            assert any("database is locked" in w for w in res.warnings)

    def test_feedback_sqlite_lock_contention_baseline_save_fails_open(self):
        """If baseline persistence encounters database locked error, shapes are still returned."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        img_bytes = create_test_image_bytes()

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [{
                    "label": "bus",
                    "box_2d": [20, 20, 180, 180],
                    "mask": [[20, 20], [180, 20], [180, 180], [20, 180]],
                }]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.2,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp), \
             patch("app.feedback.FeedbackDatabase.save_prediction_baseline", side_effect=sqlite3.OperationalError("database is locked")):

            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_RECTANGLE_MASK,
                enable_feedback=True,
            )

            assert len(res.shapes) == 2
            assert any("feedback_baseline_warning" in w for w in res.warnings)
            assert any("database is locked" in w for w in res.warnings)

    def test_feedback_db_init_failure_fails_open_with_warning(self):
        """If FeedbackDatabase fails to initialize, inference still succeeds cleanly."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        img_bytes = create_test_image_bytes()

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [{
                    "label": "pedestrian",
                    "box_2d": [30, 30, 150, 150],
                    "mask": [[30, 30], [150, 30], [150, 150], [30, 150]],
                }]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.15,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp), \
             patch("app.feedback.FeedbackDatabase", side_effect=sqlite3.OperationalError("unable to open database file")):

            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_RECTANGLE_MASK,
                enable_feedback=True,
            )

            assert len(res.shapes) == 2
            assert any("feedback_init_warning" in w for w in res.warnings)

    def test_feedback_db_configurable_timeout_and_busy_pragma(self, tmp_path):
        """FeedbackDatabase supports configurable timeout parameter and FEEDBACK_DB_TIMEOUT env."""
        db_file = tmp_path / "test_timeout.sqlite3"
        db = FeedbackDatabase(db_path=db_file, timeout=1.5)
        assert db.timeout == 1.5

        with patch.dict(os.environ, {"FEEDBACK_DB_TIMEOUT": "2.5"}):
            db2 = FeedbackDatabase(db_path=tmp_path / "test_env_timeout.sqlite3")
            assert db2.timeout == 2.5


# ==============================================================================
# 4. 9Router Timeout Structured JSON Error Tests
# ==============================================================================

class Test9RouterTimeoutStructuredJsonError:
    """Verify 9Router timeout is caught cleanly and returns structured JSON errors."""

    def test_client_send_vision_request_timeout_raises_nine_router_request_error(self):
        """requests.exceptions.Timeout produces NineRouterRequestError with status 408."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128", timeout=5.0)

        with patch.object(client.session, "post", side_effect=requests.exceptions.Timeout("Read timed out")):
            with pytest.raises(NineRouterRequestError) as exc_info:
                client.send_vision_request(
                    model="ag/gemini-3.8-flash-high",
                    image_bytes_or_b64=create_test_image_bytes(),
                    timeout=5.0,
                )

            assert exc_info.value.status_code == 408
            assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.parametrize("mod_name, file_path", [
        ("main_rect_timeout", RECT_MASK_DIR / "main.py"),
        ("main_poly_timeout", POLY_MASK_DIR / "main.py"),
        ("main_line_timeout", POLYLINE_DIR / "main.py"),
    ])
    def test_nuclio_detectors_catch_client_timeout_and_return_clean_json(self, mod_name: str, file_path: Path):
        """All 3 Nuclio detectors return structured JSON error on timeout without crashing."""
        mod = load_detector_main(mod_name, file_path)
        ctx = MockNuclioContext()

        mock_handler = MagicMock()
        mock_handler.infer.side_effect = NineRouterRequestError(
            "Request to 9Router timed out after 60.0s", status_code=408
        )
        ctx.user_data.model_handler = mock_handler

        b64_img = create_test_image_base64()
        event = MockNuclioEvent(body=json.dumps({"image": b64_img, "threshold": 0.5}))

        resp = mod.handler(ctx, event)

        assert resp.status_code == 500
        assert resp.content_type == "application/json"
        body_dict = json.loads(resp.body)
        assert isinstance(body_dict, dict)
        assert "error" in body_dict
        assert "timed out" in body_dict["error"].lower()

    @pytest.mark.parametrize("mod_name, file_path", [
        ("main_rect_req_timeout", RECT_MASK_DIR / "main.py"),
        ("main_poly_req_timeout", POLY_MASK_DIR / "main.py"),
        ("main_line_req_timeout", POLYLINE_DIR / "main.py"),
    ])
    def test_nuclio_detectors_catch_raw_requests_timeout_directly(self, mod_name: str, file_path: Path):
        """Nuclio handlers catch unhandled requests.exceptions.Timeout and return JSON."""
        mod = load_detector_main(mod_name, file_path)
        ctx = MockNuclioContext()

        mock_handler = MagicMock()
        mock_handler.infer.side_effect = requests.exceptions.Timeout("HTTP connection timed out")
        ctx.user_data.model_handler = mock_handler

        b64_img = create_test_image_base64()
        event = MockNuclioEvent(body=json.dumps({"image": b64_img, "threshold": 0.5}))

        resp = mod.handler(ctx, event)

        assert resp.status_code == 500
        assert resp.content_type == "application/json"
        body_dict = json.loads(resp.body)
        assert "error" in body_dict
        assert "timed out" in body_dict["error"].lower()


# ==============================================================================
# 5. Connection Errors Return Clean JSON Tests
# ==============================================================================

class TestConnectionErrorsReturnCleanJson:
    """Verify connection errors are caught and return structured JSON."""

    def test_client_connection_error_raises_nine_router_connection_error(self):
        """requests.exceptions.ConnectionError raises NineRouterConnectionError."""
        client = NineRouterClient(base_url="http://127.0.0.1:99999")

        with patch.object(client.session, "post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(NineRouterConnectionError) as exc_info:
                client.send_vision_request(
                    model="ag/gemini-3.8-flash-high",
                    image_bytes_or_b64=create_test_image_bytes(),
                )

            assert "Connection error when communicating with 9Router" in str(exc_info.value)

    @pytest.mark.parametrize("mod_name, file_path", [
        ("main_rect_conn", RECT_MASK_DIR / "main.py"),
        ("main_poly_conn", POLY_MASK_DIR / "main.py"),
        ("main_line_conn", POLYLINE_DIR / "main.py"),
    ])
    def test_nuclio_detectors_catch_connection_error_and_return_clean_json(self, mod_name: str, file_path: Path):
        """Nuclio detectors return clean JSON error on connection failures."""
        mod = load_detector_main(mod_name, file_path)
        ctx = MockNuclioContext()

        mock_handler = MagicMock()
        mock_handler.infer.side_effect = NineRouterConnectionError(
            "Could not connect to 9Router at http://127.0.0.1:20128"
        )
        ctx.user_data.model_handler = mock_handler

        b64_img = create_test_image_base64()
        event = MockNuclioEvent(body=json.dumps({"image": b64_img}))

        resp = mod.handler(ctx, event)

        assert resp.status_code == 500
        assert resp.content_type == "application/json"
        body_dict = json.loads(resp.body)
        assert "error" in body_dict
        assert "Could not connect" in body_dict["error"] or "Inference failed" in body_dict["error"]


# ==============================================================================
# 6. Latency Instrumentation Safety & Privacy Tests
# ==============================================================================

class TestLatencyInstrumentationSafetyAndPrivacy:
    """Verify latency logging records metrics without leaking Base64, API keys, or secrets."""

    def test_send_vision_request_logs_latency_without_leaking_base64_or_api_key(self, caplog):
        """client.send_vision_request logs metadata and duration, never Base64 or API key."""
        secret_key = "sk-ultra-secret-test-key-777"
        client = NineRouterClient(base_url="http://127.0.0.1:20128", api_key=secret_key)

        img_b64 = create_test_image_base64(width=300, height=300)
        assert len(img_b64) > 1000

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "model": "ag/gemini-3.8-flash-high",
            "choices": [{"message": {"content": '{"objects": []}'}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }

        with caplog.at_level(logging.INFO, logger="app.client"):
            with patch.object(client.session, "post", return_value=mock_resp):
                res = client.send_vision_request(
                    model="ag/gemini-3.8-flash-high",
                    image_bytes_or_b64=img_b64,
                )

                assert res.duration_seconds >= 0.0

        # Verify logs
        all_logs = " ".join(record.getMessage() for record in caplog.records)
        assert "Sending vision request" in all_logs
        assert "target_image_url_len=" in all_logs

        # ABSOLUTE PRIVACY GUARANTEES:
        assert secret_key not in all_logs, "SECRET API KEY LEAKED IN LOGS!"
        assert img_b64[:100] not in all_logs, "RAW BASE64 IMAGE DATA LEAKED IN LOGS!"
        assert "data:image/jpeg;base64," not in all_logs, "DATA URL PREFIX LEAKED IN LOGS!"

    def test_model_handler_infer_logs_latency_safely(self, caplog):
        """ModelHandler.infer logs api_latency safely without leaking image payloads."""
        mod = load_detector_model_handler("mh_latency_safe", RECT_MASK_DIR / "model_handler.py")
        secret_key = "sk-confidential-handler-key-888"

        fake_shapes = [
            {"type": "rectangle", "label": "car", "points": [10.0, 10.0, 50.0, 50.0], "group_id": 1},
            {"type": "mask", "label": "car", "points": [10.0, 10.0, 50.0, 50.0], "mask": [0, 0, 1, 1], "group_id": 1},
        ]

        fake_res = AnnotationResult(
            shapes=fake_shapes,
            parsed_objects=[],
            raw_response='{"objects": []}',
            original_dimensions=(200, 200),
            resized_dimensions=(200, 200),
            was_resized=False,
            api_duration_seconds=0.42,
            total_duration_seconds=0.45,
            model_used="ag/gemini-3.8-flash-high",
        )

        with patch("requests.Session.get") as mock_get:
            mock_get_resp = MagicMock()
            mock_get_resp.status_code = 200
            mock_get_resp.json.return_value = {
                "data": [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}]
            }
            mock_get.return_value = mock_get_resp

            handler = mod.ModelHandler(
                base_url="http://127.0.0.1:20128",
                api_key=secret_key,
                model="ag/gemini-3.8-flash-high",
            )

        with caplog.at_level(logging.INFO, logger="cvat.nuclio.ninerouter.rectangle_mask"):
            with patch.object(mod, "annotate_image", return_value=fake_res):
                shapes = handler.infer(create_test_image_bytes())
                assert len(shapes) == 2

        all_logs = " ".join(record.getMessage() for record in caplog.records)
        assert "api_latency=0.42s" in all_logs
        assert "200x200" in all_logs
        assert secret_key not in all_logs

    def test_error_sanitization_masks_api_key(self):
        """API key echoed in error responses is masked as ***."""
        secret = "sk-super-secret-key-12345"
        client = NineRouterClient(base_url="http://127.0.0.1:20128", api_key=secret)

        leaked_msg = f"upstream error: failed authorization for bearer token {secret}"
        sanitized = client._sanitize_error_text(leaked_msg)
        assert secret not in sanitized
        assert "***" in sanitized

    def test_nuclio_error_handler_masks_api_key_in_json_response(self):
        """Nuclio handler catches exceptions and masks handler.api_key before returning JSON."""
        mod = load_detector_main("main_mask_secret", RECT_MASK_DIR / "main.py")
        ctx = MockNuclioContext()

        secret_key = "sk-sensitive-api-token-999"
        mock_handler = MagicMock()
        mock_handler.api_key = secret_key
        mock_handler.infer.side_effect = RuntimeError(f"Failed calling backend with key {secret_key}")
        ctx.user_data.model_handler = mock_handler

        b64_img = create_test_image_base64()
        event = MockNuclioEvent(body=json.dumps({"image": b64_img}))

        resp = mod.handler(ctx, event)
        assert resp.status_code == 500
        body = json.loads(resp.body)

        assert secret_key not in body["error"], "API KEY EXPOSED IN JSON ERROR BODY!"
        assert "***" in body["error"]

    def test_repr_masks_api_keys_for_all_components(self):
        """repr() across NineRouterClient, ModelHandler, PipelineOptions, and AppConfig masks secrets."""
        secret = "sk-test-secret-masking-key-12345"

        # NineRouterClient
        c = NineRouterClient(api_key=secret)
        assert secret not in repr(c)
        assert "sk-...345" in repr(c)

        # PipelineOptions
        opts = PipelineOptions(image_path=Path("dummy.jpg"), ninerouter_key=secret)
        assert secret not in repr(opts)
        assert "sk-...345" in repr(opts)

        # AppConfig
        cfg = AppConfig(ninerouter_key=secret)
        assert secret not in repr(cfg)
        assert "sk-...345" in repr(cfg)

        # ModelHandler
        mod = load_detector_model_handler("mh_repr", RECT_MASK_DIR / "model_handler.py")
        with patch("requests.Session.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "data": [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}]
            }
            h = mod.ModelHandler(api_key=secret, model="ag/gemini-3.8-flash-high")
            assert secret not in repr(h)
            assert "sk-...345" in repr(h)


# ==============================================================================
# 7. Exact 14/10/7 Label Splits & Shape Emission Policies Tests
# ==============================================================================

class TestThreeDetectorsPartitionsAndEmissionPolicies:
    """Verify the 3 detectors maintain exact 14/10/7 label splits and strict emission policies."""

    def test_three_detectors_exact_14_10_7_label_split(self):
        """Exact label split: 14 foreground, 10 semantic regions, 7 lane markings (31 total)."""
        fn_rect = yaml.safe_load(open(RECT_MASK_DIR / "function.yaml", encoding="utf-8"))
        fn_poly = yaml.safe_load(open(POLY_MASK_DIR / "function.yaml", encoding="utf-8"))
        fn_line = yaml.safe_load(open(POLYLINE_DIR / "function.yaml", encoding="utf-8"))

        spec1 = json.loads(fn_rect["metadata"]["annotations"]["spec"])
        spec2 = json.loads(fn_poly["metadata"]["annotations"]["spec"])
        spec3 = json.loads(fn_line["metadata"]["annotations"]["spec"])

        names1 = [item["name"] for item in spec1]
        names2 = [item["name"] for item in spec2]
        names3 = [item["name"] for item in spec3]

        # Cardinalities
        assert len(names1) == 14, f"Detector 1 must have exactly 14 labels, got {len(names1)}"
        assert len(names2) == 10, f"Detector 2 must have exactly 10 labels, got {len(names2)}"
        assert len(names3) == 7, f"Detector 3 must have exactly 7 labels, got {len(names3)}"

        # Set partitions match canonical taxonomy tuples
        assert set(names1) == set(BOX_MASK_LABELS)
        assert set(names2) == set(POLYGON_MASK_LABELS)
        assert set(names3) == set(POLYLINE_LABELS)

        # Mutually disjoint (zero overlap)
        set1, set2, set3 = set(names1), set(names2), set(names3)
        assert len(set1.intersection(set2)) == 0, "Overlap between Detector 1 and 2!"
        assert len(set1.intersection(set3)) == 0, "Overlap between Detector 1 and 3!"
        assert len(set2.intersection(set3)) == 0, "Overlap between Detector 2 and 3!"

        # Complete partition of all 31 master labels
        union = set1.union(set2).union(set3)
        assert len(union) == 31
        assert union == set(MASTER_31_LABELS)

    def test_detector_1_strict_policy_a_emission(self):
        """Detector 1 (Rectangle + Mask): Emits paired rectangle + mask with identical group_id."""
        client = NineRouterClient()
        img_bytes = create_test_image_bytes(width=200, height=200)

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [
                    {
                        "label": "car",
                        "box_2d": [20, 20, 100, 100],
                        "mask": [[20, 20], [100, 20], [100, 100], [20, 100]],
                    },
                    # Incomplete: missing mask -> must be dropped entirely
                    {
                        "label": "pedestrian",
                        "box_2d": [110, 110, 180, 180],
                    },
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp):
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_RECTANGLE_MASK,
                enable_feedback=False,
            )

        # Only the valid paired object is emitted
        assert len(res.shapes) == 2
        types = {s["type"] for s in res.shapes}
        assert types == {"rectangle", "mask"}
        assert res.shapes[0]["group_id"] == res.shapes[1]["group_id"]
        assert res.shapes[0]["group_id"] > 0
        assert any("missing required 'mask' contour per Policy A" in w or "policy_a_incomplete" in w for w in res.warnings)

    def test_detector_2_strict_policy_b_emission(self):
        """Detector 2 (Polygon + Mask): Emits paired polygon + mask with identical group_id, boxes strictly suppressed."""
        client = NineRouterClient()
        img_bytes = create_test_image_bytes(width=200, height=200)

        fake_resp = VisionResponse(
            content=json.dumps({
                "regions": [
                    {
                        "label": "road",
                        "polygon": [[0, 50], [200, 50], [200, 200], [0, 200]],
                        "box_2d": [0, 50, 200, 200],  # Box must be suppressed
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp):
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_POLYGON_MASK,
                enable_feedback=False,
            )

        # Emits polygon + mask
        assert len(res.shapes) == 2
        types = [s["type"] for s in res.shapes]
        assert "polygon" in types
        assert "mask" in types
        assert "rectangle" not in types  # Zero bounding boxes!
        assert res.shapes[0]["group_id"] == res.shapes[1]["group_id"]
        assert any("region_box_suppressed" in w for w in res.warnings)

    def test_detector_3_strict_policy_c_emission(self):
        """Detector 3 (Polyline): Emits polyline ONLY, with NO group_id, no boxes, polygons, or masks."""
        client = NineRouterClient()
        img_bytes = create_test_image_bytes(width=200, height=200)

        fake_resp = VisionResponse(
            content=json.dumps({
                "lanes": [
                    {
                        "label": "lane/single white",
                        "polyline": [[100, 0], [100, 200]],
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp):
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_POLYLINE,
                enable_feedback=False,
            )

        # Polyline ONLY
        assert len(res.shapes) == 1
        shape = res.shapes[0]
        assert shape["type"] == "polyline"
        assert shape["label"] == "lane/single white"
        assert "group_id" not in shape  # Lane polylines are never paired

    def test_detector_3_strictly_rejects_prohibited_box_for_lane(self):
        """Detector 3 (Polyline): Strictly drops lane detection if model outputs prohibited box_2d."""
        client = NineRouterClient()
        img_bytes = create_test_image_bytes(width=200, height=200)

        fake_resp = VisionResponse(
            content=json.dumps({
                "lanes": [
                    {
                        "label": "lane/single white",
                        "polyline": [[100, 0], [100, 200]],
                        "box_2d": [98, 0, 102, 200],  # Prohibited for Policy C
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp):
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_POLYLINE,
                enable_feedback=False,
            )

        assert len(res.shapes) == 0
        assert any("prohibited 'box_2d' for Policy C lane" in w for w in res.warnings)
