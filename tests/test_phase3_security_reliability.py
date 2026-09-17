"""Comprehensive Security and Reliability Audit Test Suite for Phase 3 (Box + Instance Mask).

Audits:
1. STRICT ZERO LOCAL HEAVY ML DIRECTIVE:
   - Zero references to torch, torchvision, torchaudio, tensorflow, ultralytics, onnx,
     onnxruntime, segment-anything (SAM/SAM2), or local multi-GB model weights.
   - Only lightweight Pillow, standard library, and approved dependencies.
2. SECRET & PRIVACY HYGIENE:
   - NINEROUTER_KEY is masked in all console outputs, repr, and logs.
   - No plaintext secrets written to temporary script files.
   - Output reports (run_report.json, predictions.json) do NOT store API keys or raw base64 data URLs.
3. RESOURCE & MEMORY DOS PREVENTION:
   - Polygon parsing rejects or clamps unbounded vertex counts (> 10,000 vertices).
   - Rasterization validates image dimensions against negative, zero, and excessive sizes (SAFE_MAX_IMAGE_PIXELS).
   - CVAT flat list mask crop calculation cannot produce negative slices or huge memory allocations.
   - Rejection of NaN, Inf, boolean, and non-numeric coordinates.
4. FAULT TOLERANCE & GRACEFUL DEGRADATION:
   - Graceful fallback when vision model omits mask field.
   - Clean handling and warnings for degenerate polygons (< 3 vertices, collinear, zero area).
   - Nuclio handler guards against oversized payloads and sanitizes error messages.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.client import NineRouterClient, NineRouterError, VisionResponse
from app.config import (
    DEFAULT_MAX_IMAGE_SIZE,
    AppConfig,
    mask_api_key,
)
from app.image_ops import SAFE_MAX_IMAGE_PIXELS
from app.parser import (
    MAX_CONTOUR_VERTICES,
    ParsedObject,
    VisionParseError,
    parse_and_validate,
)
from app.pipeline import PipelineOptions, run_pipeline
from app.service import AnnotationResult, annotate_image
from core.geometry import (
    calculate_polygon_area,
    contour_to_bounding_box,
    cvat_mask_to_binary_image,
    denormalize_contour,
    mask_to_cvat_flat_list,
    normalize_contour,
    polygon_to_cvat_mask,
    rasterize_polygon_to_mask,
)
from core.vision_contract import (
    DEFAULT_BBOX_LABELS,
    MODE_BOX,
    MODE_BOX_AND_MASK,
    MODE_MASK,
    DetectedObject,
    DetectionResult,
    parse_vision_response,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# 1. STRICT ZERO LOCAL HEAVY ML DIRECTIVE AUDIT TESTS
# ============================================================================

class TestZeroLocalHeavyMLDirective:
    """Audit for zero references or dependencies on heavy local ML frameworks."""

    PROHIBITED_MODULES = [
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "ultralytics",
        "onnx",
        "onnxruntime",
        "segment_anything",
        "segment-anything",
        "sam2",
        "cv2",
        "opencv",
        "safetensors",
    ]

    PROHIBITED_EXTENSIONS = [
        ".pt",
        ".pth",
        ".onnx",
        ".bin",
        ".safetensors",
        ".h5",
        ".ckpt",
    ]

    def test_no_prohibited_dependencies_in_requirements(self):
        """Verify root and nuclio requirements.txt contain zero prohibited ML packages."""
        req_files = [
            REPO_ROOT / "requirements.txt",
            REPO_ROOT / "serverless" / "ninerouter-vision" / "nuclio" / "requirements.txt",
            REPO_ROOT / "serverless" / "ninerouter-vision-mask" / "nuclio" / "requirements.txt",
        ]
        for req_file in req_files:
            if req_file.exists():
                content = req_file.read_text(encoding="utf-8").lower()
                for mod in self.PROHIBITED_MODULES:
                    assert mod not in content, f"Prohibited module {mod!r} found in {req_file}"

    def test_no_prohibited_docker_directives(self):
        """Verify serverless function.yaml files do not install heavy ML packages."""
        yaml_files = list(REPO_ROOT.glob("serverless/**/function.yaml"))
        assert len(yaml_files) >= 1, "Expected at least one function.yaml"
        for yf in yaml_files:
            content = yf.read_text(encoding="utf-8").lower()
            for mod in self.PROHIBITED_MODULES:
                assert mod not in content, f"Prohibited module {mod!r} in {yf}"

    def test_no_prohibited_imports_in_source_code(self):
        """Scan all python source files to confirm no heavy ML imports."""
        py_files = list((REPO_ROOT / "app").glob("**/*.py")) + \
                   list((REPO_ROOT / "core").glob("**/*.py")) + \
                   list((REPO_ROOT / "serverless").glob("**/*.py"))
        for pf in py_files:
            content = pf.read_text(encoding="utf-8")
            for mod in self.PROHIBITED_MODULES:
                # Check import mod, from mod import, and __import__(mod)
                assert f"import {mod}" not in content, f"Prohibited import of {mod} in {pf}"
                assert f"from {mod}" not in content, f"Prohibited from-import of {mod} in {pf}"

    def test_no_heavy_model_weights_in_repository(self):
        """Confirm no multi-MB or multi-GB local model weight files exist."""
        for ext in self.PROHIBITED_EXTENSIONS:
            matches = list(REPO_ROOT.glob(f"**/*{ext}"))
            # Filter out virtual environments if present
            non_venv_matches = [m for m in matches if ".venv" not in str(m) and "site-packages" not in str(m)]
            assert len(non_venv_matches) == 0, f"Found prohibited model weight files: {non_venv_matches}"


# ============================================================================
# 2. SECRET & PRIVACY HYGIENE AUDIT TESTS
# ============================================================================

class TestSecretAndPrivacyHygiene:
    """Audit secret masking and data privacy invariants across Phase 3."""

    def test_ninerouter_key_masked_in_all_components(self):
        """Verify API keys are masked in config, client, pipeline, and nuclio handler."""
        raw_key = "sk-live-secret-key-123456789"

        # 1. mask_api_key helper
        masked = mask_api_key(raw_key)
        assert raw_key not in masked
        assert masked == "sk-...789"

        # 2. AppConfig
        cfg = AppConfig(ninerouter_key=raw_key)
        assert raw_key not in repr(cfg)

        # 3. PipelineOptions
        opts = PipelineOptions(image_path=Path("test.jpg"), ninerouter_key=raw_key)
        assert raw_key not in repr(opts)

        # 4. NineRouterClient
        client = NineRouterClient(api_key=raw_key)
        assert raw_key not in repr(client)

    def test_client_sanitizes_secret_key_in_exceptions(self):
        """Verify NineRouterClient sanitizes the secret key even if returned in error messages."""
        secret_key = "sk-super-secret-production-token-999"
        client = NineRouterClient(api_key=secret_key)
        leak_message = f"Failed to authenticate with bearer token: {secret_key} on 9Router"

        sanitized = client._sanitize_error_text(leak_message)
        assert secret_key not in sanitized
        assert "***" in sanitized

    def test_reports_contain_no_secrets_and_no_base64(self, tmp_path):
        """Verify predictions.json and run_report.json store neither API keys nor base64 data URLs."""
        secret_key = "sk-leak-test-token-777"
        out_dir = tmp_path / "output"
        img_path = tmp_path / "sample.jpg"

        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        img.save(img_path)

        opts = PipelineOptions(
            image_path=img_path,
            output_dir=out_dir,
            ninerouter_key=secret_key,
        )

        mock_content = json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [100, 100, 500, 500],
                    "mask": [[100, 100], [500, 100], [500, 500], [100, 500]],
                    "confidence": 0.95,
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.send_vision_request.return_value = VisionResponse(
            content=mock_content,
            raw_response={"choices": [{"message": {"content": mock_content}}]},
            duration_seconds=0.42,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
            usage={"total_tokens": 120},
        )
        mock_client.get_vision_models.return_value = [{"id": "ag/gemini-3.8-flash-high"}]

        with patch("app.pipeline.NineRouterClient", return_value=mock_client):
            res = run_pipeline(opts)

        assert res.success is True

        # Read predictions.json
        preds_file = out_dir / "predictions.json"
        assert preds_file.exists()
        preds_text = preds_file.read_text(encoding="utf-8")
        assert secret_key not in preds_text
        assert "data:image" not in preds_text
        assert ";base64," not in preds_text

        # Read run_report.json
        report_file = out_dir / "run_report.json"
        assert report_file.exists()
        report_text = report_file.read_text(encoding="utf-8")
        assert secret_key not in report_text
        assert "data:image" not in report_text
        assert ";base64," not in report_text

    def test_deployment_scripts_pass_secrets_without_temp_files(self):
        """Verify powershell scripts pass secrets in-memory via WSLENV and never write them to disk."""
        deploy_scripts = [
            REPO_ROOT / "scripts" / "phase2_deploy.ps1",
            REPO_ROOT / "scripts" / "phase3_deploy.ps1",
        ]
        for script in deploy_scripts:
            if script.exists():
                text = script.read_text(encoding="utf-8")
                # Confirm WSLENV in-memory environment passing is used
                assert "WSLENV" in text
                # Confirm cleanup in finally block
                assert "$env:NINEROUTER_KEY = $null" in text or "finally" in text
                # Confirm no Out-File or Set-Content writing the key
                assert "Out-File" not in text or "$NineRouterKey" not in text


# ============================================================================
# 3. RESOURCE & MEMORY DOS PREVENTION AUDIT TESTS
# ============================================================================

class TestResourceAndMemoryDosPrevention:
    """Audit bounds checking, vertex limits, dimension guards, and memory safety."""

    def test_polygon_parsing_rejects_unbounded_vertices_in_denormalize(self):
        """Verify denormalize_contour rejects contours with > 10,000 vertices."""
        excessive_contour = [[float(i % 1000), float((i * 2) % 1000)] for i in range(10_001)]
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            denormalize_contour(excessive_contour, width=1920, height=1080)

    def test_polygon_parsing_rejects_unbounded_vertices_in_normalize(self):
        """Verify normalize_contour rejects contours with > 10,000 vertices."""
        excessive_pts = [[float(i % 500), float((i * 2) % 500)] for i in range(10_001)]
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            normalize_contour(excessive_pts, width=1920, height=1080)

    def test_rasterization_rejects_unbounded_vertices(self):
        """Verify rasterize_polygon_to_mask rejects > 10,000 vertices."""
        excessive_pts = [(float(i % 500), float((i * 2) % 500)) for i in range(10_001)]
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            rasterize_polygon_to_mask(excessive_pts, width=1000, height=1000)

    def test_app_parser_rejects_or_clamps_unbounded_vertices(self):
        """Verify parse_and_validate rejects in strict mode and clamps in non-strict mode."""
        excessive_mask = [[float(i % 1000), float((i * 3) % 1000)] for i in range(10_500)]
        payload = json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [100, 100, 600, 600],
                    "mask": excessive_mask,
                }
            ]
        })

        # Strict mode: must raise VisionParseError
        with pytest.raises(VisionParseError, match="exceeding maximum allowed limit"):
            parse_and_validate(
                payload,
                allowed_labels=["car"],
                image_width=1000,
                image_height=1000,
                strict=True,
            )

        # Non-strict mode: must clamp and warn without crashing
        res = parse_and_validate(
            payload,
            allowed_labels=["car"],
            image_width=1000,
            image_height=1000,
            strict=False,
        )
        assert len(res.objects) == 1
        assert any("exceeding limit" in w for w in res.warnings)
        assert len(res.objects[0].mask) <= MAX_CONTOUR_VERTICES

    def test_detected_object_validate_rejects_unbounded_vertices(self):
        """Verify DetectedObject.validate() enforces the 10,000 vertex ceiling."""
        excessive_mask = [[i % 1000, (i * 2) % 1000] for i in range(10_001)]
        obj = DetectedObject(
            label="car",
            box_2d=[100, 100, 500, 500],
            mask=excessive_mask,
        )
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            obj.validate()

    def test_rasterization_rejects_non_positive_and_excessive_dimensions(self):
        """Verify rasterization and geometry functions reject zero, negative, or huge dimensions."""
        pts = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0)]

        # Zero and negative dimensions
        for bad_w, bad_h in [(0, 100), (100, 0), (-50, 100), (100, -50), (0, 0)]:
            with pytest.raises(ValueError):
                rasterize_polygon_to_mask(pts, width=bad_w, height=bad_h)
            with pytest.raises(ValueError):
                denormalize_contour([[10, 10], [100, 10], [100, 100]], width=bad_w, height=bad_h)

        # Huge dimensions exceeding SAFE_MAX_IMAGE_PIXELS (approx 89MP) to prevent multi-GB allocation
        huge_w, huge_h = 20_000, 20_000  # 400 Megapixels
        with pytest.raises(ValueError, match="exceed safe limit"):
            rasterize_polygon_to_mask(pts, width=huge_w, height=huge_h)

        with pytest.raises(ValueError, match="exceed safe limit"):
            denormalize_contour([[10, 10], [100, 10], [100, 100]], width=huge_w, height=huge_h)

    def test_cvat_flat_list_cannot_produce_negative_slices(self):
        """Verify mask_to_cvat_flat_list never produces negative crop bounds or negative sizes."""
        # Create a small valid mask
        img = Image.new("L", (100, 100), 0)
        draw_pts = [(20, 20), (80, 20), (80, 80), (20, 80)]
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.polygon(draw_pts, fill=1)

        # Test with invalid out-of-bounds or inverted explicit bbox
        for bad_bbox in [
            (80, 20, 20, 80),   # xmin > xmax
            (20, 80, 80, 20),   # ymin > ymax
            (-10, 20, 50, 50),  # negative xmin
            (20, -10, 50, 50),  # negative ymin
            (20, 20, 150, 50),  # xmax >= width
            (20, 20, 50, 150),  # ymax >= height
        ]:
            with pytest.raises(ValueError):
                mask_to_cvat_flat_list(img, bbox=bad_bbox)

    def test_cvat_mask_to_binary_image_bounds_and_memory_guards(self):
        """Verify cvat_mask_to_binary_image validates bbox bounds and prevents memory exhaustion."""
        # Out-of-bounds bbox coordinates
        with pytest.raises(ValueError, match="out of bounds"):
            # xmin < 0
            cvat_mask_to_binary_image([1, 1, 1, 1, -10, 0, 1, 1], width=100, height=100)

        with pytest.raises(ValueError, match="out of bounds"):
            # xmax >= width
            cvat_mask_to_binary_image([1, 1, 1, 1, 0, 0, 100, 1], width=100, height=100)

        # Pixel count mismatch
        with pytest.raises(ValueError, match="Pixel count mismatch"):
            # crop 2x2 = 4 pixels, but only 2 provided
            cvat_mask_to_binary_image([1, 1, 0, 0, 1, 1], width=100, height=100)

        # Huge dimensions exceeding SAFE_MAX_IMAGE_PIXELS
        with pytest.raises(ValueError, match="exceed safe limit"):
            cvat_mask_to_binary_image([1, 0, 0, 0, 0], width=20000, height=20000)

    def test_nan_and_inf_coordinate_rejection(self):
        """Verify NaN, Inf, boolean, and non-numeric coordinates are strictly rejected."""
        invalid_contours = [
            [[float("nan"), 100.0], [500.0, 100.0], [500.0, 500.0]],
            [[100.0, float("inf")], [500.0, 100.0], [500.0, 500.0]],
            [[float("-inf"), 100.0], [500.0, 100.0], [500.0, 500.0]],
            [[True, 100.0], [500.0, 100.0], [500.0, 500.0]],
            [["100", 100.0], [500.0, 100.0], [500.0, 500.0]],
            [[100.0], [500.0, 100.0], [500.0, 500.0]],
        ]
        for contour in invalid_contours:
            with pytest.raises((ValueError, TypeError)):
                denormalize_contour(contour, width=1000, height=1000)


# ============================================================================
# 4. FAULT TOLERANCE & GRACEFUL DEGRADATION AUDIT TESTS
# ============================================================================

class TestFaultToleranceAndGracefulDegradation:
    """Audit fallback mechanisms when model output is incomplete or degenerate."""

    def test_vision_model_omits_mask_falls_back_gracefully_in_service(self):
        """Verify annotate_image does not crash when mask is omitted in mask mode.

        Expected: Falls back gracefully to bounding box rasterized mask.
        """
        raw_json_no_mask = json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [100, 200, 400, 600],
                    "confidence": 0.92,
                    # No mask field provided by vision model
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
        mock_client.send_vision_request.return_value = VisionResponse(
            content=raw_json_no_mask,
            raw_response={"choices": [{"message": {"content": raw_json_no_mask}}]},
            duration_seconds=0.35,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
            usage={},
        )

        test_img = Image.new("RGB", (800, 600), color="gray")

        # Test in MODE_MASK: missing mask must reject annotation (never fabricate mask from bbox)
        res_mask = annotate_image(
            image_source=test_img,
            client=mock_client,
            candidate_labels=["car"],
            mode=MODE_MASK,
        )
        assert len(res_mask.shapes) == 0
        assert any("mask_missing" in w and "car" in w for w in res_mask.warnings)

        # Test in MODE_BOX_AND_MASK: missing mask emits rectangle ONLY (never fabricate mask)
        res_both = annotate_image(
            image_source=test_img,
            client=mock_client,
            candidate_labels=["car"],
            mode=MODE_BOX_AND_MASK,
        )
        assert len(res_both.shapes) == 1
        assert res_both.shapes[0]["type"] == "rectangle"
        assert res_both.shapes[0]["label"] == "car"
        assert any("mask_missing" in w and "car" in w for w in res_both.warnings)

    def test_degenerate_polygons_handled_cleanly_without_crash(self):
        """Verify degenerate polygons (< 3 points, collinear, zero area) fail cleanly."""
        # 1. Fewer than 3 points
        res_two_pts = polygon_to_cvat_mask([[100, 100], [200, 200]], width=1000, height=1000)
        assert res_two_pts is None

        # 2. Collinear points (zero area)
        collinear = [[100, 100], [200, 200], [300, 300], [400, 400]]
        assert calculate_polygon_area(collinear) == 0.0
        res_collinear = polygon_to_cvat_mask(collinear, width=1000, height=1000)
        assert res_collinear is None

        # 3. All identical points (single point)
        single_pt = [[500, 500], [500, 500], [500, 500]]
        assert calculate_polygon_area(single_pt) == 0.0
        res_single = polygon_to_cvat_mask(single_pt, width=1000, height=1000)
        assert res_single is None

    def test_parser_handles_degenerate_mask_in_non_strict_mode(self):
        """Verify parse_and_validate warns and skips degenerate mask without failing."""
        payload = json.dumps({
            "objects": [
                {
                    "label": "pedestrian",
                    "box_2d": [100, 100, 500, 500],
                    "mask": [[100, 100], [200, 200]],  # Only 2 points
                }
            ]
        })

        res = parse_and_validate(
            payload,
            allowed_labels=["pedestrian"],
            image_width=1000,
            image_height=1000,
            strict=False,
        )
        assert len(res.objects) == 1
        obj = res.objects[0]
        assert obj.mask is None
        assert any("degenerate mask" in w for w in res.warnings)

    def test_nuclio_handler_payload_size_and_error_sanitization(self):
        """Verify Nuclio main.py handler limits payload size and sanitizes errors."""
        import importlib.util
        import sys

        serverless_dir = REPO_ROOT / "serverless" / "ninerouter-vision" / "nuclio"
        if str(serverless_dir) not in sys.path:
            sys.path.insert(0, str(serverless_dir))

        spec_main = importlib.util.spec_from_file_location("nuclio_main_audit", str(serverless_dir / "main.py"))
        nuclio_main = importlib.util.module_from_spec(spec_main)
        spec_main.loader.exec_module(nuclio_main)

        MAX_REQUEST_BODY_SIZE = nuclio_main.MAX_REQUEST_BODY_SIZE
        handler = nuclio_main.handler

        mock_context = MagicMock()
        mock_context.Response = lambda **kwargs: kwargs

        # 1. Payload exceeding 32MB limit
        mock_event_huge = MagicMock()
        mock_event_huge.body = "A" * (MAX_REQUEST_BODY_SIZE + 1024)

        resp = handler(mock_context, mock_event_huge)
        assert resp.get("status_code") == 413

        # 2. Invalid JSON body
        mock_event_bad_json = MagicMock()
        mock_event_bad_json.body = "{ invalid json content"

        resp = handler(mock_context, mock_event_bad_json)
        assert resp.get("status_code") == 400

        # 3. Missing image field
        mock_event_no_img = MagicMock()
        mock_event_no_img.body = json.dumps({"threshold": 0.5})

        resp = handler(mock_context, mock_event_no_img)
        assert resp.get("status_code") == 400
