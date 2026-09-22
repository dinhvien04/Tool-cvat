"""Performance, Editability, and Stabilization Regression Test Suite.

Targeted, high-value regression verification suite (strictly <= 20 tests) enforcing:
1. Exactly one remote request per user click (no capability probes or model discovery during infer).
2. Policy-specific prompt generation: 14 labels for Detector 1, 10 labels for Detector 2,
   7 labels for Detector 3 with zero leakage of foreign categories.
3. Policy-specific max_tokens and image resolution configuration.
4. Feedback DB fail-open behavior under lock contention with sub-second timeout.
5. Visual few-shot crop limit (max 1 example, small crops <= 512px).
6. Controlled structured JSON timeout errors (no raw nginx HTML or uncaught exceptions).
7. No unbounded or multi-minute retries (fails fast on timeout/connection failure).
8. Native CVAT shape response formatting: rectangle (xtl, ytl, xbr, ybr), mask (RLE + bbox),
   polygon (points), polyline (points).
9. Paired group_id for rectangle+mask and polygon+mask; zero group_id for polyline.
10. No locked shape flags (occluded=False, outside=False, z_order=0).
11. Human correction preserved in feedback diffing (paired shapes treated as atomic annotations).
12. Exact 14/10/7 label partition (disjoint, summing to 31).
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from app.client import (
    NineRouterClient,
    NineRouterConnectionError,
    NineRouterError,
    NineRouterRequestError,
    VisionResponse,
)
from app.config import DEFAULT_MAX_IMAGE_SIZE
from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_BOX_MOVE,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_MASK_EDIT,
    CORRECTION_REGION_EDIT,
    CORRECTION_RELABEL,
    CorrectionDiffEngine,
    FeedbackDatabase,
)
from app.retrieval import CorrectionRetrievalEngine
from app.service import AnnotationResult, annotate_image, route_phase3b_shapes
from core.taxonomy import (
    BOX_MASK_LABELS,
    MASTER_31_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
    SHAPE_RECTANGLE,
    Taxonomy,
    get_taxonomy,
    validate_cvat_output_shapes,
)
from core.vision_contract import (
    MODE_BOX_MASK,
    MODE_FULL_31,
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
    build_polygon_mask_prompt,
    build_polyline_prompt,
    build_rectangle_mask_prompt,
    build_user_prompt,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVERLESS_DIR = REPO_ROOT / "serverless"
RECT_MASK_DIR = SERVERLESS_DIR / "ninerouter-rectangle-mask" / "nuclio"
POLY_MASK_DIR = SERVERLESS_DIR / "ninerouter-polygon-mask" / "nuclio"
POLYLINE_DIR = SERVERLESS_DIR / "ninerouter-polyline" / "nuclio"


# ==============================================================================
# Helpers & Mocks
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


def create_jpeg_bytes(width: int = 200, height: int = 200, color=(100, 150, 200)) -> bytes:
    """Generate in-memory valid JPEG bytes."""
    im = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def create_jpeg_base64(width: int = 200, height: int = 200, color=(100, 150, 200)) -> str:
    """Generate in-memory Base64-encoded JPEG."""
    raw = create_jpeg_bytes(width, height, color)
    return base64.b64encode(raw).decode("utf-8")


def load_module_from_path(module_name: str, file_path: Path):
    """Dynamically import module from filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==============================================================================
# 1. Remote Request Bounding & Zero Probes During Infer
# ==============================================================================

class TestRemoteRequestAndModelResolution:
    """Verify inference efficiency, zero probe overhead, and fast failure."""

    def test_exactly_one_remote_request_per_user_click_no_probes_or_discovery(self):
        """Req 1: Verify exactly one remote POST to 9Router per user click with zero GETs or probes."""
        mod = load_module_from_path("mh_rect_test1", RECT_MASK_DIR / "model_handler.py")
        img_bytes = create_jpeg_bytes(200, 200)

        # Mock initial model resolution during handler startup
        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "data": [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}]
        }

        with patch("requests.Session.get", return_value=models_response) as mock_get:
            handler = mod.ModelHandler(base_url="http://127.0.0.1:20128", model="ag/gemini-3.8-flash-high")

        # Now simulate user click triggering infer()
        post_response = MagicMock()
        post_response.status_code = 200
        post_response.headers = {"content-type": "application/json"}
        post_response.json.return_value = {
            "model": "ag/gemini-3.8-flash-high",
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "objects": [{
                            "label": "car",
                            "confidence": 0.95,
                            "box_2d": [100, 100, 500, 500],
                            "mask": [[100, 100], [500, 100], [500, 500], [100, 500]],
                        }]
                    })
                }
            }]
        }

        with patch("requests.Session.get") as infer_mock_get, \
             patch("requests.Session.post", return_value=post_response) as infer_mock_post, \
             patch.object(handler.client, "probe_segmentation_capability") as mock_probe_seg, \
             patch.object(handler.client, "probe_full_31_capability") as mock_probe_31:

            shapes = handler.infer(img_bytes)

            # Exactly one POST to /v1/chat/completions
            assert infer_mock_post.call_count == 1
            call_url = infer_mock_post.call_args[0][0]
            assert call_url.endswith("/v1/chat/completions")

            # Zero discovery GET calls during infer
            assert infer_mock_get.call_count == 0

            # Zero capability probes executed during user infer click
            mock_probe_seg.assert_not_called()
            mock_probe_31.assert_not_called()

            # Output contains paired shapes
            assert len(shapes) == 2

    def test_no_unbounded_or_multiminute_retries(self):
        """Req 7: Verify client fails fast on timeout/connection failure without retry loops."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128", timeout=2.0)

        with patch("requests.Session.post", side_effect=requests.exceptions.Timeout("Read timed out")) as mock_post:
            start_time = time.perf_counter()
            with pytest.raises(NineRouterRequestError) as exc_info:
                client.send_vision_request(
                    model="ag/gemini-3.8-flash-high",
                    image_bytes_or_b64=create_jpeg_bytes(100, 100),
                    prompt="Detect objects",
                )
            duration = time.perf_counter() - start_time

            # Exactly one attempt: no retry cascade, no sleep loops
            assert mock_post.call_count == 1
            assert exc_info.value.status_code == 408
            assert "timed out" in str(exc_info.value).lower()
            assert duration < 1.0


# ==============================================================================
# 2. Policy-Specific Prompting, Partitioning & Configuration
# ==============================================================================

class TestPolicyPromptAndConfiguration:
    """Verify prompt isolation, schema partition, and resolution/token bounds."""

    def test_policy_specific_prompt_generation_and_zero_foreign_leakage(self):
        """Req 2: Verify policy-specific prompts contain exact classes with zero cross-policy leakage."""
        # Detector 1: Policy A (14 Foreground Instance classes)
        prompt_a = build_user_prompt(mode=MODE_RECTANGLE_MASK)
        assert '"objects":' in prompt_a
        assert '"regions":' not in prompt_a
        assert '"lanes":' not in prompt_a
        for lbl in BOX_MASK_LABELS:
            assert lbl in prompt_a, f"Policy A prompt missing label '{lbl}'"
        for foreign_lbl in (list(POLYGON_MASK_LABELS) + list(POLYLINE_LABELS)):
            # Must not list foreign classes in allowed classes
            assert f"- {foreign_lbl}\n" not in prompt_a and f"- {foreign_lbl}" not in prompt_a

        # Detector 2: Policy B (10 Semantic Region classes)
        prompt_b = build_user_prompt(mode=MODE_POLYGON_MASK)
        assert '"regions":' in prompt_b
        assert '"objects":' not in prompt_b
        assert '"lanes":' not in prompt_b
        for lbl in POLYGON_MASK_LABELS:
            assert lbl in prompt_b, f"Policy B prompt missing label '{lbl}'"
        for foreign_lbl in (list(BOX_MASK_LABELS) + list(POLYLINE_LABELS)):
            assert f"- {foreign_lbl}\n" not in prompt_b and f"- {foreign_lbl}" not in prompt_b

        # Detector 3: Policy C (7 Lane Demarcation classes)
        prompt_c = build_user_prompt(mode=MODE_POLYLINE)
        assert '"lanes":' in prompt_c
        assert '"objects":' not in prompt_c
        assert '"regions":' not in prompt_c
        for lbl in POLYLINE_LABELS:
            assert lbl in prompt_c, f"Policy C prompt missing label '{lbl}'"
        for foreign_lbl in (list(BOX_MASK_LABELS) + list(POLYGON_MASK_LABELS)):
            assert f"- {foreign_lbl}\n" not in prompt_c and f"- {foreign_lbl}" not in prompt_c

    def test_exact_14_10_7_label_partition_disjoint_and_complete(self):
        """Req 12: Verify exact 14/10/7 label partition (disjoint and summing to 31)."""
        tax = get_taxonomy()
        assert len(BOX_MASK_LABELS) == 14
        assert len(POLYGON_MASK_LABELS) == 10
        assert len(POLYLINE_LABELS) == 7
        assert len(MASTER_31_LABELS) == 31

        set_a = set(BOX_MASK_LABELS)
        set_b = set(POLYGON_MASK_LABELS)
        set_c = set(POLYLINE_LABELS)

        assert set_a.isdisjoint(set_b), f"Overlap between A and B: {set_a & set_b}"
        assert set_b.isdisjoint(set_c), f"Overlap between B and C: {set_b & set_c}"
        assert set_a.isdisjoint(set_c), f"Overlap between A and C: {set_a & set_c}"
        assert (set_a | set_b | set_c) == set(MASTER_31_LABELS)

    def test_policy_specific_max_tokens_and_image_resolution_configuration(self):
        """Req 3: Verify max_tokens and image resolution are strictly configured and payload-enforced."""
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        # Generate oversized image (1600x1200) exceeding configured limit
        large_img = Image.new("RGB", (1600, 1200), color=(120, 140, 160))

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [
                    {
                        "label": "car",
                        "box_2d": [100, 100, 400, 400],
                        "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                        "confidence": 0.95,
                    }
                ]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        with patch.object(client, "send_vision_request", return_value=fake_resp) as mock_send:
            res = annotate_image(
                image_source=large_img,
                client=client,
                model="ag/gemini-3.8-flash-high",
                max_size=800,
                max_tokens=2048,
                mode=MODE_RECTANGLE_MASK,
                enable_feedback=False,
            )

            assert res.was_resized is True
            assert res.resized_dimensions == (800, 600)
            assert mock_send.call_count == 1
            call_kwargs = mock_send.call_args[1]
            assert call_kwargs["max_tokens"] == 2048


# ==============================================================================
# 3. Feedback DB Fail-Open, Few-Shot Bounds & Diff Engine
# ==============================================================================

class TestFeedbackSafetyAndFewShot:
    """Verify SQLite fail-open behavior, visual few-shot limits, and diff engine."""

    def test_feedback_db_fail_open_under_lock_contention_with_subsecond_timeout(self, tmp_path: Path):
        """Req 4: Verify feedback SQLite fail-open under lock contention with sub-second timeout."""
        db_path = tmp_path / "locked_feedback.sqlite3"
        f_db = FeedbackDatabase(db_path=db_path, timeout=0.2)
        assert f_db.timeout < 1.0  # strictly sub-second timeout

        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        img_bytes = create_jpeg_bytes(200, 200)

        fake_resp = VisionResponse(
            content=json.dumps({
                "objects": [{
                    "label": "car",
                    "confidence": 0.95,
                    "box_2d": [100, 100, 400, 400],
                    "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                }]
            }),
            raw_response={"choices": [{"message": {"content": "ok"}}]},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )

        start_time = time.perf_counter()
        with patch.object(client, "send_vision_request", return_value=fake_resp), \
             patch("app.retrieval.CorrectionRetrievalEngine.retrieve", side_effect=sqlite3.OperationalError("database is locked")), \
             patch("app.feedback.FeedbackDatabase.save_prediction_baseline", side_effect=sqlite3.OperationalError("database is locked")):
            res = annotate_image(
                image_source=img_bytes,
                client=client,
                model="ag/gemini-3.8-flash-high",
                mode=MODE_RECTANGLE_MASK,
                feedback_db=f_db,
                enable_feedback=True,
            )
        elapsed = time.perf_counter() - start_time

        # Fails open cleanly within sub-second threshold (< 0.5s)
        assert elapsed < 0.5
        # Returns valid CVAT shapes despite DB lock contention
        assert len(res.shapes) == 2
        assert any("database is locked" in w or "feedback" in w for w in res.warnings)

    def test_visual_few_shot_crop_limit_and_bounded_dimensions(self, tmp_path: Path):
        """Req 5: Verify visual few-shot crop limit (max 1 example, small crops <= 512px)."""
        db = FeedbackDatabase(
            db_path=tmp_path / "test_few_shot.sqlite3",
            examples_dir=tmp_path / "examples",
            max_crop_dimension=512,
        )

        # Create test image with large bounding box (700x600) exceeding 512px limit
        full_img = Image.new("RGB", (900, 700), color=(180, 200, 220))
        from app.feedback import CorrectionDiffItem

        diff_item = CorrectionDiffItem(
            correction_type=CORRECTION_RELABEL,
            ai_label="car",
            human_label="truck",
            human_shape={"type": "rectangle", "points": [50, 50, 750, 650]},
            details={"human_bbox": [50, 50, 750, 650]},
        )

        db.record_corrections([diff_item], image_hash="hash_crop_test", image=full_img)

        crops = list((tmp_path / "examples").glob("*.jpg"))
        assert len(crops) == 1
        with Image.open(crops[0]) as saved_crop:
            cw, ch = saved_crop.size
            # Privacy-preserving small crop <= 512px
            assert max(cw, ch) <= 512

        # Verify Retrieval Engine enforces max 1 example limit
        engine = CorrectionRetrievalEngine(db=db, max_examples_per_pass=1)
        res = engine.retrieve(policy=POLICY_BOX_MASK)
        assert len(res.visual_examples) <= 1
        if res.visual_examples:
            ex = res.visual_examples[0]
            assert "crop_data_url" in ex
            assert "expected_output" in ex
            assert "objects" in ex["expected_output"]

    def test_human_corrections_preserved_in_feedback_diffing(self):
        """Req 11: Verify human corrections are preserved in feedback diffing."""
        engine = CorrectionDiffEngine()

        # Case 1: Human relabeled paired detection (car -> truck)
        ai_shapes = [
            {"type": "rectangle", "label": "car", "points": [10, 10, 100, 100], "group_id": 1},
            {"type": "mask", "label": "car", "points": [10, 10, 100, 100], "mask": [0, 1], "group_id": 1},
        ]
        human_shapes = [
            {"type": "rectangle", "label": "truck", "points": [12, 12, 102, 102], "group_id": 1},
            {"type": "mask", "label": "truck", "points": [12, 12, 102, 102], "mask": [0, 1], "group_id": 1},
        ]

        diffs = engine.diff(ai_shapes, human_shapes, image_width=200, image_height=200)
        assert len(diffs) == 1
        assert diffs[0].correction_type == CORRECTION_RELABEL
        assert diffs[0].ai_label == "car"
        assert diffs[0].human_label == "truck"
        assert diffs[0].human_shape is not None

        # Case 2: Human added missing semantic region
        diffs_missing = engine.diff([], [{"type": "polygon", "label": "road", "points": [0, 50, 100, 50, 100, 100, 0, 100]}])
        assert len(diffs_missing) == 1
        assert diffs_missing[0].correction_type == CORRECTION_ADD_MISSING
        assert diffs_missing[0].human_label == "road"


# ==============================================================================
# 4. Controlled Timeout Errors & Error Handling
# ==============================================================================

class TestTimeoutErrorHandling:
    """Verify structured JSON timeout error responses and secret masking."""

    def test_controlled_structured_json_timeout_errors_no_raw_nginx_html(self):
        """Req 6: Verify 9Router timeout returns structured JSON (no raw nginx HTML or uncaught crash)."""
        mod = load_module_from_path("main_rect_err", RECT_MASK_DIR / "main.py")
        ctx = MockNuclioContext()

        # Simulate 9Router returning 504 Gateway Time-out with raw Nginx HTML body
        mock_handler = MagicMock()
        mock_handler.api_key = "secret_key_12345"
        mock_handler.infer.side_effect = NineRouterRequestError(
            "Request to 9Router timed out after 180s: <html><head><title>504 Gateway Time-out</title></head>"
            "<body><center><h1>504 Gateway Time-out</h1></center><hr><center>nginx/1.25.4</center></body></html>",
            status_code=504,
        )
        ctx.user_data.model_handler = mock_handler

        event = MockNuclioEvent(body=json.dumps({"image": create_jpeg_base64()}))
        resp = mod.handler(ctx, event)

        # Controlled JSON response, status 500
        assert resp.status_code == 500
        assert resp.content_type == "application/json"
        body = json.loads(resp.body)
        assert "error" in body
        assert "504 Gateway Time-out" in body["error"]
        assert "secret_key_12345" not in body["error"]
        # Must be valid JSON object, not a raw HTML string
        assert not resp.body.strip().startswith("<html")


# ==============================================================================
# 5. Native CVAT Shape Formatting & Unlocked Editability
# ==============================================================================

class TestCvatShapeFormattingAndEditability:
    """Verify CVAT shape schemas, paired group_id semantics, and zero locked flags."""

    def test_native_cvat_shape_response_formatting(self):
        """Req 8: Verify native CVAT shape formatting: rectangle, mask, polygon, and polyline."""
        raw_shapes = [
            # Policy A: Rectangle + Mask
            {"type": "rectangle", "label": "car", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1},
            {"type": "mask", "label": "car", "points": [10.0, 10.0, 100.0, 100.0], "mask": [1, 0, 1, 0], "group_id": 1},
            # Policy B: Polygon + Mask
            {"type": "polygon", "label": "road", "points": [0.0, 50.0, 100.0, 50.0, 100.0, 100.0, 0.0, 100.0], "group_id": 2},
            {"type": "mask", "label": "road", "points": [0.0, 50.0, 100.0, 100.0], "mask": [1, 1, 1, 1], "group_id": 2},
            # Policy C: Polyline
            {"type": "polyline", "label": "lane/single white", "points": [10.0, 20.0, 50.0, 80.0]},
        ]

        validated, warnings = validate_cvat_output_shapes(raw_shapes)
        assert len(validated) == 5

        for s in validated:
            stype = s["type"]
            if stype == "rectangle":
                pts = s["points"]
                assert len(pts) == 4
                assert pts[2] > pts[0] and pts[3] > pts[1]
            elif stype == "mask":
                assert "mask" in s
                assert isinstance(s["mask"], list)
            elif stype == "polygon":
                pts = s["points"]
                assert len(pts) >= 6 and len(pts) % 2 == 0
            elif stype == "polyline":
                pts = s["points"]
                assert len(pts) >= 4 and len(pts) % 2 == 0

    def test_paired_group_id_semantics_across_policies(self):
        """Req 9: Verify paired group_id for rectangle+mask and polygon+mask; zero group_id for polyline."""
        shapes = [
            {"type": "rectangle", "label": "car", "points": [10.0, 10.0, 80.0, 80.0], "group_id": 5},
            {"type": "mask", "label": "car", "points": [10.0, 10.0, 80.0, 80.0], "mask": [1, 1], "group_id": 5},
            {"type": "polygon", "label": "sidewalk", "points": [0.0, 0.0, 50.0, 0.0, 50.0, 50.0, 0.0, 50.0], "group_id": 8},
            {"type": "mask", "label": "sidewalk", "points": [0.0, 0.0, 50.0, 50.0], "mask": [1, 1], "group_id": 8},
            {"type": "polyline", "label": "lane/crosswalk", "points": [5.0, 10.0, 45.0, 90.0], "group_id": 99},
        ]

        validated, _ = validate_cvat_output_shapes(shapes)
        by_label = {s["label"] + "_" + s["type"]: s for s in validated}

        # Policy A shares group_id
        assert by_label["car_rectangle"]["group_id"] == by_label["car_mask"]["group_id"] == 5
        # Policy B shares group_id
        assert by_label["sidewalk_polygon"]["group_id"] == by_label["sidewalk_mask"]["group_id"] == 8
        # Policy C polyline has NO group_id (stripped)
        assert "group_id" not in by_label["lane/crosswalk_polyline"]

    def test_no_locked_shape_flags_for_cvat_editability(self):
        """Req 10: Verify all emitted shapes have occluded=False, outside=False, and z_order=0."""
        raw_shapes = [
            {"type": "rectangle", "label": "truck", "points": [20.0, 20.0, 120.0, 120.0], "group_id": 1, "occluded": False, "outside": False, "z_order": 0},
            {"type": "mask", "label": "truck", "points": [20.0, 20.0, 120.0, 120.0], "mask": [1, 1], "group_id": 1, "occluded": False, "outside": False, "z_order": 0},
            {"type": "polygon", "label": "building", "points": [0.0, 0.0, 40.0, 0.0, 40.0, 40.0, 0.0, 40.0], "group_id": 2, "occluded": False, "outside": False, "z_order": 0},
            {"type": "mask", "label": "building", "points": [0.0, 0.0, 40.0, 40.0], "mask": [1, 1], "group_id": 2, "occluded": False, "outside": False, "z_order": 0},
            {"type": "polyline", "label": "lane/road curb", "points": [0.0, 100.0, 100.0, 100.0], "occluded": False, "outside": False, "z_order": 0},
        ]

        validated, _ = validate_cvat_output_shapes(raw_shapes)
        for shape in validated:
            assert shape.get("occluded") is not True, f"Shape {shape} has occluded=True"
            assert shape.get("outside") is not True, f"Shape {shape} has outside=True"
            assert shape.get("z_order", 0) == 0, f"Shape {shape} has non-zero z_order"

    def test_end_to_end_detectors_emit_strictly_compliant_cvat_shapes(self):
        """Verify all 3 Nuclio detector main handlers emit strictly compliant CVAT shapes."""
        detectors = [
            ("main_rect", RECT_MASK_DIR / "main.py", [
                {"type": "rectangle", "label": "bus", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1},
                {"type": "mask", "label": "bus", "points": [10.0, 10.0, 100.0, 100.0], "mask": [1, 1], "group_id": 1},
            ], "rectangle_mask"),
            ("main_poly", POLY_MASK_DIR / "main.py", [
                {"type": "polygon", "label": "terrain", "points": [0.0, 0.0, 50.0, 0.0, 50.0, 50.0, 0.0, 50.0], "group_id": 1},
                {"type": "mask", "label": "terrain", "points": [0.0, 0.0, 50.0, 50.0], "mask": [1, 1], "group_id": 1},
            ], "polygon_mask"),
            ("main_line", POLYLINE_DIR / "main.py", [
                {"type": "polyline", "label": "lane/double yellow", "points": [10.0, 10.0, 80.0, 80.0]},
            ], "polyline"),
        ]

        for mod_name, main_file, mock_shapes, mode in detectors:
            mod = load_module_from_path(mod_name, main_file)
            ctx = MockNuclioContext()
            mock_handler = MagicMock()
            mock_handler.infer.return_value = mock_shapes
            ctx.user_data.model_handler = mock_handler

            event = MockNuclioEvent(body=json.dumps({"image": create_jpeg_base64()}))
            resp = mod.handler(ctx, event)
            assert resp.status_code == 200
            shapes = json.loads(resp.body)

            if mode == "rectangle_mask":
                assert len(shapes) == 2
                assert {s["type"] for s in shapes} == {"rectangle", "mask"}
                assert shapes[0]["group_id"] == shapes[1]["group_id"]
            elif mode == "polygon_mask":
                assert len(shapes) == 2
                assert {s["type"] for s in shapes} == {"polygon", "mask"}
                assert shapes[0]["group_id"] == shapes[1]["group_id"]
            elif mode == "polyline":
                assert len(shapes) == 1
                assert shapes[0]["type"] == "polyline"
                assert "group_id" not in shapes[0]


# ==============================================================================
# Native editability guard: preserve brush-editable masks beside vector shapes
# ==============================================================================

class TestNativeMaskEditabilityGuard:
    """Ensure paired AI outputs stay native/editable in CVAT even if mask-to-polygon UI conversion is enabled."""

    @staticmethod
    def _client_with_response(payload: Dict[str, Any]) -> NineRouterClient:
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        client.send_vision_request = MagicMock(
            return_value=VisionResponse(
                content=json.dumps(payload),
                raw_response={"choices": [{"message": {"content": json.dumps(payload)}}]},
                duration_seconds=0.01,
                model="ag/gemini-3.8-flash-low",
                status_code=200,
            )
        )
        return client

    def test_rectangle_mask_pair_keeps_native_mask_without_polygon_points(self):
        client = self._client_with_response({
            "objects": [{
                "label": "car",
                "confidence": 0.95,
                "box_2d": [100, 100, 700, 700],
                "mask": [[100, 100], [700, 100], [700, 700], [100, 700]],
            }]
        })
        result = annotate_image(
            image_source=Image.new("RGB", (320, 240), "white"),
            client=client,
            model="ag/gemini-3.8-flash-low",
            mode=MODE_RECTANGLE_MASK,
            threshold=0.0,
            enable_feedback=False,
        )

        assert [s["type"] for s in result.shapes] == ["mask", "rectangle"]
        mask, rectangle = result.shapes
        assert "mask" in mask and "points" not in mask
        assert "points" in rectangle
        assert mask["group_id"] == rectangle["group_id"]

    def test_polygon_mask_pair_keeps_native_brush_mask_and_editable_polygon(self):
        client = self._client_with_response({
            "regions": [{
                "label": "road",
                "confidence": 0.95,
                "polygon": [[50, 500], [950, 500], [950, 950], [50, 950]],
            }]
        })
        result = annotate_image(
            image_source=Image.new("RGB", (320, 240), "white"),
            client=client,
            model="ag/gemini-3.8-flash-low",
            mode=MODE_POLYGON_MASK,
            threshold=0.0,
            enable_feedback=False,
        )

        assert [s["type"] for s in result.shapes] == ["mask", "polygon"]
        mask, polygon = result.shapes
        assert "mask" in mask and "points" not in mask
        assert "points" in polygon and len(polygon["points"]) >= 6
        assert mask["group_id"] == polygon["group_id"]
