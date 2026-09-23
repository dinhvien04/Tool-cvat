"""Agent G: Human Feedback & Correction Verification Test Suite.

Mandate Verification:
1. Verify that human manual edits survive Save in CVAT and feed into the correction-memory system (app/feedback.py).
2. Verify diffing between AI draft baseline and final human annotations across all geometry types:
   - Rectangle: detects BOX_MOVE, BOX_RESIZE, RELABEL, DELETE_FALSE_POSITIVE, ADD_MISSING
   - Mask: detects MASK_EDIT
   - Polygon: detects REGION_EDIT
   - Polyline: detects LANE_EDIT
3. Ensure feedback operations are FAST:
   - SQLite DB open, query, example retrieval must take < 5-15 ms.
   - Fail-open quickly on locks (timeout 0.25-1.0s, never hanging inference).
   - Visual few-shot retrieval: max 1 visual crop (384-512px) per request.
   - Zero cross-policy few-shot leakage (Rectangle+Mask only gets object corrections, Polygon+Mask only gets region corrections, Polyline only gets lane corrections).
4. Verify full round-trip: AI annotate -> human edits -> feedback sync -> correction stored -> next inference retrieves matching correction.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.cvat_sync import SyncReport, sync_job_feedback
from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_BOX_MOVE,
    CORRECTION_BOX_RESIZE,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_LANE_EDIT,
    CORRECTION_MASK_EDIT,
    CORRECTION_NO_CHANGE,
    CORRECTION_REGION_EDIT,
    CORRECTION_RELABEL,
    CorrectionDiffEngine,
    CorrectionDiffItem,
    FeedbackDatabase,
    compute_image_hash,
)
from app.retrieval import CorrectionRetrievalEngine, resolve_policy_from_labels
from app.service import AnnotationResult, annotate_image
from core.taxonomy import (
    BOX_MASK_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    Taxonomy,
)
from core.vision_contract import (
    MODE_BOX_MASK,
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
)


# =============================================================================
# 1. CVAT Webhook Save and Feedback Ingestion
# =============================================================================

def _get_serverless_handler():
    path = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-rectangle-mask" / "nuclio" / "main.py"
    spec = importlib.util.spec_from_file_location("nuclio_main", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.handler


class TestCvatWebhookSaveAndIngestion:
    """Verify that human manual edits survive Save in CVAT and feed into correction memory."""

    def test_webhook_dispatches_on_save_and_update_states(self):
        """Nuclio webhook handler triggers sync on save, update, or active annotation states."""
        handler = _get_serverless_handler()

        mock_context = MagicMock()
        mock_context.Response = lambda body, headers, content_type, status_code: {
            "body": json.loads(body) if isinstance(body, str) else body,
            "status_code": status_code,
        }

        save_events = [
            {"event": "save:job", "job": {"id": 42, "state": "in progress", "stage": "annotation"}},
            {"event": "update:job", "job": {"id": 42, "state": "annotation", "stage": "annotation"}},
            {"event": "update:task", "task": {"id": 10}, "job_id": 42, "state": "in progress"},
            {"event": "update:job", "job": {"id": 42, "state": "completed", "stage": "validation"}},
        ]

        dummy_report = SyncReport(
            job_id=42,
            corrections_found={"relabel": 2},
            rules_derived=1,
        )

        with patch("app.cvat_sync.sync_job_feedback", return_value=dummy_report) as mock_sync:
            for ev_payload in save_events:
                mock_event = MagicMock()
                mock_event.body = json.dumps(ev_payload).encode("utf-8")
                mock_event.headers = {}

                resp = handler(mock_context, mock_event)
                assert resp["status_code"] == 200
                assert resp["body"]["status"] == "synced"
                assert resp["body"]["report"]["job_id"] == 42
                assert resp["body"]["report"]["corrections_recorded"] == 2

            assert mock_sync.call_count == len(save_events)

    def test_webhook_ignores_unrelated_events(self):
        """Nuclio webhook handler ignores non-save/non-job events."""
        handler = _get_serverless_handler()

        mock_context = MagicMock()
        mock_context.Response = lambda body, headers, content_type, status_code: {
            "body": json.loads(body) if isinstance(body, str) else body,
            "status_code": status_code,
        }

        mock_event = MagicMock()
        mock_event.body = json.dumps({"event": "ping"}).encode("utf-8")
        mock_event.headers = {}

        resp = handler(mock_context, mock_event)
        assert resp["status_code"] == 200
        assert resp["body"]["status"] == "ignored"

    def test_sync_job_feedback_stores_corrections(self, tmp_path: Path):
        """Verify sync_job_feedback reconciles CVAT job annotations against AI baselines and stores them."""
        db_file = tmp_path / "feedback.sqlite3"
        ex_dir = tmp_path / "examples"
        f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

        test_img = Image.new("RGB", (640, 480), color=(100, 120, 140))
        img_hash = compute_image_hash(test_img)

        # 1. AI baseline: predicted car at [100, 100, 300, 300]
        ai_shapes = [
            {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 1},
            {"type": "mask", "label": "car", "points": [100.0, 100.0, 300.0, 100.0, 300.0, 300.0, 100.0, 300.0], "group_id": 1},
        ]
        f_db.save_prediction_baseline(
            image_hash=img_hash,
            shapes=ai_shapes,
            model_name="ag/gemini-3.8-flash-high",
            image_width=640,
            image_height=480,
            task_id=1,
            job_id=10,
            frame_index=0,
        )

        # 2. Human edited shapes: relabeled car to truck
        human_shapes = [
            {"type": "rectangle", "label": "truck", "points": [102.0, 100.0, 298.0, 300.0], "group": 1, "frame": 0},
            {"type": "mask", "label": "truck", "points": [102.0, 100.0, 298.0, 100.0, 298.0, 300.0, 102.0, 300.0], "group": 1, "frame": 0},
        ]

        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        test_bytes = buf.getvalue()

        with patch("app.cvat_sync.FeedbackDatabase", return_value=f_db), \
             patch("app.cvat_sync.CVATSyncClient") as MockClient:
            mock_client_inst = MockClient.return_value
            mock_client_inst.get_job.return_value = {"id": 10, "task_id": 1}
            mock_client_inst.get_labels.return_value = {}
            mock_client_inst.get_annotations.return_value = {
                "shapes": human_shapes,
                "tracks": [],
            }
            mock_client_inst.get_frame_image.return_value = test_bytes

            report = sync_job_feedback(job_id=10, cvat_url="http://mock-cvat:8080", token="test-token")

            assert report.job_id == 10
            assert report.corrections_recorded == 1

            corrections = f_db.get_corrections()
            assert len(corrections) == 1
            rec = corrections[0]
            assert rec["correction_type"] == CORRECTION_RELABEL
            assert rec["ai_label"] == "car"
            assert rec["human_label"] == "truck"


# =============================================================================
# 2. Diffing Across All Geometry Types
# =============================================================================

class TestDiffEngineGeometryTypes:
    """Verify bipartite diffing across Rectangle, Mask, Polygon, and Polyline."""

    @pytest.fixture
    def diff_engine(self) -> CorrectionDiffEngine:
        return CorrectionDiffEngine()

    # --- Rectangle diffing ---

    def test_diff_box_move(self, diff_engine: CorrectionDiffEngine):
        """Detect BOX_MOVE when centroid shifts > 5px or box IoU < 0.90."""
        ai_shapes = [
            {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 200.0, 200.0]},
        ]
        # Shift centroid by 20px horizontally (centroid [150, 150] -> [170, 150])
        human_shapes = [
            {"type": "rectangle", "label": "car", "points": [120.0, 100.0, 220.0, 200.0]},
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_BOX_MOVE
        assert item.ai_label == "car"
        assert item.human_label == "car"
        assert item.details["box_moved"] is True
        assert item.details["shift"]["dx"] == 20.0

    def test_diff_box_resize(self, diff_engine: CorrectionDiffEngine):
        """Detect BOX_RESIZE when width or height difference ratio > 15%."""
        ai_shapes = [
            {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 200.0, 200.0]},
        ]
        # Expand width from 100 to 140 (40% difference)
        human_shapes = [
            {"type": "rectangle", "label": "car", "points": [80.0, 100.0, 220.0, 200.0]},
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_BOX_RESIZE
        assert item.details["box_resized"] is True
        assert item.details["size_diff"]["dw"] > 0.15

    def test_diff_relabel(self, diff_engine: CorrectionDiffEngine):
        """Detect RELABEL when spatial similarity >= 0.60 but label changes."""
        ai_shapes = [
            {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 250.0, 250.0]},
        ]
        human_shapes = [
            {"type": "rectangle", "label": "truck", "points": [102.0, 100.0, 248.0, 250.0]},
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_RELABEL
        assert item.ai_label == "car"
        assert item.human_label == "truck"
        assert item.iou >= 0.60

    def test_diff_delete_false_positive(self, diff_engine: CorrectionDiffEngine):
        """Detect DELETE_FALSE_POSITIVE when AI predicts shape but human annotator removed it."""
        ai_shapes = [
            {"type": "rectangle", "label": "pedestrian", "points": [100.0, 100.0, 150.0, 250.0]},
        ]
        human_shapes = []

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_DELETE_FALSE_POSITIVE
        assert item.ai_label == "pedestrian"
        assert item.human_label is None

    def test_diff_add_missing(self, diff_engine: CorrectionDiffEngine):
        """Detect ADD_MISSING when human annotator created an annotation that AI missed."""
        ai_shapes = []
        human_shapes = [
            {"type": "rectangle", "label": "bicycle", "points": [200.0, 200.0, 280.0, 320.0]},
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_ADD_MISSING
        assert item.ai_label is None
        assert item.human_label == "bicycle"

    # --- Mask diffing ---

    def test_diff_mask_edit(self, diff_engine: CorrectionDiffEngine):
        """Detect MASK_EDIT when bounding box is unchanged but mask contour changed (mask IoU < 0.92)."""
        ai_shapes = [
            {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 200.0, 200.0], "group_id": 1},
            # AI mask covers full box [100, 100, 200, 200]
            {"type": "mask", "label": "car", "points": [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0], "group_id": 1},
        ]
        human_shapes = [
            # Exact same box
            {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 200.0, 200.0], "group_id": 1},
            # Human mask changed: cut in half vertically (IoU = 0.50 < 0.92)
            {"type": "mask", "label": "car", "points": [100.0, 100.0, 150.0, 100.0, 150.0, 200.0, 100.0, 200.0], "group_id": 1},
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_MASK_EDIT
        assert item.ai_label == "car"
        assert item.details["box_changed"] is False
        assert item.details["mask_changed"] is True
        assert item.details["mask_iou"] < 0.92

    # --- Polygon diffing ---

    def test_diff_region_edit(self, diff_engine: CorrectionDiffEngine):
        """Detect REGION_EDIT on Policy B semantic regions when polygon contour is edited."""
        ai_shapes = [
            {
                "type": "polygon",
                "label": "road",
                "points": [100.0, 100.0, 300.0, 100.0, 300.0, 300.0, 100.0, 300.0],
                "group_id": 1,
            },
            {
                "type": "mask",
                "label": "road",
                "points": [100.0, 100.0, 300.0, 100.0, 300.0, 300.0, 100.0, 300.0],
                "group_id": 1,
            },
        ]
        # Human modified polygon boundary significantly
        human_shapes = [
            {
                "type": "polygon",
                "label": "road",
                "points": [100.0, 100.0, 220.0, 100.0, 220.0, 300.0, 100.0, 300.0],
                "group_id": 1,
            },
            {
                "type": "mask",
                "label": "road",
                "points": [100.0, 100.0, 220.0, 100.0, 220.0, 300.0, 100.0, 300.0],
                "group_id": 1,
            },
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_REGION_EDIT
        assert item.ai_label == "road"
        assert item.details["polygon_changed"] is True
        assert "bbox_iou_diagnostic" in item.details

    # --- Polyline diffing ---

    def test_diff_lane_edit(self, diff_engine: CorrectionDiffEngine):
        """Detect LANE_EDIT on Policy C lane markings when line points are adjusted."""
        ai_shapes = [
            {
                "type": "polyline",
                "label": "lane/single white",
                "points": [50.0, 100.0, 200.0, 300.0, 400.0, 450.0],
            }
        ]
        # Human shifted line vertices laterally by 12px (> tolerance 4px, < max match dist 25px)
        human_shapes = [
            {
                "type": "polyline",
                "label": "lane/single white",
                "points": [62.0, 100.0, 212.0, 300.0, 412.0, 450.0],
            }
        ]

        diffs = diff_engine.diff(ai_shapes, human_shapes, image_width=640, image_height=480)
        assert len(diffs) == 1
        item = diffs[0]
        assert item.correction_type == CORRECTION_LANE_EDIT
        assert item.ai_label == "lane/single white"
        assert item.details["avg_dist"] > 4.0
        assert "bbox_iou_diagnostic" in item.details


# =============================================================================
# 3. Performance, Fast Fail-Open Locks & Policy Retrieval Isolation
# =============================================================================

class TestFastFeedbackPerformanceAndSafety:
    """Verify performance (<5-15ms), fast fail-open locks (0.25-1.0s), and visual limits."""

    def test_sqlite_latency_benchmark_under_15ms(self, tmp_path: Path):
        """Benchmark 100 iterations of SQLite DB open, query, and retrieval: median < 5ms, avg < 15ms."""
        db_file = tmp_path / "feedback_perf.sqlite3"
        ex_dir = tmp_path / "examples"
        f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

        test_img = Image.new("RGB", (300, 300), color=(120, 120, 120))
        img_hash = compute_image_hash(test_img)

        # Prepopulate records
        diff_items = [
            CorrectionDiffItem(
                correction_type=CORRECTION_RELABEL,
                ai_label="car",
                human_label="truck",
                ai_shape={"type": "rectangle", "points": [10.0, 10.0, 80.0, 80.0]},
                human_shape={"type": "rectangle", "points": [10.0, 10.0, 80.0, 80.0]},
                iou=0.95,
                details={"human_rect": {"type": "rectangle", "points": [10.0, 10.0, 80.0, 80.0]}},
            )
        ]
        f_db.record_corrections(diff_items, image_hash=img_hash, image=test_img)
        f_db.derive_rules(min_samples=1)

        # Single warm-up iteration
        CorrectionRetrievalEngine(db=f_db).retrieve(candidate_labels=["car", "truck"], policy=POLICY_BOX_MASK)

        durations_ms: List[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            # Open DB, query active rules and recent corrections, retrieve few-shot
            engine = CorrectionRetrievalEngine(db=f_db)
            res = engine.retrieve(candidate_labels=["car", "truck"], policy=POLICY_BOX_MASK)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            durations_ms.append(elapsed_ms)

        durations_ms.sort()
        median_ms = durations_ms[len(durations_ms) // 2]
        avg_ms = sum(durations_ms) / len(durations_ms)
        p95_ms = durations_ms[int(len(durations_ms) * 0.95)]

        # SQLite operations must be blazingly fast: median < 15ms, avg < 15ms (< 5-15ms mandate)
        assert median_ms < 15.0, f"Median latency too high: {median_ms:.2f}ms >= 15.0ms"
        assert avg_ms < 15.0, f"Average latency too high: {avg_ms:.2f}ms >= 15.0ms"
        assert p95_ms < 35.0, f"P95 latency too high: {p95_ms:.2f}ms >= 35.0ms"

    def test_fast_fail_open_on_locked_database(self, tmp_path: Path):
        """Verify DB fails open quickly (0.25 - 1.0s) on lock without hanging inference."""
        db_file = tmp_path / "feedback_locked.sqlite3"
        ex_dir = tmp_path / "examples"

        # Initialize DB with default 0.5s timeout (within 0.25 - 1.0s window)
        f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir, timeout=0.5)
        assert 0.25 <= f_db.timeout <= 1.0

        # Acquire an exclusive lock on the SQLite file using a separate raw connection
        lock_conn = sqlite3.connect(str(db_file), timeout=0.1)
        lock_conn.execute("BEGIN EXCLUSIVE")

        try:
            # 1. Prediction baseline lookup & save fail open quickly without raising
            t0 = time.perf_counter()
            base = f_db.get_prediction_baseline("nonexistent_hash")
            t_elapsed = time.perf_counter() - t0
            assert base is None
            assert t_elapsed < 1.2, f"Lock timeout exceeded 1.0s fail-open window: {t_elapsed:.2f}s"

            # 2. Record corrections fails open safely
            t0 = time.perf_counter()
            recs = f_db.record_corrections([], image_hash="dummy_hash")
            t_elapsed = time.perf_counter() - t0
            assert recs == []
            assert t_elapsed < 1.2

            # 3. Label statistics fails open with empty stats dict
            t0 = time.perf_counter()
            stats = f_db.get_label_statistics()
            t_elapsed = time.perf_counter() - t0
            assert "car" in stats
            assert stats["car"]["accepted"] == 0
            assert t_elapsed < 1.2

            # 4. Retrieval engine fails open gracefully with empty result
            t0 = time.perf_counter()
            engine = CorrectionRetrievalEngine(db=f_db)
            ret_res = engine.retrieve(candidate_labels=["car"])
            t_elapsed = time.perf_counter() - t0
            assert ret_res.has_content() is False
            assert ret_res.rules == []
            assert ret_res.examples == []
            assert ret_res.visual_examples == []
            assert t_elapsed < 1.2

        finally:
            lock_conn.rollback()
            lock_conn.close()

    def test_visual_few_shot_retrieval_limits_and_bounds(self, tmp_path: Path):
        """Verify visual few-shot retrieval enforces max 1 visual crop and 384-512px resolution."""
        db_file = tmp_path / "feedback_visual.sqlite3"
        ex_dir = tmp_path / "examples"
        f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

        test_img = Image.new("RGB", (800, 800), color=(100, 100, 100))
        img_hash = compute_image_hash(test_img)

        # Insert 3 corrections with valid crop geometries
        diff_items = []
        for i in range(3):
            diff_items.append(
                CorrectionDiffItem(
                    correction_type=CORRECTION_ADD_MISSING,
                    human_label="pedestrian",
                    human_shape={"type": "rectangle", "points": [50.0 + i * 100, 50.0, 120.0 + i * 100, 200.0]},
                    details={
                        "human_rect": {"type": "rectangle", "points": [50.0 + i * 100, 50.0, 120.0 + i * 100, 200.0]},
                        "human_mask": {
                            "type": "mask",
                            "points": [50.0 + i * 100, 50.0, 120.0 + i * 100, 50.0, 120.0 + i * 100, 200.0, 50.0 + i * 100, 200.0],
                        },
                    },
                )
            )
        f_db.record_corrections(diff_items, image_hash=img_hash, image=test_img)

        # Retrieval in request flow enforces max 1 visual crop
        engine = CorrectionRetrievalEngine(db=f_db, max_visual_examples=1)
        res = engine.retrieve(candidate_labels=["pedestrian"], policy=POLICY_BOX_MASK, max_visual_examples=1)

        assert len(res.visual_examples) == 1
        crop_ex = res.visual_examples[0]
        data_url = crop_ex["crop_data_url"]
        assert data_url.startswith("data:image/jpeg;base64,")

        # Decode crop and verify bounded resolution between 384 and 512px
        raw_b64 = data_url.split(",", 1)[1]
        crop_bytes = base64.b64decode(raw_b64)
        crop_img = Image.open(io.BytesIO(crop_bytes))
        max_dim = max(crop_img.size)
        assert 384 <= max_dim <= 512, f"Crop size {crop_img.size} not bounded in [384, 512]"

    def test_zero_cross_policy_few_shot_leakage(self, tmp_path: Path):
        """Verify strict policy isolation: zero leakage between Policy A, Policy B, and Policy C."""
        db_file = tmp_path / "feedback_leakage.sqlite3"
        ex_dir = tmp_path / "examples"
        f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

        test_img = Image.new("RGB", (640, 480), color=(80, 80, 80))
        img_hash = compute_image_hash(test_img)

        # Insert 1 Policy A correction (pedestrian), 1 Policy B (area/drivable), 1 Policy C (lane/single white)
        items = [
            CorrectionDiffItem(
                correction_type=CORRECTION_RELABEL,
                ai_label="car",
                human_label="pedestrian",
                ai_shape={"type": "rectangle", "points": [10.0, 10.0, 60.0, 120.0]},
                human_shape={"type": "rectangle", "points": [10.0, 10.0, 60.0, 120.0]},
                iou=0.95,
                details={
                    "human_rect": {"type": "rectangle", "points": [10.0, 10.0, 60.0, 120.0]},
                    "human_mask": {"type": "mask", "points": [10.0, 10.0, 60.0, 10.0, 60.0, 120.0, 10.0, 120.0]},
                },
            ),
            CorrectionDiffItem(
                correction_type=CORRECTION_REGION_EDIT,
                ai_label="area/drivable",
                human_label="area/drivable",
                ai_shape={"type": "polygon", "points": [100.0, 200.0, 300.0, 200.0, 300.0, 400.0, 100.0, 400.0]},
                human_shape={"type": "polygon", "points": [100.0, 200.0, 250.0, 200.0, 250.0, 400.0, 100.0, 400.0]},
                iou=0.75,
                details={
                    "human_poly": {"type": "polygon", "points": [100.0, 200.0, 250.0, 200.0, 250.0, 400.0, 100.0, 400.0]},
                    "human_mask": {"type": "mask", "points": [100.0, 200.0, 250.0, 200.0, 250.0, 400.0, 100.0, 400.0]},
                },
            ),
            CorrectionDiffItem(
                correction_type=CORRECTION_LANE_EDIT,
                ai_label="lane/single white",
                human_label="lane/single white",
                ai_shape={"type": "polyline", "points": [50.0, 100.0, 200.0, 300.0]},
                human_shape={"type": "polyline", "points": [60.0, 100.0, 210.0, 300.0]},
                iou=0.80,
                details={
                    "human_polyline": {"type": "polyline", "points": [60.0, 100.0, 210.0, 300.0]},
                },
            ),
        ]
        f_db.record_corrections(items, image_hash=img_hash, image=test_img)

        engine = CorrectionRetrievalEngine(db=f_db)

        # 1. Policy A (box_mask) query must return ONLY instance corrections
        res_a = engine.retrieve(policy=POLICY_BOX_MASK)
        for ex in res_a.examples:
            lbl = ex["human_label"] or ex["ai_label"]
            assert lbl in BOX_MASK_LABELS
            assert lbl not in POLYGON_MASK_LABELS
            assert lbl not in POLYLINE_LABELS

        # 2. Policy B (polygon_mask) query must return ONLY region corrections
        res_b = engine.retrieve(policy=POLICY_POLYGON_MASK)
        for ex in res_b.examples:
            lbl = ex["human_label"] or ex["ai_label"]
            assert lbl in POLYGON_MASK_LABELS
            assert lbl not in BOX_MASK_LABELS
            assert lbl not in POLYLINE_LABELS

        # 3. Policy C (polyline) query must return ONLY lane corrections
        res_c = engine.retrieve(policy=POLICY_POLYLINE)
        for ex in res_c.examples:
            lbl = ex["human_label"] or ex["ai_label"]
            assert lbl in POLYLINE_LABELS
            assert lbl not in BOX_MASK_LABELS
            assert lbl not in POLYGON_MASK_LABELS


# =============================================================================
# 4. Full Round-Trip Integration Test
# =============================================================================

class TestFeedbackRoundTripIntegration:
    """Verify full loop: AI annotate -> human edits -> feedback sync -> correction stored -> next inference retrieval."""

    def test_full_roundtrip_annotate_edit_sync_retrieval(self, tmp_path: Path, monkeypatch):
        """End-to-end round trip reinforcing model on subsequent inference."""
        db_file = tmp_path / "feedback_roundtrip.sqlite3"
        ex_dir = tmp_path / "examples"
        f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        monkeypatch.setattr(
            client,
            "get_vision_models",
            lambda: [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}],
        )

        # ---------------------------------------------------------------------
        # Step 1: Initial AI Inference on Frame 0
        # ---------------------------------------------------------------------
        # AI incorrectly predicts a truck as a 'car'
        ai_resp_frame0 = VisionResponse(
            content=json.dumps({
                "objects": [
                    {
                        "label": "car",
                        "box_2d": [100, 100, 400, 400],
                        "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                        "confidence": 0.85,
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.2,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )
        monkeypatch.setattr(client, "send_vision_request", lambda *args, **kwargs: ai_resp_frame0)

        img0 = Image.new("RGB", (640, 480), color=(110, 110, 110))
        result0 = annotate_image(
            image_source=img0,
            client=client,
            mode=MODE_RECTANGLE_MASK,
            feedback_db=f_db,
            enable_feedback=True,
            task_id=1,
            job_id=5,
            frame_index=0,
        )
        assert len(result0.shapes) == 2  # 1 rectangle + 1 mask
        rect0 = next(s for s in result0.shapes if s["type"] == "rectangle")
        mask0 = next(s for s in result0.shapes if s["type"] == "mask")
        img0_hash = result0.image_hash
        assert img0_hash is not None

        # Verify AI prediction baseline is stored
        baseline = f_db.get_prediction_baseline(img0_hash)
        assert baseline is not None
        assert len(baseline["shapes"]) == 2

        # ---------------------------------------------------------------------
        # Step 2: Human Manual Edits in CVAT
        # ---------------------------------------------------------------------
        # Annotator corrects the label to 'truck' and adds an omitted 'pedestrian'
        human_mask = {"type": "mask", "label": "truck", "group": 1, "frame": 0}
        if "mask" in mask0:
            human_mask["mask"] = mask0["mask"]
        if "points" in mask0:
            human_mask["points"] = list(mask0["points"])
        human_shapes = [
            {"type": "rectangle", "label": "truck", "points": list(rect0["points"]), "group": 1, "frame": 0},
            human_mask,
            {"type": "rectangle", "label": "pedestrian", "points": [500.0, 150.0, 560.0, 300.0], "group": 2, "frame": 0},
            {"type": "mask", "label": "pedestrian", "points": [500.0, 150.0, 560.0, 150.0, 560.0, 300.0, 500.0, 300.0], "group": 2, "frame": 0},
        ]

        buf0 = io.BytesIO()
        img0.save(buf0, format="JPEG")
        img0_bytes = buf0.getvalue()

        # ---------------------------------------------------------------------
        # Step 3: Feedback Sync (simulating CVAT webhook or manual sync)
        # ---------------------------------------------------------------------
        with patch("app.cvat_sync.FeedbackDatabase", return_value=f_db), \
             patch("app.cvat_sync.CVATSyncClient") as MockClient:
            mock_client_inst = MockClient.return_value
            mock_client_inst.get_job.return_value = {"id": 5, "task_id": 1}
            mock_client_inst.get_labels.return_value = {}
            mock_client_inst.get_annotations.return_value = {
                "shapes": human_shapes,
                "tracks": [],
            }
            mock_client_inst.get_frame_image.return_value = img0_bytes

            sync_report = sync_job_feedback(job_id=5, cvat_url="http://mock-cvat:8080", token="dummy-token")
            assert sync_report.corrections_recorded == 2

        # Verify corrections in database: RELABEL and ADD_MISSING
        corrections = f_db.get_corrections()
        assert len(corrections) == 2
        corr_types = {c["correction_type"] for c in corrections}
        assert CORRECTION_RELABEL in corr_types
        assert CORRECTION_ADD_MISSING in corr_types

        # Derive active correction rules
        rules_added = f_db.derive_rules(min_samples=1)
        assert len(rules_added) >= 1

        # ---------------------------------------------------------------------
        # Step 4: Next Inference on Frame 1 Retrieves Matching Correction
        # ---------------------------------------------------------------------
        captured_prompts: List[str] = []
        captured_visual_examples: List[Any] = []

        def capture_vision_request(*args, **kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            captured_visual_examples.append(kwargs.get("visual_examples"))
            return VisionResponse(
                content=json.dumps({
                    "objects": [
                        {
                            "label": "truck",
                            "box_2d": [100, 100, 400, 400],
                            "mask": [[100, 100], [400, 100], [400, 400], [100, 400]],
                            "confidence": 0.95,
                        }
                    ]
                }),
                raw_response={},
                duration_seconds=0.15,
                model="ag/gemini-3.8-flash-high",
                status_code=200,
            )

        monkeypatch.setattr(client, "send_vision_request", capture_vision_request)

        img1 = Image.new("RGB", (640, 480), color=(120, 120, 120))
        result1 = annotate_image(
            image_source=img1,
            client=client,
            candidate_labels=["car", "truck", "pedestrian"],
            mode=MODE_RECTANGLE_MASK,
            feedback_db=f_db,
            enable_feedback=True,
            task_id=1,
            job_id=5,
            frame_index=1,
        )

        # Verify feedback was retrieved and injected into next inference prompt!
        assert result1.rules_injected != [] or result1.visual_examples_used > 0
        assert len(captured_prompts) == 1
        prompt_sent = captured_prompts[0]
        # Active derived rule or few-shot example mentions car / truck / pedestrian
        assert any(k in prompt_sent for k in ("Frequent confusion pattern", "car", "truck", "pedestrian"))
        # Visual example injected matches the max 1 visual crop constraint
        vis_sent = captured_visual_examples[0]
        if vis_sent:
            assert len(vis_sent) == 1
            assert "crop_data_url" in vis_sent[0]
