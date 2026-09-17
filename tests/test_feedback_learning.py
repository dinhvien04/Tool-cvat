"""Comprehensive unit and integration tests for Tool-cvat Correction Learning.

Tests:
1. FeedbackDatabase (CRUD, indices, crop saving, storage pruning, stats across 31 labels).
2. CorrectionDiffEngine (RELABEL, BOX_MOVE, BOX_RESIZE, MASK_EDIT, REGION_EDIT, LANE_EDIT,
   DELETE_FALSE_POSITIVE, ADD_MISSING, NO_CHANGE).
3. Cross-label geometric IoU matching (car -> truck relabel vs add+delete).
4. Derived rule generation (minimum sample threshold >= 3).
5. Few-shot retrieval engine (bounded, label-filtered, prompt extension).
6. Webhook HMAC-SHA256 signature verification.
7. CVAT synchronization workflow with reconciliation.
8. Service integration (prediction baseline persistence, prompt injection).
9. Nuclio dual-dispatch handler (inference vs webhook).
10. CLI subcommands (stats, list, enable/disable, clear).
"""

import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.cvat_sync import (
    CVATSyncClient,
    SyncReport,
    sync_job_feedback,
    verify_cvat_webhook_signature,
)
from app.feedback import (
    ALL_31_LABELS,
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
from app.retrieval import CorrectionRetrievalEngine, RetrievalResult
from core.geometry import calculate_box_iou, extract_shape_bbox


# =============================================================================
# 1. Hashing and Geometry Utility Tests
# =============================================================================

def test_compute_image_hash_deterministic():
    """Verify compute_image_hash is deterministic across bytes and PIL images."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    h1 = compute_image_hash(img)
    h2 = compute_image_hash(img)
    assert h1 == h2
    assert len(h1) == 64

    raw_bytes = b"sample_jpeg_image_bytes_123"
    h_bytes = compute_image_hash(raw_bytes)
    assert h_bytes == hashlib.sha256(raw_bytes).hexdigest()


def test_calculate_box_iou_correctness():
    """Verify calculate_box_iou on identical, overlapping, and disjoint boxes."""
    # Identical boxes
    box_a = [10.0, 10.0, 50.0, 50.0]
    assert calculate_box_iou(box_a, box_a) == 1.0

    # Disjoint boxes
    box_b = [100.0, 100.0, 150.0, 150.0]
    assert calculate_box_iou(box_a, box_b) == 0.0

    # 50% overlap
    box_c = [10.0, 10.0, 50.0, 90.0]  # area = 40 * 80 = 3200
    box_d = [10.0, 50.0, 50.0, 90.0]  # area = 40 * 40 = 1600 (contained in c)
    # inter = 1600, union = 3200 -> IoU = 0.5
    assert pytest.approx(calculate_box_iou(box_c, box_d), 0.01) == 0.5


def test_extract_shape_bbox_all_types():
    """Verify extract_shape_bbox extracts [x1, y1, x2, y2] from rectangle, mask, and polygon."""
    rect = {"type": "rectangle", "points": [10.5, 20.5, 100.0, 200.0]}
    assert extract_shape_bbox(rect) == [10.5, 20.5, 100.0, 200.0]

    mask = {"type": "mask", "mask": [0, 1, 1, 0, 15, 25, 115, 225]}
    assert extract_shape_bbox(mask) == [15.0, 25.0, 115.0, 225.0]

    poly = {"type": "polygon", "points": [10, 10, 50, 10, 50, 50, 10, 50]}
    assert extract_shape_bbox(poly) == [10.0, 10.0, 50.0, 50.0]


# =============================================================================
# 2. CorrectionDiffEngine Tests
# =============================================================================

def test_diff_engine_no_change():
    """Verify identical or near-identical shapes classify as NO_CHANGE."""
    engine = CorrectionDiffEngine()
    ai_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 200, 200]}]
    human_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 200, 200]}]

    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_NO_CHANGE
    assert diffs[0].ai_label == "car"
    assert diffs[0].human_label == "car"


def test_diff_engine_relabel_cross_label():
    """Verify high IoU cross-label shift classifies as RELABEL (not add + delete)."""
    engine = CorrectionDiffEngine(min_iou_relabel=0.60)
    # AI predicted car, human labeled truck on almost the exact same box
    ai_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 300, 250]}]
    human_shapes = [{"type": "rectangle", "label": "truck", "points": [102, 98, 305, 248]}]

    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_RELABEL
    assert diffs[0].ai_label == "car"
    assert diffs[0].human_label == "truck"
    assert diffs[0].iou > 0.85


def test_diff_engine_box_move():
    """Verify translation of box center with similar dimensions classifies as BOX_MOVE."""
    engine = CorrectionDiffEngine()
    ai_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 200, 200]}]
    # Box moved 25px horizontally
    human_shapes = [{"type": "rectangle", "label": "car", "points": [125, 100, 225, 200]}]

    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_BOX_MOVE
    assert diffs[0].ai_label == "car"


def test_diff_engine_box_resize():
    """Verify significant dimension changes classify as BOX_RESIZE."""
    engine = CorrectionDiffEngine()
    ai_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 200, 200]}]
    # Box widened by 50%
    human_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 260, 200]}]

    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_BOX_RESIZE


def test_diff_engine_delete_false_positive():
    """Verify unmatched AI shape classifies as DELETE_FALSE_POSITIVE."""
    engine = CorrectionDiffEngine()
    ai_shapes = [{"type": "rectangle", "label": "bicycle", "points": [400, 400, 450, 480]}]
    human_shapes = []

    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_DELETE_FALSE_POSITIVE
    assert diffs[0].ai_label == "bicycle"
    assert diffs[0].human_label is None


def test_diff_engine_add_missing():
    """Verify unmatched human shape classifies as ADD_MISSING."""
    engine = CorrectionDiffEngine()
    ai_shapes = []
    human_shapes = [{"type": "rectangle", "label": "pedestrian", "points": [50, 50, 80, 150]}]

    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_ADD_MISSING
    assert diffs[0].human_label == "pedestrian"
    assert diffs[0].ai_label is None


def test_diff_engine_region_and_lane_edit():
    """Verify region and lane modifications classify as REGION_EDIT and LANE_EDIT."""
    engine = CorrectionDiffEngine()
    # Semantic region
    ai_reg = [{"type": "mask", "label": "road", "mask": [1, 1, 0, 300, 1280, 720]}]
    human_reg = [{"type": "mask", "label": "road", "mask": [1, 1, 0, 250, 1280, 720]}]
    diffs_reg = engine.diff(ai_reg, human_reg)
    assert diffs_reg[0].correction_type == CORRECTION_REGION_EDIT

    # Lane marking
    ai_lane = [{"type": "polyline", "label": "lane/single white", "points": [100, 500, 200, 600]}]
    human_lane = [{"type": "polyline", "label": "lane/single white", "points": [120, 510, 220, 620]}]
    diffs_lane = engine.diff(ai_lane, human_lane)
    assert diffs_lane[0].correction_type == CORRECTION_LANE_EDIT


# =============================================================================
# 3. FeedbackDatabase & Storage Tests
# =============================================================================

def test_feedback_database_lifecycle(tmp_path):
    """Verify FeedbackDatabase baseline persistence, diff recording, and stats."""
    db_file = tmp_path / "feedback.sqlite3"
    crops_dir = tmp_path / "examples"
    db = FeedbackDatabase(db_path=db_file, examples_dir=crops_dir)

    # 1. Save prediction baseline
    img_hash = "a" * 64
    pred_shapes = [
        {"type": "rectangle", "label": "car", "points": [100, 100, 200, 200]},
        {"type": "rectangle", "label": "pedestrian", "points": [300, 300, 350, 400]},
    ]
    db.save_prediction_baseline(
        image_hash=img_hash,
        shapes=pred_shapes,
        model="ag/gemini-3.8-flash-high",
        mode="full_31",
        job_id=10,
        frame_index=0,
    )

    baseline = db.get_prediction_baseline(img_hash)
    assert baseline is not None
    assert baseline["model"] == "ag/gemini-3.8-flash-high"
    assert len(baseline["shapes"]) == 2

    # 2. Record human corrections
    diff_item = CorrectionDiffItem(
        correction_type=CORRECTION_RELABEL,
        ai_label="car",
        human_label="truck",
        ai_shape=pred_shapes[0],
        human_shape={"type": "rectangle", "label": "truck", "points": [100, 100, 200, 200]},
        iou=0.95,
    )
    img = Image.new("RGB", (400, 400), color="blue")
    c_ids = db.record_corrections([diff_item], image_hash=img_hash, image=img)
    assert len(c_ids) == 1

    # 3. Check statistics across 31 labels
    stats = db.get_label_statistics()
    assert len(stats) == 31
    assert stats["car"]["relabeled_from"] == 1
    assert stats["truck"]["relabeled_to"] == 1
    assert stats["car"]["total_corrections"] == 1


def test_derived_rule_synthesis(tmp_path):
    """Verify derive_rules synthesizes active rules when threshold >= min_samples."""
    db = FeedbackDatabase(db_path=tmp_path / "rules_test.sqlite3", examples_dir=tmp_path / "crops")

    # Add 3 relabel corrections: car -> truck
    for i in range(3):
        diff = CorrectionDiffItem(
            correction_type=CORRECTION_RELABEL,
            ai_label="car",
            human_label="truck",
            iou=0.9,
        )
        db.record_corrections([diff], image_hash=f"hash_{i}")

    # Add 3 false positives for motorcycle
    for i in range(3):
        diff = CorrectionDiffItem(
            correction_type=CORRECTION_DELETE_FALSE_POSITIVE,
            ai_label="motorcycle",
            human_label=None,
        )
        db.record_corrections([diff], image_hash=f"hash_fp_{i}")

    rules = db.derive_rules(min_samples=3)
    assert len(rules) == 2

    active_rules = db.get_active_rules(labels=["car", "truck"])
    assert len(active_rules) >= 1
    assert "car" in active_rules[0] and "truck" in active_rules[0]

    all_rules = db.get_active_rules()
    assert len(all_rules) == 2


def test_storage_pruning(tmp_path):
    """Verify crops are pruned when max_examples_total boundary is exceeded."""
    db = FeedbackDatabase(
        db_path=tmp_path / "prune.sqlite3",
        examples_dir=tmp_path / "crops",
        max_examples_total=3,
        max_storage_mb=50,
    )

    img = Image.new("RGB", (200, 200), color="green")
    for i in range(6):
        diff = CorrectionDiffItem(
            correction_type=CORRECTION_RELABEL,
            ai_label="car",
            human_label="truck",
            ai_shape={"type": "rectangle", "points": [10, 10, 50, 50]},
        )
        db.record_corrections([diff], image_hash=f"h_{i}", image=img)

    crops = list((tmp_path / "crops").glob("*.jpg"))
    assert len(crops) <= 3


# =============================================================================
# 4. Retrieval & Prompt Extension Tests
# =============================================================================

def test_retrieval_engine_disabled(tmp_path):
    """Verify retrieval returns empty result when feedback is disabled."""
    db = FeedbackDatabase(db_path=tmp_path / "ret_test.sqlite3", examples_dir=tmp_path / "crops")
    db.set_enabled(False)
    engine = CorrectionRetrievalEngine(db=db)

    res = engine.retrieve(candidate_labels=["car", "truck"])
    assert res.has_content() is False
    assert res.prompt_extension == ""


def test_retrieval_engine_with_active_rules_and_examples(tmp_path):
    """Verify retrieval engine injects active rules and few-shot guidance into prompt extension."""
    db = FeedbackDatabase(db_path=tmp_path / "ret_active.sqlite3", examples_dir=tmp_path / "crops")
    # Insert 3 corrections
    for i in range(3):
        diff = CorrectionDiffItem(
            correction_type=CORRECTION_RELABEL,
            ai_label="car",
            human_label="truck",
            iou=0.92,
        )
        db.record_corrections([diff], image_hash=f"h_{i}")
    db.derive_rules(min_samples=3)

    engine = CorrectionRetrievalEngine(db=db, max_examples_per_pass=2)
    res = engine.retrieve(candidate_labels=["car", "truck"])

    assert res.has_content() is True
    assert len(res.rules) >= 1
    assert len(res.examples) >= 1
    assert "Correction Rules" in res.prompt_extension
    assert "car" in res.prompt_extension and "truck" in res.prompt_extension


# =============================================================================
# 5. Webhook Signature Verification Tests
# =============================================================================

def test_webhook_signature_verification():
    """Verify HMAC-SHA256 signature verification accepts valid and rejects forged payloads."""
    secret = "my_super_secret_webhook_key_123"
    body = b'{"event": "update:job", "job": {"id": 14, "state": "completed"}}'

    valid_hex = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert verify_cvat_webhook_signature(body, valid_hex, secret) is True
    assert verify_cvat_webhook_signature(body, f"sha256={valid_hex}", secret) is True

    # Tampered body
    tampered_body = b'{"event": "update:job", "job": {"id": 99, "state": "completed"}}'
    assert verify_cvat_webhook_signature(tampered_body, valid_hex, secret) is False

    # Tampered secret
    assert verify_cvat_webhook_signature(body, valid_hex, "wrong_secret") is False

    # Missing header or secret
    assert verify_cvat_webhook_signature(body, None, secret) is False
    assert verify_cvat_webhook_signature(body, valid_hex, "") is False


# =============================================================================
# 6. CVAT Synchronization Integration Tests
# =============================================================================

def test_sync_job_feedback_flow(tmp_path):
    """Verify sync_job_feedback reconciles job annotations with AI baselines."""
    db = FeedbackDatabase(db_path=tmp_path / "sync_db.sqlite3", examples_dir=tmp_path / "crops")

    img = Image.new("RGB", (400, 400), color="yellow")
    img_bytes = b"fake_yellow_image_bytes"
    img_hash = compute_image_hash(img_bytes)

    # Pre-populate AI baseline prediction
    db.save_prediction_baseline(
        image_hash=img_hash,
        shapes=[{"type": "rectangle", "label": "car", "points": [50, 50, 150, 150]}],
        model="ag/gemini-3.8-flash-high",
        mode="full_31",
        job_id=42,
        frame_index=0,
    )

    # Mock CVAT Client
    mock_client = MagicMock(spec=CVATSyncClient)
    mock_client.get_job.return_value = {"id": 42, "task_id": 10, "state": "completed"}
    mock_client.get_labels.return_value = {1: "truck"}
    mock_client.get_annotations.return_value = {
        "shapes": [
            {
                "id": 101,
                "label_id": 1,
                "type": "rectangle",
                "frame": 0,
                "points": [52, 48, 153, 149],
            }
        ]
    }
    mock_client.get_frame_image.return_value = img_bytes

    report = sync_job_feedback(
        job_id=42,
        cvat_client=mock_client,
        db=db,
        min_rule_samples=1,
    )

    assert report.success is True
    assert report.frames_reconciled == 1
    assert report.corrections_found.get(CORRECTION_RELABEL) == 1
    assert report.rules_derived >= 1


# =============================================================================
# 7. Service Integration (annotate_image) Tests
# =============================================================================

def test_annotate_image_feedback_integration(tmp_path):
    """Verify annotate_image saves baseline and retrieves rules."""
    from app.service import annotate_image

    db = FeedbackDatabase(db_path=tmp_path / "service_fb.sqlite3", examples_dir=tmp_path / "crops")
    # Pre-add a derived rule
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO rules (source_label, target_label, rule_type, rule_text, sample_count, updated_at) "
            "VALUES ('car', 'truck', 'confusion_pair', 'Distinguish car vs truck', 5, '2026-09-17T00:00:00Z')"
        )
        conn.commit()

    mock_client = MagicMock()
    mock_client.resolve_segmentation_model.return_value = "ag/gemini-3.8-flash-high"

    fake_resp = MagicMock()
    fake_resp.content = '{"objects": [{"label": "car", "box_2d": [100, 100, 200, 200], "mask": [[100,100],[200,100],[200,200],[100,200]]}]}'
    fake_resp.duration_seconds = 0.5
    mock_client.send_vision_request.return_value = fake_resp

    img = Image.new("RGB", (200, 200), color="blue")
    res = annotate_image(
        image_source=img,
        client=mock_client,
        mode="full_31",
        feedback_db=db,
        enable_feedback=True,
    )

    # Verify baseline was persisted
    assert res.image_hash is not None
    baseline = db.get_prediction_baseline(res.image_hash)
    assert baseline is not None
    assert len(baseline["shapes"]) >= 1

    # Verify rules were retrieved and injected
    assert len(res.rules_injected) >= 1
    sent_prompt = mock_client.send_vision_request.call_args[1]["prompt"]
    assert "Distinguish car vs truck" in sent_prompt


# =============================================================================
# 8. Ambiguity Policies & Threshold Boundaries
# =============================================================================

def test_diff_engine_cross_label_below_threshold():
    """Verify cross-label pair with IoU < min_iou_relabel (e.g. 0.45) does NOT relabel.

    Instead classifies as 1 DELETE_FALSE_POSITIVE and 1 ADD_MISSING.
    """
    engine = CorrectionDiffEngine(min_iou_relabel=0.60)
    ai_shapes = [{"type": "rectangle", "label": "car", "points": [100, 100, 200, 200]}]
    # Box with partial overlap: IoU approx 0.40
    human_shapes = [{"type": "rectangle", "label": "truck", "points": [140, 100, 260, 200]}]

    diffs = engine.diff(ai_shapes, human_shapes)
    types = [d.correction_type for d in diffs]
    assert CORRECTION_RELABEL not in types
    assert CORRECTION_DELETE_FALSE_POSITIVE in types
    assert CORRECTION_ADD_MISSING in types


def test_diff_engine_preserves_strict_ambiguity_pairs():
    """Verify distinct ambiguity pairs are preserved without merging or normalization."""
    engine = CorrectionDiffEngine()

    # 1. pedestrian vs person
    ai_ped = [{"type": "rectangle", "label": "pedestrian", "points": [50, 50, 100, 200]}]
    human_person = [{"type": "rectangle", "label": "person", "points": [50, 50, 100, 200]}]
    diff1 = engine.diff(ai_ped, human_person)
    assert diff1[0].correction_type == CORRECTION_RELABEL
    assert diff1[0].ai_label == "pedestrian"
    assert diff1[0].human_label == "person"

    # 2. traffic light vs traffic_light
    ai_tl_space = [{"type": "rectangle", "label": "traffic light", "points": [200, 50, 230, 120]}]
    human_tl_under = [{"type": "rectangle", "label": "traffic_light", "points": [200, 50, 230, 120]}]
    diff2 = engine.diff(ai_tl_space, human_tl_under)
    assert diff2[0].correction_type == CORRECTION_RELABEL
    assert diff2[0].ai_label == "traffic light"
    assert diff2[0].human_label == "traffic_light"

    # 3. traffic sign vs traffic_sign
    ai_ts_space = [{"type": "rectangle", "label": "traffic sign", "points": [300, 50, 340, 100]}]
    human_ts_under = [{"type": "rectangle", "label": "traffic_sign", "points": [300, 50, 340, 100]}]
    diff3 = engine.diff(ai_ts_space, human_ts_under)
    assert diff3[0].correction_type == CORRECTION_RELABEL
    assert diff3[0].ai_label == "traffic sign"
    assert diff3[0].human_label == "traffic_sign"


def test_all_31_labels_tracked_in_statistics(tmp_path):
    """Verify all 31 labels from master taxonomy are present in feedback statistics."""
    db = FeedbackDatabase(db_path=tmp_path / "stats_31.sqlite3", examples_dir=tmp_path / "crops")
    stats = db.get_label_statistics()

    assert len(stats) == 31
    for label in ALL_31_LABELS:
        assert label in stats
        assert "accepted" in stats[label]
        assert "relabeled_from" in stats[label]
        assert "relabeled_to" in stats[label]
        assert "total_corrections" in stats[label]


def test_derive_rules_threshold_boundary(tmp_path):
    """Verify that 2 samples do NOT trigger a rule, but exactly 3 samples DO."""
    db = FeedbackDatabase(db_path=tmp_path / "thresh_test.sqlite3", examples_dir=tmp_path / "crops")

    # Add 2 samples
    for i in range(2):
        db.record_corrections(
            [CorrectionDiffItem(correction_type=CORRECTION_RELABEL, ai_label="car", human_label="bus", iou=0.8)],
            image_hash=f"h_{i}",
        )
    rules_2 = db.derive_rules(min_samples=3)
    assert len(rules_2) == 0

    # Add 3rd sample
    db.record_corrections(
        [CorrectionDiffItem(correction_type=CORRECTION_RELABEL, ai_label="car", human_label="bus", iou=0.85)],
        image_hash="h_2",
    )
    rules_3 = db.derive_rules(min_samples=3)
    assert len(rules_3) == 1
    assert "car" in rules_3[0]["rule_text"] and "bus" in rules_3[0]["rule_text"]


# =============================================================================
# 9. Nuclio Dual-Dispatch Handler Tests
# =============================================================================

def _get_phase3b_nuclio_main():
    import importlib.util
    import sys
    nuclio_dir = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-vision-31" / "nuclio"
    if str(nuclio_dir) not in sys.path:
        sys.path.insert(0, str(nuclio_dir))

    spec = importlib.util.spec_from_file_location("phase3b_nuclio_main", str(nuclio_dir / "main.py"))
    main_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_mod)
    return main_mod


class _MockResponse:
    def __init__(self, body, headers=None, content_type="application/json", status_code=200):
        self.body = body
        self.headers = headers or {}
        self.content_type = content_type
        self.status_code = status_code


class _MockContext:
    def __init__(self):
        from types import SimpleNamespace
        self.logger = MagicMock()
        self.user_data = SimpleNamespace()
        self.Response = _MockResponse


class _MockEvent:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}


def test_nuclio_handler_dual_dispatch_webhook(monkeypatch):
    """Verify Nuclio handler processes CVAT webhook update:job events and handles signature validation."""
    nuclio_main = _get_phase3b_nuclio_main()
    ctx = _MockContext()
    secret = "secret_123"
    monkeypatch.setenv("CVAT_WEBHOOK_SECRET", secret)

    payload = json.dumps({"event": "update:job", "job": {"id": 99, "state": "completed"}})
    valid_sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # 1. Invalid signature returns 401
    evt_bad_sig = _MockEvent(body=payload, headers={"X-CVAT-Signature": "invalid_sig"})
    resp_bad = nuclio_main.handler(ctx, evt_bad_sig)
    assert resp_bad.status_code == 401
    assert "Invalid webhook signature" in resp_bad.body

    # 2. Valid signature calls sync and returns 200
    evt_valid = _MockEvent(body=payload, headers={"X-CVAT-Signature": valid_sig})
    with patch("app.cvat_sync.sync_job_feedback") as mock_sync:
        mock_sync.return_value = SyncReport(job_id=99, frames_reconciled=1, success=True)
        resp_valid = nuclio_main.handler(ctx, evt_valid)
        assert resp_valid.status_code == 200
        data = json.loads(resp_valid.body)
        assert data["status"] == "synced"
        assert data["report"]["job_id"] == 99


def test_nuclio_handler_dual_dispatch_inference():
    """Verify Nuclio handler processes detector inference when payload contains image."""
    nuclio_main = _get_phase3b_nuclio_main()
    ctx = _MockContext()
    mock_handler = MagicMock()
    mock_handler.infer.return_value = [{"type": "rectangle", "label": "car", "points": [10, 10, 50, 50]}]
    ctx.user_data.model_handler = mock_handler

    # Send base64 image request
    import base64
    b64 = base64.b64encode(b"fake_jpeg").decode("ascii")
    evt = _MockEvent(body=json.dumps({"image": b64, "threshold": 0.5}))

    resp = nuclio_main.handler(ctx, evt)
    assert resp.status_code == 200
    shapes = json.loads(resp.body)
    assert len(shapes) == 1
    assert shapes[0]["label"] == "car"


# =============================================================================
# 10. CLI Subcommands Tests
# =============================================================================

def test_cli_subcommands_dispatch(capsys):
    """Verify CLI subcommand handlers (stats, list, toggle, clear)."""
    import importlib.util
    root_main_path = Path(__file__).resolve().parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("root_main", str(root_main_path))
    cli_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_main)

    # 1. Stats
    code = cli_main.handle_feedback_stats([])
    assert code == 0
    captured = capsys.readouterr()
    assert "Correction Memory - Per-Label Statistics" in captured.out
    assert "pedestrian" in captured.out
    assert "traffic_sign" in captured.out

    # 2. Toggle
    code = cli_main.handle_feedback_toggle(False)
    assert code == 0
    captured = capsys.readouterr()
    assert "DISABLED" in captured.out

    code = cli_main.handle_feedback_toggle(True)
    assert code == 0
    captured = capsys.readouterr()
    assert "ENABLED" in captured.out

    # 3. Clear
    code = cli_main.handle_feedback_clear(["--all"])
    assert code == 0
    captured = capsys.readouterr()
    assert "cleared" in captured.out

    # 4. List
    code = cli_main.handle_feedback_list(["--limit", "10"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Correction Records" in captured.out

