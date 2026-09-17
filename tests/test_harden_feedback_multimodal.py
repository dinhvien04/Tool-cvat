"""Comprehensive hardening tests for Tool-cvat unified implementation:
1. Canonical Visual Fingerprinting across PIL, PNG, JPEG, file paths, and CVAT downloads.
2. Baseline reconciliation and prediction lookup.
3. Multimodal few-shot retrieval with structured annotations and bounded limits.
4. Resilient fallback on corrupt or missing crop files.
5. CVAT Webhook HMAC-SHA256 signature verification over raw bytes and case-insensitive headers.
6. Instrumentation for visual_examples_used and text_rules_used.
7. Webhook inspection and management helpers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.cvat_sync import (
    CVATSyncClient,
    extract_cvat_event_header,
    extract_cvat_signature_header,
    verify_cvat_webhook_signature,
)
from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_RELABEL,
    CorrectionDiffItem,
    FeedbackDatabase,
    compute_image_hash,
    compute_visual_fingerprint,
)
from app.retrieval import CorrectionRetrievalEngine, RetrievalResult
from app.service import AnnotationResult, annotate_image


# -----------------------------------------------------------------------------
# 1. Visual Fingerprinting & Identity Tests
# -----------------------------------------------------------------------------

def test_visual_fingerprint_equality_across_formats(tmp_path: Path):
    """Verify compute_image_hash produces identical fingerprint across PIL, PNG bytes, and disk files."""
    # Create distinct pattern image
    im = Image.new("RGB", (100, 80), color=(128, 64, 32))
    for x in range(20, 40):
        for y in range(20, 40):
            im.putpixel((x, y), (255, 255, 0))

    hash_pil = compute_image_hash(im)
    assert isinstance(hash_pil, str) and len(hash_pil) == 64

    # Test alias
    assert compute_visual_fingerprint(im) == hash_pil

    # Encode as lossless PNG bytes
    buf_png = io.BytesIO()
    im.save(buf_png, format="PNG")
    png_bytes = buf_png.getvalue()
    hash_png = compute_image_hash(png_bytes)
    assert hash_png == hash_pil

    # Save to disk as PNG and reload
    png_file = tmp_path / "test_frame.png"
    png_file.write_bytes(png_bytes)
    hash_disk_png = compute_image_hash(png_file)
    assert hash_disk_png == hash_pil

    # Re-read via Image.open(png_file)
    with Image.open(png_file) as reloaded:
        assert compute_image_hash(reloaded) == hash_pil


def test_visual_fingerprint_non_image_fallback():
    """Non-image bytes gracefully fall back to raw SHA-256 without crashing."""
    raw_mock_bytes = b"non_image_arbitrary_binary_stream_xyz_12345"
    expected_raw_sha = hashlib.sha256(raw_mock_bytes).hexdigest()

    res_hash = compute_image_hash(raw_mock_bytes)
    assert res_hash == expected_raw_sha


def test_baseline_prediction_lookup_after_cvat_frame_download(tmp_path: Path):
    """Simulate CVAT serverless detector storing baseline and subsequent CVAT frame download."""
    db_file = tmp_path / "test_feedback.sqlite3"
    f_db = FeedbackDatabase(db_path=db_file)

    # 1. Original frame in detector invocation
    detector_img = Image.new("RGB", (120, 90), color=(40, 50, 60))
    detector_hash = compute_image_hash(detector_img)

    ai_shapes = [
        {"type": "rectangle", "label": "car", "points": [10.0, 10.0, 80.0, 70.0], "confidence": "0.92"},
        {"type": "rectangle", "label": "pedestrian", "points": [85.0, 15.0, 105.0, 75.0], "confidence": "0.85"},
    ]
    f_db.save_prediction_baseline(
        image_hash=detector_hash,
        shapes=ai_shapes,
        model="ag/gemini-3.8-flash-high",
        mode="box",
        task_id=101,
        job_id=202,
        frame_index=5,
    )

    # 2. Later, CVAT frame is downloaded via /api/jobs/{id}/data?type=frame
    buf = io.BytesIO()
    detector_img.save(buf, format="PNG")
    cvat_downloaded_bytes = buf.getvalue()

    downloaded_hash = compute_image_hash(cvat_downloaded_bytes)
    assert downloaded_hash == detector_hash

    # 3. Retrieve baseline using downloaded frame hash
    baseline = f_db.get_prediction_baseline(downloaded_hash)
    assert baseline is not None
    assert len(baseline["shapes"]) == 2
    assert baseline["shapes"][0]["label"] == "car"
    assert baseline["task_id"] == 101
    assert baseline["job_id"] == 202


# -----------------------------------------------------------------------------
# 2. Multimodal Few-Shot Retrieval Tests
# -----------------------------------------------------------------------------

def test_retrieval_engine_returns_real_visual_examples(tmp_path: Path):
    """Verify CorrectionRetrievalEngine generates real visual examples with base64 data URLs."""
    db_file = tmp_path / "feedback.sqlite3"
    ex_dir = tmp_path / "examples"
    f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

    # Create dummy base image
    base_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    img_hash = compute_image_hash(base_img)

    # Create correction with crop
    diff_item = CorrectionDiffItem(
        correction_type=CORRECTION_RELABEL,
        ai_label="truck",
        human_label="bus",
        ai_shape={"type": "rectangle", "points": [20.0, 20.0, 100.0, 100.0]},
        human_shape={"type": "rectangle", "points": [20.0, 20.0, 100.0, 100.0]},
        iou=1.0,
    )
    f_db.record_corrections(
        corrections=[diff_item],
        image_hash=img_hash,
        image=base_img,
    )

    engine = CorrectionRetrievalEngine(db=f_db, max_examples_per_pass=3, use_visual_examples=True)
    res: RetrievalResult = engine.retrieve(candidate_labels=["bus", "truck"])

    assert len(res.visual_examples) == 1
    vex = res.visual_examples[0]
    assert vex["correction_type"] == CORRECTION_RELABEL
    assert vex["crop_data_url"].startswith("data:image/jpeg;base64,")
    assert "bus" in vex["description"]
    assert "objects" in vex["expected_output"]
    assert vex["expected_output"]["objects"][0]["label"] == "bus"


def test_retrieval_engine_bounded_examples(tmp_path: Path):
    """Verify CorrectionRetrievalEngine respects max_examples_per_pass."""
    db_file = tmp_path / "feedback.sqlite3"
    ex_dir = tmp_path / "examples"
    f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

    base_img = Image.new("RGB", (200, 200), color=(80, 80, 80))
    img_hash = compute_image_hash(base_img)

    # Insert 5 corrections
    diff_items = []
    for i in range(5):
        diff_items.append(
            CorrectionDiffItem(
                correction_type=CORRECTION_ADD_MISSING,
                human_label="pedestrian",
                human_shape={"type": "rectangle", "points": [10.0 + i * 10, 10.0, 30.0 + i * 10, 50.0]},
            )
        )
    f_db.record_corrections(corrections=diff_items, image_hash=img_hash, image=base_img)

    # Bounded to 2
    engine = CorrectionRetrievalEngine(db=f_db, max_examples_per_pass=2, use_visual_examples=True)
    res = engine.retrieve(candidate_labels=["pedestrian"])

    assert len(res.visual_examples) == 2
    assert len(res.examples) == 2


def test_retrieval_engine_corrupt_or_missing_crop_handled_gracefully(tmp_path: Path):
    """Corrupted or missing crop files must not crash retrieval and fall back safely."""
    db_file = tmp_path / "feedback.sqlite3"
    ex_dir = tmp_path / "examples"
    f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

    base_img = Image.new("RGB", (100, 100), color=(50, 50, 50))
    img_hash = compute_image_hash(base_img)

    diff_item = CorrectionDiffItem(
        correction_type=CORRECTION_DELETE_FALSE_POSITIVE,
        ai_label="car",
        ai_shape={"type": "rectangle", "points": [10.0, 10.0, 50.0, 50.0]},
    )
    f_db.record_corrections(corrections=[diff_item], image_hash=img_hash, image=base_img)

    # Corrupt the saved crop file
    crop_files = list(ex_dir.glob("*.jpg"))
    assert len(crop_files) == 1
    crop_files[0].write_bytes(b"corrupted_bytes_not_an_image")

    engine = CorrectionRetrievalEngine(db=f_db, max_examples_per_pass=3, use_visual_examples=True)
    res = engine.retrieve(candidate_labels=["car"])

    # Gracefully skipped corrupt crop
    assert len(res.visual_examples) == 0
    # Text summary and rules still intact
    assert len(res.examples) == 1


# -----------------------------------------------------------------------------
# 3. CVAT Webhook HMAC-SHA256 Signature Verification Tests
# -----------------------------------------------------------------------------

def test_webhook_signature_verification_full():
    """Verify HMAC verification over raw bytes, headers, and edge cases."""
    secret = "cvat_webhook_test_secret_xyz"
    payload = b'{"event":"update:job","job":{"id":42,"state":"completed"}}'

    valid_hmac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    # 1. Valid with sha256= prefix
    assert verify_cvat_webhook_signature(payload, f"sha256={valid_hmac}", secret) is True

    # 2. Valid bare hex digest
    assert verify_cvat_webhook_signature(payload, valid_hmac, secret) is True

    # 3. Invalid signature
    assert verify_cvat_webhook_signature(payload, "invalid_sig_abc", secret) is False

    # 4. Missing signature
    assert verify_cvat_webhook_signature(payload, None, secret) is False
    assert verify_cvat_webhook_signature(payload, "", secret) is False

    # 5. Missing secret
    assert verify_cvat_webhook_signature(payload, valid_hmac, "") is False

    # 6. Raw bytes sensitivity: modified whitespace fails
    modified_payload = b'{"event": "update:job", "job": {"id": 42, "state": "completed"}}'
    assert verify_cvat_webhook_signature(modified_payload, valid_hmac, secret) is False


def test_case_insensitive_header_lookup():
    """Verify case-insensitive extraction of signature and event headers."""
    h1 = {"X-Signature-256": "sig123", "X-CVAT-Event": "ping"}
    assert extract_cvat_signature_header(h1) == "sig123"
    assert extract_cvat_event_header(h1) == "ping"

    h2 = {"x-signature-256": "sig456", "x-cvat-event": "update:job"}
    assert extract_cvat_signature_header(h2) == "sig456"
    assert extract_cvat_event_header(h2) == "update:job"

    h3 = {"X-CVAT-Signature": "sig789", "X-Event": "create:task"}
    assert extract_cvat_signature_header(h3) == "sig789"
    assert extract_cvat_event_header(h3) == "create:task"

    h4 = {"Content-Type": "application/json"}
    assert extract_cvat_signature_header(h4) is None
    assert extract_cvat_event_header(h4) is None


# -----------------------------------------------------------------------------
# 4. Multimodal Few-Shot Client & Service Instrumentation Tests
# -----------------------------------------------------------------------------

def test_client_send_vision_request_interleaved_payload(monkeypatch):
    """Verify NineRouterClient builds interleaved content array for visual few-shot."""
    client = NineRouterClient(base_url="http://127.0.0.1:20128")

    captured_payload = {}

    def mock_post(url, json=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"objects": []}'}}],
            "model": "ag/gemini-3.8-flash-high",
        }
        return mock_resp

    monkeypatch.setattr(client.session, "post", mock_post)

    target_bytes = b"fake_target_image_bytes"
    visual_ex = [
        {
            "description": "Relabeled: truck -> bus",
            "crop_data_url": "data:image/jpeg;base64,samplecrop1",
            "expected_output_json": '{"objects": [{"label": "bus", "box_2d": [100, 100, 900, 900]}]}',
        }
    ]

    resp = client.send_vision_request(
        model="ag/gemini-3.8-flash-high",
        image_bytes_or_b64=target_bytes,
        visual_examples=visual_ex,
    )
    assert resp.content == '{"objects": []}'

    # Inspect captured payload
    user_msg = captured_payload["messages"][1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    assert isinstance(content, list)
    # Expected interleaved elements: text desc, crop image_url, expected json, target text, target image_url
    assert len(content) == 5
    assert content[0]["type"] == "text" and "Corrected Example #1" in content[0]["text"]
    assert content[1]["type"] == "image_url" and content[1]["image_url"]["url"] == "data:image/jpeg;base64,samplecrop1"
    assert content[2]["type"] == "text" and "Expected structured annotation" in content[2]["text"]
    assert content[3]["type"] == "text" and "Target Image to Annotate" in content[3]["text"]
    assert content[4]["type"] == "image_url"


def test_annotate_image_instrumentation_counts(monkeypatch, tmp_path: Path):
    """Verify AnnotationResult tracks visual_examples_used and text_rules_used."""
    db_file = tmp_path / "feedback.sqlite3"
    ex_dir = tmp_path / "examples"
    f_db = FeedbackDatabase(db_path=db_file, examples_dir=ex_dir)

    base_img = Image.new("RGB", (150, 150), color=(70, 70, 70))
    img_hash = compute_image_hash(base_img)

    diff_item = CorrectionDiffItem(
        correction_type=CORRECTION_RELABEL,
        ai_label="car",
        human_label="truck",
        ai_shape={"type": "rectangle", "points": [10.0, 10.0, 80.0, 80.0]},
        human_shape={"type": "rectangle", "points": [10.0, 10.0, 80.0, 80.0]},
        iou=1.0,
    )
    f_db.record_corrections([diff_item], image_hash=img_hash, image=base_img)

    client = NineRouterClient(base_url="http://127.0.0.1:20128")

    def mock_send(model, image_bytes_or_b64, **kwargs):
        return VisionResponse(
            content='{"objects": [{"label": "truck", "box_2d": [100, 100, 800, 800], "confidence": 0.95}]}',
            raw_response={},
            duration_seconds=0.25,
            model=model,
            status_code=200,
        )

    monkeypatch.setattr(client, "send_vision_request", mock_send)
    monkeypatch.setattr(client, "resolve_vision_model", lambda: "ag/gemini-3.8-flash-high")

    res = annotate_image(
        image_source=base_img,
        client=client,
        feedback_db=f_db,
        enable_feedback=True,
    )

    assert isinstance(res, AnnotationResult)
    assert res.visual_examples_used == 1
    assert len(res.shapes) == 1
    assert res.shapes[0]["label"] == "truck"


# -----------------------------------------------------------------------------
# 5. Webhook Setup Helper Tests
# -----------------------------------------------------------------------------

def test_cvat_sync_client_webhook_setup_and_inspection(monkeypatch):
    """Verify CVATSyncClient inspects and registers or updates webhooks."""
    client = CVATSyncClient(base_url="http://localhost:18080", token="test_token")

    existing_webhooks = []

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": existing_webhooks}
        return resp

    def mock_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.status_code = 201
        created = dict(json)
        created["id"] = 77
        existing_webhooks.append(created)
        resp.json.return_value = created
        return resp

    def mock_patch(url, json=None, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": 77, **json}
        return resp

    monkeypatch.setattr(client.session, "get", mock_get)
    monkeypatch.setattr(client.session, "post", mock_post)
    monkeypatch.setattr(client.session, "patch", mock_patch)

    # 1. Create when none exists
    res1 = client.setup_webhook(target_url="http://nuclio-nuclio-ninerouter-vision-31:8080", secret="mysecret")
    assert res1["action"] == "created"
    assert res1["id"] == 77

    # 2. Update when existing matches target_url or description
    res2 = client.setup_webhook(target_url="http://nuclio-nuclio-ninerouter-vision-31:8080", secret="newsecret")
    assert res2["action"] == "updated"
    assert res2["id"] == 77
