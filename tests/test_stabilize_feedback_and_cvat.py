"""Comprehensive stabilization test suite for Tool-cvat.

Tests:
1. Specification validation for CVAT Lambda Manager compatibility:
   - Valid YAML passes.
   - Poisoned YAML (missing annotations, invalid spec JSON, duplicate IDs) fails.
   - Full 31-label detector schema integrity.
2. True geometry comparison in CorrectionDiffEngine:
   - True mask IoU detects modified polygon contours even when bounding-box IoU is high (>0.92).
   - True line stroke overlap detects modified lane lines even when bounding-box IoU is high.
   - Identical masks/lanes correctly evaluate to NO_CHANGE.
3. Two-level image identity and perceptual hash matching:
   - Exact decoded-RGB SHA-256 fingerprint matching (level 1).
   - Lightweight dHash perceptual hash matching with Hamming distance <= 5 (level 2).
   - Job ID + frame index fallback (level 3).
4. Full-31 visual few-shot schema routing:
   - Instance objects route to objects[] with box_2d.
   - Semantic regions route to regions[] with mask and NO box_2d.
   - Lane markings route to lanes[] with mask and NO box_2d.
   - Negative examples (DELETE_FALSE_POSITIVE) emit full three-array contract {"objects": [], "regions": [], "lanes": []}.
5. CVAT webhook project scoping and raw-bytes HMAC verification.
"""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from PIL import Image, ImageDraw

from app.cvat_sync import CVATSyncClient, verify_cvat_webhook_signature
from app.feedback import (
    ALL_31_LABELS,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_LANE_EDIT,
    CORRECTION_MASK_EDIT,
    CORRECTION_NO_CHANGE,
    CORRECTION_REGION_EDIT,
    CorrectionDiffEngine,
    FeedbackDatabase,
    _compute_shape_iou,
    compute_image_hash,
    compute_perceptual_hash,
    hamming_distance,
)
from app.retrieval import CorrectionRetrievalEngine
from core.geometry import polygon_to_cvat_mask
from core.taxonomy import GROUP_INSTANCE, GROUP_LANE, GROUP_REGION, Taxonomy
from scripts.validate_function_spec import validate_function_yaml


# =============================================================================
# 1. Specification Validation Tests (Prevent Poisoning CVAT Lambda Manager)
# =============================================================================


class TestSpecificationValidation:
    """Test validate_function_spec.py rules preventing CVAT UI crashes."""

    def test_valid_31_detector_spec(self):
        spec_path = Path("serverless/ninerouter-vision-31/nuclio/function.yaml")
        is_valid, errors = validate_function_yaml(spec_path)
        assert is_valid is True, f"Expected valid spec, got errors: {errors}"
        assert len(errors) == 0

    def test_missing_annotations_poison_fails(self, tmp_path: Path):
        poisoned_yaml = tmp_path / "poisoned_function.yaml"
        poisoned_yaml.write_text(
            yaml.dump({
                "metadata": {"name": "test-poison"},
                "spec": {"handler": "main:handler"},
            }),
            encoding="utf-8",
        )
        is_valid, errors = validate_function_yaml(poisoned_yaml)
        assert is_valid is False
        assert any("metadata.annotations" in e for e in errors)

    def test_invalid_json_spec_fails(self, tmp_path: Path):
        bad_json_yaml = tmp_path / "bad_json.yaml"
        bad_json_yaml.write_text(
            yaml.dump({
                "metadata": {
                    "name": "test-bad-json",
                    "annotations": {
                        "name": "Test Detector",
                        "type": "detector",
                        "spec": "{unquoted_invalid_json",
                    },
                },
            }),
            encoding="utf-8",
        )
        is_valid, errors = validate_function_yaml(bad_json_yaml)
        assert is_valid is False
        assert any("Invalid JSON in 'metadata.annotations.spec'" in e for e in errors)

    def test_duplicate_label_id_fails(self, tmp_path: Path):
        dup_yaml = tmp_path / "dup_id.yaml"
        dup_yaml.write_text(
            yaml.dump({
                "metadata": {
                    "name": "test-dup",
                    "annotations": {
                        "name": "Test Dup",
                        "type": "detector",
                        "spec": json.dumps([
                            {"id": 1, "name": "car", "type": "rectangle"},
                            {"id": 1, "name": "truck", "type": "rectangle"},
                        ]),
                    },
                },
            }),
            encoding="utf-8",
        )
        is_valid, errors = validate_function_yaml(dup_yaml)
        assert is_valid is False
        assert any("Duplicate label id" in e for e in errors)

    def test_unsupported_cvat_label_type_fails(self, tmp_path: Path):
        bad_type_yaml = tmp_path / "bad_type.yaml"
        bad_type_yaml.write_text(
            yaml.dump({
                "metadata": {
                    "name": "test-bad-type",
                    "annotations": {
                        "name": "Test Bad Type",
                        "type": "detector",
                        "spec": json.dumps([
                            {"id": 1, "name": "car", "type": "3d_cuboid_unsupported"},
                        ]),
                    },
                },
            }),
            encoding="utf-8",
        )
        is_valid, errors = validate_function_yaml(bad_type_yaml)
        assert is_valid is False
        assert any("Unsupported label type" in e for e in errors)

    def test_31_detector_missing_label_fails(self, tmp_path: Path):
        partial_yaml = tmp_path / "partial_31.yaml"
        partial_yaml.write_text(
            yaml.dump({
                "metadata": {
                    "name": "ninerouter-vision-31",
                    "annotations": {
                        "name": "9Router Vision 31-Label",
                        "type": "detector",
                        "spec": json.dumps([
                            {"id": 1, "name": "car", "type": "any"},
                        ]),
                    },
                },
            }),
            encoding="utf-8",
        )
        is_valid, errors = validate_function_yaml(partial_yaml)
        assert is_valid is False
        assert any("must have exactly 31 labels" in e for e in errors)


# =============================================================================
# 2. True Geometry Comparison Tests (Mask IoU & Lane Overlap)
# =============================================================================


class TestTrueGeometryComparison:
    """Test that CorrectionDiffEngine accurately distinguishes modified contours and lanes."""

    def test_modified_region_contour_detected_despite_high_bbox_iou(self):
        """A semantic region with high bbox IoU but modified polygon contour must be REGION_EDIT."""
        engine = CorrectionDiffEngine()

        # Both shapes share similar outer bounding boxes [100, 100, 500, 500] (bbox IoU ~0.95)
        # But Shape A is a full square, while Shape B is a triangle with half the area
        ai_shape = {
            "type": "polygon",
            "label": "road",
            "points": [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0],
        }
        human_shape = {
            "type": "polygon",
            "label": "road",
            "points": [100.0, 100.0, 500.0, 100.0, 300.0, 500.0],
        }

        diffs = engine.diff([ai_shape], [human_shape], image_width=1280, image_height=720)
        assert len(diffs) == 1
        item = diffs[0]

        # Must NOT be misclassified as NO_CHANGE!
        assert item.correction_type == CORRECTION_REGION_EDIT
        assert item.ai_label == "road"
        assert item.human_label == "road"
        assert item.iou < 0.88
        assert "mask_iou" in item.details
        assert "bbox_iou" in item.details

    def test_identical_region_contour_evaluates_to_no_change(self):
        """Identical polygon contour must evaluate to NO_CHANGE."""
        engine = CorrectionDiffEngine()
        shape = {
            "type": "polygon",
            "label": "vegetation",
            "points": [200.0, 200.0, 400.0, 200.0, 400.0, 400.0, 200.0, 400.0],
        }
        diffs = engine.diff([shape], [shape], image_width=1280, image_height=720)
        assert len(diffs) == 1
        assert diffs[0].correction_type == CORRECTION_NO_CHANGE
        assert diffs[0].iou >= 0.95

    def test_modified_lane_polyline_detected_despite_high_bbox_iou(self):
        """A lane line shifted or realigned within the same box must be LANE_EDIT."""
        engine = CorrectionDiffEngine()

        # Both polylines span between y=100 and y=500 within x=100..200 (exact bbox IoU = 1.0)
        # But Line A runs top-left to bottom-right, Line B runs top-right to bottom-left
        ai_lane = {
            "type": "polyline",
            "label": "lane/single white",
            "points": [100.0, 100.0, 200.0, 500.0],
        }
        human_lane = {
            "type": "polyline",
            "label": "lane/single white",
            "points": [200.0, 100.0, 100.0, 500.0],
        }

        diffs = engine.diff([ai_lane], [human_lane], image_width=1280, image_height=720)
        assert len(diffs) == 1
        item = diffs[0]

        # Must detect lane line realignment
        assert item.correction_type == CORRECTION_LANE_EDIT
        assert item.ai_label == "lane/single white"
        assert item.iou < 0.85
        assert "lane_iou" in item.details

    def test_identical_lane_polyline_evaluates_to_no_change(self):
        """Identical lane polyline must evaluate to NO_CHANGE."""
        engine = CorrectionDiffEngine()
        lane = {
            "type": "polyline",
            "label": "lane/double yellow",
            "points": [150.0, 100.0, 300.0, 400.0, 450.0, 700.0],
        }
        diffs = engine.diff([lane], [lane], image_width=1280, image_height=720)
        assert len(diffs) == 1
        assert diffs[0].correction_type == CORRECTION_NO_CHANGE
        assert diffs[0].iou >= 0.95

    def test_instance_mask_contour_adjustment(self):
        """Instance mask contour adjustment within same bbox is detected as MASK_EDIT."""
        engine = CorrectionDiffEngine()

        # Create two binary masks with different interior shapes
        im1 = Image.new("L", (100, 100), 0)
        im2 = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(im1).rectangle([10, 10, 90, 90], fill=1)
        ImageDraw.Draw(im2).polygon([(10, 10), (90, 10), (50, 90)], fill=1)

        from core.geometry import mask_to_cvat_flat_list
        flat1 = mask_to_cvat_flat_list(im1)
        flat2 = mask_to_cvat_flat_list(im2)

        shape1 = {"type": "mask", "label": "car", "mask": flat1}
        shape2 = {"type": "mask", "label": "car", "mask": flat2}

        diffs = engine.diff([shape1], [shape2], image_width=100, image_height=100)
        assert len(diffs) == 1
        assert diffs[0].correction_type == CORRECTION_MASK_EDIT
        assert diffs[0].iou < 0.88


# =============================================================================
# 3. Two-Level Image Identity and Perceptual Hashing Tests
# =============================================================================


class TestTwoLevelImageIdentity:
    """Test exact decoded-RGB fingerprint + secondary dHash perceptual fallback."""

    def test_exact_fingerprint_identical_across_formats(self):
        im = Image.new("RGB", (64, 64), (120, 80, 200))
        h1 = compute_image_hash(im)

        png_buf = io.BytesIO()
        im.save(png_buf, format="PNG")
        h2 = compute_image_hash(png_buf.getvalue())

        assert h1 == h2
        assert len(h1) == 64

    def test_perceptual_hash_robust_to_jpeg_reencoding(self):
        im = Image.new("RGB", (128, 128), (50, 100, 150))
        d = ImageDraw.Draw(im)
        d.rectangle([20, 20, 100, 100], fill=(200, 50, 50))
        d.line([(0, 0), (128, 128)], fill=(255, 255, 0), width=5)

        phash_orig = compute_perceptual_hash(im)
        assert len(phash_orig) == 16

        # Re-encode as lossy JPEG
        jpg_buf = io.BytesIO()
        im.save(jpg_buf, format="JPEG", quality=75)
        reencoded = Image.open(jpg_buf)

        phash_reencoded = compute_perceptual_hash(reencoded)
        dist = hamming_distance(phash_orig, phash_reencoded)
        assert dist <= 5, f"Expected Hamming distance <= 5, got {dist}"

    def test_perceptual_hash_differs_on_distinct_images(self):
        im1 = Image.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(im1).rectangle([0, 0, 32, 64], fill=(255, 255, 255))

        im2 = Image.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(im2).rectangle([0, 0, 64, 32], fill=(255, 255, 255))

        ph1 = compute_perceptual_hash(im1)
        ph2 = compute_perceptual_hash(im2)
        dist = hamming_distance(ph1, ph2)
        assert dist > 10

    def test_two_level_baseline_lookup(self, tmp_path: Path):
        db = FeedbackDatabase(db_path=tmp_path / "test_feedback.sqlite3")

        # Create original image
        im = Image.new("RGB", (128, 128), (40, 120, 200))
        ImageDraw.Draw(im).rectangle([20, 20, 80, 80], fill=(220, 30, 30))

        exact_hash = compute_image_hash(im)
        phash = compute_perceptual_hash(im)

        pred_shapes = [{"type": "rectangle", "label": "car", "points": [20, 20, 80, 80]}]
        db.save_prediction_baseline(
            image_hash=exact_hash,
            shapes=pred_shapes,
            model="test-model",
            mode="full_31",
            task_id=10,
            job_id=20,
            frame_index=0,
            perceptual_hash=phash,
        )

        # 1. Primary match: exact image_hash
        match1 = db.get_prediction_baseline(image_hash=exact_hash)
        assert match1 is not None
        assert match1["match_type"] == "exact_fingerprint"
        assert match1["image_hash"] == exact_hash

        # 2. Secondary match: simulate lossy re-encoded image with DIFFERENT exact hash
        jpg_buf = io.BytesIO()
        im.save(jpg_buf, format="JPEG", quality=70)
        reencoded = Image.open(jpg_buf)
        different_exact_hash = compute_image_hash(reencoded)
        reencoded_phash = compute_perceptual_hash(reencoded)

        # Confirm exact hash differs due to JPEG compression
        assert different_exact_hash != exact_hash

        # But secondary perceptual hash lookup succeeds!
        match2 = db.get_prediction_baseline(
            image_hash=different_exact_hash,
            perceptual_hash=reencoded_phash,
        )
        assert match2 is not None
        assert match2["match_type"] == "perceptual_hash"
        assert match2["hamming_distance"] <= 5
        assert match2["shapes"][0]["label"] == "car"

        # 3. Tertiary match: job_id + frame_index fallback
        match3 = db.get_prediction_baseline(
            image_hash="completely_unrelated_hash",
            perceptual_hash="ffffffffffffffff",
            job_id=20,
            frame_index=0,
        )
        assert match3 is not None
        assert match3["match_type"] == "job_frame_fallback"


# =============================================================================
# 4. Multimodal Few-Shot Three-Array Contract Routing Tests
# =============================================================================


class TestFewShotSchemaRouting:
    """Test that visual few-shot examples route instances, regions, and lanes properly."""

    def test_instance_object_routes_to_objects(self, tmp_path: Path):
        db = FeedbackDatabase(
            db_path=tmp_path / "fb.sqlite3",
            examples_dir=tmp_path / "examples",
        )
        # Create a mock crop
        crop_path = tmp_path / "examples" / "crop1.jpg"
        Image.new("RGB", (64, 64), (100, 100, 100)).save(crop_path)

        record = {
            "image_hash": "hash1",
            "correction_type": "RELABEL",
            "ai_label": "car",
            "human_label": "truck",
            "crop_path": str(crop_path),
            "details_json": json.dumps({
                "crop_coords": [0, 0, 64, 64],
                "human_rect": {"type": "rectangle", "points": [10, 10, 50, 50]},
                "human_mask": {"type": "mask", "points": [10, 10, 50, 10, 50, 50, 10, 50]},
            }),
            "human_shape_json": json.dumps({
                "type": "rectangle",
                "label": "truck",
                "points": [10, 10, 50, 50],
            }),
            "created_at": "2026-09-18T12:00:00Z",
        }

        with db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (:image_hash, :correction_type, :ai_label, :human_label,
                        :crop_path, :details_json, :human_shape_json, :created_at)
                """,
                record,
            )
            conn.commit()

        retrieval = CorrectionRetrievalEngine(db=db)
        res = retrieval.retrieve(candidate_labels=["truck", "car"])
        assert len(res.visual_examples) == 1
        vex = res.visual_examples[0]
        out = vex["expected_output"]

        # Must route exclusively to objects[] with atomic box_2d and mask per Policy A
        assert len(out["objects"]) == 1
        assert len(out["regions"]) == 0
        assert len(out["lanes"]) == 0
        assert out["objects"][0]["label"] == "truck"
        assert "box_2d" in out["objects"][0]
        assert "mask" in out["objects"][0]

    def test_semantic_region_routes_to_regions_without_box_2d(self, tmp_path: Path):
        db = FeedbackDatabase(
            db_path=tmp_path / "fb.sqlite3",
            examples_dir=tmp_path / "examples",
        )
        crop_path = tmp_path / "examples" / "crop_road.jpg"
        Image.new("RGB", (64, 64), (80, 80, 80)).save(crop_path)

        record = {
            "image_hash": "hash_road",
            "correction_type": "REGION_EDIT",
            "ai_label": "road",
            "human_label": "road",
            "crop_path": str(crop_path),
            "details_json": json.dumps({"crop_coords": [0, 0, 64, 64]}),
            "human_shape_json": json.dumps({
                "type": "polygon",
                "label": "road",
                "points": [0, 30, 64, 30, 64, 64, 0, 64],
            }),
            "created_at": "2026-09-18T12:00:00Z",
        }

        with db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (:image_hash, :correction_type, :ai_label, :human_label,
                        :crop_path, :details_json, :human_shape_json, :created_at)
                """,
                record,
            )
            conn.commit()

        retrieval = CorrectionRetrievalEngine(db=db)
        res = retrieval.retrieve(candidate_labels=["road"])
        assert len(res.visual_examples) == 1
        out = res.visual_examples[0]["expected_output"]

        # Must route exclusively to regions[] with polygon per Policy B, NO box_2d!
        assert len(out["objects"]) == 0
        assert len(out["regions"]) == 1
        assert len(out["lanes"]) == 0
        assert out["regions"][0]["label"] == "road"
        assert "polygon" in out["regions"][0]
        assert "box_2d" not in out["regions"][0]

    def test_lane_marking_routes_to_lanes_without_box_2d(self, tmp_path: Path):
        db = FeedbackDatabase(
            db_path=tmp_path / "fb.sqlite3",
            examples_dir=tmp_path / "examples",
        )
        crop_path = tmp_path / "examples" / "crop_lane.jpg"
        Image.new("RGB", (64, 64), (50, 50, 50)).save(crop_path)

        record = {
            "image_hash": "hash_lane",
            "correction_type": "LANE_EDIT",
            "ai_label": "lane/single white",
            "human_label": "lane/single white",
            "crop_path": str(crop_path),
            "details_json": json.dumps({"crop_coords": [0, 0, 64, 64]}),
            "human_shape_json": json.dumps({
                "type": "polyline",
                "label": "lane/single white",
                "points": [10, 60, 50, 10],
            }),
            "created_at": "2026-09-18T12:00:00Z",
        }

        with db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (:image_hash, :correction_type, :ai_label, :human_label,
                        :crop_path, :details_json, :human_shape_json, :created_at)
                """,
                record,
            )
            conn.commit()

        retrieval = CorrectionRetrievalEngine(db=db)
        res = retrieval.retrieve(candidate_labels=["lane/single white"])
        assert len(res.visual_examples) == 1
        out = res.visual_examples[0]["expected_output"]

        # Must route exclusively to lanes[] with polyline per Policy C, NO box_2d!
        assert len(out["objects"]) == 0
        assert len(out["regions"]) == 0
        assert len(out["lanes"]) == 1
        assert out["lanes"][0]["label"] == "lane/single white"
        assert "polyline" in out["lanes"][0]
        assert "box_2d" not in out["lanes"][0]

    def test_negative_example_full_three_array_contract(self, tmp_path: Path):
        """DELETE_FALSE_POSITIVE must emit objects:[], regions:[], lanes:[]."""
        db = FeedbackDatabase(
            db_path=tmp_path / "fb.sqlite3",
            examples_dir=tmp_path / "examples",
        )
        crop_path = tmp_path / "examples" / "crop_neg.jpg"
        Image.new("RGB", (64, 64), (120, 120, 120)).save(crop_path)

        record = {
            "image_hash": "hash_neg",
            "correction_type": "DELETE_FALSE_POSITIVE",
            "ai_label": "pole",
            "human_label": None,
            "crop_path": str(crop_path),
            "details_json": json.dumps({"crop_coords": [10, 10, 50, 50]}),
            "human_shape_json": None,
            "created_at": "2026-09-18T12:00:00Z",
        }

        with db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (:image_hash, :correction_type, :ai_label, :human_label,
                        :crop_path, :details_json, :human_shape_json, :created_at)
                """,
                record,
            )
            conn.commit()

        retrieval = CorrectionRetrievalEngine(db=db)
        res = retrieval.retrieve(candidate_labels=["pole"])
        assert len(res.visual_examples) == 1
        vex = res.visual_examples[0]
        out = vex["expected_output"]

        # Full 3-array empty contract
        assert out == {"objects": [], "regions": [], "lanes": []}
        assert "negative example" in vex["description"]


# =============================================================================
# 5. Webhook Verification and Project Scoping Tests
# =============================================================================


class TestWebhookHardening:
    """Test raw-bytes HMAC signature verification and project-scoped registration."""

    def test_raw_bytes_hmac_verification(self):
        import hashlib
        import hmac

        secret = "super_secret_webhook_key"
        raw_body = b'{"event":"update:job","job":{"id":42,"status":"completed"}}'
        sig = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        # Exact raw bytes must verify True
        assert verify_cvat_webhook_signature(raw_body, sig, secret) is True

        # Tampered raw bytes must verify False
        tampered_body = b'{"event":"update:job","job":{"id":42,"status":"rejected"}}'
        assert verify_cvat_webhook_signature(tampered_body, sig, secret) is False

        # Non-bytes input (like arbitrary objects) must be rejected
        assert verify_cvat_webhook_signature(12345, sig, secret) is False

    def test_webhook_setup_with_project_id(self, monkeypatch):
        """Test that setup_webhook includes project_id and type='project'."""
        client = CVATSyncClient(base_url="http://mock-cvat:8080")

        # Mock list_webhooks to return empty list
        monkeypatch.setattr(client, "list_webhooks", lambda project_id=None: [])

        captured_payload = {}

        def mock_create_webhook(target_url, events=None, secret=None, description=None, project_id=None):
            captured_payload["target_url"] = target_url
            captured_payload["secret"] = secret
            captured_payload["project_id"] = project_id
            captured_payload["type"] = "project" if project_id else "organization"
            return {"id": 99, **captured_payload}

        monkeypatch.setattr(client, "create_webhook", mock_create_webhook)

        res = client.setup_webhook(
            target_url="http://detector:8080",
            project_id=14,
            secret="test_secret",
        )

        assert res["action"] == "created"
        assert captured_payload["project_id"] == 14
        assert captured_payload["type"] == "project"
        assert captured_payload["secret"] == "test_secret"
