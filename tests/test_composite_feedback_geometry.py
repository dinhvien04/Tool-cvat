"""Test suite for Composite Feedback Geometry (Policy A, Policy B, and Paired Shape Storage/Reconstruction).

Audits and validates:
1. Policy B composite diffing:
   - Independent polygon similarity / shape_iou and mask_iou.
   - BBox IoU kept for diagnostics only.
   - polygon unchanged, mask unchanged -> NO_CHANGE
   - polygon changed, mask unchanged -> REGION_EDIT
   - polygon unchanged, mask changed -> REGION_EDIT
   - both changed -> REGION_EDIT
   - Details stores polygon_mask_iou / shape_iou, mask_iou, bbox_iou_diagnostic, polygon_changed, mask_changed.
2. Policy A composite diffing:
   - box unchanged, mask changed -> MASK_EDIT
   - box moved, mask unchanged -> BOX_MOVE
   - box resized, mask unchanged -> BOX_RESIZE
   - box changed AND mask changed -> details records BOTH flags, box_iou, mask_iou, box_change_type.
3. Paired shape storage & reconstruction:
   - FeedbackDatabase / CorrectionDatabase stores and reconstructs BOTH member shapes for Policy A and Policy B.
   - Reconstructs for same-label edits, RELABEL, and ADD_MISSING.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_BOX_MOVE,
    CORRECTION_BOX_RESIZE,
    CORRECTION_MASK_EDIT,
    CORRECTION_NO_CHANGE,
    CORRECTION_REGION_EDIT,
    CORRECTION_RELABEL,
    CorrectionDatabase,
    CorrectionDiffEngine,
    FeedbackDatabase,
    reconstruct_shapes,
)


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================


def make_policy_b_pair(
    label: str,
    group_id: int,
    poly_points: List[float],
    mask_bbox: List[int],
) -> List[Dict[str, Any]]:
    """Build paired Policy B annotation (polygon + mask) sharing group_id."""
    x1, y1, x2, y2 = mask_bbox
    # CVAT mask flat format: [rle_length, rle_val..., x1, y1, x2, y2]
    w = x2 - x1
    h = y2 - y1
    mask_flat = [1] * (w * h) + [x1, y1, x2, y2]
    return [
        {
            "label": label,
            "type": "polygon",
            "points": poly_points,
            "group_id": group_id,
        },
        {
            "label": label,
            "type": "mask",
            "points": [float(x1), float(y1), float(x2), float(y2)],
            "mask": mask_flat,
            "group_id": group_id,
        },
    ]


def make_policy_a_pair(
    label: str,
    group_id: int,
    box_points: List[float],
    mask_bbox: List[int],
) -> List[Dict[str, Any]]:
    """Build paired Policy A annotation (rectangle + mask) sharing group_id."""
    x1, y1, x2, y2 = mask_bbox
    w = x2 - x1
    h = y2 - y1
    mask_flat = [1] * (w * h) + [x1, y1, x2, y2]
    return [
        {
            "label": label,
            "type": "rectangle",
            "points": box_points,
            "group_id": group_id,
        },
        {
            "label": label,
            "type": "mask",
            "points": [float(x1), float(y1), float(x2), float(y2)],
            "mask": mask_flat,
            "group_id": group_id,
        },
    ]


# =============================================================================
# 1. Critical Bug #3: Policy B Composite Geometry Tests
# =============================================================================


def test_policy_b_both_unchanged_evaluates_to_no_change():
    """Policy B: Identical polygon and identical mask -> NO_CHANGE."""
    poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    ai = make_policy_b_pair("road", 1, poly, [100, 100, 500, 500])
    human = make_policy_b_pair("road", 1, poly, [100, 100, 500, 500])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_NO_CHANGE
    assert item.details["polygon_changed"] is False
    assert item.details["mask_changed"] is False
    assert item.details["shape_iou"] >= 0.98
    assert item.details["mask_iou"] >= 0.98
    assert "bbox_iou_diagnostic" in item.details
    assert "polygon_mask_iou" in item.details


def test_policy_b_polygon_changed_mask_unchanged_evaluates_to_region_edit():
    """Policy B: Polygon contour edited by annotator while mask unchanged -> REGION_EDIT."""
    ai_poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]  # square
    # Human changed polygon to a triangle
    human_poly = [100.0, 100.0, 500.0, 100.0, 300.0, 500.0]

    ai = make_policy_b_pair("road", 1, ai_poly, [100, 100, 500, 500])
    human = make_policy_b_pair("road", 1, human_poly, [100, 100, 500, 500])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_REGION_EDIT
    assert item.details["polygon_changed"] is True
    assert item.details["mask_changed"] is False
    assert item.details["shape_iou"] < 0.85
    assert item.details["mask_iou"] >= 0.98
    assert "polygon contour edited" in item.details["reason"]


def test_policy_b_polygon_unchanged_mask_changed_evaluates_to_region_edit():
    """Policy B: Polygon unchanged while mask boundary edited by annotator -> REGION_EDIT."""
    poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    # Mask changed from 400x400 to 200x200
    ai = make_policy_b_pair("sidewalk", 2, poly, [100, 100, 500, 500])
    human = make_policy_b_pair("sidewalk", 2, poly, [100, 100, 300, 300])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_REGION_EDIT
    assert item.details["polygon_changed"] is False
    assert item.details["mask_changed"] is True
    assert item.details["shape_iou"] >= 0.98
    assert item.details["mask_iou"] < 0.50
    assert "mask boundary edited" in item.details["reason"]


def test_policy_b_both_changed_evaluates_to_region_edit():
    """Policy B: Both polygon and mask edited -> REGION_EDIT with both flags true."""
    ai_poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    human_poly = [100.0, 100.0, 500.0, 100.0, 300.0, 500.0]
    ai = make_policy_b_pair("building", 3, ai_poly, [100, 100, 500, 500])
    human = make_policy_b_pair("building", 3, human_poly, [100, 100, 300, 300])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_REGION_EDIT
    assert item.details["polygon_changed"] is True
    assert item.details["mask_changed"] is True
    assert item.details["shape_iou"] < 0.85
    assert item.details["mask_iou"] < 0.50
    assert "both polygon" in item.details["reason"]


# =============================================================================
# 2. Policy A Composite Geometry Tests
# =============================================================================


def test_policy_a_box_unchanged_mask_changed_evaluates_to_mask_edit():
    """Policy A: Box identical, mask boundary modified -> MASK_EDIT."""
    box = [100.0, 100.0, 300.0, 300.0]
    ai = make_policy_a_pair("car", 1, box, [100, 100, 300, 300])
    # Human changed mask to half size
    human = make_policy_a_pair("car", 1, box, [100, 100, 200, 200])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_MASK_EDIT
    assert item.details["box_changed"] is False
    assert item.details["mask_changed"] is True
    assert item.details["box_iou"] >= 0.99
    assert item.details["mask_iou"] < 0.50


def test_policy_a_box_moved_mask_unchanged_evaluates_to_box_move():
    """Policy A: Box shifted by annotator, mask unchanged -> BOX_MOVE."""
    ai_box = [100.0, 100.0, 300.0, 300.0]
    human_box = [130.0, 130.0, 330.0, 330.0]  # shifted by 30px
    mask = [100, 100, 300, 300]

    ai = make_policy_a_pair("truck", 1, ai_box, mask)
    human = make_policy_a_pair("truck", 1, human_box, mask)

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_BOX_MOVE
    assert item.details["box_changed"] is True
    assert item.details["mask_changed"] is False
    assert item.details.get("box_moved") is True


def test_policy_a_box_resized_mask_unchanged_evaluates_to_box_resize():
    """Policy A: Box dimensions resized by >15%, mask unchanged -> BOX_RESIZE."""
    ai_box = [100.0, 100.0, 200.0, 200.0]  # 100x100
    human_box = [100.0, 100.0, 250.0, 250.0]  # 150x150 (50% increase)
    mask = [100, 100, 200, 200]

    ai = make_policy_a_pair("bus", 1, ai_box, mask)
    human = make_policy_a_pair("bus", 1, human_box, mask)

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_BOX_RESIZE
    assert item.details["box_changed"] is True
    assert item.details["mask_changed"] is False
    assert item.details.get("box_resized") is True


def test_policy_a_both_box_and_mask_changed_records_both_flags():
    """Policy A: Both box and mask modified -> details records both flags and neither is lost."""
    ai_box = [100.0, 100.0, 200.0, 200.0]
    human_box = [100.0, 100.0, 260.0, 260.0]  # resized box
    ai_mask = [100, 100, 200, 200]
    human_mask = [100, 100, 150, 150]  # modified mask

    ai = make_policy_a_pair("car", 1, ai_box, ai_mask)
    human = make_policy_a_pair("car", 1, human_box, human_mask)

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.details["box_changed"] is True
    assert item.details["mask_changed"] is True
    assert item.details["box_change_type"] in (CORRECTION_BOX_RESIZE, CORRECTION_BOX_MOVE)
    assert "box_iou" in item.details
    assert "mask_iou" in item.details
    assert item.details["mask_iou"] < 0.60


# =============================================================================
# 3. Storage and Reconstruction of Paired Shapes
# =============================================================================


def test_storage_and_reconstruction_policy_a(tmp_path: Path):
    """Verify storing and reconstructing BOTH rect and mask for Policy A."""
    db = FeedbackDatabase(
        db_path=tmp_path / "test_fb.sqlite3",
        examples_dir=tmp_path / "examples",
    )
    assert CorrectionDatabase is FeedbackDatabase

    ai = make_policy_a_pair("car", 10, [100.0, 100.0, 200.0, 200.0], [100, 100, 200, 200])
    human = make_policy_a_pair("car", 10, [100.0, 100.0, 200.0, 200.0], [100, 100, 180, 180])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human)
    assert len(diffs) == 1

    img = Image.new("RGB", (500, 500), color="blue")
    c_ids = db.record_corrections(diffs, image_hash="hash_a", image=img)
    assert len(c_ids) == 1

    records = db.get_corrections(limit=1)
    assert len(records) == 1
    rec = records[0]

    # Reconstruct human shapes
    human_reconstructed = reconstruct_shapes(rec, target="human")
    assert len(human_reconstructed) == 2
    types = {s["type"] for s in human_reconstructed}
    assert types == {"rectangle", "mask"}

    # Reconstruct AI shapes
    ai_reconstructed = reconstruct_shapes(rec, target="ai")
    assert len(ai_reconstructed) == 2
    ai_types = {s["type"] for s in ai_reconstructed}
    assert ai_types == {"rectangle", "mask"}


def test_storage_and_reconstruction_policy_b(tmp_path: Path):
    """Verify storing and reconstructing BOTH poly and mask for Policy B."""
    db = FeedbackDatabase(
        db_path=tmp_path / "test_fb.sqlite3",
        examples_dir=tmp_path / "examples",
    )

    ai = make_policy_b_pair("vegetation", 20, [0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0], [0, 0, 100, 100])
    human = make_policy_b_pair("vegetation", 20, [0.0, 0.0, 80.0, 0.0, 80.0, 80.0, 0.0, 80.0], [0, 0, 80, 80])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human)
    assert len(diffs) == 1

    c_ids = db.record_corrections(diffs, image_hash="hash_b")
    assert len(c_ids) == 1

    records = db.get_corrections(limit=1)
    rec = records[0]

    human_reconstructed = db.reconstruct_shapes(rec, target="human")
    assert len(human_reconstructed) == 2
    types = {s["type"] for s in human_reconstructed}
    assert types == {"polygon", "mask"}


def test_storage_and_reconstruction_add_missing(tmp_path: Path):
    """Verify ADD_MISSING stores and reconstructs paired shapes (both rect+mask or poly+mask)."""
    db = FeedbackDatabase(
        db_path=tmp_path / "test_fb.sqlite3",
        examples_dir=tmp_path / "examples",
    )

    ai_shapes: List[Dict[str, Any]] = []
    # Human manually added missing bicycle (Policy A) and missing road (Policy B)
    human_shapes = (
        make_policy_a_pair("bicycle", 30, [50.0, 50.0, 120.0, 120.0], [50, 50, 120, 120])
        + make_policy_b_pair("road", 31, [0.0, 400.0, 800.0, 400.0, 800.0, 800.0, 0.0, 800.0], [0, 400, 800, 800])
    )

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes)
    assert len(diffs) == 2
    assert all(d.correction_type == CORRECTION_ADD_MISSING for d in diffs)

    c_ids = db.record_corrections(diffs, image_hash="hash_missing")
    assert len(c_ids) == 2

    records = db.get_corrections(limit=10)
    assert len(records) == 2

    # Verify bicycle has rect + mask reconstructed
    bike_rec = next(r for r in records if r["human_label"] == "bicycle")
    bike_shapes = reconstruct_shapes(bike_rec, target="human")
    assert len(bike_shapes) == 2
    assert {s["type"] for s in bike_shapes} == {"rectangle", "mask"}

    # Verify road has poly + mask reconstructed
    road_rec = next(r for r in records if r["human_label"] == "road")
    road_shapes = reconstruct_shapes(road_rec, target="human")
    assert len(road_shapes) == 2
    assert {s["type"] for s in road_shapes} == {"polygon", "mask"}


def test_storage_and_reconstruction_relabel(tmp_path: Path):
    """Verify RELABEL stores and reconstructs paired shapes for both human and AI."""
    db = FeedbackDatabase(
        db_path=tmp_path / "test_fb.sqlite3",
        examples_dir=tmp_path / "examples",
    )

    # AI predicted car (rect + mask), human relabeled to truck (rect + mask)
    ai = make_policy_a_pair("car", 40, [100.0, 100.0, 300.0, 300.0], [100, 100, 300, 300])
    human = make_policy_a_pair("truck", 40, [100.0, 100.0, 300.0, 300.0], [100, 100, 300, 300])

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai, human)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_RELABEL

    db.record_corrections(diffs, image_hash="hash_relabel")
    records = db.get_corrections(limit=1)
    rec = records[0]

    human_shapes = reconstruct_shapes(rec, target="human")
    assert len(human_shapes) == 2
    assert {s["type"] for s in human_shapes} == {"rectangle", "mask"}

    ai_shapes = reconstruct_shapes(rec, target="ai")
    assert len(ai_shapes) == 2
    assert {s["type"] for s in ai_shapes} == {"rectangle", "mask"}
