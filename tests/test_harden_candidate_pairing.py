"""Regression test suite for hardened candidate pairing and diffing in app/feedback.py.

Verifies:
1. Section 13: Policy A Candidate Matching
   - Combined similarity from box_iou and mask_iou (max(box_iou, mask_iou)).
   - Significant box move with identical mask still matches as same semantic instance (BOX_MOVE).
   - Prevents spurious DELETE_FALSE_POSITIVE + ADD_MISSING.
2. Section 14: Policy B Candidate Matching
   - Evaluates BOTH polygon and mask.
   - Candidate similarity = max(polygon_iou, mask_iou) with conservative threshold.
   - Heavy polygon edit with stable mask still pairs as same region (REGION_EDIT).
   - Diffing independently records polygon_changed, mask_changed, shape_iou, polygon_iou,
     mask_iou, and bbox_iou_diagnostic.
3. Section 15: Policy C Matching
   - Dilated line stroke overlap (configurable lane_compare_width_px = 5) + average distance.
   - Coarse bbox only as diagnostic (bbox_iou_diagnostic).
4. Section 16: Cross-Label RELABEL Matching
   - Policy A (e.g. car -> truck) classifies as RELABEL with strong geometry match.
   - Policy B (e.g. terrain -> vegetation) classifies as RELABEL with strong geometry match.
   - Policy C (e.g. lane/single white -> lane/single yellow) classifies as RELABEL with strong stroke/distance match.
   - Strict 3-policy isolation preserved across groups.
"""

from __future__ import annotations

import pytest
from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_BOX_MOVE,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_LANE_EDIT,
    CORRECTION_NO_CHANGE,
    CORRECTION_REGION_EDIT,
    CORRECTION_RELABEL,
    CorrectionDiffEngine,
)


def make_policy_a_pair(label: str, gid: int, box: list[float], mask_rle_rect: list[int]) -> list[dict]:
    """Helper creating paired Policy A (rectangle + mask) annotation."""
    x1, y1, x2, y2 = mask_rle_rect
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return [
        {
            "label": label,
            "type": "rectangle",
            "points": list(box),
            "group_id": gid,
        },
        {
            "label": label,
            "type": "mask",
            "points": [float(x1), float(y1), float(x2), float(y2)],
            "mask": [1, 0, 0, w, h],
            "group_id": gid,
        },
    ]


def make_policy_b_pair(label: str, gid: int, poly_pts: list[float], mask_rle_rect: list[int]) -> list[dict]:
    """Helper creating paired Policy B (polygon + mask) annotation."""
    x1, y1, x2, y2 = mask_rle_rect
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return [
        {
            "label": label,
            "type": "polygon",
            "points": list(poly_pts),
            "group_id": gid,
        },
        {
            "label": label,
            "type": "mask",
            "points": [float(x1), float(y1), float(x2), float(y2)],
            "mask": [1, 0, 0, w, h],
            "group_id": gid,
        },
    ]


# ==============================================================================
# Section 13: Policy A Candidate Matching
# ==============================================================================


def test_section13_policy_a_box_moved_significantly_mask_identical_matches():
    """Policy A: Box moved significantly (low box IoU < 0.20) but mask identical.

    Must pair as same semantic instance and classify as BOX_MOVE,
    NOT DELETE_FALSE_POSITIVE + ADD_MISSING.
    """
    engine = CorrectionDiffEngine()

    # AI predicted box at [100, 100, 300, 300] (200x200) with mask at [100, 100, 300, 300]
    ai_shapes = make_policy_a_pair("car", 1, [100.0, 100.0, 300.0, 300.0], [100, 100, 300, 300])

    # Human moved box to [230, 230, 430, 430] (box IoU ~ 0.08, far below min_iou_match=0.30)
    # but left mask identical at [100, 100, 300, 300]
    human_shapes = make_policy_a_pair("car", 1, [230.0, 230.0, 430.0, 430.0], [100, 100, 300, 300])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_BOX_MOVE
    assert item.ai_label == "car"
    assert item.human_label == "car"
    assert item.details["box_changed"] is True
    assert item.details["box_moved"] is True
    assert item.details["mask_changed"] is False
    assert item.details["box_iou"] < 0.20
    assert item.details["mask_iou"] >= 0.95
    assert "shape_iou" in item.details
    assert "bbox_iou_diagnostic" in item.details


# ==============================================================================
# Section 14: Policy B Candidate Matching
# ==============================================================================


def test_section14_policy_b_polygon_edited_heavily_mask_stable_matches():
    """Policy B: Annotator heavily edits polygon contour (poly IoU < 0.20) but mask is stable.

    Must pair using candidate similarity = max(polygon_iou, mask_iou) and classify
    as REGION_EDIT, NOT DELETE_FALSE_POSITIVE + ADD_MISSING.
    """
    engine = CorrectionDiffEngine()

    # AI predicted large road polygon [100, 100] to [500, 500] and mask [100, 100, 500, 500]
    ai_poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    ai_shapes = make_policy_b_pair("road", 2, ai_poly, [100, 100, 500, 500])

    # Human heavily edited polygon to a thin strip [100, 100] to [150, 500] (poly IoU ~ 0.12)
    # but mask remained identical [100, 100, 500, 500]
    human_poly = [100.0, 100.0, 150.0, 100.0, 150.0, 500.0, 100.0, 500.0]
    human_shapes = make_policy_b_pair("road", 2, human_poly, [100, 100, 500, 500])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_REGION_EDIT
    assert item.ai_label == "road"
    assert item.human_label == "road"

    # Verify independent geometry tracking
    assert item.details["polygon_changed"] is True
    assert item.details["mask_changed"] is False
    assert item.details["polygon_iou"] < 0.25
    assert item.details["mask_iou"] >= 0.95
    assert "shape_iou" in item.details
    assert "polygon_iou" in item.details
    assert "mask_iou" in item.details
    assert "bbox_iou_diagnostic" in item.details


# ==============================================================================
# Section 15: Policy C Matching
# ==============================================================================


def test_section15_policy_c_dilated_stroke_and_distance_matching():
    """Policy C: Matches based on dilated stroke IoU and average distance.

    Coarse bbox is recorded as diagnostic.
    """
    engine = CorrectionDiffEngine(lane_compare_width_px=5)

    # AI lane line
    ai_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    }
    # Human lane line slightly shifted by 2px (within lane_tolerance_px=3px)
    human_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [102.0, 200.0, 302.0, 400.0, 502.0, 600.0],
    }

    diffs = engine.diff([ai_lane], [human_lane], image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_NO_CHANGE
    assert item.details["stroke_iou"] > 0.40
    assert item.details["lane_shift_px"] <= 3.0
    assert "bbox_iou_diagnostic" in item.details
    assert "bbox_iou" in item.details


# ==============================================================================
# Section 16: Cross-Label RELABEL Matching across All 3 Policies
# ==============================================================================


def test_section16_relabel_policy_a_car_to_truck():
    """Policy A cross-label: car -> truck with matching geometry classifies as RELABEL."""
    engine = CorrectionDiffEngine(min_iou_relabel=0.60)

    ai_shapes = make_policy_a_pair("car", 10, [100.0, 100.0, 300.0, 250.0], [100, 100, 300, 250])
    human_shapes = make_policy_a_pair("truck", 10, [102.0, 98.0, 305.0, 248.0], [100, 100, 300, 250])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_RELABEL
    assert item.ai_label == "car"
    assert item.human_label == "truck"
    assert item.iou >= 0.85
    assert "box_iou" in item.details
    assert "mask_iou" in item.details
    assert "bbox_iou_diagnostic" in item.details


def test_section16_relabel_policy_b_terrain_to_vegetation():
    """Policy B cross-label: terrain -> vegetation with matching shape classifies as RELABEL."""
    engine = CorrectionDiffEngine(min_iou_relabel=0.60)

    poly = [100.0, 100.0, 400.0, 100.0, 400.0, 400.0, 100.0, 400.0]
    ai_shapes = make_policy_b_pair("terrain", 20, poly, [100, 100, 400, 400])
    human_shapes = make_policy_b_pair("vegetation", 20, poly, [100, 100, 400, 400])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_RELABEL
    assert item.ai_label == "terrain"
    assert item.human_label == "vegetation"
    assert item.iou >= 0.95
    assert "shape_iou" in item.details
    assert "polygon_iou" in item.details
    assert "mask_iou" in item.details
    assert "polygon_mask_iou" in item.details
    assert "bbox_iou_diagnostic" in item.details


def test_section16_relabel_policy_c_white_to_yellow_lane():
    """Policy C cross-label: lane/single white -> lane/single yellow classifies as RELABEL."""
    engine = CorrectionDiffEngine(min_iou_relabel=0.60, lane_compare_width_px=5)

    ai_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    }
    human_lane = {
        "type": "polyline",
        "label": "lane/single yellow",
        "points": [101.0, 201.0, 301.0, 401.0, 501.0, 601.0],
    }

    diffs = engine.diff([ai_lane], [human_lane], image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]
    assert item.correction_type == CORRECTION_RELABEL
    assert item.ai_label == "lane/single white"
    assert item.human_label == "lane/single yellow"
    assert item.iou >= 0.60
    assert "stroke_iou" in item.details
    assert "lane_shift_px" in item.details
    assert "bbox_iou_diagnostic" in item.details
