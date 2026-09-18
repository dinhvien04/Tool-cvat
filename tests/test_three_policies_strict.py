"""Comprehensive Test Suite for the 3 Strict Annotation Policies and Runtime Validator.

Architecture:
- POLICY A (14 instance labels): RECTANGLE + MASK (shared group_id; never box-only or mask-only)
- POLICY B (10 region labels): POLYGON + MASK (shared group_id; never box, never single shape)
- POLICY C (7 lane labels): POLYLINE ONLY (zero fallback to polygon/mask; drops if extraction fails)
- Output Validator: validate_cvat_output_shapes() rejects any policy violation
- Feedback Learning: CorrectionDiffEngine groups paired shapes as single semantic annotations
"""

import json
import pytest
from typing import Any, Dict, List
from PIL import Image

from core.taxonomy import (
    Taxonomy,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    BOX_MASK_LABELS,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    MASTER_31_LABELS,
    SHAPE_RECTANGLE,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
    validate_cvat_output_shapes,
    get_taxonomy,
)
from core.line_geometry import (
    ALL_LANE_LABELS,
    LANE_CROSSWALK,
    LANE_SINGLE_WHITE,
    LANE_DOUBLE_YELLOW,
    LANE_ROAD_CURB,
    lane_shape_pipeline,
    polyline_to_cvat_polyline,
    denormalize_polyline,
    extract_lane_centerline,
)
from app.service import route_phase3b_shapes, MAX_OBJECTS, MAX_REGIONS, MAX_LANES
from app.feedback import (
    CorrectionDiffEngine,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_ADD_MISSING,
    CORRECTION_NO_CHANGE,
    CORRECTION_RELABEL,
    CORRECTION_MASK_EDIT,
    CORRECTION_BOX_MOVE,
    CORRECTION_BOX_RESIZE,
    CORRECTION_REGION_EDIT,
    CORRECTION_LANE_EDIT,
)
from app.retrieval import CorrectionRetrievalEngine
from app.parser import ParsedObject
from core.vision_contract import build_full_31_prompt


@pytest.fixture
def tax() -> Taxonomy:
    return get_taxonomy()


# ==============================================================================
# 1. Schema & Taxonomy Partition Verification (31 Labels across 3 Policies)
# ==============================================================================

def test_exact_31_labels_partition(tax: Taxonomy):
    """Verify that all 31 labels belong to exactly one policy with zero overlap."""
    assert len(MASTER_31_LABELS) == 31
    assert len(BOX_MASK_LABELS) == 14
    assert len(POLYGON_MASK_LABELS) == 10
    assert len(POLYLINE_LABELS) == 7

    set_a = set(BOX_MASK_LABELS)
    set_b = set(POLYGON_MASK_LABELS)
    set_c = set(POLYLINE_LABELS)

    assert set_a.isdisjoint(set_b)
    assert set_b.isdisjoint(set_c)
    assert set_a.isdisjoint(set_c)
    assert (set_a | set_b | set_c) == set(MASTER_31_LABELS)


def test_policy_a_allowed_shapes(tax: Taxonomy):
    """Policy A (14 instance labels) must allow rectangle + mask ONLY (no polygon, no polyline)."""
    for lbl in BOX_MASK_LABELS:
        meta = tax.get_label_info(lbl)
        assert meta.policy == POLICY_BOX_MASK
        assert meta.is_box_mask is True
        assert meta.is_instance is True
        assert set(meta.allowed_shapes) == {SHAPE_RECTANGLE, SHAPE_MASK}
        assert meta.supports_bounding_box is True
        assert meta.supports_mask is True
        assert meta.supports_polygon is False
        assert meta.supports_polyline is False


def test_policy_b_allowed_shapes(tax: Taxonomy):
    """Policy B (10 region labels) must allow polygon + mask ONLY (no rectangle, no polyline)."""
    for lbl in POLYGON_MASK_LABELS:
        meta = tax.get_label_info(lbl)
        assert meta.policy == POLICY_POLYGON_MASK
        assert meta.is_polygon_mask is True
        assert meta.is_region is True
        assert set(meta.allowed_shapes) == {SHAPE_POLYGON, SHAPE_MASK}
        assert meta.supports_bounding_box is False
        assert meta.supports_mask is True
        assert meta.supports_polygon is True
        assert meta.supports_polyline is False


def test_policy_c_allowed_shapes(tax: Taxonomy):
    """Policy C (7 lane labels, including crosswalk) must allow polyline ONLY."""
    for lbl in POLYLINE_LABELS:
        meta = tax.get_label_info(lbl)
        assert meta.policy == POLICY_POLYLINE
        assert meta.is_polyline is True
        assert meta.is_lane is True
        assert meta.allowed_shapes == (SHAPE_POLYLINE,)
        assert meta.supports_polyline is True
        assert meta.supports_bounding_box is False
        assert meta.supports_mask is False
        assert meta.supports_polygon is False


# ==============================================================================
# 2. Policy C Lane Pipeline (Polyline Only, No Auto Fallbacks)
# ==============================================================================

def test_policy_c_crosswalk_centerline_extraction():
    """Verify lane/crosswalk extracts a traversal polyline centerline."""
    crosswalk_contour = [[100, 200], [800, 200], [800, 400], [100, 400]]
    shape = lane_shape_pipeline(
        label=LANE_CROSSWALK,
        contour=crosswalk_contour,
        width=1000,
        height=1000,
        confidence=0.92,
        preferred_geometry="polyline",
    )
    assert shape is not None
    assert shape["type"] == SHAPE_POLYLINE
    assert shape["label"] == LANE_CROSSWALK
    assert len(shape["points"]) >= 4


def test_policy_c_linear_lane_extraction():
    """Verify linear lane lines emit polyline centerline."""
    lane_contour = [[100, 800], [115, 800], [455, 300], [445, 300]]
    shape = lane_shape_pipeline(
        label=LANE_SINGLE_WHITE,
        contour=lane_contour,
        width=1000,
        height=1000,
        confidence=0.95,
        preferred_geometry="polyline",
    )
    assert shape is not None
    assert shape["type"] == SHAPE_POLYLINE
    assert shape["label"] == LANE_SINGLE_WHITE


def test_policy_c_fails_and_drops_without_fallback():
    """Under Policy C, if polyline centerline extraction fails, return None (NO fallback)."""
    degenerate_contour = [[100, 100], [200, 100], [200, 200], [100, 200]]  # square patch (aspect ratio 1.0 < 2.5)
    shape = lane_shape_pipeline(
        label=LANE_DOUBLE_YELLOW,
        contour=degenerate_contour,
        width=1000,
        height=1000,
        confidence=0.80,
        preferred_geometry="polyline",
        allow_fallback=False,
    )
    assert shape is None  # Strictly dropped!


def test_section_8_direct_polyline_to_cvat_polyline():
    """Section 8: Model polyline normalized [0..1000] directly converts to CVAT polyline shape.

    Preserves exact point order, skips PCA thinning / centerline resampling.
    """
    pts = [[100, 200], [250, 400], [600, 850]]
    shape = polyline_to_cvat_polyline(
        polyline=pts,
        width=1000,
        height=1000,
        label=LANE_SINGLE_WHITE,
        confidence=0.96,
    )
    assert shape is not None
    assert shape["type"] == SHAPE_POLYLINE
    assert shape["label"] == LANE_SINGLE_WHITE
    assert shape["confidence"] == "0.96"
    assert shape["points"] == [100.0, 200.0, 250.0, 400.0, 600.0, 850.0]


def test_section_8_crosswalk_direct_polyline_traversal():
    """Section 8: lane/crosswalk follows direct polyline traversal path."""
    traversal_pts = [[200, 300], [500, 310], [800, 290]]
    shape = polyline_to_cvat_polyline(
        polyline=traversal_pts,
        width=1920,
        height=1080,
        label=LANE_CROSSWALK,
        confidence=0.91,
    )
    assert shape is not None
    assert shape["type"] == SHAPE_POLYLINE
    assert shape["label"] == LANE_CROSSWALK
    assert shape["points"] == [
        round((200 / 1000.0) * 1920, 2), round((300 / 1000.0) * 1080, 2),
        round((500 / 1000.0) * 1920, 2), round((310 / 1000.0) * 1080, 2),
        round((800 / 1000.0) * 1920, 2), round((290 / 1000.0) * 1080, 2),
    ]


# ==============================================================================
# 3. Runtime Output Validator: validate_cvat_output_shapes
# ==============================================================================

def test_validator_accepts_valid_three_policy_shapes(tax: Taxonomy):
    """Validator accepts paired Policy A (box+mask), paired Policy B (poly+mask), and Policy C (polyline)."""
    shapes = [
        # Policy A: car (paired box + mask sharing group_id 1)
        {
            "label": "car",
            "type": "rectangle",
            "points": [100.0, 150.0, 300.0, 350.0],
            "group_id": 1,
        },
        {
            "label": "car",
            "type": "mask",
            "points": [100.0, 150.0, 300.0, 350.0],
            "mask": [1, 0, 1, 100, 150, 300, 350],
            "group_id": 1,
        },
        # Policy B: road (paired polygon + mask sharing group_id 2)
        {
            "label": "road",
            "type": "polygon",
            "points": [0.0, 500.0, 1000.0, 500.0, 1000.0, 1000.0, 0.0, 1000.0],
            "group_id": 2,
        },
        {
            "label": "road",
            "type": "mask",
            "points": [0.0, 500.0, 1000.0, 1000.0],
            "mask": [1, 1, 1, 0, 500, 1000, 1000],
            "group_id": 2,
        },
        # Policy C: lane/single white (polyline ONLY)
        {
            "label": "lane/single white",
            "type": "polyline",
            "points": [400.0, 500.0, 420.0, 700.0, 450.0, 1000.0],
        },
    ]

    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 5
    assert len(warnings) == 0


def test_validator_rejects_policy_a_box_only(tax: Taxonomy):
    """Validator drops Policy A instance if mask is missing."""
    shapes = [
        {
            "label": "pedestrian",
            "type": "rectangle",
            "points": [50.0, 100.0, 80.0, 200.0],
            "group_id": 1,
        }
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_a_violation" in w for w in warnings)


def test_validator_rejects_policy_a_mask_only(tax: Taxonomy):
    """Validator drops Policy A instance if rectangle is missing."""
    shapes = [
        {
            "label": "truck",
            "type": "mask",
            "points": [50.0, 100.0, 280.0, 400.0],
            "mask": [1, 1, 50, 100, 280, 400],
            "group_id": 1,
        }
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_a_violation" in w for w in warnings)


def test_validator_rejects_policy_b_polygon_only(tax: Taxonomy):
    """Validator drops Policy B region if mask is missing."""
    shapes = [
        {
            "label": "sidewalk",
            "type": "polygon",
            "points": [10.0, 400.0, 200.0, 400.0, 200.0, 600.0, 10.0, 600.0],
            "group_id": 10,
        }
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_b_violation" in w for w in warnings)


def test_validator_rejects_policy_b_bounding_box(tax: Taxonomy):
    """Validator drops Policy B region if rectangle is present."""
    shapes = [
        {
            "label": "vegetation",
            "type": "rectangle",
            "points": [10.0, 10.0, 300.0, 300.0],
            "group_id": 11,
        },
        {
            "label": "vegetation",
            "type": "mask",
            "points": [10.0, 10.0, 300.0, 300.0],
            "mask": [1, 10, 10, 300, 300],
            "group_id": 11,
        }
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_b_violation" in w for w in warnings)


def test_validator_rejects_policy_c_non_polyline(tax: Taxonomy):
    """Validator drops any lane marking that is not a polyline."""
    shapes = [
        {
            "label": "lane/crosswalk",
            "type": "polygon",
            "points": [100.0, 200.0, 400.0, 200.0, 400.0, 300.0, 100.0, 300.0],
        },
        {
            "label": "lane/double yellow",
            "type": "rectangle",
            "points": [500.0, 400.0, 520.0, 800.0],
        }
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert len(warnings) == 2
    assert all("policy_c_violation" in w for w in warnings)


# ==============================================================================
# 4. route_phase3b_shapes Integration
# ==============================================================================

def test_route_phase3b_shapes_enforces_three_policies():
    """Verify route_phase3b_shapes filters out non-compliant shapes and validates output."""
    raw_shapes = [
        # Valid Policy A pair
        {"label": "bus", "type": "rectangle", "points": [100.0, 100.0, 400.0, 500.0], "group_id": 1},
        {"label": "bus", "type": "mask", "points": [100.0, 100.0, 400.0, 500.0], "mask": [1, 100, 100, 400, 500], "group_id": 1},
        # Invalid Policy B box (suppressed)
        {"label": "building", "type": "rectangle", "points": [0.0, 0.0, 500.0, 400.0], "group_id": 2},
        # Valid Policy B pair
        {"label": "sky", "type": "polygon", "points": [0.0, 0.0, 1000.0, 0.0, 1000.0, 300.0, 0.0, 300.0], "group_id": 3},
        {"label": "sky", "type": "mask", "points": [0.0, 0.0, 1000.0, 300.0], "mask": [1, 0, 0, 1000, 300], "group_id": 3},
        # Valid Policy C polyline
        {"label": "lane/road curb", "type": "polyline", "points": [50.0, 600.0, 70.0, 900.0]},
        # Invalid Policy C mask (suppressed)
        {"label": "lane/single yellow", "type": "mask", "points": [50.0, 600.0, 70.0, 900.0], "mask": [1, 50, 600, 70, 900]},
    ]

    routed, warnings = route_phase3b_shapes(raw_shapes)
    # Expected in routed: bus (box+mask), sky (poly+mask), lane/road curb (polyline) = 5 shapes
    assert len(routed) == 5
    types_by_label = {s["label"]: [] for s in routed}
    for s in routed:
        types_by_label[s["label"]].append(s["type"])

    assert set(types_by_label["bus"]) == {"rectangle", "mask"}
    assert set(types_by_label["sky"]) == {"polygon", "mask"}
    assert types_by_label["lane/road curb"] == ["polyline"]


# ==============================================================================
# 5. Feedback Learning (CorrectionDiffEngine) Paired Shape Grouping
# ==============================================================================

def test_diff_engine_treats_paired_shapes_as_single_annotation():
    """Verify that deleting a paired (box+mask) AI annotation results in 1 diff item, not 2."""
    ai_shapes = [
        # Car with paired box + mask
        {"label": "car", "type": "rectangle", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 1},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 1},
    ]
    human_shapes: List[Dict[str, Any]] = []  # Annotator deleted the car

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_DELETE_FALSE_POSITIVE
    assert diffs[0].ai_label == "car"
    assert diffs[0].details.get("group_id") == 1
    assert diffs[0].details.get("sub_shapes_count") == 2


def test_diff_engine_human_added_paired_annotation():
    """Verify that adding a paired (polygon+mask) human annotation results in 1 diff item, not 2."""
    ai_shapes: List[Dict[str, Any]] = []
    human_shapes = [
        # Road with paired polygon + mask
        {"label": "road", "type": "polygon", "points": [0.0, 500.0, 1000.0, 500.0, 1000.0, 1000.0, 0.0, 1000.0], "group_id": 5},
        {"label": "road", "type": "mask", "points": [0.0, 500.0, 1000.0, 1000.0], "mask": [1, 0, 500, 1000, 1000], "group_id": 5},
    ]

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_ADD_MISSING
    assert diffs[0].human_label == "road"
    assert diffs[0].details.get("group_id") == 5
    assert diffs[0].details.get("sub_shapes_count") == 2


def test_diff_engine_paired_shapes_accepted_no_change():
    """Verify that identical paired shapes between AI and human produce NO_CHANGE."""
    ai_shapes = [
        {"label": "bicycle", "type": "rectangle", "points": [50.0, 50.0, 150.0, 150.0], "group_id": 1},
        {"label": "bicycle", "type": "mask", "points": [50.0, 50.0, 150.0, 150.0], "mask": [1, 50, 50, 150, 150], "group_id": 1},
    ]
    human_shapes = [
        {"label": "bicycle", "type": "rectangle", "points": [50.0, 50.0, 150.0, 150.0], "group_id": 1},
        {"label": "bicycle", "type": "mask", "points": [50.0, 50.0, 150.0, 150.0], "mask": [1, 50, 50, 150, 150], "group_id": 1},
    ]

    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_NO_CHANGE
    assert diffs[0].ai_label == "bicycle"


# ==============================================================================
# 6. Specific 13 Regression Tests for 3 Strict Annotation Policies
# ==============================================================================

def test_policy_a_drops_incomplete_instance_atomic(tax: Taxonomy):
    """Regression Test 1: Incomplete Policy A instance (missing box or missing mask) is dropped atomically."""
    # Rectangle only -> dropped
    box_only = [
        {"label": "pedestrian", "type": "rectangle", "points": [10.0, 20.0, 30.0, 40.0], "group_id": 1}
    ]
    val_box, warn_box = validate_cvat_output_shapes(box_only, taxonomy=tax)
    assert len(val_box) == 0
    assert any("policy_a_violation" in w for w in warn_box)

    routed_box, _ = route_phase3b_shapes(box_only)
    assert len(routed_box) == 0

    # Mask only -> dropped
    mask_only = [
        {"label": "pedestrian", "type": "mask", "points": [10.0, 20.0, 30.0, 40.0], "mask": [1, 10, 20, 30, 40], "group_id": 1}
    ]
    val_mask, warn_mask = validate_cvat_output_shapes(mask_only, taxonomy=tax)
    assert len(val_mask) == 0
    assert any("policy_a_violation" in w for w in warn_mask)

    routed_mask, _ = route_phase3b_shapes(mask_only)
    assert len(routed_mask) == 0


def test_policy_a_emits_both_rectangle_and_mask_with_shared_group_id(tax: Taxonomy):
    """Regression Test 2: Valid Policy A instance emits both rectangle and mask sharing exact group_id."""
    raw = [
        {"label": "car", "type": "rectangle", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 42},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 42},
    ]
    validated, warnings = validate_cvat_output_shapes(raw, taxonomy=tax)
    assert len(validated) == 2
    assert len(warnings) == 0
    assert validated[0]["group_id"] == 42
    assert validated[1]["group_id"] == 42
    assert {validated[0]["type"], validated[1]["type"]} == {"rectangle", "mask"}


def test_policy_b_drops_rectangle_and_emits_polygon_plus_mask(tax: Taxonomy):
    """Regression Test 3: Policy B regions drop any bounding box and emit polygon + mask with shared group_id."""
    raw = [
        {"label": "road", "type": "rectangle", "points": [0.0, 500.0, 1000.0, 1000.0], "group_id": 7},
        {"label": "road", "type": "polygon", "points": [0.0, 500.0, 1000.0, 500.0, 1000.0, 1000.0, 0.0, 1000.0], "group_id": 7},
        {"label": "road", "type": "mask", "points": [0.0, 500.0, 1000.0, 1000.0], "mask": [1, 0, 500, 1000, 1000], "group_id": 7},
    ]
    routed, warnings = route_phase3b_shapes(raw)
    assert len(routed) == 2
    types = {s["type"] for s in routed}
    assert types == {"polygon", "mask"}
    assert all(s["group_id"] == 7 for s in routed)


def test_policy_c_crosswalk_emits_polyline_only(tax: Taxonomy):
    """Regression Test 4: lane/crosswalk strictly emits polyline only (never rectangle, polygon, or mask)."""
    meta = tax.get_label_info(LANE_CROSSWALK)
    assert meta.policy == POLICY_POLYLINE
    assert meta.allowed_shapes == (SHAPE_POLYLINE,)
    assert meta.supports_bounding_box is False
    assert meta.supports_polygon is False
    assert meta.supports_mask is False

    crosswalk_contour = [[100, 200], [800, 200], [800, 400], [100, 400]]
    shape = lane_shape_pipeline(
        label=LANE_CROSSWALK,
        contour=crosswalk_contour,
        width=1000,
        height=1000,
        confidence=0.90,
    )
    assert shape is not None
    assert shape["type"] == "polyline"
    assert shape["label"] == LANE_CROSSWALK


def test_policy_c_lane_drop_on_failure_no_fallback():
    """Regression Test 5: Lane extraction failure returns None and drops without fallback."""
    degenerate_contour = [[100, 100], [200, 100], [200, 200], [100, 200]]  # square patch (aspect ratio 1.0 < 2.5)
    shape = lane_shape_pipeline(
        label=LANE_SINGLE_WHITE,
        contour=degenerate_contour,
        width=1000,
        height=1000,
        confidence=0.85,
    )
    assert shape is None  # Strictly returns None; no fallback


def test_route_shapes_strict_policy_a(tax: Taxonomy):
    """Regression Test 6: tax.route_shapes requires both box and mask for Policy A instances."""
    # When both box and mask are present: emits both
    assert tax.route_shapes("car", requested_mode="box_and_mask", has_box=True, has_mask=True) == [
        SHAPE_RECTANGLE,
        SHAPE_MASK,
    ]
    # Incomplete instance: dropped
    assert tax.route_shapes("car", requested_mode="box_and_mask", has_box=True, has_mask=False) == []
    assert tax.route_shapes("car", requested_mode="box_and_mask", has_box=False, has_mask=True) == []
    # Pole is Policy A: emits both when complete, dropped when incomplete
    assert tax.route_shapes("pole", requested_mode="box_and_mask", has_box=True, has_mask=True) == [
        SHAPE_RECTANGLE,
        SHAPE_MASK,
    ]
    assert tax.route_shapes("pole", requested_mode="box_and_mask", has_box=True, has_mask=False) == []


def test_route_shapes_strict_policy_b(tax: Taxonomy):
    """Regression Test 7: tax.route_shapes emits polygon + mask and never rectangle for Policy B regions."""
    assert tax.route_shapes("road", requested_mode="box_and_mask", has_box=True, has_mask=True) == [
        SHAPE_POLYGON,
        SHAPE_MASK,
    ]
    assert tax.route_shapes("building", requested_mode="box_and_mask", has_box=False, has_mask=True) == [
        SHAPE_POLYGON,
        SHAPE_MASK,
    ]
    assert tax.route_shapes("vegetation", requested_mode="box_and_mask", has_box=True, has_mask=False) == []


def test_route_shapes_strict_policy_c(tax: Taxonomy):
    """Regression Test 8: tax.route_shapes emits polyline only for Policy C lanes."""
    assert tax.route_shapes("lane/single white", requested_mode="box_and_mask", has_mask=True) == [SHAPE_POLYLINE]
    assert tax.route_shapes("lane/crosswalk", requested_mode="box_and_mask", has_mask=True) == [SHAPE_POLYLINE]
    assert tax.route_shapes("lane/crosswalk", requested_mode="box_and_mask", has_box=True, has_mask=False) == []


def test_feedback_diff_paired_shape_mask_edit():
    """Regression Test 9: Human mask edit with unchanged box produces CORRECTION_MASK_EDIT."""
    ai_shapes = [
        {"label": "car", "type": "rectangle", "points": [100.0, 100.0, 200.0, 200.0], "group_id": 1},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 200.0, 200.0], "mask": [1] * 100 + [100, 100, 200, 200], "group_id": 1},
    ]
    # Human modified mask contour, box identical
    human_shapes = [
        {"label": "car", "type": "rectangle", "points": [100.0, 100.0, 200.0, 200.0], "group_id": 1},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 150.0, 150.0], "mask": [1] * 25 + [100, 100, 150, 150], "group_id": 1},
    ]
    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_MASK_EDIT
    assert diffs[0].details.get("mask_changed") is True
    assert diffs[0].details.get("box_changed") is False


def test_feedback_diff_paired_shape_box_move():
    """Regression Test 10: Human box shift with unchanged mask produces CORRECTION_BOX_MOVE."""
    ai_shapes = [
        {"label": "car", "type": "rectangle", "points": [100.0, 100.0, 200.0, 200.0], "group_id": 1},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 200.0, 200.0], "mask": [1] * 100 + [100, 100, 200, 200], "group_id": 1},
    ]
    # Human shifted box by 20px, mask unchanged
    human_shapes = [
        {"label": "car", "type": "rectangle", "points": [120.0, 120.0, 220.0, 220.0], "group_id": 1},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 200.0, 200.0], "mask": [1] * 100 + [100, 100, 200, 200], "group_id": 1},
    ]
    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_BOX_MOVE
    assert diffs[0].details.get("box_changed") is True


def test_feedback_diff_paired_shape_no_change():
    """Regression Test 11: Identical paired shapes between AI and human produce CORRECTION_NO_CHANGE."""
    ai_shapes = [
        {"label": "truck", "type": "rectangle", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 1},
        {"label": "truck", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 1},
    ]
    human_shapes = [
        {"label": "truck", "type": "rectangle", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 1},
        {"label": "truck", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 1},
    ]
    engine = CorrectionDiffEngine()
    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_NO_CHANGE
    assert diffs[0].ai_label == "truck"


def test_feedback_retrieval_visual_few_shot_policy_b_polygon_schema(tmp_path):
    """Regression Test 12: Few-shot visual retrieval uses 'polygon' key for Policy B regions."""
    crop_file = tmp_path / "crop_test_region.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(crop_file)

    class MockDb:
        def resolve_crop_path(self, p):
            return crop_file

    retriever = CorrectionRetrievalEngine(db=MockDb())
    examples = [
        {
            "id": 1,
            "crop_path": str(crop_file),
            "correction_type": "REGION_EDIT",
            "ai_label": "road",
            "human_label": "road",
            "details_json": json.dumps({
                "crop_coords": [100.0, 100.0, 300.0, 300.0],
                "human_poly": {"points": [100.0, 100.0, 300.0, 100.0, 300.0, 300.0, 100.0, 300.0]},
            }),
            "human_shape_json": json.dumps({
                "type": "polygon",
                "points": [100.0, 100.0, 300.0, 100.0, 300.0, 300.0, 100.0, 300.0],
            }),
        }
    ]

    visual = retriever._build_visual_examples(examples)
    assert len(visual) == 1
    out = visual[0]["expected_output"]
    assert "regions" in out
    assert len(out["regions"]) == 1
    assert out["regions"][0]["label"] == "road"
    assert "polygon" in out["regions"][0]
    assert "mask" not in out["regions"][0]
    assert out["objects"] == []
    assert out["lanes"] == []


def test_feedback_retrieval_visual_few_shot_policy_c_polyline_schema(tmp_path):
    """Regression Test 13: Few-shot visual retrieval uses 'polyline' key for Policy C lanes."""
    crop_file = tmp_path / "crop_test_lane.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(crop_file)

    class MockDb:
        def resolve_crop_path(self, p):
            return crop_file

    retriever = CorrectionRetrievalEngine(db=MockDb())
    examples = [
        {
            "id": 2,
            "crop_path": str(crop_file),
            "correction_type": "LANE_EDIT",
            "ai_label": "lane/single white",
            "human_label": "lane/single white",
            "details_json": json.dumps({
                "crop_coords": [100.0, 100.0, 300.0, 300.0],
            }),
            "human_shape_json": json.dumps({
                "type": "polyline",
                "points": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0],
            }),
        }
    ]

    visual = retriever._build_visual_examples(examples)
    assert len(visual) == 1
    out = visual[0]["expected_output"]
    assert "lanes" in out
    assert len(out["lanes"]) == 1
    assert out["lanes"][0]["label"] == "lane/single white"
    assert "polyline" in out["lanes"][0]
    assert "mask" not in out["lanes"][0]
    assert out["objects"] == []
    assert out["regions"] == []


def test_feedback_retrieval_visual_few_shot_policy_a_atomic_success(tmp_path):
    """Regression Test 14: Policy A visual example succeeds when BOTH box_2d and mask are present."""
    crop_file = tmp_path / "crop_test_instance.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(crop_file)

    class MockDb:
        def resolve_crop_path(self, p):
            return crop_file

    retriever = CorrectionRetrievalEngine(db=MockDb())
    examples = [
        {
            "id": 3,
            "crop_path": str(crop_file),
            "correction_type": "RELABEL",
            "ai_label": "car",
            "human_label": "truck",
            "details_json": json.dumps({
                "crop_coords": [10.0, 10.0, 90.0, 90.0],
                "human_rect": {"points": [15.0, 15.0, 85.0, 85.0]},
                "human_mask": {"points": [15.0, 15.0, 85.0, 15.0, 85.0, 85.0, 15.0, 85.0]},
            }),
            "human_shape_json": json.dumps({
                "type": "rectangle",
                "points": [15.0, 15.0, 85.0, 85.0],
            }),
        }
    ]

    visual = retriever._build_visual_examples(examples)
    assert len(visual) == 1
    out = visual[0]["expected_output"]
    assert len(out["objects"]) == 1
    assert out["objects"][0]["label"] == "truck"
    assert "box_2d" in out["objects"][0]
    assert "mask" in out["objects"][0]
    assert len(out["objects"][0]["mask"]) >= 3
    assert out["regions"] == []
    assert out["lanes"] == []


def test_feedback_retrieval_visual_few_shot_policy_a_skips_box_only(tmp_path):
    """Regression Test 15: Policy A visual example is SKIPPED when mask is missing (never box-only)."""
    crop_file = tmp_path / "crop_test_box_only.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(crop_file)

    class MockDb:
        def resolve_crop_path(self, p):
            return crop_file

    retriever = CorrectionRetrievalEngine(db=MockDb())
    examples = [
        {
            "id": 4,
            "crop_path": str(crop_file),
            "correction_type": "BOX_MOVE",
            "ai_label": "car",
            "human_label": "car",
            "details_json": json.dumps({
                "crop_coords": [10.0, 10.0, 90.0, 90.0],
                "human_rect": {"points": [15.0, 15.0, 85.0, 85.0]},
                # No mask present!
            }),
            "human_shape_json": json.dumps({
                "type": "rectangle",
                "points": [15.0, 15.0, 85.0, 85.0],
            }),
        }
    ]

    visual = retriever._build_visual_examples(examples)
    # Must skip Policy A example completely to avoid teaching remote model box-only violation
    assert len(visual) == 0


def test_feedback_retrieval_visual_few_shot_policy_a_skips_mask_only(tmp_path):
    """Regression Test 16: Policy A visual example is SKIPPED when box is missing (never mask-only)."""
    crop_file = tmp_path / "crop_test_mask_only.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(crop_file)

    class MockDb:
        def resolve_crop_path(self, p):
            return crop_file

    retriever = CorrectionRetrievalEngine(db=MockDb())
    examples = [
        {
            "id": 5,
            "crop_path": str(crop_file),
            "correction_type": "MASK_EDIT",
            "ai_label": "car",
            "human_label": "car",
            "details_json": json.dumps({
                "crop_coords": [10.0, 10.0, 90.0, 90.0],
                # No human_rect present!
                "human_mask": {"points": [15.0, 15.0, 85.0, 15.0, 85.0, 85.0, 15.0, 85.0]},
            }),
            "human_shape_json": json.dumps({
                "type": "mask",
                "points": [15.0, 15.0, 85.0, 15.0, 85.0, 85.0, 15.0, 85.0],
            }),
        }
    ]

    visual = retriever._build_visual_examples(examples)
    # Must skip Policy A example completely when box is missing
    assert len(visual) == 0


def test_feedback_retrieval_visual_few_shot_negative_delete_false_positive(tmp_path):
    """Regression Test 17: DELETE_FALSE_POSITIVE emits clean empty arrays across all 3 policies."""
    crop_file = tmp_path / "crop_test_fp.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(crop_file)

    class MockDb:
        def resolve_crop_path(self, p):
            return crop_file

    retriever = CorrectionRetrievalEngine(db=MockDb())
    examples = [
        {
            "id": 6,
            "crop_path": str(crop_file),
            "correction_type": "DELETE_FALSE_POSITIVE",
            "ai_label": "car",
            "human_label": None,
            "details_json": json.dumps({
                "crop_coords": [10.0, 10.0, 50.0, 50.0],
            }),
            "human_shape_json": None,
        }
    ]

    visual = retriever._build_visual_examples(examples)
    assert len(visual) == 1
    out = visual[0]["expected_output"]
    assert out == {"objects": [], "regions": [], "lanes": []}
    assert "False positive" in visual[0]["description"]


# ==============================================================================
# 7. Model Vision Contract Schema & Policy Boundary Verification
# ==============================================================================

def test_vision_contract_build_full_31_prompt_schema():
    """Regression Test 18: build_full_31_prompt aligns with strict 3-tier schema."""
    prompt = build_full_31_prompt()

    # Verify presence of the 3 primary categories in output schema
    assert '"objects": [' in prompt
    assert '"regions": [' in prompt
    assert '"lanes": [' in prompt

    # Verify Policy A: box_2d + mask
    assert '"box_2d": [ymin, xmin, ymax, xmax]' in prompt
    assert '"mask": [[x, y], [x, y], ...]' in prompt

    # Verify Policy B: polygon (never mask, never box_2d in output schema)
    assert '"polygon": [[x, y], [x, y], ...]' in prompt

    # Verify Policy C: polyline (never polygon, never mask, never box_2d in output schema)
    assert '"polyline": [[x, y], [x, y], ...]' in prompt

    # Verify strict negative rules in prompt text
    assert "Do NOT provide box_2d or mask for regions" in prompt
    assert "Do NOT provide polygon, mask, or box_2d for lanes" in prompt


# ==============================================================================
# 8. Policy-Specific Candidate Pairing in CorrectionDiffEngine
# ==============================================================================

def test_lane_candidate_pairing_uses_stroke_overlap_not_bbox():
    """Regression Test 19: Lane pairing in diff() uses stroke overlap rather than bbox overlap.

    Two lanes with intersecting bounding boxes but zero stroke overlap must NOT be paired.
    """
    engine = CorrectionDiffEngine()

    # Two diagonal polylines forming an 'X' shape within the same bounding box [100, 100, 500, 500]
    # Line 1: Top-left to bottom-right
    ai_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [100.0, 100.0, 500.0, 500.0],
    }
    # Line 2: Shifted parallel line with zero stroke overlap despite bounding box overlap
    # Shifted horizontally by 80px: [180, 100] to [580, 500]
    human_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [250.0, 100.0, 650.0, 500.0],
    }

    # When line stroke width is small (e.g. 5px), their dilated strokes have 0 overlap
    diffs = engine.diff([ai_lane], [human_lane], image_width=1000, image_height=1000)

    # Since stroke overlap is 0.0 (< min_iou_match=0.30), they must NOT be matched as same-line edit
    # Instead, one is DELETE_FALSE_POSITIVE and one is ADD_MISSING
    corr_types = {d.correction_type for d in diffs}
    assert CORRECTION_DELETE_FALSE_POSITIVE in corr_types
    assert CORRECTION_ADD_MISSING in corr_types
    assert CORRECTION_LANE_EDIT not in corr_types


def test_lane_candidate_pairing_with_overlapping_strokes_matches():
    """Regression Test 20: Lane lines with overlapping strokes ARE matched as candidates."""
    engine = CorrectionDiffEngine()

    # Nearly identical polylines with high stroke overlap
    ai_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    }
    human_lane = {
        "type": "polyline",
        "label": "lane/single white",
        "points": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    }

    diffs = engine.diff([ai_lane], [human_lane], image_width=1000, image_height=1000)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_NO_CHANGE
    assert diffs[0].ai_label == "lane/single white"
    assert diffs[0].iou >= 0.90


def test_region_candidate_pairing_uses_true_shape_overlap_not_bbox():
    """Regression Test 21: Region pairing in diff() uses true shape overlap rather than bbox overlap.

    Two regions sharing bounding box but having zero pixel overlap must NOT be paired.
    """
    engine = CorrectionDiffEngine()

    # Two non-overlapping triangles that share the same bounding box [100, 100, 500, 500]
    # Triangle 1: Top-left triangle
    ai_region = {
        "type": "polygon",
        "label": "road",
        "points": [100.0, 100.0, 400.0, 100.0, 100.0, 400.0],
    }
    # Triangle 2: Bottom-right triangle (disjoint from Triangle 1)
    human_region = {
        "type": "polygon",
        "label": "road",
        "points": [500.0, 200.0, 500.0, 500.0, 200.0, 500.0],
    }

    diffs = engine.diff([ai_region], [human_region], image_width=1000, image_height=1000)

    # Disjoint shape means shape IoU is 0.0 -> must NOT pair into REGION_EDIT or NO_CHANGE
    corr_types = {d.correction_type for d in diffs}
    assert CORRECTION_DELETE_FALSE_POSITIVE in corr_types
    assert CORRECTION_ADD_MISSING in corr_types
    assert CORRECTION_REGION_EDIT not in corr_types


def test_region_candidate_pairing_with_overlapping_shape_matches():
    """Regression Test 22: Regions with overlapping polygon shapes ARE matched as candidate pairs."""
    engine = CorrectionDiffEngine()

    region1 = {
        "type": "polygon",
        "label": "vegetation",
        "points": [100.0, 100.0, 400.0, 100.0, 400.0, 400.0, 100.0, 400.0],
    }
    region2 = {
        "type": "polygon",
        "label": "vegetation",
        "points": [100.0, 100.0, 400.0, 100.0, 400.0, 400.0, 100.0, 400.0],
    }

    diffs = engine.diff([region1], [region2], image_width=1000, image_height=1000)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_NO_CHANGE
    assert diffs[0].ai_label == "vegetation"
    assert diffs[0].iou >= 0.95


# ==============================================================================
# 9. Policy B Independent Geometry Diffing (Polygon & Mask)
# ==============================================================================

def test_policy_b_independent_polygon_and_mask_evaluation():
    """Regression Test 23: Policy B paired shapes evaluate polygon and mask independently with diagnostic bbox."""
    engine = CorrectionDiffEngine()

    # Paired AI region (polygon + mask)
    ai_shapes = [
        {"label": "road", "type": "polygon", "points": [0.0, 500.0, 1000.0, 500.0, 1000.0, 1000.0, 0.0, 1000.0], "group_id": 10},
        {"label": "road", "type": "mask", "points": [0.0, 500.0, 1000.0, 1000.0], "mask": [1, 0, 500, 1000, 1000], "group_id": 10},
    ]
    # Human modified polygon contour, mask identical
    human_shapes = [
        {"label": "road", "type": "polygon", "points": [0.0, 600.0, 1000.0, 600.0, 1000.0, 1000.0, 0.0, 1000.0], "group_id": 10},
        {"label": "road", "type": "mask", "points": [0.0, 500.0, 1000.0, 1000.0], "mask": [1, 0, 500, 1000, 1000], "group_id": 10},
    ]

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]

    assert item.correction_type == CORRECTION_REGION_EDIT
    assert "polygon_changed" in item.details
    assert "mask_changed" in item.details
    assert "polygon_iou" in item.details
    assert "mask_iou" in item.details
    assert "shape_iou" in item.details
    assert "polygon_similarity" in item.details
    assert "polygon_mask_iou" in item.details
    assert "bbox_iou_diagnostic" in item.details
    assert "bbox_iou" in item.details

    assert item.details["polygon_changed"] is True
    assert item.details["mask_changed"] is False
    assert item.details["polygon_iou"] < 0.90
    assert item.details["mask_iou"] >= 0.95


def test_policy_b_mask_edit_only_triggers_region_edit():
    """Regression Test 24: Editing only the mask contour in Policy B triggers REGION_EDIT."""
    engine = CorrectionDiffEngine()

    ai_shapes = [
        {"label": "building", "type": "polygon", "points": [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0], "group_id": 20},
        {"label": "building", "type": "mask", "points": [100.0, 100.0, 500.0, 500.0], "mask": [1, 100, 100, 500, 500], "group_id": 20},
    ]
    # Human modified mask, polygon identical
    human_shapes = [
        {"label": "building", "type": "polygon", "points": [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0], "group_id": 20},
        {"label": "building", "type": "mask", "points": [200.0, 200.0, 400.0, 400.0], "mask": [1, 200, 200, 400, 400], "group_id": 20},
    ]

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]

    assert item.correction_type == CORRECTION_REGION_EDIT
    assert item.details["polygon_changed"] is False
    assert item.details["mask_changed"] is True


def test_policy_b_paired_shapes_accepted_no_change():
    """Regression Test 25: Both polygon and mask identical produce CORRECTION_NO_CHANGE."""
    engine = CorrectionDiffEngine()

    shapes = [
        {"label": "sidewalk", "type": "polygon", "points": [50.0, 400.0, 300.0, 400.0, 300.0, 600.0, 50.0, 600.0], "group_id": 30},
        {"label": "sidewalk", "type": "mask", "points": [50.0, 400.0, 300.0, 600.0], "mask": [1, 50, 400, 300, 600], "group_id": 30},
    ]

    diffs = engine.diff(shapes, shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_NO_CHANGE
    assert diffs[0].details["polygon_changed"] is False
    assert diffs[0].details["mask_changed"] is False


# ==============================================================================
# 10. Policy A Independent Geometry Diffing (Box & Mask)
# ==============================================================================

def test_policy_a_independent_box_and_mask_evaluation():
    """Regression Test 26: Policy A paired shapes evaluate box and mask independently."""
    engine = CorrectionDiffEngine()

    ai_shapes = [
        {"label": "car", "type": "rectangle", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 40},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 40},
    ]
    # Human shifted rectangle, mask unchanged
    human_shapes = [
        {"label": "car", "type": "rectangle", "points": [130.0, 130.0, 330.0, 330.0], "group_id": 40},
        {"label": "car", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 40},
    ]

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]

    assert item.correction_type == CORRECTION_BOX_MOVE
    assert item.details["box_changed"] is True
    assert item.details["mask_changed"] is False
    assert "box_iou" in item.details
    assert "mask_iou" in item.details
    assert "shift" in item.details
    assert "size_diff" in item.details


def test_policy_a_box_resize_detected():
    """Regression Test 27: Resizing bounding box by >15% triggers CORRECTION_BOX_RESIZE."""
    engine = CorrectionDiffEngine()

    ai_shapes = [
        {"label": "bus", "type": "rectangle", "points": [100.0, 100.0, 300.0, 300.0], "group_id": 50},
        {"label": "bus", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 50},
    ]
    # Human enlarged box width and height significantly
    human_shapes = [
        {"label": "bus", "type": "rectangle", "points": [100.0, 100.0, 450.0, 450.0], "group_id": 50},
        {"label": "bus", "type": "mask", "points": [100.0, 100.0, 300.0, 300.0], "mask": [1, 100, 100, 300, 300], "group_id": 50},
    ]

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)
    assert len(diffs) == 1
    item = diffs[0]

    assert item.correction_type == CORRECTION_BOX_RESIZE
    assert item.details["box_changed"] is True
    assert item.details["box_resized"] is True


def test_pole_strictly_follows_policy_a_paired_shape_contract(tax: Taxonomy):
    """Regression Test 28: Pole is strictly Policy A (requires paired rectangle and mask, dropped if incomplete)."""
    meta = tax.get_label_info("pole")
    assert meta.policy == POLICY_BOX_MASK
    assert meta.is_box_mask is True
    assert meta.is_instance is True
    assert set(meta.allowed_shapes) == {SHAPE_RECTANGLE, SHAPE_MASK}
    assert meta.supports_bounding_box is True
    assert meta.supports_mask is True
    assert meta.supports_polygon is False
    assert meta.supports_polyline is False

    # Standalone pole box without mask -> dropped atomically
    box_only = [{"label": "pole", "type": "rectangle", "points": [100.0, 100.0, 120.0, 500.0], "group_id": 1}]
    validated, warnings = validate_cvat_output_shapes(box_only, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_a_violation" in w for w in warnings)

    # Standalone pole mask without box -> dropped atomically
    mask_only = [{"label": "pole", "type": "mask", "points": [100.0, 100.0, 120.0, 500.0], "mask": [1, 100, 100, 120, 500], "group_id": 1}]
    validated, warnings = validate_cvat_output_shapes(mask_only, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_a_violation" in w for w in warnings)

    # Complete paired pole (box + mask sharing group_id) -> accepted!
    paired = [
        {"label": "pole", "type": "rectangle", "points": [100.0, 100.0, 120.0, 500.0], "group_id": 1},
        {"label": "pole", "type": "mask", "points": [100.0, 100.0, 120.0, 500.0], "mask": [1, 100, 100, 120, 500], "group_id": 1},
    ]
    validated, warnings = validate_cvat_output_shapes(paired, taxonomy=tax)
    assert len(validated) == 2
    assert len(warnings) == 0


