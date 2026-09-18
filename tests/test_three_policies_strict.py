"""Comprehensive Test Suite for the 3 Strict Annotation Policies and Runtime Validator.

Architecture:
- POLICY A (14 instance labels): RECTANGLE + MASK (shared group_id; never box-only or mask-only)
- POLICY B (10 region labels): POLYGON + MASK (shared group_id; never box, never single shape)
- POLICY C (7 lane labels): POLYLINE ONLY (zero fallback to polygon/mask; drops if extraction fails)
- Output Validator: validate_cvat_output_shapes() rejects any policy violation
- Feedback Learning: CorrectionDiffEngine groups paired shapes as single semantic annotations
"""

import pytest
from typing import Any, Dict, List

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
    lane_shape_pipeline,
)
from app.service import route_phase3b_shapes, MAX_OBJECTS, MAX_REGIONS, MAX_LANES
from app.feedback import (
    CorrectionDiffEngine,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_ADD_MISSING,
    CORRECTION_NO_CHANGE,
    CORRECTION_RELABEL,
)


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
