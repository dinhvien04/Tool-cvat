"""Comprehensive Test Suite for Atomic Paired Outputs and Group ID Integrity (Sections 10, 11, 12).

Section 10 (Critical Atomicity):
- Policy A: Both rectangle and mask must be valid. If either fails, emit NEITHER. Both must share group_id.
- Policy B: Both polygon and mask must be valid. If either fails, emit NEITHER. Both must share group_id.
            Bounding box strictly prohibited.
- Policy C: Emit valid polyline only. Zero fallback to polygon or mask. No group needed.

Section 11 (Final Output Validator validate_cvat_output_shapes):
- Policy A: Exactly 1 rectangle and 1 mask per group. Drop incomplete groups.
- Policy B: Exactly 1 polygon and 1 mask per group. Drop incomplete groups.
- Policy C: Must be polyline only (no group needed).
- Explicitly reject:
  * car rectangle only -> drop
  * car mask only -> drop
  * road polygon only -> drop
  * road mask only -> drop
  * lane polygon -> drop
  * lane mask -> drop

Section 12 (Group ID Allocation):
- Ensure distinct instances never share the same group_id.
- Ensure Policy A paired shapes share group_id.
- Ensure Policy B paired shapes share group_id.
- Unique allocation in app/service.py preventing collisions.
"""

import pytest
from typing import Any, Dict, List

from core.vision_contract import (
    DetectionResult,
    DetectedObject,
    DetectedRegion,
    DetectedLane,
)
from core.taxonomy import (
    Taxonomy,
    get_taxonomy,
    validate_cvat_output_shapes,
    SHAPE_RECTANGLE,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
)
from app.parser import ParsedObject, ParseResult
from app.service import route_phase3b_shapes


@pytest.fixture
def tax() -> Taxonomy:
    return get_taxonomy()


# ==============================================================================
# Section 10: DetectionResult.to_cvat_annotations() Atomicity
# ==============================================================================

def test_section10_policy_a_atomic_success():
    """Policy A: Valid box + valid mask contour emits both rectangle and mask sharing group_id."""
    obj = DetectedObject(
        label="car",
        box_2d=[100, 100, 400, 500],  # [ymin, xmin, ymax, xmax] in [0, 1000]
        mask=[[100, 100], [400, 100], [400, 500], [100, 500]],
        confidence=0.95,
    )
    det = DetectionResult(objects=[obj])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 2
    types = [s["type"] for s in shapes]
    assert "rectangle" in types
    assert "mask" in types
    assert shapes[0]["group_id"] == shapes[1]["group_id"]
    assert isinstance(shapes[0]["group_id"], int)
    assert shapes[0]["group_id"] > 0


def test_section10_policy_a_degenerate_box_emits_neither():
    """Policy A: Empty/degenerate bounding box (xmin >= xmax or ymin >= ymax) drops entire annotation."""
    # Degenerate box: xmin (500) == xmax (500)
    obj = DetectedObject(
        label="car",
        box_2d=[100, 500, 400, 500],
        mask=[[100, 500], [400, 500], [400, 500], [100, 500]],
        confidence=0.88,
    )
    det = DetectionResult(objects=[obj])
    shapes = det.to_cvat_annotations(width=1000, height=1000)
    assert len(shapes) == 0


def test_section10_policy_a_missing_mask_emits_neither():
    """Policy A: Missing or invalid mask contour drops entire annotation (no box-only fallback)."""
    obj = DetectedObject(
        label="pedestrian",
        box_2d=[200, 300, 600, 400],
        mask=None,  # No mask
        confidence=0.92,
    )
    det = DetectionResult(objects=[obj])
    shapes = det.to_cvat_annotations(width=1000, height=1000)
    assert len(shapes) == 0


def test_section10_policy_b_atomic_success():
    """Policy B: Valid region polygon emits polygon + mask sharing group_id, no rectangle."""
    reg = DetectedRegion(
        label="road",
        polygon=[[100, 500], [900, 500], [900, 900], [100, 900]],
        confidence=0.89,
    )
    det = DetectionResult(regions=[reg])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 2
    types = [s["type"] for s in shapes]
    assert "polygon" in types
    assert "mask" in types
    assert "rectangle" not in types
    assert shapes[0]["group_id"] == shapes[1]["group_id"]
    assert isinstance(shapes[0]["group_id"], int)


def test_section10_policy_b_degenerate_polygon_emits_neither():
    """Policy B: Collinear/zero-area polygon contour drops entire annotation."""
    # Collinear points forming zero area
    reg = DetectedRegion(
        label="sidewalk",
        polygon=[[100, 100], [200, 200], [300, 300]],
        confidence=0.75,
    )
    det = DetectionResult(regions=[reg])
    shapes = det.to_cvat_annotations(width=1000, height=1000)
    assert len(shapes) == 0


def test_section10_policy_c_polyline_only_no_group():
    """Policy C: Lane emits polyline only, zero fallback, and no group_id."""
    lane = DetectedLane(
        label="lane/single white",
        polyline=[[200, 500], [300, 700], [400, 900]],
        confidence=0.91,
    )
    det = DetectionResult(lanes=[lane])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 1
    assert shapes[0]["type"] == "polyline"
    assert shapes[0]["label"] == "lane/single white"
    assert "group_id" not in shapes[0]


def test_section10_policy_c_degenerate_polyline_emits_neither():
    """Policy C: Polyline with < 2 vertices drops completely."""
    lane = DetectedLane(
        label="lane/double yellow",
        polyline=[[500, 500]],  # Only 1 vertex (invalid)
        confidence=0.85,
    )
    # DetectedLane.validate() raises ValueError for < 2 points
    with pytest.raises(ValueError):
        lane.validate()


# ==============================================================================
# Section 11: Final Output Validator validate_cvat_output_shapes()
# ==============================================================================

def test_section11_rejects_car_rectangle_only(tax: Taxonomy):
    """Validator drops Policy A instance if mask is missing."""
    shapes = [
        {"label": "car", "type": "rectangle", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1}
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_a_violation" in w for w in warnings)


def test_section11_rejects_car_mask_only(tax: Taxonomy):
    """Validator drops Policy A instance if rectangle is missing."""
    shapes = [
        {"label": "car", "type": "mask", "points": [10.0, 10.0, 100.0, 100.0], "mask": [1, 10, 10, 100, 100], "group_id": 1}
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_a_violation" in w for w in warnings)


def test_section11_rejects_road_polygon_only(tax: Taxonomy):
    """Validator drops Policy B region if mask is missing."""
    shapes = [
        {"label": "road", "type": "polygon", "points": [0.0, 500.0, 1000.0, 500.0, 1000.0, 1000.0, 0.0, 1000.0], "group_id": 1}
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_b_violation" in w for w in warnings)


def test_section11_rejects_road_mask_only(tax: Taxonomy):
    """Validator drops Policy B region if polygon is missing."""
    shapes = [
        {"label": "road", "type": "mask", "points": [0.0, 500.0, 1000.0, 1000.0], "mask": [1, 0, 500, 1000, 1000], "group_id": 1}
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_b_violation" in w for w in warnings)


def test_section11_rejects_lane_polygon(tax: Taxonomy):
    """Validator drops lane marking if shape is polygon."""
    shapes = [
        {"label": "lane/crosswalk", "type": "polygon", "points": [100.0, 200.0, 300.0, 200.0, 300.0, 400.0, 100.0, 400.0]}
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_c_violation" in w for w in warnings)


def test_section11_rejects_lane_mask(tax: Taxonomy):
    """Validator drops lane marking if shape is mask."""
    shapes = [
        {"label": "lane/single white", "type": "mask", "points": [100.0, 200.0, 300.0, 400.0], "mask": [1, 100, 200, 300, 400]}
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("policy_c_violation" in w for w in warnings)


def test_section11_lane_polyline_strips_group_id(tax: Taxonomy):
    """Validator ensures Policy C polylines have group_id stripped."""
    shapes = [
        {
            "label": "lane/road curb",
            "type": "polyline",
            "points": [10.0, 20.0, 30.0, 40.0],
            "group_id": 99,  # Should be stripped
        }
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 1
    assert validated[0]["type"] == "polyline"
    assert "group_id" not in validated[0]


# ==============================================================================
# Section 12: Group ID Allocation & Collision Prevention
# ==============================================================================

def test_section12_distinct_instances_sharing_group_id_rejected(tax: Taxonomy):
    """Validator drops annotations if distinct entities share the same group_id."""
    # car pair with group_id=1 and pedestrian pair with group_id=1
    shapes = [
        {"label": "car", "type": "rectangle", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1},
        {"label": "car", "type": "mask", "points": [10.0, 10.0, 100.0, 100.0], "mask": [1, 10, 10, 100, 100], "group_id": 1},
        {"label": "pedestrian", "type": "rectangle", "points": [200.0, 200.0, 250.0, 350.0], "group_id": 1},
        {"label": "pedestrian", "type": "mask", "points": [200.0, 200.0, 250.0, 350.0], "mask": [1, 200, 200, 250, 350], "group_id": 1},
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("shared across distinct instances" in w for w in warnings)


def test_section12_cross_policy_group_id_sharing_rejected(tax: Taxonomy):
    """Validator drops annotations if a Policy A instance and Policy B region share group_id."""
    shapes = [
        {"label": "car", "type": "rectangle", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 5},
        {"label": "car", "type": "mask", "points": [10.0, 10.0, 100.0, 100.0], "mask": [1, 10, 10, 100, 100], "group_id": 5},
        {"label": "vegetation", "type": "polygon", "points": [0.0, 0.0, 200.0, 0.0, 200.0, 200.0, 0.0, 200.0], "group_id": 5},
        {"label": "vegetation", "type": "mask", "points": [0.0, 0.0, 200.0, 200.0], "mask": [1, 0, 0, 200, 200], "group_id": 5},
    ]
    validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
    assert len(validated) == 0
    assert any("shared across distinct instances" in w for w in warnings)


def test_section12_invalid_group_id_types_rejected(tax: Taxonomy):
    """Validator rejects non-integer, bool, zero, or negative group_ids."""
    invalid_cases = [
        True,   # bool
        False,  # bool
        "1",    # string
        0,      # zero
        -1,     # negative
    ]
    for bad_gid in invalid_cases:
        shapes = [
            {"label": "truck", "type": "rectangle", "points": [10.0, 10.0, 100.0, 100.0], "group_id": bad_gid},
            {"label": "truck", "type": "mask", "points": [10.0, 10.0, 100.0, 100.0], "mask": [1, 10, 10, 100, 100], "group_id": bad_gid},
        ]
        validated, warnings = validate_cvat_output_shapes(shapes, taxonomy=tax)
        assert len(validated) == 0
        assert any("policy_a_violation" in w for w in warnings)


def test_section12_detection_result_sequential_distinct_group_ids():
    """DetectionResult.to_cvat_annotations assigns distinct sequential group_ids."""
    obj1 = DetectedObject(
        label="car",
        box_2d=[100, 100, 400, 400],
        mask=[[100, 100], [400, 100], [400, 400], [100, 400]],
    )
    obj2 = DetectedObject(
        label="bus",
        box_2d=[500, 500, 800, 800],
        mask=[[500, 500], [800, 500], [800, 800], [500, 800]],
    )
    reg1 = DetectedRegion(
        label="road",
        polygon=[[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
    )
    lane1 = DetectedLane(
        label="lane/single white",
        polyline=[[200, 500], [300, 700], [400, 900]],
    )

    det = DetectionResult(objects=[obj1, obj2], regions=[reg1], lanes=[lane1])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    # 2 shapes for obj1, 2 shapes for obj2, 2 shapes for reg1, 1 shape for lane1 = 7 shapes
    assert len(shapes) == 7

    car_shapes = [s for s in shapes if s["label"] == "car"]
    bus_shapes = [s for s in shapes if s["label"] == "bus"]
    road_shapes = [s for s in shapes if s["label"] == "road"]
    lane_shapes = [s for s in shapes if s["label"] == "lane/single white"]

    assert len(car_shapes) == 2
    assert len(bus_shapes) == 2
    assert len(road_shapes) == 2
    assert len(lane_shapes) == 1

    car_gid = car_shapes[0]["group_id"]
    bus_gid = bus_shapes[0]["group_id"]
    road_gid = road_shapes[0]["group_id"]

    assert car_shapes[0]["group_id"] == car_shapes[1]["group_id"] == car_gid
    assert bus_shapes[0]["group_id"] == bus_shapes[1]["group_id"] == bus_gid
    assert road_shapes[0]["group_id"] == road_shapes[1]["group_id"] == road_gid

    # All group_ids must be unique across distinct entities
    assert len({car_gid, bus_gid, road_gid}) == 3
    assert "group_id" not in lane_shapes[0]
