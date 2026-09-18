"""Tests for Phase 3B unified scene response parser (objects, regions, lanes).

Covers:
- Strict 3-policy geometry validation:
  * Policy A objects: box_2d + mask (rejects polygon, polyline, missing box, missing mask)
  * Policy B regions: polygon only (rejects mask, polyline, box_2d as required geometry)
  * Policy C lanes: polyline only (rejects mask, polygon, box_2d, never rasterizes as cvat_mask)
- First-class ParsedObject geometry attributes (instance_mask, region_polygon, lane_polyline)
- Legacy migration helper (migrate_legacy_full31_item) with explicit opt-in and warnings
- Strict category label boundaries (preventing label cross-contamination across policies)
"""

import json
from typing import Any, Dict, List
import pytest

from app.parser import (
    ParsedObject,
    ParseResult,
    VisionParseError,
    clean_json_string,
    parse_and_validate,
    migrate_legacy_full31_item,
    parse_legacy_full31_geometry,
)
from core.vision_contract import ALL_31_LABELS


def test_parse_unified_scene_response():
    """Verify parsing a valid unified scene response with objects, regions, and lanes."""
    raw_payload = {
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "confidence": 0.95,
                "mask": [[200, 100], [600, 100], [600, 500], [200, 500]],
            },
            {
                "label": "pole",
                "box_2d": [50, 800, 400, 820],
                "confidence": 0.90,
                "mask": [[800, 50], [820, 50], [820, 400], [800, 400]],
            },
        ],
        "regions": [
            {
                "label": "road",
                "confidence": 0.98,
                "polygon": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
            },
            {
                "label": "sky",
                "confidence": 0.99,
                "polygon": [[0, 0], [1000, 0], [1000, 400], [0, 400]],
            },
        ],
        "lanes": [
            {
                "label": "lane/single white",
                "confidence": 0.92,
                "polyline": [[500, 500], [505, 500], [520, 1000], [515, 1000]],
            },
            {
                "label": "lane/crosswalk",
                "confidence": 0.88,
                "polyline": [[200, 600], [800, 600], [800, 700], [200, 700]],
            },
        ],
    }

    res = parse_and_validate(
        json.dumps(raw_payload),
        allowed_labels=list(ALL_31_LABELS),
        image_width=1000,
        image_height=1000,
        strict=True,
    )

    assert len(res.objects) == 2
    assert len(res.regions) == 2
    assert len(res.lanes) == 2
    assert len(res.all_items) == 6

    # Verify objects: Policy A (box_2d, instance_mask, group_id)
    car = res.objects[0]
    assert car.label == "car"
    assert car.box_2d == [100, 200, 500, 600]
    assert car.group_id is not None
    assert car.instance_mask is not None
    assert car.region_polygon is None
    assert car.lane_polyline is None
    assert car.mask == car.instance_mask  # backward compatibility property

    # Verify regions: Policy B (region_polygon, cvat_mask derived, no box, no group_id)
    road = res.regions[0]
    assert road.label == "road"
    assert road.box_2d is None
    assert road.pixel_box is None
    assert road.group_id is None
    assert road.region_polygon is not None
    assert road.instance_mask is None
    assert road.lane_polyline is None
    assert road.cvat_mask is not None

    # Verify lanes: Policy C (lane_polyline, pixel_polyline, NO cvat_mask, no box, no group_id)
    lane = res.lanes[0]
    assert lane.label == "lane/single white"
    assert lane.box_2d is None
    assert lane.group_id is None
    assert lane.lane_polyline is not None
    assert lane.pixel_polyline is not None
    assert lane.region_polygon is None
    assert lane.instance_mask is None
    assert lane.cvat_mask is None  # CRITICAL Section 6: Never rasterized as polygon mask!

    # Verify to_dict() structure
    d = res.to_dict()
    assert "objects" in d and len(d["objects"]) == 2
    assert "regions" in d and len(d["regions"]) == 2
    assert "lanes" in d and len(d["lanes"]) == 2
    assert "mask" in d["objects"][0]
    assert "polygon" in d["regions"][0]
    assert "polyline" in d["lanes"][0]


def test_parse_region_only_response():
    """Verify response containing only 'regions' parses without error."""
    raw_payload = {
        "regions": [
            {
                "label": "vegetation",
                "confidence": 0.91,
                "polygon": [[0, 300], [400, 300], [400, 600], [0, 600]],
            }
        ]
    }

    res = parse_and_validate(
        json.dumps(raw_payload),
        allowed_labels=list(ALL_31_LABELS),
        image_width=800,
        image_height=600,
        strict=True,
    )

    assert len(res.objects) == 0
    assert len(res.regions) == 1
    assert len(res.lanes) == 0
    assert res.regions[0].label == "vegetation"
    assert res.regions[0].region_polygon is not None
    assert res.regions[0].cvat_mask is not None


def test_parse_lane_only_response():
    """Verify response containing only 'lanes' parses without error."""
    raw_payload = {
        "lanes": [
            {
                "label": "lane/double yellow",
                "confidence": 0.89,
                "polyline": [[490, 500], [510, 500], [510, 1000], [490, 1000]],
            }
        ]
    }

    res = parse_and_validate(
        json.dumps(raw_payload),
        allowed_labels=list(ALL_31_LABELS),
        image_width=800,
        image_height=600,
        strict=True,
    )

    assert len(res.lanes) == 1
    assert res.lanes[0].label == "lane/double yellow"
    assert res.lanes[0].lane_polyline is not None
    assert res.lanes[0].cvat_mask is None


def test_parse_strict_rejects_missing_box_on_instance():
    """Verify missing box_2d on an instance label raises in strict mode."""
    raw_payload = {
        "objects": [
            {
                "label": "car",
                "confidence": 0.95,
                "mask": [[200, 100], [600, 100], [600, 500], [200, 500]],
            }
        ]
    }

    with pytest.raises(VisionParseError, match="invalid box_2d"):
        parse_and_validate(
            json.dumps(raw_payload),
            allowed_labels=list(ALL_31_LABELS),
            image_width=1000,
            image_height=1000,
            strict=True,
        )


def test_parse_non_strict_skips_invalid_item_with_warning():
    """Verify non-strict mode skips invalid items and logs warnings."""
    raw_payload = {
        "objects": [
            {
                "label": "alien_spaceship",  # Unknown label
                "box_2d": [100, 200, 300, 400],
            },
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "confidence": 0.95,
            },
        ]
    }

    res = parse_and_validate(
        json.dumps(raw_payload),
        allowed_labels=list(ALL_31_LABELS),
        image_width=1000,
        image_height=1000,
        strict=False,
    )

    assert len(res.objects) == 1
    assert res.objects[0].label == "car"
    assert any("not in allowed labels" in w for w in res.warnings)


def test_legacy_migration_helper():
    """Verify migrate_legacy_full31_item converts legacy mask/points and emits warnings."""
    # Test region migration
    region_item = {
        "label": "road",
        "confidence": 0.95,
        "mask": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
    }
    migrated_reg, reg_warns = migrate_legacy_full31_item(region_item, "regions")
    assert "polygon" in migrated_reg
    assert "mask" not in migrated_reg
    assert any("legacy_geometry_migrated" in w for w in reg_warns)

    # Test lane migration from mask
    lane_item_mask = {
        "label": "lane/single white",
        "confidence": 0.90,
        "mask": [[500, 500], [520, 1000]],
    }
    migrated_lane1, lane_warns1 = migrate_legacy_full31_item(lane_item_mask, "lanes")
    assert "polyline" in migrated_lane1
    assert "mask" not in migrated_lane1
    assert any("legacy_geometry_migrated" in w for w in lane_warns1)

    # Test lane migration from points
    lane_item_pts = {
        "label": "lane/single yellow",
        "confidence": 0.90,
        "points": [[400, 400], [410, 800]],
    }
    migrated_lane2, lane_warns2 = parse_legacy_full31_geometry(lane_item_pts, "lanes")
    assert "polyline" in migrated_lane2
    assert "points" not in migrated_lane2
    assert any("legacy_geometry_migrated" in w for w in lane_warns2)

    # Test parse_and_validate with enable_legacy_migration=True
    legacy_payload = {
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "confidence": 0.95,
                "mask": [[200, 100], [600, 100], [600, 500], [200, 500]],
            }
        ],
        "regions": [
            {
                "label": "road",
                "confidence": 0.98,
                "mask": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
            }
        ],
        "lanes": [
            {
                "label": "lane/single white",
                "confidence": 0.92,
                "mask": [[500, 500], [505, 500], [520, 1000], [515, 1000]],
            }
        ],
    }

    # Strict without migration MUST fail
    with pytest.raises(VisionParseError, match="prohibited geometry 'mask' for Policy B"):
        parse_and_validate(
            json.dumps(legacy_payload),
            allowed_labels=list(ALL_31_LABELS),
            image_width=1000,
            image_height=1000,
            strict=True,
            enable_legacy_migration=False,
        )

    # Strict with migration MUST succeed and record warning
    migrated_res = parse_and_validate(
        json.dumps(legacy_payload),
        allowed_labels=list(ALL_31_LABELS),
        image_width=1000,
        image_height=1000,
        strict=True,
        enable_legacy_migration=True,
    )
    assert len(migrated_res.objects) == 1
    assert len(migrated_res.regions) == 1
    assert len(migrated_res.lanes) == 1
    assert any("legacy_geometry_migrated" in w for w in migrated_res.warnings)
    assert migrated_res.regions[0].region_polygon is not None
    assert migrated_res.lanes[0].lane_polyline is not None


def test_strict_rejects_object_forbidden_geometries():
    """Verify strict rejection of forbidden geometries for Policy A objects."""
    # Reject polygon on object
    poly_payload = {
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "polygon": [[200, 100], [600, 100], [600, 500]],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited geometry 'polygon' for Policy A"):
        parse_and_validate(poly_payload, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Reject polyline on object
    line_payload = {
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "polyline": [[200, 100], [600, 100]],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited geometry 'polyline' for Policy A"):
        parse_and_validate(line_payload, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Reject points-only on object
    pts_payload = {
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "points": [[200, 100], [600, 100]],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited points-only geometry"):
        parse_and_validate(pts_payload, allowed_labels=list(ALL_31_LABELS), strict=True)


def test_strict_rejects_object_missing_mask_in_full31():
    """Verify Policy A object missing mask in full_31 mode is strictly rejected."""
    payload = {
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
            }
        ],
        "regions": [],
        "lanes": [],
    }
    with pytest.raises(VisionParseError, match="missing required 'mask' contour per Policy A"):
        parse_and_validate(payload, allowed_labels=list(ALL_31_LABELS), mode="full_31", strict=True)


def test_strict_rejects_region_forbidden_geometries():
    """Verify strict rejection of forbidden geometries for Policy B regions."""
    # Reject polyline for region
    line_payload = {
        "regions": [
            {
                "label": "road",
                "polyline": [[0, 500], [1000, 500]],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited geometry 'polyline' for Policy B"):
        parse_and_validate(line_payload, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Reject box_2d as required geometry for region
    box_payload = {
        "regions": [
            {
                "label": "road",
                "box_2d": [500, 0, 1000, 1000],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited box_2d as required geometry for Policy B"):
        parse_and_validate(box_payload, allowed_labels=list(ALL_31_LABELS), strict=True)


def test_strict_rejects_lane_forbidden_geometries():
    """Verify strict rejection of forbidden geometries for Policy C lanes."""
    # Reject mask for lane
    mask_payload = {
        "lanes": [
            {
                "label": "lane/single white",
                "mask": [[500, 500], [510, 500], [510, 1000], [500, 1000]],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited geometry 'mask' for Policy C"):
        parse_and_validate(mask_payload, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Reject polygon for lane
    poly_payload = {
        "lanes": [
            {
                "label": "lane/single white",
                "polygon": [[500, 500], [510, 500], [510, 1000], [500, 1000]],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited geometry 'polygon' for Policy C"):
        parse_and_validate(poly_payload, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Reject box_2d for lane
    box_payload = {
        "lanes": [
            {
                "label": "lane/single white",
                "box_2d": [500, 500, 1000, 510],
            }
        ]
    }
    with pytest.raises(VisionParseError, match="prohibited 'box_2d' for Policy C"):
        parse_and_validate(box_payload, allowed_labels=list(ALL_31_LABELS), strict=True)


def test_lane_never_generates_cvat_mask():
    """CRITICAL Section 6: Polyline must never be rasterized as CVAT mask."""
    payload = {
        "lanes": [
            {
                "label": "lane/single white",
                "polyline": [[500, 500], [510, 600], [520, 800], [530, 1000]],
            }
        ]
    }
    res = parse_and_validate(
        payload,
        allowed_labels=list(ALL_31_LABELS),
        image_width=1000,
        image_height=1000,
        strict=True,
    )
    lane = res.lanes[0]
    assert lane.cvat_mask is None
    assert lane.to_cvat_mask_shape() is None
    poly_shape = lane.to_cvat_polyline_shape()
    assert poly_shape is not None
    assert poly_shape["type"] == "polyline"


def test_strict_category_label_cross_contamination():
    """Verify labels belonging to other policies are rejected from incompatible categories."""
    # Policy B label inside objects
    payload_b_in_obj = {
        "objects": [
            {
                "label": "road",
                "box_2d": [500, 0, 1000, 1000],
                "mask": [[0, 500], [1000, 500], [1000, 1000]],
            }
        ],
        "regions": [],
        "lanes": [],
    }
    with pytest.raises(VisionParseError, match="belongs to semantic regions"):
        parse_and_validate(payload_b_in_obj, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Policy C label inside objects
    payload_c_in_obj = {
        "objects": [
            {
                "label": "lane/single white",
                "box_2d": [500, 500, 1000, 520],
                "mask": [[500, 500], [520, 500], [520, 1000]],
            }
        ],
        "regions": [],
        "lanes": [],
    }
    with pytest.raises(VisionParseError, match="belongs to lane markings"):
        parse_and_validate(payload_c_in_obj, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Policy A label inside regions
    payload_a_in_reg = {
        "objects": [],
        "regions": [
            {
                "label": "car",
                "polygon": [[200, 100], [600, 100], [600, 500]],
            }
        ],
        "lanes": [],
    }
    with pytest.raises(VisionParseError, match="belongs to object instances"):
        parse_and_validate(payload_a_in_reg, allowed_labels=list(ALL_31_LABELS), strict=True)

    # Policy A label inside lanes
    payload_a_in_lane = {
        "objects": [],
        "regions": [],
        "lanes": [
            {
                "label": "car",
                "polyline": [[200, 100], [600, 100]],
            }
        ],
    }
    with pytest.raises(VisionParseError, match="belongs to object instances"):
        parse_and_validate(payload_a_in_lane, allowed_labels=list(ALL_31_LABELS), strict=True)
