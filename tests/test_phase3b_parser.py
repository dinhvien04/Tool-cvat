"""Tests for Phase 3B unified scene response parser (objects, regions, lanes)."""

import json
from typing import Any, Dict, List
import pytest

from app.parser import (
    ParsedObject,
    ParseResult,
    VisionParseError,
    clean_json_string,
    parse_and_validate,
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
                "mask": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
            },
            {
                "label": "sky",
                "confidence": 0.99,
                "mask": [[0, 0], [1000, 0], [1000, 400], [0, 400]],
            },
        ],
        "lanes": [
            {
                "label": "lane/single white",
                "confidence": 0.92,
                "mask": [[500, 500], [505, 500], [520, 1000], [515, 1000]],
            },
            {
                "label": "lane/crosswalk",
                "confidence": 0.88,
                "mask": [[200, 600], [800, 600], [800, 700], [200, 700]],
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

    # Verify objects have box_2d and group_id
    car = res.objects[0]
    assert car.label == "car"
    assert car.box_2d == [100, 200, 500, 600]
    assert car.group_id is not None

    # Verify regions have no box_2d and group_id is None
    road = res.regions[0]
    assert road.label == "road"
    assert road.box_2d is None
    assert road.pixel_box is None
    assert road.group_id is None
    assert road.cvat_mask is not None

    # Verify lanes have no box_2d and group_id is None
    lane = res.lanes[0]
    assert lane.label == "lane/single white"
    assert lane.box_2d is None
    assert lane.group_id is None
    assert lane.mask is not None

    # Verify to_dict() structure
    d = res.to_dict()
    assert "objects" in d and len(d["objects"]) == 2
    assert "regions" in d and len(d["regions"]) == 2
    assert "lanes" in d and len(d["lanes"]) == 2


def test_parse_region_only_response():
    """Verify response containing only 'regions' parses without error."""
    raw_payload = {
        "regions": [
            {
                "label": "vegetation",
                "confidence": 0.91,
                "mask": [[0, 300], [400, 300], [400, 600], [0, 600]],
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


def test_parse_lane_only_response():
    """Verify response containing only 'lanes' parses without error."""
    raw_payload = {
        "lanes": [
            {
                "label": "lane/double yellow",
                "confidence": 0.89,
                "mask": [[490, 500], [510, 500], [510, 1000], [490, 1000]],
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
