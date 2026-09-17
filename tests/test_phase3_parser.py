"""Tests for Phase 3 parser extensions (Box + Instance Mask / Segmentation)."""

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

SAMPLE_ALLOWED_LABELS = ["car", "pedestrian", "bicycle", "traffic light", "bus"]


# ============================================================================
# 1. Parsing Model Responses with both box_2d and mask
# ============================================================================


def test_parse_box_and_mask_valid():
    """Verify parsing a valid model response containing both box_2d and mask contour."""
    raw_json = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 500, 600],
                "confidence": 0.94,
                "mask": [
                    [200, 100],
                    [600, 100],
                    [600, 500],
                    [200, 500],
                ],
            }
        ]
    })

    res = parse_and_validate(
        raw_json,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=1000,
        image_height=1000,
        strict=True,
    )

    assert len(res.objects) == 1
    obj = res.objects[0]
    assert obj.label == "car"
    assert obj.box_2d == [100, 200, 500, 600]
    assert obj.confidence == 0.94
    assert obj.mask == [[200.0, 100.0], [600.0, 100.0], [600.0, 500.0], [200.0, 500.0]]
    assert obj.pixel_box == [200.0, 100.0, 600.0, 500.0]
    assert obj.pixel_polygon is not None
    assert obj.cvat_mask is not None

    # Check to_dict()
    d = obj.to_dict()
    assert "mask" in d
    assert "pixel_polygon" in d
    assert "group_id" in d

    # Check to_cvat_mask_shape()
    mask_shape = obj.to_cvat_mask_shape()
    assert mask_shape is not None
    assert mask_shape["type"] == "mask"
    assert mask_shape["label"] == "car"
    assert mask_shape["confidence"] == "0.94"
    assert isinstance(mask_shape["mask"], list)
    assert len(mask_shape["mask"]) > 4


def test_parse_box_and_mask_from_markdown_fenced_response():
    """Verify parser extracts JSON wrapped inside markdown code fences."""
    fenced = """```json
    {
      "objects": [
        {
          "label": "pedestrian",
          "box_2d": [300, 400, 700, 500],
          "confidence": "88%",
          "mask": [
            [400, 300],
            [500, 300],
            [500, 700],
            [400, 700]
          ]
        }
      ]
    }
    ```"""

    res = parse_and_validate(
        fenced,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=800,
        image_height=600,
        strict=True,
    )

    assert len(res.objects) == 1
    obj = res.objects[0]
    assert obj.label == "pedestrian"
    assert obj.confidence == 0.88
    assert obj.cvat_mask is not None


def test_parse_openai_chat_completion_structure():
    """Verify parser extracts content from OpenAI-compatible chat completion payload."""
    chat_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "objects": [
                            {
                                "label": "bicycle",
                                "box_2d": [100, 100, 300, 300],
                                "confidence": 0.85,
                                "mask": [
                                    [100, 100],
                                    [300, 100],
                                    [300, 300],
                                    [100, 300]
                                ]
                            }
                        ]
                    })
                }
            }
        ]
    }

    res = parse_and_validate(
        chat_payload,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=640,
        image_height=480,
    )
    assert len(res.objects) == 1
    assert res.objects[0].label == "bicycle"
    assert res.objects[0].cvat_mask is not None


# ============================================================================
# 2. Parsing Model Responses with Missing mask Field
# ============================================================================


def test_parse_missing_mask_graceful_fallback():
    """Verify parser gracefully handles detections where model omits the mask field."""
    raw_json = json.dumps({
        "objects": [
            {
                "label": "bus",
                "box_2d": [150, 250, 450, 650],
                "confidence": 0.91,
                # 'mask' is deliberately omitted
            }
        ]
    })

    # Strict mode should still succeed because mask is an optional attribute in detection
    res_strict = parse_and_validate(
        raw_json,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=1280,
        image_height=720,
        strict=True,
    )

    assert len(res_strict.objects) == 1
    obj = res_strict.objects[0]
    assert obj.label == "bus"
    assert obj.box_2d == [150, 250, 450, 650]
    assert obj.mask is None
    assert obj.pixel_polygon is None
    assert obj.cvat_mask is None

    # Rectangle shape works, mask shape returns None
    rect_shape = obj.to_cvat_shape()
    assert rect_shape["type"] == "rectangle"
    assert obj.to_cvat_mask_shape() is None


# ============================================================================
# 3. Parsing Model Responses with Invalid mask Formats
# ============================================================================


def test_parse_invalid_mask_strict_raises():
    """Verify strict mode raises VisionParseError on invalid mask structures."""
    invalid_cases = [
        # Non-list mask (string)
        {"label": "car", "box_2d": [100, 100, 300, 300], "mask": "not_a_list"},
        # Non-list mask (dict)
        {"label": "car", "box_2d": [100, 100, 300, 300], "mask": {"points": [100, 200]}},
        # Degenerate: fewer than 3 vertices
        {"label": "car", "box_2d": [100, 100, 300, 300], "mask": [[100, 100], [200, 200]]},
        # Vertex is not a 2-element sequence (e.g. 3 elements)
        {"label": "car", "box_2d": [100, 100, 300, 300], "mask": [[100, 100, 50], [200, 200], [300, 300]]},
        # Non-numeric coordinate in vertex
        {"label": "car", "box_2d": [100, 100, 300, 300], "mask": [["bad", "coord"], [200, 200], [300, 300]]},
    ]

    for item in invalid_cases:
        payload = json.dumps({"objects": [item]})
        with pytest.raises(VisionParseError):
            parse_and_validate(
                payload,
                allowed_labels=SAMPLE_ALLOWED_LABELS,
                image_width=800,
                image_height=600,
                strict=True,
            )


def test_parse_invalid_mask_non_strict_warning():
    """Verify non-strict mode skips corrupt mask, logs warning, and keeps valid box_2d."""
    payload = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 100, 300, 300],
                "confidence": 0.85,
                "mask": "corrupted_non_list_mask",
            }
        ]
    })

    res = parse_and_validate(
        payload,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=800,
        image_height=600,
        strict=False,
    )

    assert len(res.objects) == 1
    assert res.objects[0].label == "car"
    assert res.objects[0].box_2d == [100, 100, 300, 300]
    assert res.objects[0].mask is None
    assert len(res.warnings) > 0
    assert any("mask" in w.lower() for w in res.warnings)


# ============================================================================
# 4. Preserving Model Confidence and Threshold Filtering
# ============================================================================


def test_parse_confidence_formats_and_preservation():
    """Verify parsing and preservation of various confidence representations."""
    test_data = json.dumps({
        "objects": [
            {"label": "car", "box_2d": [100, 100, 200, 200], "confidence": 0.95},
            {"label": "pedestrian", "box_2d": [200, 200, 300, 300], "confidence": "0.82"},
            {"label": "bicycle", "box_2d": [300, 300, 400, 400], "confidence": "76%"},
            {"label": "bus", "box_2d": [400, 400, 500, 500]},  # omitted confidence
        ]
    })

    res = parse_and_validate(test_data, allowed_labels=SAMPLE_ALLOWED_LABELS)
    assert len(res.objects) == 4

    assert res.objects[0].confidence == 0.95
    assert res.objects[1].confidence == 0.82
    assert res.objects[2].confidence == 0.76
    assert res.objects[3].confidence is None


def test_parse_out_of_bounds_mask_coordinates_clamped():
    """Verify mask coordinates outside [0, 1000] are clamped without crashing."""
    payload = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 100, 400, 400],
                "mask": [
                    [-50, -50],
                    [1050, -20],
                    [1050, 1080],
                    [-10, 800],
                ],
            }
        ]
    })

    res = parse_and_validate(
        payload,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=500,
        image_height=500,
        strict=True,
    )

    assert len(res.objects) == 1
    obj = res.objects[0]
    assert obj.cvat_mask is not None
    # Trailing bbox coords should be inside [0, 499]
    xmin, ymin, xmax, ymax = obj.cvat_mask[-4:]
    assert 0 <= xmin <= 499
    assert 0 <= ymin <= 499
    assert 0 <= xmax <= 499
    assert 0 <= ymax <= 499
