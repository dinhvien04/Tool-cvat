"""Comprehensive tests for app.parser module.

Covers:
- Malformed LLM responses (missing objects, non-list objects, preambles, trailing commas, single quotes, non-JSON)
- Coordinate validation & conversion (negative, >1000, inverted, degenerate, clamping, float rounding)
- Exact label validation (intentional ambiguities, non-bbox label rejection)
"""

import json
import pytest

from app.parser import (
    ParsedObject,
    ParseResult,
    VisionParseError,
    clean_json_string,
    parse_and_validate,
)
from core.vision_contract import DEFAULT_BBOX_LABELS

SAMPLE_ALLOWED_LABELS = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "traffic light",
    "traffic_light",
    "traffic sign",
    "traffic_sign",
    "person",
]


# =====================================================================
# 1. Malformed LLM Responses
# =====================================================================

def test_clean_json_string_markdown_fences():
    """Verify markdown fences removal."""
    text = """Here is the result:
```json
{
  "objects": [
    {"label": "car", "box_2d": [100, 200, 300, 400]}
  ]
}
```
Thanks!"""
    cleaned = clean_json_string(text)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")
    data = json.loads(cleaned)
    assert len(data["objects"]) == 1


def test_clean_json_string_trailing_commas():
    """Verify removal of trailing commas."""
    text = '{"objects": [{"label": "car", "box_2d": [100, 200, 300, 400],},],}'
    cleaned = clean_json_string(text)
    data = json.loads(cleaned)
    assert len(data["objects"]) == 1


def test_parse_missing_objects_key():
    """Verify missing 'objects' key raises VisionParseError."""
    # Dict with 'detections' instead of 'objects'
    raw_detections = json.dumps({
        "detections": [
            {"label": "car", "box_2d": [100, 200, 300, 400]}
        ]
    })
    with pytest.raises(VisionParseError, match="Missing 'objects' key"):
        parse_and_validate(raw_detections, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # Completely empty dict {}
    with pytest.raises(VisionParseError, match="Missing 'objects' key"):
        parse_and_validate("{}", allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # Dict with error message
    raw_error = json.dumps({"error": "Rate limit exceeded"})
    with pytest.raises(VisionParseError, match="Missing 'objects' key"):
        parse_and_validate(raw_error, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)


def test_parse_objects_not_a_list():
    """Verify 'objects' value as string, dict, int, or null raises VisionParseError."""
    # 'objects' is a string
    raw_str = json.dumps({"objects": "car"})
    with pytest.raises(VisionParseError, match="Expected 'objects' to be a list"):
        parse_and_validate(raw_str, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # 'objects' is a dict instead of a list
    raw_dict = json.dumps({"objects": {"label": "car", "box_2d": [100, 200, 300, 400]}})
    with pytest.raises(VisionParseError, match="Expected 'objects' to be a list"):
        parse_and_validate(raw_dict, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # 'objects' is an integer
    raw_int = json.dumps({"objects": 42})
    with pytest.raises(VisionParseError, match="Expected 'objects' to be a list"):
        parse_and_validate(raw_int, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # 'objects' is null
    raw_null = '{"objects": null}'
    with pytest.raises(VisionParseError, match="Expected 'objects' to be a list"):
        parse_and_validate(raw_null, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)


def test_parse_text_preamble_and_commentary():
    """Verify text commentary before and after code block is stripped."""
    raw = """I analyzed the road scene thoroughly. Here is the bounding box annotation:
```json
{
  "objects": [
    {
      "label": "car",
      "box_2d": [120, 250, 480, 620]
    }
  ]
}
```
Total 1 vehicle detected. Let me know if you need additional annotations."""

    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].label == "car"
    assert res.objects[0].box_2d == [120, 250, 480, 620]


def test_parse_text_preamble_without_code_fences():
    """Verify text commentary before and after raw JSON is cleanly extracted."""
    raw = """Autonomous driving perception output:
{"objects": [{"label": "pedestrian", "box_2d": [300, 400, 700, 550]}]}
End of transmission."""

    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].label == "pedestrian"
    assert res.objects[0].box_2d == [300, 400, 700, 550]


def test_parse_single_quotes_instead_of_double_quotes():
    """Verify parsing Python-style single-quoted dictionaries."""
    raw = "{'objects': [{'label': 'car', 'box_2d': [100, 200, 300, 400]}]}"
    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].label == "car"
    assert res.objects[0].box_2d == [100, 200, 300, 400]


def test_parse_single_quotes_with_trailing_commas():
    """Verify single quotes combined with trailing commas."""
    raw = "{'objects': [{'label': 'truck', 'box_2d': [150, 250, 450, 650],},],}"
    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].label == "truck"
    assert res.objects[0].box_2d == [150, 250, 450, 650]


def test_parse_completely_invalid_non_json():
    """Verify non-JSON outputs raise VisionParseError."""
    # Conversational refusal
    with pytest.raises(VisionParseError):
        parse_and_validate("I cannot analyze this image.", allowed_labels=SAMPLE_ALLOWED_LABELS)

    # HTML error page
    html_error = "<html><head><title>502 Bad Gateway</title></head><body>502 Bad Gateway</body></html>"
    with pytest.raises(VisionParseError):
        parse_and_validate(html_error, allowed_labels=SAMPLE_ALLOWED_LABELS)

    # Empty string
    with pytest.raises(VisionParseError):
        parse_and_validate("", allowed_labels=SAMPLE_ALLOWED_LABELS)

    # Whitespace only
    with pytest.raises(VisionParseError):
        parse_and_validate("   \n\t  ", allowed_labels=SAMPLE_ALLOWED_LABELS)

    # JSON primitive number or null
    with pytest.raises(VisionParseError):
        parse_and_validate("12345", allowed_labels=SAMPLE_ALLOWED_LABELS)

    with pytest.raises(VisionParseError):
        parse_and_validate("null", allowed_labels=SAMPLE_ALLOWED_LABELS)


# =====================================================================
# 2. Coordinate Validation & Conversion
# =====================================================================

def test_parse_and_validate_success():
    """Test successful parsing and coordinate conversion."""
    raw = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 600, 800]  # [ymin, xmin, ymax, xmax] in [0, 1000]
            },
            {
                "label": "pedestrian",
                "box_2d": [400, 100, 900, 250]
            }
        ]
    })

    width, height = 1920, 1080
    res = parse_and_validate(
        raw,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=width,
        image_height=height,
        strict=True,
    )

    assert len(res.objects) == 2
    car = res.objects[0]
    assert car.label == "car"
    assert car.box_2d == [100, 200, 600, 800]
    # x1 = (200/1000)*1920 = 384.0, y1 = (100/1000)*1080 = 108.0
    # x2 = (800/1000)*1920 = 1536.0, y2 = (600/1000)*1080 = 648.0
    assert car.pixel_box == [384.0, 108.0, 1536.0, 648.0]

    cvat_shape = car.to_cvat_shape()
    assert cvat_shape["type"] == "rectangle"
    assert cvat_shape["label"] == "car"
    assert cvat_shape["points"] == [384.0, 108.0, 1536.0, 648.0]


def test_parse_negative_coordinates():
    """Verify negative coordinates [-10, -5, 500, 600] handling."""
    raw = json.dumps({
        "objects": [{"label": "car", "box_2d": [-10, -5, 500, 600]}]
    })

    # In strict mode: raises VisionParseError
    with pytest.raises(VisionParseError, match="out of \\[0, 1000\\] bounds"):
        parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # In non-strict mode: clamped to [0, 0, 500, 600]
    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res.objects) == 1
    assert res.objects[0].box_2d == [0, 0, 500, 600]
    assert any("Clamped out of bounds" in w for w in res.warnings)


def test_parse_coordinates_exceeding_1000():
    """Verify coordinates > 1000 [100, 200, 1200, 800] handling."""
    raw = json.dumps({
        "objects": [{"label": "car", "box_2d": [100, 200, 1200, 800]}]
    })

    # In strict mode: raises VisionParseError
    with pytest.raises(VisionParseError, match="out of \\[0, 1000\\] bounds"):
        parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # In non-strict mode: clamped to [100, 200, 1000, 800]
    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res.objects) == 1
    assert res.objects[0].box_2d == [100, 200, 1000, 800]
    assert any("Clamped out of bounds" in w for w in res.warnings)


def test_parse_inverted_coordinates():
    """Verify inverted coordinates (ymin > ymax or xmin > xmax)."""
    # Inverted y: ymin=600 > ymax=200
    raw_inv_y = json.dumps({
        "objects": [{"label": "car", "box_2d": [600, 100, 200, 300]}]
    })
    with pytest.raises(VisionParseError, match="inverted y coordinates"):
        parse_and_validate(raw_inv_y, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    res_y = parse_and_validate(raw_inv_y, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res_y.objects) == 1
    assert res_y.objects[0].box_2d == [200, 100, 600, 300]

    # Inverted x: xmin=800 > xmax=200
    raw_inv_x = json.dumps({
        "objects": [{"label": "car", "box_2d": [100, 800, 300, 200]}]
    })
    with pytest.raises(VisionParseError, match="inverted x coordinates"):
        parse_and_validate(raw_inv_x, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    res_x = parse_and_validate(raw_inv_x, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res_x.objects) == 1
    assert res_x.objects[0].box_2d == [100, 200, 300, 800]


def test_parse_degenerate_boxes():
    """Verify degenerate boxes (ymin == ymax or xmin == xmax)."""
    # ymin == ymax
    raw_zero_h = json.dumps({
        "objects": [{"label": "car", "box_2d": [200, 100, 200, 300]}]
    })
    with pytest.raises(VisionParseError, match="zero area"):
        parse_and_validate(raw_zero_h, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    res_h = parse_and_validate(raw_zero_h, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res_h.objects) == 0

    # xmin == xmax
    raw_zero_w = json.dumps({
        "objects": [{"label": "car", "box_2d": [100, 300, 400, 300]}]
    })
    with pytest.raises(VisionParseError, match="zero area"):
        parse_and_validate(raw_zero_w, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    res_w = parse_and_validate(raw_zero_w, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res_w.objects) == 0


def test_parse_clamping_to_image_boundaries():
    """Verify coordinate clamping to image boundaries [width, height]."""
    raw = json.dumps({
        "objects": [
            {"label": "car", "box_2d": [0, 0, 1000, 1000]}
        ]
    })
    res = parse_and_validate(
        raw,
        allowed_labels=SAMPLE_ALLOWED_LABELS,
        image_width=1280,
        image_height=720,
        strict=True,
    )
    assert len(res.objects) == 1
    assert res.objects[0].pixel_box == [0.0, 0.0, 1280.0, 720.0]

    # Invalid image dimensions
    with pytest.raises(VisionParseError, match="Invalid image dimensions"):
        parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, image_width=0, image_height=720)


def test_parse_float_coordinates_rounding():
    """Verify float coordinates round accurately to integers."""
    raw = json.dumps({
        "objects": [
            {"label": "car", "box_2d": [100.4, 200.6, 600.2, 800.8]}
        ]
    })
    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].box_2d == [100, 201, 600, 801]


def test_parse_non_numeric_coordinates():
    """Verify non-numeric coordinates fail validation."""
    raw = json.dumps({
        "objects": [
            {"label": "car", "box_2d": ["abc", 200, 300, 400]}
        ]
    })
    with pytest.raises(VisionParseError, match="cannot be converted to int"):
        parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)

    # In non-strict mode: skipped with warning
    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=False)
    assert len(res.objects) == 0
    assert len(res.warnings) == 1


# =====================================================================
# 3. Exact Label Validation
# =====================================================================

def test_intentional_label_distinctions_preserved():
    """Verify intentional distinctions are preserved verbatim and not aliased."""
    raw = json.dumps({
        "objects": [
            {"label": "pedestrian", "box_2d": [10, 10, 20, 20]},
            {"label": "person", "box_2d": [30, 30, 40, 40]},
            {"label": "traffic light", "box_2d": [50, 50, 60, 60]},
            {"label": "traffic_light", "box_2d": [70, 70, 80, 80]},
            {"label": "traffic sign", "box_2d": [90, 90, 100, 100]},
            {"label": "traffic_sign", "box_2d": [110, 110, 120, 120]},
        ]
    })

    res = parse_and_validate(raw, allowed_labels=SAMPLE_ALLOWED_LABELS, strict=True)
    labels = [obj.label for obj in res.objects]
    assert labels == [
        "pedestrian",
        "person",
        "traffic light",
        "traffic_light",
        "traffic sign",
        "traffic_sign",
    ]


def test_pedestrian_vs_person_strict_validation():
    """Verify 'pedestrian' vs 'person' ambiguity is strictly validated."""
    # When ONLY 'pedestrian' is requested: 'person' must fail in strict mode
    raw_person = json.dumps({
        "objects": [{"label": "person", "box_2d": [100, 100, 200, 200]}]
    })
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        parse_and_validate(raw_person, allowed_labels=["pedestrian"], strict=True)

    # When ONLY 'person' is requested: 'pedestrian' must fail in strict mode
    raw_pedestrian = json.dumps({
        "objects": [{"label": "pedestrian", "box_2d": [100, 100, 200, 200]}]
    })
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        parse_and_validate(raw_pedestrian, allowed_labels=["person"], strict=True)


def test_traffic_light_spaced_vs_underscored_strict_validation():
    """Verify 'traffic light' vs 'traffic_light' is strictly validated."""
    # Request spaced: underscored fails
    raw_under = json.dumps({
        "objects": [{"label": "traffic_light", "box_2d": [50, 50, 100, 100]}]
    })
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        parse_and_validate(raw_under, allowed_labels=["traffic light"], strict=True)

    # Request underscored: spaced fails
    raw_space = json.dumps({
        "objects": [{"label": "traffic light", "box_2d": [50, 50, 100, 100]}]
    })
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        parse_and_validate(raw_space, allowed_labels=["traffic_light"], strict=True)


def test_traffic_sign_spaced_vs_underscored_strict_validation():
    """Verify 'traffic sign' vs 'traffic_sign' is strictly validated."""
    # Request spaced: underscored fails
    raw_under = json.dumps({
        "objects": [{"label": "traffic_sign", "box_2d": [50, 50, 100, 100]}]
    })
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        parse_and_validate(raw_under, allowed_labels=["traffic sign"], strict=True)

    # Request underscored: spaced fails
    raw_space = json.dumps({
        "objects": [{"label": "traffic sign", "box_2d": [50, 50, 100, 100]}]
    })
    with pytest.raises(VisionParseError, match="not in allowed labels"):
        parse_and_validate(raw_space, allowed_labels=["traffic_sign"], strict=True)


def test_non_bbox_labels_rejected_against_default():
    """Verify non-bbox labels (road, sky, building, etc.) are rejected against default bbox list."""
    non_bbox_samples = [
        "road",
        "sky",
        "building",
        "sidewalk",
        "vegetation",
        "pole",
        "wall",
        "fence",
        "terrain",
    ]

    for non_bbox in non_bbox_samples:
        raw = json.dumps({
            "objects": [{"label": non_bbox, "box_2d": [100, 100, 500, 500]}]
        })

        # Against default 13 bbox labels in strict mode: must raise VisionParseError
        with pytest.raises(VisionParseError, match="not in allowed labels"):
            parse_and_validate(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)

        # In non-strict mode: skipped with warning
        res = parse_and_validate(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=False)
        assert len(res.objects) == 0
        assert len(res.warnings) == 1

    # But if explicitly allowed, it should succeed
    raw_road = json.dumps({
        "objects": [{"label": "road", "box_2d": [500, 0, 1000, 1000]}]
    })
    res_road = parse_and_validate(raw_road, allowed_labels=["road", "car"], strict=True)
    assert len(res_road.objects) == 1
    assert res_road.objects[0].label == "road"
