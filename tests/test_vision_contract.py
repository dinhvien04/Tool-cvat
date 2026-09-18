"""Unit tests for Vision Contract and Prompt Specification.

Covers:
- Exact 31 labels schema and 13 default bbox labels
- Intentional label distinction preservation
- Coordinate conversion, clamping, and validation
- DetectedObject.validate edge cases and failure modes
- parse_vision_response malformed responses, bounds checks, and label contracts
"""

import json
from pathlib import Path

import pytest
import yaml

from core.vision_contract import (
    ALL_31_LABELS,
    DEFAULT_BBOX_LABELS,
    MODE_FULL_31,
    SYSTEM_PROMPT_FULL_31,
    DetectedLane,
    DetectedObject,
    DetectedRegion,
    DetectionResult,
    box_2d_to_cvat_rect,
    build_full_31_prompt,
    build_openai_vision_payload,
    build_user_prompt,
    cvat_rect_to_box_2d,
    load_label_config,
    parse_vision_response,
)


def test_labels_count_and_content():
    """Verify exact 31 labels and 13 bbox labels."""
    assert len(ALL_31_LABELS) == 31
    assert len(DEFAULT_BBOX_LABELS) == 13

    # Verify all 13 bbox labels are a subset of all 31 labels
    for bbox_label in DEFAULT_BBOX_LABELS:
        assert bbox_label in ALL_31_LABELS

    # Critical requirement: Intentional distinctions must NOT be merged or renamed
    assert "pedestrian" in ALL_31_LABELS
    assert "person" in ALL_31_LABELS
    assert "pedestrian" != "person"

    assert "traffic light" in ALL_31_LABELS
    assert "traffic_light" in ALL_31_LABELS
    assert "traffic light" != "traffic_light"

    assert "traffic sign" in ALL_31_LABELS
    assert "traffic_sign" in ALL_31_LABELS
    assert "traffic sign" != "traffic_sign"


def test_labels_yaml_file():
    """Verify labels.yaml matches specifications exactly."""
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "labels.yaml"
    assert yaml_path.exists()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_labels = data["all_labels"]
    bbox_labels = data["bbox_labels"]
    non_bbox_labels = data["non_bbox_labels"]

    assert len(all_labels) == 31
    assert len(bbox_labels) == 13
    assert len(non_bbox_labels) == 18

    assert set(all_labels) == set(ALL_31_LABELS)
    assert set(bbox_labels) == set(DEFAULT_BBOX_LABELS)

    # Ensure no duplicates
    assert len(all_labels) == len(set(all_labels))
    assert len(bbox_labels) == len(set(bbox_labels))


def test_coordinate_conversion_roundtrip():
    """Verify coordinate translation between [ymin, xmin, ymax, xmax] in [0, 1000] and CVAT pixels."""
    width, height = 1920, 1080

    # Normal box: ymin=100, xmin=200, ymax=600, xmax=800
    box_2d = [100, 200, 600, 800]
    xtl, ytl, xbr, ybr = box_2d_to_cvat_rect(box_2d, width, height)

    assert xtl == round((200 / 1000.0) * 1920, 2)  # 384.0
    assert ytl == round((100 / 1000.0) * 1080, 2)  # 108.0
    assert xbr == round((800 / 1000.0) * 1920, 2)  # 1536.0
    assert ybr == round((600 / 1000.0) * 1080, 2)  # 648.0

    roundtrip_box = cvat_rect_to_box_2d(xtl, ytl, xbr, ybr, width, height)
    assert roundtrip_box == box_2d


def test_cvat_rect_clamping():
    """Verify clamping to [0, 1000] bounds."""
    width, height = 1000, 1000

    box = cvat_rect_to_box_2d(-50, -20, 1200, 1500, width, height)
    assert box == [0, 0, 1000, 1000]


def test_box_2d_to_cvat_rect_invalid_dimensions():
    """Verify invalid image dimensions raise ValueError."""
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        box_2d_to_cvat_rect([100, 200, 300, 400], width=0, height=100)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        box_2d_to_cvat_rect([100, 200, 300, 400], width=100, height=-10)


def test_cvat_rect_to_box_2d_invalid_dimensions():
    """Verify invalid image dimensions in cvat_rect_to_box_2d raise ValueError."""
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        cvat_rect_to_box_2d(0, 0, 100, 100, width=0, height=500)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        cvat_rect_to_box_2d(0, 0, 100, 100, width=500, height=-1)


def test_prompt_generation():
    """Verify system and user prompt generation and payload structure."""
    user_prompt = build_user_prompt()
    assert "Allowed labels" in user_prompt
    assert "pedestrian" in user_prompt
    assert "person" in user_prompt
    assert "traffic light" in user_prompt
    assert "traffic_light" in user_prompt
    assert "traffic sign" in user_prompt
    assert "traffic_sign" in user_prompt

    payload = build_openai_vision_payload(
        image_base64="aGVsbG8=",
        model="gpt-4o",
    )

    assert payload["model"] == "gpt-4o"
    assert payload["response_format"] == {"type": "json_object"}
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"

    user_content = payload["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# =====================================================================
# DetectedObject.validate tests
# =====================================================================

def test_detected_object_validate_success():
    """Verify valid DetectedObject passes validation."""
    obj = DetectedObject(label="car", box_2d=[100, 200, 600, 800])
    obj.validate(allowed_labels=["car", "truck"])

    # CVAT rect conversion
    rect = obj.to_cvat_rect(width=1000, height=1000)
    assert rect["type"] == "rectangle"
    assert rect["label"] == "car"
    assert rect["points"] == [200.0, 100.0, 800.0, 600.0]


def test_detected_object_validate_invalid_label():
    """Verify invalid labels raise ValueError."""
    # Empty string
    with pytest.raises(ValueError, match="Invalid label"):
        DetectedObject(label="", box_2d=[0, 0, 100, 100]).validate()

    # Whitespace only
    with pytest.raises(ValueError, match="Invalid label"):
        DetectedObject(label="   ", box_2d=[0, 0, 100, 100]).validate()

    # Disallowed label
    with pytest.raises(ValueError, match="not in allowed labels"):
        DetectedObject(label="bicycle", box_2d=[0, 0, 100, 100]).validate(allowed_labels=["car"])


def test_detected_object_validate_invalid_box_format():
    """Verify invalid box_2d format raises ValueError."""
    # Length not 4
    with pytest.raises(ValueError, match="must be a list/tuple of 4"):
        DetectedObject(label="car", box_2d=[100, 200, 300]).validate()

    # Non-integer float
    with pytest.raises(ValueError, match="must be an integer"):
        DetectedObject(label="car", box_2d=[100.5, 200, 300, 400]).validate()

    # Non-numeric
    with pytest.raises(ValueError, match="must be an integer"):
        DetectedObject(label="car", box_2d=["100", 200, 300, 400]).validate()


def test_detected_object_validate_out_of_bounds():
    """Verify out of bounds coordinates raise ValueError."""
    # Negative coordinate
    with pytest.raises(ValueError, match="Coordinates out of bounds"):
        DetectedObject(label="car", box_2d=[-1, 100, 300, 400]).validate()

    # Coordinate > 1000
    with pytest.raises(ValueError, match="Coordinates out of bounds"):
        DetectedObject(label="car", box_2d=[100, 100, 1001, 400]).validate()


def test_detected_object_validate_inverted_and_degenerate():
    """Verify inverted and degenerate coordinates raise ValueError."""
    # ymin >= ymax
    with pytest.raises(ValueError, match="Invalid vertical coordinates"):
        DetectedObject(label="car", box_2d=[500, 100, 200, 300]).validate()

    with pytest.raises(ValueError, match="Invalid vertical coordinates"):
        DetectedObject(label="car", box_2d=[300, 100, 300, 300]).validate()

    # xmin >= xmax
    with pytest.raises(ValueError, match="Invalid horizontal coordinates"):
        DetectedObject(label="car", box_2d=[100, 500, 300, 200]).validate()

    with pytest.raises(ValueError, match="Invalid horizontal coordinates"):
        DetectedObject(label="car", box_2d=[100, 200, 300, 200]).validate()


# =====================================================================
# parse_vision_response edge case tests
# =====================================================================

def test_parse_valid_json_response():
    """Test parsing clean JSON response."""
    raw = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [150, 250, 450, 650]
            },
            {
                "label": "traffic light",
                "box_2d": [50, 800, 120, 850]
            },
            {
                "label": "person",
                "box_2d": [400, 100, 700, 180]
            }
        ]
    })

    result = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS))
    assert len(result.objects) == 3
    assert result.objects[0].label == "car"
    assert result.objects[0].box_2d == [150, 250, 450, 650]
    assert result.objects[1].label == "traffic light"
    assert result.objects[2].label == "person"


def test_parse_markdown_wrapped_response():
    """Test extracting JSON from markdown code fences."""
    raw = """Here is the detection output:
```json
{
  "objects": [
    {
      "label": "pedestrian",
      "box_2d": [200, 300, 500, 400]
    }
  ]
}
```
Hope this helps!"""

    result = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS))
    assert len(result.objects) == 1
    assert result.objects[0].label == "pedestrian"
    assert result.objects[0].box_2d == [200, 300, 500, 400]


def test_parse_empty_objects():
    """Test handling of empty detections."""
    raw = '{"objects": []}'
    result = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS))
    assert len(result.objects) == 0


def test_parse_vision_response_missing_objects_key():
    """Verify parse_vision_response raises ValueError when 'objects' key is missing."""
    raw = json.dumps({"detections": [{"label": "car", "box_2d": [100, 100, 200, 200]}]})
    with pytest.raises(ValueError, match="Missing 'objects' key"):
        parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS))

    with pytest.raises(ValueError, match="Missing 'objects' key"):
        parse_vision_response("{}", allowed_labels=list(DEFAULT_BBOX_LABELS))


def test_parse_vision_response_objects_not_list():
    """Verify parse_vision_response raises ValueError when 'objects' is not a list."""
    raw_str = json.dumps({"objects": "car"})
    with pytest.raises(ValueError, match="Expected 'objects' to be a list"):
        parse_vision_response(raw_str, allowed_labels=list(DEFAULT_BBOX_LABELS))

    raw_dict = json.dumps({"objects": {"label": "car"}})
    with pytest.raises(ValueError, match="Expected 'objects' to be a list"):
        parse_vision_response(raw_dict, allowed_labels=list(DEFAULT_BBOX_LABELS))


def test_parse_vision_response_non_json():
    """Verify parse_vision_response raises ValueError on non-JSON output."""
    with pytest.raises(ValueError, match="Failed to parse vision model response"):
        parse_vision_response("I cannot process this image.")

    with pytest.raises(ValueError, match="Failed to parse vision model response"):
        parse_vision_response("<html><body>Error</body></html>")

    with pytest.raises(ValueError, match="Failed to parse vision model response"):
        parse_vision_response("")


def test_parse_vision_response_negative_coords():
    """Verify negative coordinates in parse_vision_response."""
    raw = json.dumps({
        "objects": [{"label": "car", "box_2d": [-10, -5, 500, 600]}]
    })
    # In strict mode: raises ValueError
    with pytest.raises(ValueError, match="out of bounds"):
        parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)

    # In non-strict mode: clamps to [0, 0, 500, 600]
    res = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=False)
    assert len(res.objects) == 1
    assert res.objects[0].box_2d == [0, 0, 500, 600]


def test_parse_vision_response_coords_exceeding_1000():
    """Verify coordinates > 1000 in parse_vision_response."""
    raw = json.dumps({
        "objects": [{"label": "car", "box_2d": [100, 200, 1200, 800]}]
    })
    # In strict mode: raises ValueError
    with pytest.raises(ValueError, match="out of bounds"):
        parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)

    # In non-strict mode: clamps to 1000
    res = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=False)
    assert len(res.objects) == 1
    assert res.objects[0].box_2d == [100, 200, 1000, 800]


def test_parse_vision_response_float_coords_rounding():
    """Verify float coordinates round to integers in parse_vision_response."""
    raw = json.dumps({
        "objects": [{"label": "car", "box_2d": [100.4, 200.6, 600.2, 800.8]}]
    })
    res = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].box_2d == [100, 201, 600, 801]


def test_parse_vision_response_single_quotes_and_trailing_commas():
    """Verify single quotes and trailing commas in parse_vision_response."""
    raw = "{'objects': [{'label': 'car', 'box_2d': [100, 200, 300, 400],},],}"
    res = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)
    assert len(res.objects) == 1
    assert res.objects[0].label == "car"
    assert res.objects[0].box_2d == [100, 200, 300, 400]


def test_parse_disallowed_label_rejection():
    """Test rejection when model outputs unallowed label."""
    raw = json.dumps({
        "objects": [
            {
                "label": "airplane",  # Not in allowed 13 bbox labels
                "box_2d": [100, 100, 300, 300]
            }
        ]
    })

    with pytest.raises(ValueError, match="not in allowed list"):
        parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)

    # In non-strict mode, invalid labels are skipped
    result = parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=False)
    assert len(result.objects) == 0


def test_non_bbox_labels_rejected_in_vision_contract():
    """Verify non-bbox labels (road, sky, etc.) are rejected against default bbox list."""
    for non_bbox in ["road", "sky", "building", "pole"]:
        raw = json.dumps({
            "objects": [{"label": non_bbox, "box_2d": [100, 100, 300, 300]}]
        })
        with pytest.raises(ValueError, match="not in allowed list"):
            parse_vision_response(raw, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)


def test_parse_invalid_coordinates():
    """Test rejection of invalid coordinates."""
    raw_inverted = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [500, 100, 200, 300]  # ymin > ymax
            }
        ]
    })

    with pytest.raises(ValueError, match="inverted y coordinates"):
        parse_vision_response(raw_inverted, allowed_labels=list(DEFAULT_BBOX_LABELS), strict=True)


# =====================================================================
# Phase 3B 3-Policy Unified Vision Contract Tests
# =====================================================================

def test_system_prompt_full_31_schema():
    """Verify SYSTEM_PROMPT_FULL_31 defines the 3 strict annotation policies."""
    assert "Policy A — Foreground Instances" in SYSTEM_PROMPT_FULL_31
    assert "Policy B — Semantic Regions" in SYSTEM_PROMPT_FULL_31
    assert "Policy C — Lane Demarcations & Crosswalks" in SYSTEM_PROMPT_FULL_31
    # Check schema keys
    assert '"objects": [' in SYSTEM_PROMPT_FULL_31
    assert '"box_2d":' in SYSTEM_PROMPT_FULL_31
    assert '"regions": [' in SYSTEM_PROMPT_FULL_31
    assert '"polygon":' in SYSTEM_PROMPT_FULL_31
    assert '"lanes": [' in SYSTEM_PROMPT_FULL_31
    assert '"polyline":' in SYSTEM_PROMPT_FULL_31
    # Check that regions explicitly do not request box_2d or mask
    assert "Do NOT provide box_2d or mask for regions" in SYSTEM_PROMPT_FULL_31
    # Check that lanes explicitly do not request polygon, mask, or box_2d
    assert "Do NOT provide polygon, mask, or box_2d for lanes" in SYSTEM_PROMPT_FULL_31


def test_build_full_31_prompt():
    """Verify build_full_31_prompt constructs prompt with exact 3-policy schema."""
    prompt = build_full_31_prompt()
    # Check policy sections
    assert "Policy A — Foreground Instances (14 classes):" in prompt
    assert "Policy B — Semantic Regions (10 classes):" in prompt
    assert "Policy C — Lane Demarcations & Crosswalks (7 classes):" in prompt
    # Check rules
    assert "Do NOT provide 'box_2d' or 'mask' for regions" in prompt
    assert "Do NOT provide 'polygon', 'mask', or 'box_2d' for lanes" in prompt
    # Check labels present in respective sections
    assert "- car" in prompt
    assert "- pedestrian" in prompt
    assert "- road" in prompt
    assert "- vegetation" in prompt
    assert "- lane/single white" in prompt
    assert "- lane/crosswalk" in prompt
    # Schema check
    assert '"objects":' in prompt
    assert '"regions":' in prompt
    assert '"lanes":' in prompt
    assert '"polygon": [[x, y], [x, y], ...]' in prompt
    assert '"polyline": [[x, y], [x, y], ...]' in prompt


def test_build_full_31_prompt_filtered_and_no_conf():
    """Verify build_full_31_prompt with subset of labels and no confidence field."""
    subset = ["car", "road", "lane/single white"]
    prompt = build_full_31_prompt(allowed_labels=subset, include_confidence=False)
    assert "- car" in prompt
    assert "- road" in prompt
    assert "- lane/single white" in prompt
    # Other labels not requested must be omitted
    assert "- pedestrian" not in prompt
    assert "- sky" not in prompt
    assert "- lane/crosswalk" not in prompt
    assert '"confidence"' not in prompt


def test_build_openai_vision_payload_full_31():
    """Verify build_openai_vision_payload correctly uses SYSTEM_PROMPT_FULL_31 for MODE_FULL_31."""
    payload = build_openai_vision_payload(
        image_base64="fake_base64_data",
        mode=MODE_FULL_31,
    )
    assert payload["model"] == "gpt-4o"
    assert payload["response_format"] == {"type": "json_object"}
    messages = payload["messages"]
    assert len(messages) == 2
    # System prompt must be SYSTEM_PROMPT_FULL_31
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT_FULL_31
    # User prompt must contain 3-policy structure
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    text_part = next(c for c in user_content if c["type"] == "text")
    assert "Policy A — Foreground Instances" in text_part["text"]
    assert "Policy B — Semantic Regions" in text_part["text"]
    assert "Policy C — Lane Demarcations & Crosswalks" in text_part["text"]


def test_detected_region_and_lane_dataclasses():
    """Verify validation and CVAT conversion for DetectedRegion and DetectedLane."""
    # 1. DetectedRegion
    reg = DetectedRegion(
        label="road",
        polygon=[[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
        confidence=0.92,
    )
    reg.validate(allowed_labels=ALL_31_LABELS)
    cvat_poly = reg.to_cvat_polygon(width=1280, height=720)
    assert cvat_poly["type"] == "polygon"
    assert cvat_poly["label"] == "road"
    assert cvat_poly["confidence"] == 0.92
    assert cvat_poly["points"] == [0.0, 360.0, 1280.0, 360.0, 1280.0, 720.0, 0.0, 720.0]

    cvat_mask = reg.to_cvat_mask(width=1280, height=720)
    assert cvat_mask is not None
    assert cvat_mask["type"] == "mask"
    assert cvat_mask["label"] == "road"

    # Region validation error on < 3 points
    with pytest.raises(ValueError, match="at least 3 vertices"):
        DetectedRegion(label="road", polygon=[[0, 500], [1000, 500]]).validate()

    # 2. DetectedLane
    lane = DetectedLane(
        label="lane/single white",
        polyline=[[500, 400], [400, 900]],
        confidence=0.88,
    )
    lane.validate(allowed_labels=ALL_31_LABELS)
    cvat_line = lane.to_cvat_polyline(width=1280, height=720)
    assert cvat_line["type"] == "polyline"
    assert cvat_line["label"] == "lane/single white"
    assert cvat_line["confidence"] == 0.88
    assert cvat_line["points"] == [640.0, 288.0, 512.0, 648.0]

    # Lane validation error on < 2 points
    with pytest.raises(ValueError, match="at least 2 vertices"):
        DetectedLane(label="lane/single white", polyline=[[500, 400]]).validate()


def test_parse_vision_response_full_31_unified():
    """Verify parse_vision_response correctly parses unified 3-policy response."""
    raw = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 200, 300, 400],
                "mask": [[200, 100], [400, 100], [400, 300], [200, 300]],
                "confidence": 0.95,
            }
        ],
        "regions": [
            {
                "label": "road",
                "polygon": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
                "confidence": 0.90,
            }
        ],
        "lanes": [
            {
                "label": "lane/single white",
                "polyline": [[500, 500], [400, 1000]],
                "confidence": 0.88,
            }
        ],
    })

    result = parse_vision_response(raw, allowed_labels=ALL_31_LABELS)
    assert len(result.objects) == 1
    assert len(result.regions) == 1
    assert len(result.lanes) == 1

    assert result.objects[0].label == "car"
    assert result.objects[0].box_2d == [100, 200, 300, 400]
    assert result.objects[0].mask == [[200, 100], [400, 100], [400, 300], [200, 300]]
    assert result.objects[0].confidence == 0.95

    assert result.regions[0].label == "road"
    assert result.regions[0].polygon == [[0, 500], [1000, 500], [1000, 1000], [0, 1000]]
    assert result.regions[0].confidence == 0.90

    assert result.lanes[0].label == "lane/single white"
    assert result.lanes[0].polyline == [[500, 500], [400, 1000]]
    assert result.lanes[0].confidence == 0.88

    # Verify conversion to full CVAT annotations
    cvat_annos = result.to_cvat_annotations(width=1280, height=720)
    # Policy A emits rectangle + mask (2 shapes)
    # Policy B emits polygon + mask (2 shapes)
    # Policy C emits polyline (1 shape)
    # Total = 5 shapes
    assert len(cvat_annos) == 5

    types = [a["type"] for a in cvat_annos]
    assert types.count("rectangle") == 1
    assert types.count("mask") == 2
    assert types.count("polygon") == 1
    assert types.count("polyline") == 1

    # Check grouping: instance rect & mask have same group_id
    inst_shapes = [a for a in cvat_annos if a["label"] == "car"]
    assert len(inst_shapes) == 2
    assert inst_shapes[0]["group_id"] == inst_shapes[1]["group_id"]
    assert inst_shapes[0]["group_id"] > 0

    # Region poly & mask have same group_id
    reg_shapes = [a for a in cvat_annos if a["label"] == "road"]
    assert len(reg_shapes) == 2
    assert reg_shapes[0]["group_id"] == reg_shapes[1]["group_id"]
    assert reg_shapes[0]["group_id"] > 0

    # Lane has no group_id (or None)
    lane_shape = next(a for a in cvat_annos if a["label"] == "lane/single white")
    assert lane_shape["type"] == "polyline"
    assert lane_shape.get("group_id") is None

