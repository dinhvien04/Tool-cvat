"""Tests for Phase 3 service module (Box + Mask Annotation Service)."""

import io
import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.service import AnnotationResult, annotate_image
from core.vision_contract import MODE_BOX, MODE_BOX_AND_MASK, MODE_MASK


@pytest.fixture
def sample_pil_image():
    return Image.new("RGB", (800, 600), color="blue")


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NineRouterClient)
    client.get_health.return_value = {"ok": True}
    return client


def make_vision_response(objects_data, model="ag/gemini-3.8-flash-high"):
    return VisionResponse(
        content=json.dumps({"objects": objects_data}),
        raw_response={},
        duration_seconds=0.65,
        model=model,
        status_code=200,
    )


# ============================================================================
# 1. Test annotate_image with output_mode="box"
# ============================================================================


def test_annotate_image_output_mode_box(sample_pil_image, mock_client):
    """Verify output_mode='box' returns pure CVAT rectangle shapes."""
    fake_resp = make_vision_response([
        {
            "label": "car",
            "box_2d": [100, 200, 400, 600],
            "confidence": 0.95,
            "mask": [
                [200, 100],
                [600, 100],
                [600, 400],
                [200, 400],
            ],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="box",
    )

    assert isinstance(result, AnnotationResult)
    assert result.mode == "box"
    assert len(result.shapes) == 1

    shape = result.shapes[0]
    assert shape["type"] == "rectangle"
    assert shape["label"] == "car"
    assert shape["confidence"] == "0.95"
    assert "mask" not in shape
    assert len(shape["points"]) == 4

    # Helper methods
    assert len(result.to_cvat_rectangles()) == 1
    assert len(result.to_cvat_masks()) == 0


# ============================================================================
# 2. Test annotate_image with output_mode="mask"
# ============================================================================


def test_annotate_image_output_mode_mask(sample_pil_image, mock_client):
    """Verify output_mode='mask' returns native CVAT mask shapes with flat list and bbox."""
    fake_resp = make_vision_response([
        {
            "label": "car",
            "box_2d": [100, 200, 400, 600],
            "confidence": 0.95,
            "mask": [
                [200, 100],
                [600, 100],
                [600, 400],
                [200, 400],
            ],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="mask",
    )

    assert isinstance(result, AnnotationResult)
    assert result.mode == "mask"
    assert len(result.shapes) == 1

    shape = result.shapes[0]
    assert shape["type"] == "mask"
    assert shape["label"] == "car"
    assert shape["confidence"] == "0.95"
    assert "mask" in shape
    assert "points" in shape
    assert isinstance(shape["points"], list)
    assert len(shape["points"]) >= 6  # Flattened polygon vertices [x1, y1, x2, y2, ...]

    mask_list = shape["mask"]
    assert isinstance(mask_list, list)
    assert len(mask_list) > 4

    # Check trailing bounding box [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = mask_list[-4:]
    assert 0 <= xmin <= xmax < 800
    assert 0 <= ymin <= ymax < 600

    # Check pixel count matches crop dimensions
    crop_w = xmax - xmin + 1
    crop_h = ymax - ymin + 1
    expected_len = crop_w * crop_h + 4
    assert len(mask_list) == expected_len

    # Helper methods
    assert len(result.to_cvat_rectangles()) == 0
    assert len(result.to_cvat_masks()) == 1


# ============================================================================
# 3. Test annotate_image with output_mode="box_and_mask"
# ============================================================================


def test_annotate_image_output_mode_box_and_mask(sample_pil_image, mock_client):
    """Verify output_mode='box_and_mask' returns both rectangle and mask shapes with group_id."""
    fake_resp = make_vision_response([
        {
            "label": "pedestrian",
            "box_2d": [200, 300, 500, 400],
            "confidence": 0.88,
            "mask": [
                [300, 200],
                [400, 200],
                [400, 500],
                [300, 500],
            ],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="box_and_mask",
    )

    assert isinstance(result, AnnotationResult)
    assert result.mode == "box_and_mask"
    # Returns 2 shapes: 1 rectangle, 1 mask
    assert len(result.shapes) == 2

    rect_shapes = result.to_cvat_rectangles()
    mask_shapes = result.to_cvat_masks()
    assert len(rect_shapes) == 1
    assert len(mask_shapes) == 1

    rect = rect_shapes[0]
    mask = mask_shapes[0]

    assert rect["type"] == "rectangle"
    assert rect["label"] == "pedestrian"
    assert rect["confidence"] == "0.88"

    assert mask["type"] == "mask"
    assert mask["label"] == "pedestrian"
    assert mask["confidence"] == "0.88"

    # Both share the same group_id for instance correlation
    assert "group_id" in rect
    assert "group_id" in mask
    assert rect["group_id"] == mask["group_id"]


# ============================================================================
# 4. Mode Alias and Backward Compatibility Tests
# ============================================================================


def test_annotate_image_mode_alias_support(sample_pil_image, mock_client):
    """Verify 'mode' parameter acts as an alias for 'output_mode'."""
    fake_resp = make_vision_response([
        {
            "label": "bus",
            "box_2d": [100, 100, 500, 500],
            "mask": [[100, 100], [500, 100], [500, 500], [100, 500]],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    # Calling with mode='mask'
    res_alias = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        mode="mask",
    )
    assert res_alias.mode == "mask"
    assert len(res_alias.shapes) == 1
    assert res_alias.shapes[0]["type"] == "mask"


# ============================================================================
# 5. Non-Fabrication of Masks from Bounding Boxes (Issue 1 & 2 Regressions)
# ============================================================================


def test_annotate_image_mask_mode_rejects_missing_contour_without_fabrication(sample_pil_image, mock_client):
    """Verify mask mode rejects detection and adds warning when model omits contour (no box-to-mask fallback)."""
    fake_resp = make_vision_response([
        {
            "label": "bicycle",
            "box_2d": [150, 200, 450, 500],
            "confidence": 0.82,
            # 'mask' is deliberately omitted by model
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="mask",
    )

    # In mask mode, missing contour must reject annotation completely
    assert len(result.shapes) == 0
    assert any("mask_missing" in w and "bicycle" in w for w in result.warnings)


def test_annotate_image_box_and_mask_mode_emits_box_only_when_contour_missing(sample_pil_image, mock_client):
    """Verify box_and_mask mode emits rectangle only with warning when model omits contour."""
    fake_resp = make_vision_response([
        {
            "label": "truck",
            "box_2d": [100, 100, 400, 500],
            "confidence": 0.91,
            # 'mask' is omitted
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="box_and_mask",
    )

    # Emits rectangle ONLY; does NOT invent rectangular mask
    assert len(result.shapes) == 1
    rect = result.shapes[0]
    assert rect["type"] == "rectangle"
    assert rect["label"] == "truck"
    assert "mask" not in rect
    assert len(result.to_cvat_masks()) == 0
    assert any("mask_missing" in w and "truck" in w for w in result.warnings)


def test_cvat_detection_result_converter_dual_path_contract(sample_pil_image, mock_client):
    """Verify emitted mask shape satisfies both paths of CVAT DetectionResultConverter:
    1. conv_mask_to_poly=True  -> uses anno['points'] to produce polygon
    2. conv_mask_to_poly=False -> uses anno['mask'] to produce native mask
    """
    fake_resp = make_vision_response([
        {
            "label": "car",
            "box_2d": [100, 200, 400, 600],
            "confidence": 0.95,
            "mask": [
                [200, 100],
                [600, 100],
                [600, 400],
                [200, 400],
            ],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="mask",
    )

    assert len(result.shapes) == 1
    anno = result.shapes[0]

    # Both keys must be present in detector annotation
    assert anno["type"] == "mask"
    assert "mask" in anno
    assert "points" in anno

    # Simulate CVAT DetectionResultConverter (views.py:893-911)
    # Path A: conv_mask_to_poly = True
    shape_a = {
        "type": anno["type"],
        "points": (anno.get("mask", []) if anno["type"] == "mask" else anno.get("points", [])),
    }
    if anno["type"] == "mask" and "points" in anno and True:  # conv_mask_to_poly = True
        shape_a["type"] = "polygon"
        shape_a["points"] = anno["points"]

    assert shape_a["type"] == "polygon"
    assert len(shape_a["points"]) >= 6
    assert isinstance(shape_a["points"][0], (int, float))

    # Path B: conv_mask_to_poly = False
    shape_b = {
        "type": anno["type"],
        "points": (anno.get("mask", []) if anno["type"] == "mask" else anno.get("points", [])),
    }
    if anno["type"] == "mask" and "points" in anno and False:  # conv_mask_to_poly = False
        shape_b["type"] = "polygon"
        shape_b["points"] = anno["points"]
    elif anno["type"] == "mask":
        [xtl, ytl, xbr, ybr] = shape_b["points"][-4:]
        cut_points = shape_b["points"][:-4]
        # Verify cut_points is valid crop mask and coordinates form valid bounding box
        assert len(cut_points) > 0
        assert xtl <= xbr
        assert ytl <= ybr

    assert shape_b["type"] == "mask"


def test_annotate_image_box_and_mask_invalid_box_emits_mask_only(sample_pil_image, mock_client):
    """Verify that in box_and_mask mode, an object with an invalid box but valid mask emits mask ONLY."""
    from app.parser import ParsedObject, ParseResult

    # Simulate an object with no valid pixel_box but a valid cvat_mask and pixel_polygon
    mock_obj = ParsedObject(
        label="car",
        box_2d=[0, 0, 0, 0],
        pixel_box=None,  # No valid box
        confidence=0.90,
        cvat_mask=[1, 1, 1, 1, 10, 20, 30, 40],
        pixel_polygon=[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
        group_id=1,
    )

    with patch("app.service.parse_and_validate") as mock_parse:
        mock_parse.return_value = ParseResult(objects=[mock_obj], raw_text="{}", warnings=[])
        fake_resp = make_vision_response([])
        mock_client.send_vision_request.return_value = fake_resp

        result = annotate_image(
            image_source=sample_pil_image,
            client=mock_client,
            output_mode="box_and_mask",
        )

        assert len(result.shapes) == 1
        shape = result.shapes[0]
        assert shape["type"] == "mask"
        assert shape["label"] == "car"
        assert "points" in shape
        assert any("box_missing" in w and "car" in w for w in result.warnings)


# ============================================================================
# 6. Confidence Threshold Filtering with Masks
# ============================================================================


def test_annotate_image_mask_threshold_filtering(sample_pil_image, mock_client):
    """Verify confidence threshold filtering applies correctly to mask detections."""
    fake_resp = make_vision_response([
        {"label": "car", "box_2d": [100, 100, 300, 300], "confidence": 0.92, "mask": [[100, 100], [300, 100], [300, 300], [100, 300]]},
        {"label": "bicycle", "box_2d": [400, 400, 600, 600], "confidence": 0.45, "mask": [[400, 400], [600, 400], [600, 600], [400, 600]]},
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        threshold=0.50,
        output_mode="box_and_mask",
    )

    # Bicycle (0.45) is filtered out; only car (0.92) remains
    assert len(result.shapes) == 2  # 1 rectangle + 1 mask for 'car'
    labels = [s["label"] for s in result.shapes]
    assert all(label == "car" for label in labels)


# ============================================================================
# 7. Vision Prompt Construction for Mask vs Box Modes
# ============================================================================


def test_annotate_image_prompts_for_mask_mode(sample_pil_image, mock_client):
    """Verify prompt sent to 9Router includes instance segmentation instructions when in mask mode."""
    fake_resp = make_vision_response([])
    mock_client.send_vision_request.return_value = fake_resp

    annotate_image(
        image_source=sample_pil_image,
        client=mock_client,
        output_mode="mask",
    )

    call_args = mock_client.send_vision_request.call_args
    sent_prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
    assert "mask" in sent_prompt.lower()
    assert "segmentation" in sent_prompt.lower()
