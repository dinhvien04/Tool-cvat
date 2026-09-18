"""Section 21 Comprehensive Regression Test Suite for Tool-cvat.

Directly verifies and enforces all 26 requirements specified in Section 21:
--------------------------------------------------------------------------------
1. Mode Propagation Tests (Requirements 1–3):
   - Req 1: annotate_image(full_31) -> send_vision_request receives mode=MODE_FULL_31.
   - Req 2: Payload system message matches SYSTEM_PROMPT_FULL_31.
   - Req 3: Payload does NOT contain bbox-only system contract (SYSTEM_PROMPT).

2. Parser Strictness Tests (Requirements 4–12):
   - Req 4: Policy A object requires box (missing box_2d is rejected/dropped).
   - Req 5: Policy A object requires mask (missing mask is rejected/dropped).
   - Req 6: Policy A object polygon is rejected (cannot substitute polygon for mask).
   - Req 7: Policy B region requires polygon (missing polygon is rejected/dropped).
   - Req 8: Policy B region mask-only is rejected (cannot substitute mask for polygon).
   - Req 9: Policy C lane requires polyline (missing polyline is rejected/dropped).
   - Req 10: Policy C lane mask is rejected (cannot supply mask for lane).
   - Req 11: Policy C lane polygon is rejected (cannot supply polygon for lane).
   - Req 12: Policy C lane polyline is never rasterized as polygon mask.

3. Polyline Geometry Tests (Requirements 13–16):
   - Req 13: Model polyline maps directly to CVAT pixel coordinates.
   - Req 14: Polyline point order is strictly preserved.
   - Req 15: Crosswalk polyline direct path without polygon/box conversion.
   - Req 16: No lane_shape_pipeline call for valid current polyline.

4. Strict Atomicity Tests (Requirements 17–21):
   - Req 17: DetectedObject mask fail -> no rectangle emitted.
   - Req 18: DetectedRegion mask derivation fail -> no polygon emitted.
   - Req 19: Valid object -> exactly rectangle + mask with identical integer group_id > 0.
   - Req 20: Valid region -> exactly polygon + mask with identical integer group_id > 0.
   - Req 21: Valid lane -> exactly polyline with no group_id.

5. Feedback Matching & Diffing Tests (Requirements 22–26):
   - Req 22: Policy A box low IoU but mask strong match -> same annotation (BOX_MOVE).
   - Req 23: Policy B polygon changed but mask overlaps -> REGION_EDIT, not DELETE + ADD.
   - Req 24: Policy B mask changed but polygon overlaps -> REGION_EDIT.
   - Req 25: Lane shifted moderately -> LANE_EDIT.
   - Req 26: Lane relabel with same geometry -> RELABEL.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.client import NineRouterClient
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
from app.parser import (
    ParsedObject,
    ParseResult,
    VisionParseError,
    parse_and_validate,
)
from app.service import (
    annotate_image,
    polyline_to_cvat_polyline,
    route_phase3b_shapes,
)
from core.taxonomy import (
    get_taxonomy,
    validate_cvat_output_shapes,
)
from core.vision_contract import (
    MODE_BOX,
    MODE_FULL_31,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FULL_31,
    DetectedLane,
    DetectedObject,
    DetectedRegion,
    DetectionResult,
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def make_policy_a_pair(label: str, gid: int, box: list[float], mask_rle_rect: list[int]) -> list[dict]:
    """Create paired Policy A (rectangle + mask) CVAT shape dictionary."""
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
            "mask": [1] * (w * h) + [int(x1), int(y1), int(x2), int(y2)],
            "group_id": gid,
        },
    ]


def make_policy_b_pair(label: str, gid: int, poly_pts: list[float], mask_rle_rect: list[int]) -> list[dict]:
    """Create paired Policy B (polygon + mask) CVAT shape dictionary."""
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
            "mask": [1] * (w * h) + [int(x1), int(y1), int(x2), int(y2)],
            "group_id": gid,
        },
    ]


# ==============================================================================
# Group 1: Mode Propagation Tests (Requirements 1–3)
# ==============================================================================

def test_req01_annotate_image_propagates_mode_full_31():
    """Req 1: annotate_image(full_31) forwards mode=MODE_FULL_31 to send_vision_request."""
    mock_client = MagicMock()
    mock_client.send_vision_request.return_value = MagicMock(
        content='{"objects": [], "regions": [], "lanes": []}',
        raw_content='{"objects": [], "regions": [], "lanes": []}',
    )

    img = Image.new("RGB", (640, 480), color="white")
    annotate_image(
        image_source=img,
        client=mock_client,
        mode=MODE_FULL_31,
    )

    mock_client.send_vision_request.assert_called_once()
    call_kwargs = mock_client.send_vision_request.call_args[1]
    assert call_kwargs.get("mode") == MODE_FULL_31


def test_req02_payload_system_message_matches_system_prompt_full_31():
    """Req 2: Vision request payload system message equals SYSTEM_PROMPT_FULL_31 when mode=MODE_FULL_31."""
    client = NineRouterClient(base_url="http://mock-9router:8080/v1", api_key="test_key")
    with patch.object(client.session, "post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"objects": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )
        client.send_vision_request(
            model="ag/gemini-3.8-flash-high",
            image_bytes_or_b64=b"dummy_bytes",
            mode=MODE_FULL_31,
        )

        assert mock_post.called
        payload = mock_post.call_args[1]["json"]
        messages = payload.get("messages", [])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == SYSTEM_PROMPT_FULL_31


def test_req03_payload_does_not_contain_bbox_only_system_contract():
    """Req 3: Vision request payload does NOT contain bbox-only system contract (SYSTEM_PROMPT)."""
    client = NineRouterClient(base_url="http://mock-9router:8080/v1", api_key="test_key")
    with patch.object(client.session, "post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"objects": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )
        client.send_vision_request(
            model="ag/gemini-3.8-flash-high",
            image_bytes_or_b64=b"dummy_bytes",
            mode=MODE_FULL_31,
        )

        assert mock_post.called
        payload = mock_post.call_args[1]["json"]
        messages = payload.get("messages", [])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        content = system_msgs[0]["content"]

        # Strictly verify SYSTEM_PROMPT (bbox-only) is not used
        assert content != SYSTEM_PROMPT
        assert "Return ONLY valid JSON with an 'objects' list containing detected objects" not in content
        # Ensure 3-policy structure is described
        assert "Policy A" in content or "objects" in content
        assert "regions" in content
        assert "lanes" in content


# ==============================================================================
# Group 2: Parser Strictness Tests (Requirements 4–12)
# ==============================================================================

def test_req04_parser_object_requires_box():
    """Req 4: Policy A object requires box_2d. Missing box is rejected."""
    raw = json.dumps({
        "objects": [
            {
                "label": "car",
                "mask": [[100, 100], [200, 100], [200, 200], [100, 200]],
            }
        ],
        "regions": [],
        "lanes": [],
    })

    # In strict mode: raises VisionParseError
    with pytest.raises(VisionParseError, match="invalid box_2d"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)

    # In non-strict mode: dropped with warning
    res = parse_and_validate(raw, strict=False, mode=MODE_FULL_31)
    assert len(res.objects) == 0
    assert any("invalid box_2d" in w for w in res.warnings)


def test_req05_parser_object_requires_mask():
    """Req 5: Policy A object requires mask. Missing mask is rejected in full_31 mode."""
    raw = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 100, 200, 200],
            }
        ],
        "regions": [],
        "lanes": [],
    })

    # In strict mode: raises VisionParseError
    with pytest.raises(VisionParseError, match="missing required 'mask' contour per Policy A"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)

    # In non-strict mode: dropped
    res = parse_and_validate(raw, strict=False, mode=MODE_FULL_31)
    assert len(res.objects) == 0
    assert any("missing required 'mask'" in w for w in res.warnings)


def test_req06_parser_object_polygon_rejected():
    """Req 6: Policy A object polygon is rejected. Allowed geometry is mask only."""
    raw = json.dumps({
        "objects": [
            {
                "label": "car",
                "box_2d": [100, 100, 200, 200],
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
            }
        ],
        "regions": [],
        "lanes": [],
    })

    with pytest.raises(VisionParseError, match="prohibited geometry 'polygon' for Policy A object"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)


def test_req07_parser_region_requires_polygon():
    """Req 7: Policy B region requires polygon. Missing polygon is rejected."""
    raw = json.dumps({
        "objects": [],
        "regions": [
            {
                "label": "road",
            }
        ],
        "lanes": [],
    })

    with pytest.raises(VisionParseError, match="missing required 'polygon' field per Policy B"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)


def test_req08_parser_region_mask_only_rejected():
    """Req 8: Policy B region mask-only is rejected. Allowed geometry is polygon only."""
    raw = json.dumps({
        "objects": [],
        "regions": [
            {
                "label": "road",
                "mask": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
            }
        ],
        "lanes": [],
    })

    with pytest.raises(VisionParseError, match="prohibited geometry 'mask' for Policy B region"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)


def test_req09_parser_lane_requires_polyline():
    """Req 9: Policy C lane requires polyline. Missing polyline is rejected."""
    raw = json.dumps({
        "objects": [],
        "regions": [],
        "lanes": [
            {
                "label": "lane/single white",
            }
        ],
    })

    with pytest.raises(VisionParseError, match="missing required 'polyline' field per Policy C"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)


def test_req10_parser_lane_mask_rejected():
    """Req 10: Policy C lane mask is rejected. Allowed geometry is polyline only."""
    raw = json.dumps({
        "objects": [],
        "regions": [],
        "lanes": [
            {
                "label": "lane/single white",
                "mask": [[100, 200], [300, 400], [500, 600]],
            }
        ],
    })

    with pytest.raises(VisionParseError, match="prohibited geometry 'mask' for Policy C lane"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)


def test_req11_parser_lane_polygon_rejected():
    """Req 11: Policy C lane polygon is rejected. Allowed geometry is polyline only."""
    raw = json.dumps({
        "objects": [],
        "regions": [],
        "lanes": [
            {
                "label": "lane/single white",
                "polygon": [[100, 200], [300, 400], [500, 600]],
            }
        ],
    })

    with pytest.raises(VisionParseError, match="prohibited geometry 'polygon' for Policy C lane"):
        parse_and_validate(raw, strict=True, mode=MODE_FULL_31)


def test_req12_parser_lane_polyline_never_rasterized_as_polygon_mask():
    """Req 12: Policy C lane polyline is never rasterized as polygon mask."""
    raw = json.dumps({
        "objects": [],
        "regions": [],
        "lanes": [
            {
                "label": "lane/single white",
                "polyline": [[100, 200], [300, 400], [500, 600]],
            }
        ],
    })

    res = parse_and_validate(raw, image_width=1000, image_height=1000, strict=True, mode=MODE_FULL_31)
    assert len(res.lanes) == 1
    lane_obj = res.lanes[0]
    assert lane_obj.cvat_mask is None
    assert lane_obj.pixel_polygon is None
    assert lane_obj.pixel_polyline is not None

    # Verify downstream shape routing strictly generates polyline
    det_lane = DetectedLane(label=lane_obj.label, polyline=lane_obj.lane_polyline)
    shapes = DetectionResult(lanes=[det_lane]).to_cvat_annotations(width=1000, height=1000)
    routed_shapes, _ = route_phase3b_shapes(shapes)
    assert len(routed_shapes) == 1
    assert routed_shapes[0]["type"] == "polyline"
    assert "mask" not in routed_shapes[0]


# ==============================================================================
# Group 3: Polyline Geometry Tests (Requirements 13–16)
# ==============================================================================

def test_req13_model_polyline_maps_directly_to_cvat_pixels():
    """Req 13: Model polyline coordinates map directly to CVAT pixel coordinates."""
    norm_polyline = [[100, 200], [500, 600], [800, 900]]
    width, height = 1920, 1080

    shape = polyline_to_cvat_polyline(
        polyline=norm_polyline,
        width=width,
        height=height,
        label="lane/single white",
    )

    assert shape is not None
    assert shape["type"] == "polyline"
    assert shape["label"] == "lane/single white"

    # Direct linear projection: px = (norm_x / 1000.0) * width, py = (norm_y / 1000.0) * height
    expected_pts = [
        192.0,  # 100 / 1000 * 1920
        216.0,  # 200 / 1000 * 1080
        960.0,  # 500 / 1000 * 1920
        648.0,  # 600 / 1000 * 1080
        1536.0, # 800 / 1000 * 1920
        972.0,  # 900 / 1000 * 1080
    ]
    assert shape["points"] == expected_pts


def test_req14_polyline_point_order_preserved():
    """Req 14: Polyline point order is strictly preserved (no sorting, reversing, or convex hull)."""
    # Non-monotonic zig-zag polyline
    norm_polyline = [[100, 800], [600, 200], [300, 900], [900, 100]]
    width, height = 1000, 1000

    shape = polyline_to_cvat_polyline(
        polyline=norm_polyline,
        width=width,
        height=height,
        label="lane/single white",
    )

    assert shape is not None
    expected_pts = [100.0, 800.0, 600.0, 200.0, 300.0, 900.0, 900.0, 100.0]
    assert shape["points"] == expected_pts

    # Also test via DetectedLane dataclass
    det_lane = DetectedLane(label="lane/single white", polyline=norm_polyline)
    cvat_lane = det_lane.to_cvat_polyline(width=width, height=height)
    assert cvat_lane["points"] == expected_pts


def test_req15_crosswalk_polyline_direct_path():
    """Req 15: Crosswalk (Policy C) is generated as direct polyline path without polygon/box conversion."""
    norm_polyline = [[200, 700], [800, 700]]
    width, height = 1000, 1000

    shape = polyline_to_cvat_polyline(
        polyline=norm_polyline,
        width=width,
        height=height,
        label="lane/crosswalk",
    )

    assert shape is not None
    assert shape["type"] == "polyline"
    assert shape["label"] == "lane/crosswalk"
    assert shape["points"] == [200.0, 700.0, 800.0, 700.0]
    assert "mask" not in shape
    assert shape.get("group_id") in (None, 0)


def test_req16_no_lane_shape_pipeline_call_for_valid_current_polyline():
    """Req 16: No lane_shape_pipeline call for valid current model polyline."""
    raw_json = json.dumps({
        "objects": [],
        "regions": [],
        "lanes": [{"label": "lane/single white", "polyline": [[100, 200], [500, 600]]}],
    })
    mock_client = MagicMock()
    mock_client.send_vision_request.return_value = MagicMock(content=raw_json, raw_content=raw_json)
    img = Image.new("RGB", (640, 480), color="white")

    with patch("core.line_geometry.lane_shape_pipeline") as mock_pipeline:
        ann_res = annotate_image(
            image_source=img,
            client=mock_client,
            mode=MODE_FULL_31,
        )

        assert len(ann_res.shapes) == 1
        assert ann_res.shapes[0]["type"] == "polyline"
        mock_pipeline.assert_not_called()


# ==============================================================================
# Group 4: Strict Atomicity Tests (Requirements 17–21)
# ==============================================================================

def test_req17_atomicity_object_mask_fail_drops_rectangle():
    """Req 17: DetectedObject mask failure emits no rectangle (strict all-or-nothing)."""
    # Object with valid box but missing/failing mask
    obj = DetectedObject(
        label="car",
        box_2d=[100, 100, 400, 400],
        mask=None,  # No mask
        confidence=0.9,
    )
    det = DetectionResult(objects=[obj])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    # Must be completely dropped (atomicity)
    assert len(shapes) == 0


def test_req18_atomicity_region_mask_fail_drops_polygon():
    """Req 18: DetectedRegion mask derivation failure emits no polygon (strict all-or-nothing)."""
    # Degenerate collinear polygon producing zero area
    reg = DetectedRegion(
        label="road",
        polygon=[[100, 100], [100, 100], [100, 100]],
        confidence=0.9,
    )
    det = DetectionResult(regions=[reg])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 0


def test_req19_atomicity_valid_object_emits_exactly_rectangle_and_mask_same_group():
    """Req 19: Valid Policy A object emits exactly 1 rectangle and 1 mask sharing integer group_id > 0."""
    obj = DetectedObject(
        label="car",
        box_2d=[100, 100, 400, 400],
        mask=[[100, 100], [400, 100], [400, 400], [100, 400]],
        confidence=0.95,
    )
    det = DetectionResult(objects=[obj])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 2
    types = {s["type"] for s in shapes}
    assert types == {"rectangle", "mask"}

    rect = next(s for s in shapes if s["type"] == "rectangle")
    mask = next(s for s in shapes if s["type"] == "mask")

    assert rect["group_id"] == mask["group_id"]
    assert isinstance(rect["group_id"], int)
    assert rect["group_id"] > 0

    # Output validator pass
    validated, dropped = validate_cvat_output_shapes(shapes)
    assert len(validated) == 2
    assert len(dropped) == 0


def test_req20_atomicity_valid_region_emits_exactly_polygon_and_mask_same_group():
    """Req 20: Valid Policy B region emits exactly 1 polygon and 1 mask sharing integer group_id > 0."""
    reg = DetectedRegion(
        label="road",
        polygon=[[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
        confidence=0.98,
    )
    det = DetectionResult(regions=[reg])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 2
    types = {s["type"] for s in shapes}
    assert types == {"polygon", "mask"}

    poly = next(s for s in shapes if s["type"] == "polygon")
    mask = next(s for s in shapes if s["type"] == "mask")

    assert poly["group_id"] == mask["group_id"]
    assert isinstance(poly["group_id"], int)
    assert poly["group_id"] > 0

    # Output validator pass
    validated, dropped = validate_cvat_output_shapes(shapes)
    assert len(validated) == 2
    assert len(dropped) == 0


def test_req21_atomicity_valid_lane_emits_exactly_polyline_no_group():
    """Req 21: Valid Policy C lane emits exactly 1 polyline with no group_id."""
    lane = DetectedLane(
        label="lane/single white",
        polyline=[[100, 500], [900, 500]],
        confidence=0.93,
    )
    det = DetectionResult(lanes=[lane])
    shapes = det.to_cvat_annotations(width=1000, height=1000)

    assert len(shapes) == 1
    assert shapes[0]["type"] == "polyline"
    assert shapes[0]["label"] == "lane/single white"
    assert shapes[0].get("group_id") in (None, 0)

    # Output validator pass
    validated, dropped = validate_cvat_output_shapes(shapes)
    assert len(validated) == 1
    assert len(dropped) == 0


# ==============================================================================
# Group 5: Feedback Matching Tests (Requirements 22–26)
# ==============================================================================

def test_req22_feedback_matching_policy_a_box_moved_mask_matched():
    """Req 22: Policy A box low IoU but mask strong match -> same annotation (BOX_MOVE).

    Prevents spurious DELETE_FALSE_POSITIVE + ADD_MISSING.
    """
    engine = CorrectionDiffEngine()

    # AI prediction: box [100, 100, 300, 300], mask [100, 100, 300, 300]
    ai_shapes = make_policy_a_pair("car", 1, [100.0, 100.0, 300.0, 300.0], [100, 100, 300, 300])

    # Human moved box to [230, 230, 430, 430] (box IoU ~ 0.08 < 0.30 threshold),
    # but left mask identical at [100, 100, 300, 300]
    human_shapes = make_policy_a_pair("car", 1, [230.0, 230.0, 430.0, 430.0], [100, 100, 300, 300])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_BOX_MOVE
    assert diffs[0].ai_label == "car"
    assert diffs[0].human_label == "car"
    assert not any(d.correction_type == CORRECTION_DELETE_FALSE_POSITIVE for d in diffs)
    assert not any(d.correction_type == CORRECTION_ADD_MISSING for d in diffs)


def test_req23_feedback_matching_policy_b_polygon_changed_mask_overlaps_region_edit():
    """Req 23: Policy B polygon changed but mask overlaps -> REGION_EDIT, not DELETE + ADD."""
    engine = CorrectionDiffEngine()

    # AI prediction: road with polygon [100, 100, 500, 100, 500, 500, 100, 500] and mask [100, 100, 500, 500]
    ai_poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    ai_shapes = make_policy_b_pair("road", 1, ai_poly, [100, 100, 500, 500])

    # Human edited polygon contour: notched upper side, polygon IoU ~ 0.75 < 0.92,
    # but mask stays at [100, 100, 500, 500]
    human_poly = [100.0, 100.0, 300.0, 200.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    human_shapes = make_policy_b_pair("road", 1, human_poly, [100, 100, 500, 500])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_REGION_EDIT
    assert diffs[0].details.get("polygon_changed") is True
    assert not any(d.correction_type == CORRECTION_DELETE_FALSE_POSITIVE for d in diffs)
    assert not any(d.correction_type == CORRECTION_ADD_MISSING for d in diffs)


def test_req24_feedback_matching_policy_b_mask_changed_polygon_overlaps_region_edit():
    """Req 24: Policy B mask changed but polygon overlaps -> REGION_EDIT."""
    engine = CorrectionDiffEngine()

    poly = [100.0, 100.0, 500.0, 100.0, 500.0, 500.0, 100.0, 500.0]
    ai_shapes = make_policy_b_pair("road", 1, poly, [100, 100, 500, 500])

    # Human kept polygon identical but edited mask crop boundary (e.g. height cropped from 500 to 350)
    human_shapes = make_policy_b_pair("road", 1, poly, [100, 100, 500, 350])

    diffs = engine.diff(ai_shapes, human_shapes, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_REGION_EDIT
    assert diffs[0].details.get("mask_changed") is True


def test_req25_feedback_matching_lane_shifted_moderately_lane_edit():
    """Req 25: Lane shifted moderately -> LANE_EDIT."""
    engine = CorrectionDiffEngine(lane_tolerance_px=3.0)

    ai_lane = [
        {
            "label": "lane/single white",
            "type": "polyline",
            "points": [100.0, 200.0, 500.0, 200.0],
        }
    ]

    # Human shifted line by 8px vertically (> 3.0px tolerance)
    human_lane = [
        {
            "label": "lane/single white",
            "type": "polyline",
            "points": [100.0, 208.0, 500.0, 208.0],
        }
    ]

    diffs = engine.diff(ai_lane, human_lane, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_LANE_EDIT
    assert diffs[0].details.get("avg_dist") == pytest.approx(8.0, abs=0.5)


def test_req26_feedback_matching_lane_relabel_same_geometry_relabel():
    """Req 26: Lane relabel with same geometry -> RELABEL."""
    engine = CorrectionDiffEngine()

    ai_lane = [
        {
            "label": "lane/single white",
            "type": "polyline",
            "points": [100.0, 200.0, 500.0, 200.0],
        }
    ]

    # Human relabeled geometry from white to yellow
    human_lane = [
        {
            "label": "lane/single yellow",
            "type": "polyline",
            "points": [100.0, 200.0, 500.0, 200.0],
        }
    ]

    diffs = engine.diff(ai_lane, human_lane, image_width=1000, image_height=1000)

    assert len(diffs) == 1
    assert diffs[0].correction_type == CORRECTION_RELABEL
    assert diffs[0].ai_label == "lane/single white"
    assert diffs[0].human_label == "lane/single yellow"
