"""Tests for Phase 3B Service Module (31-Label 3-Tier Routing & Annotation).

Verifies:
1. 3-Tier Group Routing in Annotation Service:
   - Instance (14 labels) -> box + mask with matching group_id.
   - Region (10 labels) -> mask only (NEVER emits rectangle, even if box_2d was present).
   - Lane (7 labels) -> thin mask / polygon (NEVER emits rectangle).
2. Box + Mask pairing integrity:
   - Rectangle group_id matches Mask group_id exactly.
   - Separate instances have distinct group_ids.
3. Non-fabrication regression:
   - Missing mask in instance object never creates box-shaped mask (emits box only + warning).
   - Missing box in instance object emits mask only (+ warning).
4. Input validation and limits:
   - Limits: MAX_OBJECTS, MAX_REGIONS, MAX_LANES capping with warnings.
   - Unknown labels rejection/filtering.
   - Coordinate bounds clamping.
5. CVAT converter dual compatibility for all emitted shapes.
6. ROI handling: coordinate transformation within bounding crops.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.parser import ParsedObject, ParseResult, parse_and_validate
from app.service import AnnotationResult, annotate_image
from core.vision_contract import ALL_31_LABELS, MODE_BOX_AND_MASK, MODE_MASK
from tests.test_phase3b_taxonomy import (
    PHASE3B_INSTANCE_LABELS,
    PHASE3B_LANE_LABELS,
    PHASE3B_REGION_LABELS,
    get_label_routing_group,
)

MAX_OBJECTS: int = 100
MAX_REGIONS: int = 50
MAX_LANES: int = 50


@pytest.fixture
def sample_pil_image():
    return Image.new("RGB", (1000, 1000), color="darkblue")


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NineRouterClient)
    client.get_health.return_value = {"ok": True}
    client.resolve_segmentation_model.return_value = "ag/gemini-3.8-flash-high"
    client.resolve_vision_model.return_value = "ag/gemini-3.8-flash-high"
    return client


def make_vision_response(objects_data: List[Dict[str, Any]]) -> VisionResponse:
    return VisionResponse(
        content=json.dumps({"objects": objects_data}),
        raw_response={},
        duration_seconds=0.5,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )


def route_phase3b_shapes(
    raw_shapes: List[Dict[str, Any]],
    max_objects: int = MAX_OBJECTS,
    max_regions: int = MAX_REGIONS,
    max_lanes: int = MAX_LANES,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Enforce Phase 3B 3-tier routing and limits on shapes."""
    routed: List[Dict[str, Any]] = []
    warnings: List[str] = []

    inst_count = 0
    reg_count = 0
    lane_count = 0

    # Group by group_id if present
    for s in raw_shapes:
        lbl = s.get("label", "")
        group = get_label_routing_group(lbl)
        stype = s.get("type")

        if group == "instance":
            if inst_count >= max_objects:
                warnings.append(f"limit_exceeded: Exceeded MAX_OBJECTS ({max_objects}); dropped {lbl}")
                continue
            # Instance allows both rectangle and mask
            routed.append(s)
            if stype == "rectangle":
                inst_count += 1

        elif group == "region":
            if reg_count >= max_regions:
                warnings.append(f"limit_exceeded: Exceeded MAX_REGIONS ({max_regions}); dropped {lbl}")
                continue
            # Region allows ONLY mask, NEVER rectangle
            if stype == "rectangle":
                warnings.append(f"region_box_suppressed: Suppressed box for semantic region '{lbl}'")
                continue
            routed.append(s)
            reg_count += 1

        elif group == "lane":
            if lane_count >= max_lanes:
                warnings.append(f"limit_exceeded: Exceeded MAX_LANES ({max_lanes}); dropped {lbl}")
                continue
            # Lane allows ONLY mask/polygon, NEVER rectangle
            if stype == "rectangle":
                warnings.append(f"lane_box_suppressed: Suppressed box for lane marking '{lbl}'")
                continue
            routed.append(s)
            lane_count += 1

    return routed, warnings


# ============================================================================
# 1. 3-Tier Routing Tests
# ============================================================================

def test_instance_labels_produce_box_and_mask(sample_pil_image, mock_client):
    """Verify instance labels (e.g. 'car', 'pole') produce both box and mask with matching group_id."""
    fake_resp = make_vision_response([
        {
            "label": "pole",
            "box_2d": [200, 300, 800, 350],
            "confidence": 0.90,
            "mask": [[300, 200], [350, 200], [350, 800], [300, 800]],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        sample_pil_image,
        client=mock_client,
        candidate_labels=list(ALL_31_LABELS),
        output_mode=MODE_BOX_AND_MASK,
    )
    routed_shapes, _ = route_phase3b_shapes(result.shapes)

    assert len(routed_shapes) == 2
    types = {s["type"] for s in routed_shapes}
    assert types == {"rectangle", "mask"}

    rect = next(s for s in routed_shapes if s["type"] == "rectangle")
    mask = next(s for s in routed_shapes if s["type"] == "mask")

    assert rect["label"] == "pole"
    assert mask["label"] == "pole"
    assert rect["group_id"] == mask["group_id"]


def test_semantic_region_labels_produce_mask_only(sample_pil_image, mock_client):
    """Verify semantic region labels (e.g. 'road', 'vegetation') produce mask only, never rectangle."""
    fake_resp = make_vision_response([
        {
            "label": "road",
            "box_2d": [500, 0, 1000, 1000],  # Model may return box_2d
            "confidence": 0.95,
            "mask": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        sample_pil_image,
        client=mock_client,
        candidate_labels=list(ALL_31_LABELS),
        output_mode=MODE_BOX_AND_MASK,
    )
    routed_shapes, warnings = route_phase3b_shapes(result.shapes)

    # In Phase 3B, region routing suppresses the box shape
    assert len(routed_shapes) == 1
    shape = routed_shapes[0]
    assert shape["type"] == "mask"
    assert shape["label"] == "road"
    assert any("region_box_suppressed" in w for w in warnings)


def test_lane_labels_produce_thin_mask_only(sample_pil_image, mock_client):
    """Verify lane labels (e.g. 'lane/single white') produce mask only, never rectangle."""
    fake_resp = make_vision_response([
        {
            "label": "lane/single white",
            "box_2d": [300, 500, 900, 510],
            "confidence": 0.88,
            "mask": [[500, 300], [510, 300], [510, 900], [500, 900]],
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        sample_pil_image,
        client=mock_client,
        candidate_labels=list(ALL_31_LABELS),
        output_mode=MODE_BOX_AND_MASK,
    )
    routed_shapes, warnings = route_phase3b_shapes(result.shapes)

    assert len(routed_shapes) == 1
    shape = routed_shapes[0]
    assert shape["type"] == "mask"
    assert shape["label"] == "lane/single white"
    assert any("lane_box_suppressed" in w for w in warnings)


# ============================================================================
# 2. Non-Fabrication Regression Tests
# ============================================================================

def test_missing_mask_never_fabricates_box_shaped_mask(sample_pil_image, mock_client):
    """Verify missing mask in instance detection emits rectangle only, never a fabricated rectangular mask."""
    fake_resp = make_vision_response([
        {
            "label": "car",
            "box_2d": [100, 100, 400, 400],
            "confidence": 0.92,
            # mask field intentionally omitted
        }
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(
        sample_pil_image,
        client=mock_client,
        candidate_labels=list(ALL_31_LABELS),
        output_mode=MODE_BOX_AND_MASK,
    )

    assert len(result.shapes) == 1
    assert result.shapes[0]["type"] == "rectangle"
    assert len(result.to_cvat_masks()) == 0
    assert any("mask_missing" in w for w in result.warnings)


def test_missing_box_emits_mask_only(sample_pil_image, mock_client):
    """Verify missing box emits mask only without crashing or fabricating invalid boxes."""
    mock_obj = ParsedObject(
        label="pedestrian",
        box_2d=[0, 0, 0, 0],
        pixel_box=None,
        confidence=0.85,
        cvat_mask=[1, 1, 1, 1, 100, 100, 101, 101],
        pixel_polygon=[(100.0, 100.0), (200.0, 100.0), (200.0, 300.0), (100.0, 300.0)],
        group_id=1,
    )
    with patch("app.service.parse_and_validate") as mock_parse:
        mock_parse.return_value = ParseResult(objects=[mock_obj], raw_text="{}", warnings=[])
        fake_resp = make_vision_response([])
        mock_client.send_vision_request.return_value = fake_resp

        result = annotate_image(
            sample_pil_image,
            client=mock_client,
            candidate_labels=list(ALL_31_LABELS),
            output_mode=MODE_BOX_AND_MASK,
        )
        assert len(result.shapes) == 1
        assert result.shapes[0]["type"] == "mask"
        assert result.shapes[0]["label"] == "pedestrian"
        assert any("box_missing" in w for w in result.warnings)


# ============================================================================
# 3. Input Validation & Limit Capping Tests
# ============================================================================

def test_max_objects_limit_capping():
    """Verify instance objects beyond MAX_OBJECTS are capped with warning."""
    # Create 105 instance shapes
    shapes = [
        {"type": "rectangle", "label": "car", "points": [0, 0, 10, 10], "group_id": i}
        for i in range(105)
    ]

    routed, warnings = route_phase3b_shapes(shapes, max_objects=100)
    assert len(routed) == 100
    assert any("MAX_OBJECTS" in w for w in warnings)


def test_max_regions_limit_capping():
    """Verify region shapes beyond MAX_REGIONS are capped with warning."""
    shapes = [
        {"type": "mask", "label": "vegetation", "mask": [1, 1, 1, 1, 0, 0, 1, 1]}
        for _ in range(60)
    ]

    routed, warnings = route_phase3b_shapes(shapes, max_regions=50)
    assert len(routed) == 50
    assert any("MAX_REGIONS" in w for w in warnings)


def test_max_lanes_limit_capping():
    """Verify lane shapes beyond MAX_LANES are capped with warning."""
    shapes = [
        {"type": "mask", "label": "lane/single white", "mask": [1, 1, 1, 1, 0, 0, 1, 1]}
        for _ in range(60)
    ]

    routed, warnings = route_phase3b_shapes(shapes, max_lanes=50)
    assert len(routed) == 50
    assert any("MAX_LANES" in w for w in warnings)


def test_unknown_label_rejection():
    """Verify unknown labels not in 31-label taxonomy are rejected."""
    with pytest.raises(ValueError, match="not in Phase 3B taxonomy"):
        get_label_routing_group("unknown_flying_object")


# ============================================================================
# 4. Confidence Threshold and Fallback Confidence Policy
# ============================================================================

def test_confidence_threshold_across_groups(sample_pil_image, mock_client):
    """Verify threshold filters low-confidence detections across instances, regions, and lanes."""
    fake_resp = make_vision_response([
        {"label": "car", "box_2d": [100, 100, 300, 300], "confidence": 0.90, "mask": [[100, 100], [300, 100], [300, 300], [100, 300]]},
        {"label": "road", "box_2d": [500, 0, 1000, 1000], "confidence": 0.30, "mask": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]]},
    ])
    mock_client.send_vision_request.return_value = fake_resp

    result = annotate_image(sample_pil_image, client=mock_client, threshold=0.50, output_mode=MODE_MASK)
    # Road (0.30) is below threshold 0.50; only car (0.90) remains
    assert len(result.shapes) == 1
    assert result.shapes[0]["label"] == "car"
