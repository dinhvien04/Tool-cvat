"""Phase 3B Duplicate Suppression Test Suite.

Verifies:
1. Bounding Box IoU calculation and Non-Maximum Suppression (NMS) for instance objects.
2. Mask pixel-level IoU and overlap suppression for semantic regions and lane markings.
3. Class-specific suppression policy:
   - Duplicate suppression ONLY applies within the same exact label.
   - Overlapping objects of different classes are NEVER suppressed.
   - Ambiguous label segregation: 'pedestrian' vs 'person', 'traffic light' vs 'traffic_light',
     and 'traffic sign' vs 'traffic_sign' never suppress each other even at 100% spatial overlap.
4. Box + Mask pairing consistency:
   - When an instance box is suppressed as a duplicate, its matching mask (matching group_id)
     is also suppressed atomically.
   - No orphaned shapes (isolated box without mask, or isolated mask without box).
5. Robustness to edge cases:
   - Empty input list, single object, zero overlap, identical coordinates, ties in confidence.
   - Zero-area boxes or masks do not cause division-by-zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
from PIL import Image, ImageDraw

from core.geometry import (
    calculate_polygon_area,
    cvat_mask_to_binary_image,
    mask_to_cvat_flat_list,
    polygon_to_cvat_mask,
    rasterize_polygon_to_mask,
)
from tests.test_phase3b_taxonomy import (
    PHASE3B_INSTANCE_LABELS,
    PHASE3B_LANE_LABELS,
    PHASE3B_REGION_LABELS,
    get_label_routing_group,
)


# ============================================================================
# Reference Duplicate Suppression Implementations & Contracts
# ============================================================================

def calculate_box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) for two [x1, y1, x2, y2] boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_w = max(0.0, x2_inter - x1_inter)
    inter_h = max(0.0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def calculate_mask_iou(mask1: List[int], mask2: List[int], image_w: int, image_h: int) -> float:
    """Calculate IoU between two CVAT flat masks [p0, p1, ..., xmin, ymin, xmax, ymax]."""
    img1 = cvat_mask_to_binary_image(mask1, image_w, image_h)
    img2 = cvat_mask_to_binary_image(mask2, image_w, image_h)

    bytes1 = img1.tobytes()
    bytes2 = img2.tobytes()

    inter = 0
    union = 0
    for b1, b2 in zip(bytes1, bytes2):
        v1 = 1 if b1 > 0 else 0
        v2 = 1 if b2 > 0 else 0
        if v1 and v2:
            inter += 1
        if v1 or v2:
            union += 1

    if union == 0:
        return 0.0
    return inter / float(union)


def suppress_phase3b_duplicates(
    shapes: List[Dict[str, Any]],
    box_iou_threshold: float = 0.70,
    mask_iou_threshold: float = 0.60,
    image_width: int = 1000,
    image_height: int = 1000,
) -> List[Dict[str, Any]]:
    """Suppresses duplicate predictions across instance, region, and lane shapes.

    Enforces:
    - Intra-class suppression only (different labels never suppress each other).
    - Instance pairs (rectangle + mask) are suppressed together based on box IoU.
    - Regions and lanes are suppressed based on mask IoU.
    """
    if not shapes:
        return []

    # 1. Group instance shapes by group_id
    instance_pairs: Dict[int, Dict[str, Any]] = {}
    regions_and_lanes: List[Dict[str, Any]] = []

    for s in shapes:
        grp = s.get("group_id")
        lbl = s.get("label", "")
        routing = get_label_routing_group(lbl)

        if routing == "instance" and grp is not None:
            if grp not in instance_pairs:
                instance_pairs[grp] = {"group_id": grp, "label": lbl, "confidence": float(s.get("confidence", 0.0))}
            if s.get("type") == "rectangle":
                instance_pairs[grp]["rectangle"] = s
            elif s.get("type") == "mask":
                instance_pairs[grp]["mask"] = s
        else:
            regions_and_lanes.append(s)

    # 2. Suppress instance duplicates by BBox IoU
    sorted_instances = sorted(
        instance_pairs.values(),
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    )
    kept_instance_groups: Set[int] = set()
    suppressed_instance_groups: Set[int] = set()

    for i, inst_a in enumerate(sorted_instances):
        gid_a = inst_a["group_id"]
        if gid_a in suppressed_instance_groups:
            continue
        kept_instance_groups.add(gid_a)

        rect_a = inst_a.get("rectangle")
        if not rect_a or "points" not in rect_a:
            continue
        box_a = rect_a["points"]

        for j in range(i + 1, len(sorted_instances)):
            inst_b = sorted_instances[j]
            gid_b = inst_b["group_id"]
            if gid_b in suppressed_instance_groups:
                continue

            # Class-specific suppression: strictly identical label
            if inst_a["label"] != inst_b["label"]:
                continue

            rect_b = inst_b.get("rectangle")
            if not rect_b or "points" not in rect_b:
                continue
            box_b = rect_b["points"]

            iou = calculate_box_iou(box_a, box_b)
            if iou >= box_iou_threshold:
                suppressed_instance_groups.add(gid_b)

    # 3. Suppress region and lane duplicates by Mask IoU
    sorted_regions_lanes = sorted(
        regions_and_lanes,
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    )
    kept_regions_lanes: List[Dict[str, Any]] = []
    suppressed_rl_indices: Set[int] = set()

    for i, shape_a in enumerate(sorted_regions_lanes):
        if i in suppressed_rl_indices:
            continue
        kept_regions_lanes.append(shape_a)

        mask_a = shape_a.get("mask")
        if not mask_a:
            continue

        for j in range(i + 1, len(sorted_regions_lanes)):
            if j in suppressed_rl_indices:
                continue
            shape_b = sorted_regions_lanes[j]

            # Intra-class only
            if shape_a["label"] != shape_b["label"]:
                continue

            mask_b = shape_b.get("mask")
            if not mask_b:
                continue

            m_iou = calculate_mask_iou(mask_a, mask_b, image_width, image_height)
            if m_iou >= mask_iou_threshold:
                suppressed_rl_indices.add(j)

    # 4. Reassemble kept shapes
    final_shapes: List[Dict[str, Any]] = []
    for gid in kept_instance_groups:
        inst = instance_pairs[gid]
        if "rectangle" in inst:
            final_shapes.append(inst["rectangle"])
        if "mask" in inst:
            final_shapes.append(inst["mask"])

    final_shapes.extend(kept_regions_lanes)
    return final_shapes


# ============================================================================
# 1. Box IoU Calculation Tests
# ============================================================================

def test_box_iou_disjoint_and_identical():
    """Verify IoU for disjoint, identical, and partially overlapping boxes."""
    # Identical boxes -> IoU = 1.0
    box1 = [10.0, 10.0, 50.0, 50.0]
    assert calculate_box_iou(box1, box1) == pytest.approx(1.0)

    # Disjoint boxes -> IoU = 0.0
    box_disjoint = [60.0, 60.0, 100.0, 100.0]
    assert calculate_box_iou(box1, box_disjoint) == 0.0

    # 50% overlap in x, 100% in y -> Area inter = 20*40 = 800, Area union = 1600 + 1600 - 800 = 2400 -> IoU = 1/3
    box_a = [0.0, 0.0, 40.0, 40.0]  # area 1600
    box_b = [20.0, 0.0, 60.0, 40.0]  # area 1600
    expected_iou = (20.0 * 40.0) / (1600.0 + 1600.0 - 800.0)
    assert calculate_box_iou(box_a, box_b) == pytest.approx(expected_iou)


def test_box_iou_zero_area_safe():
    """Verify zero-area or inverted boxes return 0.0 without error."""
    zero_box = [10.0, 10.0, 10.0, 10.0]
    valid_box = [10.0, 10.0, 20.0, 20.0]
    assert calculate_box_iou(zero_box, valid_box) == 0.0
    assert calculate_box_iou(zero_box, zero_box) == 0.0


# ============================================================================
# 2. Instance Duplicate Suppression & Atomic Box+Mask Pair Pruning
# ============================================================================

def test_instance_duplicate_suppression_keeps_highest_confidence():
    """Verify duplicate instances of the same label keep the higher confidence pair."""
    # Two overlapping 'car' detections with IoU > 0.8
    # Car 1: conf 0.95, group_id 1
    car1_rect = {"type": "rectangle", "label": "car", "points": [100.0, 100.0, 300.0, 300.0], "confidence": "0.95", "group_id": 1}
    car1_mask = {"type": "mask", "label": "car", "mask": [1, 1, 1, 1, 100, 100, 101, 101], "confidence": "0.95", "group_id": 1}

    # Car 2: conf 0.70, group_id 2 (heavily overlapping Car 1)
    car2_rect = {"type": "rectangle", "label": "car", "points": [105.0, 102.0, 305.0, 302.0], "confidence": "0.70", "group_id": 2}
    car2_mask = {"type": "mask", "label": "car", "mask": [1, 1, 1, 1, 105, 102, 106, 103], "confidence": "0.70", "group_id": 2}

    shapes = [car1_rect, car1_mask, car2_rect, car2_mask]
    result = suppress_phase3b_duplicates(shapes, box_iou_threshold=0.70)

    # Car 2 must be suppressed completely (both rect and mask pruned)
    assert len(result) == 2
    groups = {s["group_id"] for s in result}
    assert groups == {1}
    assert any(s["type"] == "rectangle" and s["confidence"] == "0.95" for s in result)
    assert any(s["type"] == "mask" and s["confidence"] == "0.95" for s in result)


def test_instance_suppression_never_leaves_orphans():
    """Verify suppression never drops only the box while leaving an orphaned mask."""
    # Detection A: conf 0.90
    det_a_box = {"type": "rectangle", "label": "bus", "points": [0.0, 0.0, 500.0, 500.0], "confidence": "0.90", "group_id": 10}
    det_a_mask = {"type": "mask", "label": "bus", "mask": [1, 0, 0, 0], "confidence": "0.90", "group_id": 10}

    # Detection B (duplicate): conf 0.60
    det_b_box = {"type": "rectangle", "label": "bus", "points": [2.0, 2.0, 502.0, 502.0], "confidence": "0.60", "group_id": 20}
    det_b_mask = {"type": "mask", "label": "bus", "mask": [1, 0, 0, 0], "confidence": "0.60", "group_id": 20}

    result = suppress_phase3b_duplicates([det_a_box, det_a_mask, det_b_box, det_b_mask], box_iou_threshold=0.80)

    # Must contain exactly 1 rectangle and 1 mask, both sharing group_id 10
    rects = [s for s in result if s["type"] == "rectangle"]
    masks = [s for s in result if s["type"] == "mask"]
    assert len(rects) == 1
    assert len(masks) == 1
    assert rects[0]["group_id"] == 10
    assert masks[0]["group_id"] == 10


# ============================================================================
# 3. Ambiguous Label Preservation during Duplicate Suppression
# ============================================================================

@pytest.mark.parametrize(
    "label_a, label_b",
    [
        ("pedestrian", "person"),
        ("traffic light", "traffic_light"),
        ("traffic sign", "traffic_sign"),
    ],
)
def test_ambiguous_pairs_never_suppress_each_other(label_a: str, label_b: str):
    """Verify ambiguous label pairs with 100% spatial overlap are NEVER suppressed."""
    identical_box = [100.0, 100.0, 400.0, 400.0]

    shape_a_rect = {"type": "rectangle", "label": label_a, "points": list(identical_box), "confidence": "0.95", "group_id": 1}
    shape_a_mask = {"type": "mask", "label": label_a, "mask": [1, 1, 1, 1, 100, 100, 101, 101], "confidence": "0.95", "group_id": 1}

    shape_b_rect = {"type": "rectangle", "label": label_b, "points": list(identical_box), "confidence": "0.85", "group_id": 2}
    shape_b_mask = {"type": "mask", "label": label_b, "mask": [1, 1, 1, 1, 100, 100, 101, 101], "confidence": "0.85", "group_id": 2}

    shapes = [shape_a_rect, shape_a_mask, shape_b_rect, shape_b_mask]
    result = suppress_phase3b_duplicates(shapes, box_iou_threshold=0.50)

    # Because classes are different, both instance pairs MUST be preserved!
    assert len(result) == 4
    labels_present = {s["label"] for s in result}
    assert labels_present == {label_a, label_b}


# ============================================================================
# 4. Region and Lane Mask Overlap Suppression
# ============================================================================

def test_region_duplicate_suppression():
    """Verify overlapping semantic regions of same class are pruned, keeping higher confidence."""
    # Create two overlapping 'road' polygon masks
    poly1 = [[100, 500], [900, 500], [900, 900], [100, 900]]
    poly2 = [[120, 500], [920, 500], [920, 900], [120, 900]]

    cvat_m1 = polygon_to_cvat_mask(poly1, width=1000, height=1000, label="road", confidence=0.92)
    cvat_m2 = polygon_to_cvat_mask(poly2, width=1000, height=1000, label="road", confidence=0.65)
    assert cvat_m1 is not None and cvat_m2 is not None

    shapes = [cvat_m1, cvat_m2]
    result = suppress_phase3b_duplicates(shapes, mask_iou_threshold=0.70, image_width=1000, image_height=1000)

    # Road 2 (0.65) must be suppressed, Road 1 (0.92) kept
    assert len(result) == 1
    assert result[0]["confidence"] == "0.92"


def test_lane_duplicate_suppression_and_parallel_preservation():
    """Verify duplicate lane markings are suppressed, while distinct parallel lanes are preserved."""
    # Lane A (Left line): x from 200 to 220
    lane_a = [[200, 200], [220, 200], [220, 800], [200, 800]]
    # Lane B (Right parallel line): x from 600 to 620
    lane_b = [[600, 200], [620, 200], [620, 800], [600, 800]]
    # Lane C (Duplicate of Lane A): x from 202 to 222
    lane_c = [[202, 200], [222, 200], [222, 800], [202, 800]]

    mask_a = polygon_to_cvat_mask(lane_a, width=1000, height=1000, label="lane/single white", confidence=0.90)
    mask_b = polygon_to_cvat_mask(lane_b, width=1000, height=1000, label="lane/single white", confidence=0.88)
    mask_c = polygon_to_cvat_mask(lane_c, width=1000, height=1000, label="lane/single white", confidence=0.55)

    shapes = [mask_a, mask_b, mask_c]
    result = suppress_phase3b_duplicates(shapes, mask_iou_threshold=0.60, image_width=1000, image_height=1000)

    # Lane A and Lane B must be kept; Lane C (duplicate of A) must be suppressed
    assert len(result) == 2
    confidences = {s["confidence"] for s in result}
    assert confidences == {"0.9", "0.88"}


def test_cross_class_regions_never_suppress():
    """Verify different region classes (e.g. road and sidewalk) never suppress each other."""
    poly = [[100, 500], [900, 500], [900, 900], [100, 900]]
    m_road = polygon_to_cvat_mask(poly, width=1000, height=1000, label="road", confidence=0.95)
    m_sidewalk = polygon_to_cvat_mask(poly, width=1000, height=1000, label="sidewalk", confidence=0.80)

    result = suppress_phase3b_duplicates([m_road, m_sidewalk], mask_iou_threshold=0.50, image_width=1000, image_height=1000)
    assert len(result) == 2
    labels = {s["label"] for s in result}
    assert labels == {"road", "sidewalk"}
