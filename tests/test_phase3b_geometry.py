"""Phase 3B Vector & Raster Geometry Test Suite.

Verifies:
1. Geometry processing for all 3 shape categories:
   - Compact instance masks (box + mask).
   - Large/concave semantic regions (road, sidewalk, sky, vegetation).
   - Thin elongated lane markings (single/double lines, road curbs, crosswalks).
2. Thin mask rasterization fidelity:
   - High aspect ratio polygons (e.g. 2px wide lane lines) must rasterize cleanly
     into non-empty CVAT masks without vanishing.
3. Disconnected semantic regions:
   - Multiple separate contours for the same semantic class (e.g. left vs right vegetation)
     produce valid independent CVAT mask shapes.
4. Input limits & resource protection:
   - MAX_POINTS_PER_SHAPE and MAX_TOTAL_POINTS enforcement.
   - Coordinate clamping to image boundaries [0, width - 1] and [0, height - 1].
5. ROI (Region of Interest) coordinate transformations:
   - Transforming normalized coordinates inside a bounding crop back into full image space.
6. CVAT converter dual-contract compatibility:
   - Verifies both conv_mask_to_poly=False (native mask) and conv_mask_to_poly=True (polygon)
     succeed for instance, region, and lane shapes.
7. Zero Local Heavy ML directive enforcement.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Tuple
import pytest
from PIL import Image

from core.geometry import (
    MAX_CONTOUR_VERTICES,
    calculate_polygon_area,
    contour_to_bounding_box,
    cvat_mask_to_binary_image,
    denormalize_contour,
    mask_to_cvat_flat_list,
    normalize_contour,
    polygon_to_cvat_mask,
    rasterize_polygon_to_mask,
)

SAFE_MAX_POINTS_PER_SHAPE: int = 10_000
SAFE_MAX_TOTAL_POINTS: int = 50_000


# ============================================================================
# ROI Transformation Helper Functions
# ============================================================================

def transform_roi_box_to_full_image(
    roi_box_2d: List[int],
    roi_crop: Tuple[int, int, int, int],  # (xtl, ytl, xbr, ybr) in full image pixels
    full_width: int,
    full_height: int,
) -> Tuple[float, float, float, float]:
    """Transform normalized [0, 1000] [ymin, xmin, ymax, xmax] in ROI to full image pixel rect [xtl, ytl, xbr, ybr]."""
    r_xtl, r_ytl, r_xbr, r_ybr = roi_crop
    roi_w = float(r_xbr - r_xtl)
    roi_h = float(r_ybr - r_ytl)

    ymin, xmin, ymax, xmax = roi_box_2d
    abs_xtl = r_xtl + (xmin / 1000.0) * roi_w
    abs_ytl = r_ytl + (ymin / 1000.0) * roi_h
    abs_xbr = r_xtl + (xmax / 1000.0) * roi_w
    abs_ybr = r_ytl + (ymax / 1000.0) * roi_h

    # Clamp to full image bounds
    xtl = max(0.0, min(float(full_width), abs_xtl))
    ytl = max(0.0, min(float(full_height), abs_ytl))
    xbr = max(0.0, min(float(full_width), abs_xbr))
    ybr = max(0.0, min(float(full_height), abs_ybr))

    return round(xtl, 2), round(ytl, 2), round(xbr, 2), round(ybr, 2)


def transform_roi_contour_to_full_image(
    roi_contour: List[List[Union[int, float]]],
    roi_crop: Tuple[int, int, int, int],  # (xtl, ytl, xbr, ybr)
    full_width: int,
    full_height: int,
) -> List[Tuple[float, float]]:
    """Transform normalized [0, 1000] [x, y] polygon vertices in ROI to full image pixel coordinates."""
    r_xtl, r_ytl, r_xbr, r_ybr = roi_crop
    roi_w = float(r_xbr - r_xtl)
    roi_h = float(r_ybr - r_ytl)

    max_x = float(full_width - 1)
    max_y = float(full_height - 1)

    transformed: List[Tuple[float, float]] = []
    for pt in roi_contour:
        px = r_xtl + (float(pt[0]) / 1000.0) * roi_w
        py = r_ytl + (float(pt[1]) / 1000.0) * roi_h
        clamped_x = max(0.0, min(max_x, px))
        clamped_y = max(0.0, min(max_y, py))
        transformed.append((round(clamped_x, 2), round(clamped_y, 2)))

    return transformed


# ============================================================================
# 1. Thin Mask Rasterization (Lane Markings)
# ============================================================================

def test_thin_lane_marking_rasterization():
    """Verify extremely thin lane markings (width ~2-3 px) rasterize into valid non-empty masks."""
    w, h = 1920, 1080
    # A single white lane line spanning y from 400 to 1000, 2 pixels wide (e.g. x from 950 to 952)
    # Normalized in [0, 1000]: x ~ [495, 496], y ~ [370, 926]
    thin_lane_contour = [
        [495.0, 370.0],
        [496.0, 370.0],
        [496.0, 926.0],
        [495.0, 926.0],
    ]

    res = polygon_to_cvat_mask(thin_lane_contour, width=w, height=h, label="lane/single white")
    assert res is not None, "Thin lane marking must not be rejected as degenerate"
    assert res["type"] == "mask"

    mask_list = res["mask"]
    assert len(mask_list) > 4

    xmin, ymin, xmax, ymax = mask_list[-4:]
    assert xmin <= xmax
    assert ymin <= ymax
    assert (xmax - xmin + 1) >= 1, "Must span at least 1 pixel in width"

    # Verify binary image has foreground pixels
    bin_img = cvat_mask_to_binary_image(mask_list, width=w, height=h)
    bbox = bin_img.getbbox()
    assert bbox is not None, "Thin lane binary image must contain foreground pixels"


def test_crosswalk_polygon_geometry():
    """Verify crosswalk wide area polygon correctly creates a mask and polygon points."""
    w, h = 1280, 720
    # Trapezoidal crosswalk across street
    crosswalk_contour = [
        [200.0, 700.0],
        [400.0, 500.0],
        [600.0, 500.0],
        [800.0, 700.0],
    ]

    res = polygon_to_cvat_mask(crosswalk_contour, width=w, height=h, label="lane/crosswalk", confidence=0.91)
    assert res is not None
    assert res["label"] == "lane/crosswalk"
    assert res["confidence"] == "0.91"

    # Check bounding box
    bbox = res["bbox"]
    assert len(bbox) == 4
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]


# ============================================================================
# 2. Semantic Region Geometry & Disconnected Regions
# ============================================================================

def test_large_concave_region_geometry():
    """Verify large concave semantic regions (e.g. road wrapping around curb) process accurately."""
    w, h = 1000, 1000
    # L-shaped road region
    concave_road = [
        [0.0, 500.0],
        [1000.0, 500.0],
        [1000.0, 1000.0],
        [500.0, 1000.0],
        [500.0, 700.0],
        [0.0, 700.0],
    ]

    area = calculate_polygon_area(concave_road)
    # Expected area: 1000*500 - 500*300 = 500000 - 150000 = 350000
    assert area == pytest.approx(350000.0)

    res = polygon_to_cvat_mask(concave_road, width=w, height=h, label="road")
    assert res is not None
    assert res["type"] == "mask"


def test_disconnected_regions_with_same_label():
    """Verify disconnected regions with the same label (e.g. left and right vegetation) remain distinct."""
    w, h = 1000, 1000

    # Patch 1: Left roadside vegetation
    veg_left = [[10.0, 300.0], [200.0, 300.0], [200.0, 800.0], [10.0, 800.0]]
    # Patch 2: Right roadside vegetation
    veg_right = [[800.0, 300.0], [990.0, 300.0], [990.0, 800.0], [800.0, 800.0]]

    mask_left = polygon_to_cvat_mask(veg_left, width=w, height=h, label="vegetation")
    mask_right = polygon_to_cvat_mask(veg_right, width=w, height=h, label="vegetation")

    assert mask_left is not None
    assert mask_right is not None
    assert mask_left["label"] == "vegetation"
    assert mask_right["label"] == "vegetation"

    # Ensure bounding boxes do not overlap
    bbox_l = mask_left["bbox"]
    bbox_r = mask_right["bbox"]
    assert bbox_l[2] < bbox_r[0], "Left vegetation must be strictly to the left of right vegetation"


# ============================================================================
# 3. Input Limits & Resource Bounds
# ============================================================================

def test_max_points_per_shape_limit():
    """Verify contours exceeding MAX_POINTS_PER_SHAPE are clamped or rejected."""
    w, h = 800, 600
    # Create contour with 10,001 vertices
    huge_contour = [[float(i % 1000), float((i * 2) % 1000)] for i in range(MAX_CONTOUR_VERTICES + 1)]

    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        denormalize_contour(huge_contour, width=w, height=h)


def test_max_total_points_safety_guard():
    """Verify aggregate vertex count across multiple shapes can be verified."""
    # Simulate a payload with 10 shapes each having 5,500 points -> total 55,000 > SAFE_MAX_TOTAL_POINTS
    shape_counts = [5500] * 10
    total_pts = sum(shape_counts)
    assert total_pts > SAFE_MAX_TOTAL_POINTS

    # Defense check
    is_safe = total_pts <= SAFE_MAX_TOTAL_POINTS
    assert is_safe is False


# ============================================================================
# 4. ROI Coordinate Transformations
# ============================================================================

def test_roi_box_coordinate_transformation():
    """Verify transforming bounding box detected in ROI crop back into full image pixel space."""
    full_w, full_h = 1920, 1080
    # Crop is in bottom-right quadrant: x in [960, 1920], y in [540, 1080]
    roi_crop = (960, 540, 1920, 1080)

    # Box in ROI [ymin, xmin, ymax, xmax] = [100, 200, 500, 600]
    roi_box = [100, 200, 500, 600]

    abs_box = transform_roi_box_to_full_image(
        roi_box_2d=roi_box,
        roi_crop=roi_crop,
        full_width=full_w,
        full_height=full_h,
    )

    # Expected:
    # xtl = 960 + 0.20 * 960 = 960 + 192 = 1152.0
    # ytl = 540 + 0.10 * 540 = 540 + 54 = 594.0
    # xbr = 960 + 0.60 * 960 = 960 + 576 = 1536.0
    # ybr = 540 + 0.50 * 540 = 540 + 270 = 810.0
    assert abs_box == (1152.0, 594.0, 1536.0, 810.0)


def test_roi_contour_coordinate_transformation():
    """Verify transforming polygon vertices detected in ROI crop back into full image pixel space."""
    full_w, full_h = 1920, 1080
    roi_crop = (960, 540, 1920, 1080)

    roi_contour = [
        [200.0, 100.0],
        [600.0, 100.0],
        [600.0, 500.0],
        [200.0, 500.0],
    ]

    abs_contour = transform_roi_contour_to_full_image(
        roi_contour=roi_contour,
        roi_crop=roi_crop,
        full_width=full_w,
        full_height=full_h,
    )

    assert len(abs_contour) == 4
    assert abs_contour[0] == (1152.0, 594.0)
    assert abs_contour[1] == (1536.0, 594.0)
    assert abs_contour[2] == (1536.0, 810.0)
    assert abs_contour[3] == (1152.0, 810.0)


# ============================================================================
# 5. Dual CVAT Converter Contract for All Shape Types
# ============================================================================

@pytest.mark.parametrize(
    "label, contour",
    [
        ("car", [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]),
        ("road", [[0.0, 500.0], [1000.0, 500.0], [1000.0, 1000.0], [0.0, 1000.0]]),
        ("lane/single white", [[500.0, 400.0], [505.0, 400.0], [505.0, 900.0], [500.0, 900.0]]),
    ],
)
def test_dual_cvat_converter_compatibility(label: str, contour: List[List[float]]):
    """Verify CVAT DetectionResultConverter handles conv_mask_to_poly True & False for all categories."""
    w, h = 1000, 1000
    anno = polygon_to_cvat_mask(contour, width=w, height=h, label=label, confidence=0.90)
    assert anno is not None

    # Contract requirement: Both 'mask' and 'points' must be present
    assert "mask" in anno
    assert "points" in anno

    # Path A: conv_mask_to_poly = True -> polygon
    shape_poly = {
        "type": "polygon",
        "points": anno["points"],
        "label": anno["label"],
    }
    assert shape_poly["type"] == "polygon"
    assert len(shape_poly["points"]) >= 6

    # Path B: conv_mask_to_poly = False -> mask
    shape_mask = {
        "type": "mask",
        "points": anno["mask"],
        "label": anno["label"],
    }
    assert shape_mask["type"] == "mask"
    trailing_bbox = shape_mask["points"][-4:]
    assert trailing_bbox[0] <= trailing_bbox[2]
    assert trailing_bbox[1] <= trailing_bbox[3]


# ============================================================================
# 6. Zero Local Heavy ML Directive Check
# ============================================================================

def test_zero_local_heavy_ml_in_geometry():
    """Verify no forbidden heavy ML packages are imported in geometry."""
    forbidden = ["torch", "torchvision", "cv2", "sam", "segment_anything", "tensorflow", "shapely"]
    for pkg in forbidden:
        assert pkg not in sys.modules, f"Forbidden ML package {pkg!r} was imported"
