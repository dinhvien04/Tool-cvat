"""Comprehensive unit and integration tests for core/geometry.py.

Verifies:
1. Gemini normalized polygon [x, y] to pixel coordinate denormalization.
2. Tight integer bounding box calculation and 0-area detection.
3. Shoelace polygon area calculation for CW, CCW, and collinear vertices.
4. Pure PIL polygon rasterization into 0/1 binary masks.
5. CVAT native mask format crop extraction and flattening.
6. Full end-to-end polygon_to_cvat_mask pipeline with edge case handling.
7. cvat_mask_to_binary_image reverse reconstruction and round-trip fidelity.
8. Zero local heavy ML dependency enforcement (no torch, cv2, SAM, torchvision).
"""

import math
import sys
import pytest
from PIL import Image

from core.geometry import (
    calculate_polygon_area,
    contour_to_bounding_box,
    cvat_mask_to_binary_image,
    denormalize_contour,
    mask_to_cvat_flat_list,
    normalize_contour,
    overlay_mask_on_image,
    polygon_to_cvat_mask,
    rasterize_polygon_to_mask,
)


# ---------------------------------------------------------------------------
# Strict Dependency Constraint Check
# ---------------------------------------------------------------------------

def test_zero_heavy_ml_imports():
    """Verify that core.geometry does not import or require heavy ML libraries."""
    import core.geometry as geom_module

    forbidden_modules = ["torch", "torchvision", "cv2", "segment_anything", "sam", "tensorflow"]
    for mod_name in forbidden_modules:
        assert mod_name not in sys.modules, f"Forbidden heavy ML module {mod_name!r} was imported"
        assert not hasattr(geom_module, mod_name), f"core.geometry has reference to {mod_name!r}"


# ---------------------------------------------------------------------------
# calculate_polygon_area Tests
# ---------------------------------------------------------------------------

def test_calculate_polygon_area_basic():
    """Verify area for standard shapes and orientation independence."""
    # Rectangle 10x20 -> Area = 200
    rect_cw = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
    rect_ccw = [(0.0, 0.0), (0.0, 20.0), (10.0, 20.0), (10.0, 0.0)]
    assert calculate_polygon_area(rect_cw) == pytest.approx(200.0)
    assert calculate_polygon_area(rect_ccw) == pytest.approx(200.0)

    # Triangle base=10, height=10 -> Area = 50
    tri = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert calculate_polygon_area(tri) == pytest.approx(50.0)


def test_calculate_polygon_area_degenerate():
    """Verify degenerate shapes return 0.0 area."""
    # Collinear points on diagonal
    collinear_diag = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    assert calculate_polygon_area(collinear_diag) == pytest.approx(0.0)

    # Horizontal collinear line
    collinear_h = [(5.0, 10.0), (15.0, 10.0), (25.0, 10.0)]
    assert calculate_polygon_area(collinear_h) == pytest.approx(0.0)

    # Vertical collinear line
    collinear_v = [(10.0, 5.0), (10.0, 15.0), (10.0, 25.0)]
    assert calculate_polygon_area(collinear_v) == pytest.approx(0.0)

    # Identical repeated points
    repeated = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]
    assert calculate_polygon_area(repeated) == pytest.approx(0.0)

    # Fewer than 3 points
    assert calculate_polygon_area([(10.0, 10.0), (20.0, 20.0)]) == 0.0
    assert calculate_polygon_area([(10.0, 10.0)]) == 0.0
    assert calculate_polygon_area([]) == 0.0


# ---------------------------------------------------------------------------
# denormalize_contour Tests
# ---------------------------------------------------------------------------

def test_denormalize_contour_valid():
    """Verify standard denormalization from [0, 1000] to [0, width-1] and [0, height-1]."""
    # 1000x500 image: max_x = 999, max_y = 499
    contour = [
        [0, 0],
        [1000, 0],
        [1000, 1000],
        [0, 1000],
    ]
    pts = denormalize_contour(contour, width=1001, height=501)
    assert pts == [
        (0.0, 0.0),
        (1000.0, 0.0),
        (1000.0, 500.0),
        (0.0, 500.0),
    ]

    # Check midpoint: 500 in [0, 1000] -> exactly half of (width - 1)
    mid_contour = [[500, 500], [600, 600], [500, 600]]
    mid_pts = denormalize_contour(mid_contour, width=201, height=101)
    assert mid_pts[0] == (100.0, 50.0)


def test_denormalize_contour_clamping():
    """Verify small out-of-bounds coordinate overshoot is clamped properly."""
    contour = [
        [-10, -5],
        [1010, 500],
        [500, 1020],
    ]
    pts = denormalize_contour(contour, width=101, height=101, clamp=True)
    assert pts[0] == (0.0, 0.0)
    assert pts[1] == (100.0, 50.0)
    assert pts[2] == (50.0, 100.0)


def test_denormalize_contour_gross_out_of_bounds_rejected():
    """Verify coordinates that wildly exceed normalized range are rejected."""
    # x = 1500 is way beyond [-100, 1100]
    with pytest.raises(ValueError, match="grossly out of normalized range"):
        denormalize_contour([[1500, 200], [500, 500], [200, 200]], width=100, height=100)

    # Negative gross overshoot
    with pytest.raises(ValueError, match="grossly out of normalized range"):
        denormalize_contour([[-300, 200], [500, 500], [200, 200]], width=100, height=100)


def test_denormalize_contour_strict_mode():
    """Verify strict mode rejects any coordinate outside [0, 1000]."""
    with pytest.raises(ValueError, match="out of bounds"):
        denormalize_contour([[-1, 500], [500, 500], [200, 200]], width=100, height=100, clamp=False)

    with pytest.raises(ValueError, match="out of bounds"):
        denormalize_contour([[1001, 500], [500, 500], [200, 200]], width=100, height=100, clamp=False)


def test_denormalize_contour_invalid_inputs():
    """Verify rejection of invalid types, dimensions, and NaN/Inf."""
    # Invalid image dimensions
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        denormalize_contour([[0, 0], [100, 100], [50, 100]], width=0, height=100)
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        denormalize_contour([[0, 0], [100, 100], [50, 100]], width=100, height=-5)

    # Empty or fewer than 3 points
    with pytest.raises(ValueError, match="at least 3 points"):
        denormalize_contour([], width=100, height=100)
    with pytest.raises(ValueError, match="at least 3 points"):
        denormalize_contour([[10, 10], [20, 20]], width=100, height=100)

    # Optional min_points=0 allows empty
    assert denormalize_contour([], width=100, height=100, min_points=0) == []

    # Non-list contour
    with pytest.raises(TypeError):
        denormalize_contour("invalid", width=100, height=100)  # type: ignore

    # Non-2-element points
    with pytest.raises(ValueError, match="2-element sequence"):
        denormalize_contour([[100, 100], [200], [300, 300]], width=100, height=100)

    # Boolean coordinate injection guard
    with pytest.raises(TypeError, match="boolean"):
        denormalize_contour([[True, 100], [200, 200], [300, 300]], width=100, height=100)

    # Non-numeric coordinate
    with pytest.raises(TypeError, match="numeric"):
        denormalize_contour([["abc", 100], [200, 200], [300, 300]], width=100, height=100)  # type: ignore

    # NaN and Inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        denormalize_contour([[float("nan"), 100], [200, 200], [300, 300]], width=100, height=100)
    with pytest.raises(ValueError, match="NaN or Inf"):
        denormalize_contour([[float("inf"), 100], [200, 200], [300, 300]], width=100, height=100)


@pytest.mark.parametrize(
    "width, height",
    [
        (1920, 1080),  # Full HD 16:9
        (400, 400),    # Square
        (100, 100),    # Small
    ],
)
def test_denormalize_contour_various_resolutions(width: int, height: int):
    """Verify denormalization between [0, 1000] and pixel space for multiple aspect ratios."""
    contour = [
        [0, 0],
        [1000, 0],
        [1000, 1000],
        [0, 1000],
    ]
    pts = denormalize_contour(contour, width=width, height=height)
    assert pts[0] == (0.0, 0.0)
    assert pts[1] == (float(width - 1), 0.0)
    assert pts[2] == (float(width - 1), float(height - 1))
    assert pts[3] == (0.0, float(height - 1))


def test_normalize_contour_and_round_trip():
    """Verify normalization and denormalization round-trip between pixel space and [0, 1000]."""
    width, height = 1920, 1080
    original_norm = [
        [100, 150],
        [600, 200],
        [750, 800],
        [200, 900],
    ]
    pixel_pts = denormalize_contour(original_norm, width=width, height=height)
    rt_norm = normalize_contour(pixel_pts, width=width, height=height)

    for orig, rt in zip(original_norm, rt_norm):
        assert abs(orig[0] - rt[0]) <= 2
        assert abs(orig[1] - rt[1]) <= 2


def test_coordinate_ordering_polygon_vs_box_2d():
    """Verify polygon points are [x, y] while box_2d is [ymin, xmin, ymax, xmax].

    On non-square images (e.g. 1920x1080), flipping x and y leads to massive errors.
    This test verifies that x always scales with width and y always scales with height.
    """
    width, height = 1920, 1080
    from core.vision_contract import box_2d_to_cvat_rect

    # Polygon vertex: x=100 (horizontal), y=800 (vertical)
    polygon_points = [[100, 800], [500, 800], [500, 900]]
    pixel_pts = denormalize_contour(polygon_points, width=width, height=height)

    px, py = pixel_pts[0]
    expected_x = round((100 / 1000.0) * (width - 1), 2)
    expected_y = round((800 / 1000.0) * (height - 1), 2)
    assert px == expected_x
    assert py == expected_y

    # Contrast with box_2d: [ymin=800, xmin=100, ymax=900, xmax=500]
    box_2d = [800, 100, 900, 500]
    xtl, ytl, xbr, ybr = box_2d_to_cvat_rect(box_2d, width=width, height=height)

    # In box_2d, first coordinate is ymin (vertical, scaled with height)
    # In polygon, first coordinate is x (horizontal, scaled with width)
    assert abs(px - xtl) <= 2
    assert abs(py - ytl) <= 2


# ---------------------------------------------------------------------------
# contour_to_bounding_box Tests
# ---------------------------------------------------------------------------

def test_contour_to_bounding_box_valid():
    """Verify tight integer bounding box computation and clamping."""
    pts = [(10.2, 20.8), (50.7, 20.8), (50.7, 80.1), (10.2, 80.1)]
    bbox = contour_to_bounding_box(pts, width=100, height=100)
    # min_x=10.2 -> 10, max_x=50.7 -> 51, min_y=20.8 -> 21, max_y=80.1 -> 80
    assert bbox == (10, 21, 51, 80)
    assert 0 <= bbox[0] <= bbox[2] < 100
    assert 0 <= bbox[1] <= bbox[3] < 100


def test_contour_to_bounding_box_boundary_clamping():
    """Verify clamping to image boundaries [0, width-1] and [0, height-1]."""
    pts = [(-5.0, -10.0), (120.0, -10.0), (120.0, 150.0), (-5.0, 150.0)]
    bbox = contour_to_bounding_box(pts, width=100, height=100)
    assert bbox == (0, 0, 99, 99)


def test_contour_to_bounding_box_zero_area():
    """Verify that zero-area / collinear bounding boxes raise ValueError."""
    # Horizontal line (min_y == max_y)
    line_h = [(10.0, 20.0), (30.0, 20.0), (50.0, 20.0)]
    with pytest.raises(ValueError, match="0 area"):
        contour_to_bounding_box(line_h, width=100, height=100)

    # Vertical line (min_x == max_x)
    line_v = [(20.0, 10.0), (20.0, 30.0), (20.0, 50.0)]
    with pytest.raises(ValueError, match="0 area"):
        contour_to_bounding_box(line_v, width=100, height=100)

    # Single point repeated
    pt_rep = [(20.0, 20.0), (20.0, 20.0), (20.0, 20.0)]
    with pytest.raises(ValueError, match="0 area"):
        contour_to_bounding_box(pt_rep, width=100, height=100)

    # Allowed when allow_zero_area is True
    assert contour_to_bounding_box(line_h, width=100, height=100, allow_zero_area=True) == (10, 20, 50, 20)


def test_contour_to_bounding_box_invalid_inputs():
    """Verify invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError, match="at least 3 points"):
        contour_to_bounding_box([(10.0, 10.0), (20.0, 20.0)], width=100, height=100)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        contour_to_bounding_box([(10.0, 10.0), (20.0, 20.0), (10.0, 20.0)], width=-1, height=100)


# ---------------------------------------------------------------------------
# rasterize_polygon_to_mask Tests
# ---------------------------------------------------------------------------

def test_rasterize_polygon_to_mask_values():
    """Verify mask is mode 'L' with exact 0 background and 1 foreground values."""
    width, height = 50, 50
    pts = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    mask_img = rasterize_polygon_to_mask(pts, width=width, height=height)

    assert mask_img.mode == "L"
    assert mask_img.size == (width, height)

    # Center pixel should be foreground 1
    assert mask_img.getpixel((20, 20)) == 1

    # Outside pixel should be background 0
    assert mask_img.getpixel((5, 5)) == 0
    assert mask_img.getpixel((45, 45)) == 0

    # Ensure ONLY 0 and 1 values exist across entire image
    unique_vals = set(mask_img.tobytes())
    assert unique_vals.issubset({0, 1})
    assert 1 in unique_vals


def test_rasterize_polygon_to_mask_degenerate():
    """Verify collinear / zero-area polygons raise ValueError."""
    collinear = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    with pytest.raises(ValueError, match="zero area"):
        rasterize_polygon_to_mask(collinear, width=50, height=50)


# ---------------------------------------------------------------------------
# mask_to_cvat_flat_list Tests
# ---------------------------------------------------------------------------

def test_mask_to_cvat_flat_list_format():
    """Verify crop flattening, trailing bbox, and length calculation."""
    width, height = 100, 100
    mask_img = Image.new("L", (width, height), 0)

    # Set a 4x3 foreground rectangle from (10, 20) to (13, 22)
    # xmin=10, xmax=13 (width=4), ymin=20, ymax=22 (height=3)
    for y in range(20, 23):
        for x in range(10, 14):
            mask_img.putpixel((x, y), 1)

    flat_mask = mask_to_cvat_flat_list(mask_img)
    assert flat_mask is not None

    # Expected: 4 * 3 = 12 pixels + 4 bbox values = 16 elements
    assert len(flat_mask) == 16
    trailing_bbox = flat_mask[-4:]
    assert trailing_bbox == [10, 20, 13, 22]

    # All crop pixels should be 1
    crop_pixels = flat_mask[:-4]
    assert crop_pixels == [1] * 12


def test_mask_to_cvat_flat_list_empty_image():
    """Verify empty image (no foreground pixels) returns None."""
    empty_img = Image.new("L", (50, 50), 0)
    assert mask_to_cvat_flat_list(empty_img) is None


def test_mask_to_cvat_flat_list_with_explicit_bbox():
    """Verify explicit bounding box extraction."""
    mask_img = Image.new("L", (50, 50), 0)
    mask_img.putpixel((15, 15), 1)

    # Explicit bbox surrounding (15, 15): [10, 10, 20, 20] -> 11x11 crop = 121 pixels + 4 = 125
    flat_mask = mask_to_cvat_flat_list(mask_img, bbox=(10, 10, 20, 20))
    assert flat_mask is not None
    assert len(flat_mask) == 121 + 4
    assert flat_mask[-4:] == [10, 10, 20, 20]
    assert sum(flat_mask[:-4]) == 1


# ---------------------------------------------------------------------------
# polygon_to_cvat_mask Pipeline Tests
# ---------------------------------------------------------------------------

def test_polygon_to_cvat_mask_success():
    """Verify full pipeline: denormalize -> rasterize -> crop -> flat list dict."""
    # Normalized [0, 1000] square centered in 1000x1000 image
    contour = [
        [200, 200],
        [400, 200],
        [400, 400],
        [200, 400],
    ]
    res = polygon_to_cvat_mask(
        contour,
        width=1001,
        height=1001,
        label="car",
        confidence=0.95,
    )
    assert res is not None
    assert "mask" in res
    assert "bbox" in res
    assert res["label"] == "car"
    assert res["type"] == "mask"
    assert res["confidence"] == "0.95"

    flat_list = res["mask"]
    bbox = res["bbox"]
    assert len(flat_list) > 4
    assert flat_list[-4:] == bbox

    # Verify bbox corresponds to [200, 200, 400, 400] in pixel space
    xmin, ymin, xmax, ymax = bbox
    assert xmin == 200
    assert ymin == 200
    assert xmax == 400
    assert ymax == 400


def test_polygon_to_cvat_mask_edge_cases():
    """Verify edge cases return None without raising unhandled exceptions."""
    w, h = 100, 100

    # Empty contour
    assert polygon_to_cvat_mask([], width=w, height=h) is None

    # Degenerate: < 3 points
    assert polygon_to_cvat_mask([[10, 10], [20, 20]], width=w, height=h) is None

    # Degenerate: collinear points enclosing zero area
    collinear = [[100, 100], [200, 200], [300, 300]]
    assert polygon_to_cvat_mask(collinear, width=w, height=h) is None

    # Degenerate: duplicate points
    assert polygon_to_cvat_mask([[50, 50], [50, 50], [50, 50]], width=w, height=h) is None

    # Invalid image dimensions
    assert polygon_to_cvat_mask([[10, 10], [50, 10], [50, 50]], width=0, height=h) is None
    assert polygon_to_cvat_mask([[10, 10], [50, 10], [50, 50]], width=w, height=-10) is None

    # Invalid point format
    assert polygon_to_cvat_mask([[10], [50, 10], [50, 50]], width=w, height=h) is None  # type: ignore

    # Grossly out-of-bounds coordinates
    assert polygon_to_cvat_mask([[5000, 5000], [6000, 5000], [6000, 6000]], width=w, height=h) is None


# ---------------------------------------------------------------------------
# cvat_mask_to_binary_image and Round-Trip Fidelity Tests
# ---------------------------------------------------------------------------

def test_cvat_mask_to_binary_image_basic():
    """Verify reconstruction of full binary image from CVAT flat mask."""
    # 1 pixel at (5, 8) in 20x20 image
    cvat_mask = [1, 5, 8, 5, 8]
    img = cvat_mask_to_binary_image(cvat_mask, width=20, height=20)
    assert img.size == (20, 20)
    assert img.mode == "L"
    assert img.getpixel((5, 8)) == 1
    assert img.getpixel((4, 8)) == 0
    assert img.getpixel((5, 7)) == 0
    assert img.getbbox() == (5, 8, 6, 9)


def test_cvat_mask_to_binary_image_invalid():
    """Verify validation of mask list, bounds, and pixel count."""
    # Too short
    with pytest.raises(ValueError, match="too short"):
        cvat_mask_to_binary_image([1, 2, 3], width=10, height=10)

    # Pixel count mismatch: bbox [0, 0, 1, 1] requires 2x2 = 4 pixels, but 3 provided
    with pytest.raises(ValueError, match="Pixel count mismatch"):
        cvat_mask_to_binary_image([1, 1, 1, 0, 0, 1, 1], width=10, height=10)

    # Out of bounds bbox
    with pytest.raises(ValueError, match="out of bounds"):
        cvat_mask_to_binary_image([1, 0, 0, 15, 15], width=10, height=10)


@pytest.mark.parametrize(
    "width,height,contour",
    [
        # Standard square
        (100, 100, [[200, 200], [600, 200], [600, 600], [200, 600]]),
        # Triangle
        (200, 150, [[100, 100], [800, 100], [400, 700]]),
        # Non-symmetric polygon (pentagon)
        (300, 200, [[300, 100], [700, 200], [600, 800], [200, 700], [100, 300]]),
        # High resolution
        (1920, 1080, [[100, 100], [900, 100], [900, 900], [100, 900]]),
    ],
)
def test_full_round_trip_fidelity(width, height, contour):
    """Verify that forward pipeline and reverse reconstruction match rasterized mask exactly."""
    # 1. Pipeline: contour -> CVAT mask dict
    cvat_res = polygon_to_cvat_mask(contour, width=width, height=height)
    assert cvat_res is not None
    flat_mask = cvat_res["mask"]

    # 2. Reverse: CVAT mask -> reconstructed binary image
    reconstructed_img = cvat_mask_to_binary_image(flat_mask, width=width, height=height)

    # 3. Direct reference rasterization
    pixel_points = denormalize_contour(contour, width=width, height=height)
    reference_img = rasterize_polygon_to_mask(pixel_points, width=width, height=height)

    # 4. Pixel-for-pixel identity
    assert reconstructed_img.size == reference_img.size
    assert reconstructed_img.mode == reference_img.mode
    assert reconstructed_img.tobytes() == reference_img.tobytes()


# ---------------------------------------------------------------------------
# overlay_mask_on_image Tests
# ---------------------------------------------------------------------------

def test_overlay_mask_on_image():
    """Verify visual blending with green tint on foreground only."""
    bg_color = (200, 200, 200)
    orig_img = Image.new("RGB", (20, 20), bg_color)

    mask = Image.new("L", (20, 20), 0)
    # Foreground at (10, 10)
    mask.putpixel((10, 10), 1)

    overlaid = overlay_mask_on_image(orig_img, mask, color=(0, 255, 0), alpha=0.5)
    assert overlaid.size == (20, 20)
    assert overlaid.mode == "RGB"

    # Background pixel untouched
    assert overlaid.getpixel((0, 0)) == bg_color

    # Foreground pixel tinted with green
    fg_pixel = overlaid.getpixel((10, 10))
    assert fg_pixel != bg_color
    # Green channel should be boosted
    assert fg_pixel[1] > fg_pixel[0]
    assert fg_pixel[1] > fg_pixel[2]
