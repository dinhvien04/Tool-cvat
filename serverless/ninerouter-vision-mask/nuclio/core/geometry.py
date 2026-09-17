"""Vector and Raster Geometry Module for CVAT x 9Router Mask Conversions.

This module provides production-grade conversions between Gemini normalized polygon
contours [0, 1000] and CVAT native raster masks.

Strict Constraints:
- ZERO LOCAL HEAVY ML: Relies exclusively on Pillow (PIL.Image, PIL.ImageDraw)
  and Python standard library. No PyTorch, torchvision, OpenCV (cv2), SAM, or TensorFlow.
- Coordinate order distinction:
  * Gemini bounding boxes (`box_2d`) are [ymin, xmin, ymax, xmax] (y first).
  * Gemini polygon contour points are [x, y] (horizontal x first, vertical y second).
- CVAT native mask format:
  * Tight bounding box [xmin, ymin, xmax, ymax] (inclusive integer pixel coordinates).
  * Crop binary mask to [xmin, ymin, xmax, ymax]: shape is (ymax - ymin + 1, xmax - xmin + 1).
  * Crop flattened in row-major order into a 1D list of 0 and 1 integers.
  * Append [xmin, ymin, xmax, ymax] as the last 4 elements:
    [p0, p1, p2, ..., xmin, ymin, xmax, ymax].
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw

NORMALIZATION_MAX: float = 1000.0
MAX_CONTOUR_VERTICES: int = 10_000
SAFE_MAX_IMAGE_PIXELS: int = 89_478_485


def calculate_polygon_area(pixel_points: Sequence[Sequence[Union[int, float]]]) -> float:
    """Compute unsigned 2D polygon area using the Shoelace formula (Gauss's area formula).

    Handles both clockwise and counter-clockwise vertex orderings.

    Args:
        pixel_points: Sequence of [x, y] or (x, y) coordinates in pixel space.

    Returns:
        Unsigned area of the polygon as a float (0.0 if fewer than 3 points or collinear).
    """
    n = len(pixel_points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = float(pixel_points[i][0]), float(pixel_points[i][1])
        x_j, y_j = float(pixel_points[j][0]), float(pixel_points[j][1])
        area += x_i * y_j
        area -= x_j * y_i

    return abs(area) / 2.0


def denormalize_contour(
    contour: List[List[Union[int, float]]],
    width: int,
    height: int,
    clamp: bool = True,
    min_points: int = 3,
) -> List[Tuple[float, float]]:
    """Convert Gemini normalized [0, 1000] polygon vertices to image pixel coordinates.

    Scales [0, 1000] coordinates to [0, width - 1] and [0, height - 1], clamped.
    Rejects invalid coordinates, non-numeric values, NaN/Inf, and out-of-bounds coordinates.

    Coordinate Order:
    - Gemini polygon contours use [x, y] format (horizontal x first, vertical y second),
      unlike box_2d which uses [ymin, xmin, ymax, xmax].

    Args:
        contour: List of [x, y] coordinates normalized in [0, 1000].
        width: Image width in pixels (> 0).
        height: Image height in pixels (> 0).
        clamp: If True, coordinates within reasonable boundary overshoot [-100, 1100]
               are clamped to [0, width-1] and [0, height-1]. If False, strict [0, 1000] is enforced.
        min_points: Minimum number of vertices required (default 3 for a valid polygon;
                    0 allows empty contours returning []).

    Returns:
        List of (x, y) float tuples in image pixel coordinates clamped within
        [0.0, float(width - 1)] and [0.0, float(height - 1)].

    Raises:
        ValueError: If width/height <= 0, contour contains < min_points, coordinates
                    are NaN/Inf, or coordinates are grossly out of bounds.
        TypeError: If contour or point coordinates have invalid types.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={width}, height={height} (both must be positive integers)"
        )

    if width * height > SAFE_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions {width}x{height} ({width * height} pixels) exceed safe limit of {SAFE_MAX_IMAGE_PIXELS} pixels"
        )

    if not isinstance(contour, (list, tuple)):
        raise TypeError(f"Contour must be a list or tuple of points, got {type(contour).__name__}")

    if len(contour) > MAX_CONTOUR_VERTICES:
        raise ValueError(
            f"Contour vertex count ({len(contour)}) exceeds maximum allowed limit ({MAX_CONTOUR_VERTICES})"
        )

    if len(contour) < min_points:
        raise ValueError(
            f"Contour must contain at least {min_points} points to form a polygon, got {len(contour)}"
        )

    if len(contour) == 0:
        return []

    max_x_px = float(width - 1)
    max_y_px = float(height - 1)

    denormalized: List[Tuple[float, float]] = []
    for idx, pt in enumerate(contour):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(
                f"Contour point at index {idx} must be a 2-element sequence [x, y], got {pt!r}"
            )

        x, y = pt[0], pt[1]

        # Guard against boolean types (isinstance(True, int) is True in Python)
        if isinstance(x, bool) or isinstance(y, bool):
            raise TypeError(f"Contour point at index {idx} contains boolean coordinates: {[x, y]}")

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError(
                f"Contour point at index {idx} coordinates must be numeric, got {[type(x).__name__, type(y).__name__]}"
            )

        x_f, y_f = float(x), float(y)
        if math.isnan(x_f) or math.isnan(y_f) or math.isinf(x_f) or math.isinf(y_f):
            raise ValueError(f"Contour point at index {idx} coordinates cannot be NaN or Inf: {[x, y]}")

        if clamp:
            # Allow model overshoot within [-100.0, 1100.0] and clamp to [0, 1000]
            if x_f < -100.0 or x_f > 1100.0 or y_f < -100.0 or y_f > 1100.0:
                raise ValueError(
                    f"Contour point at index {idx} coordinate [{x_f}, {y_f}] is grossly out of normalized range [0, 1000]"
                )
            x_norm = max(0.0, min(NORMALIZATION_MAX, x_f))
            y_norm = max(0.0, min(NORMALIZATION_MAX, y_f))
        else:
            if x_f < 0.0 or x_f > NORMALIZATION_MAX or y_f < 0.0 or y_f > NORMALIZATION_MAX:
                raise ValueError(
                    f"Contour point at index {idx} coordinate [{x_f}, {y_f}] is out of bounds [0, 1000]"
                )
            x_norm, y_norm = x_f, y_f

        # Scale to [0, width - 1] and [0, height - 1]
        px = (x_norm / NORMALIZATION_MAX) * max_x_px
        py = (y_norm / NORMALIZATION_MAX) * max_y_px

        px_clamped = max(0.0, min(max_x_px, px))
        py_clamped = max(0.0, min(max_y_px, py))

        denormalized.append((round(px_clamped, 2), round(py_clamped, 2)))

    return denormalized


def normalize_contour(
    pixel_points: Sequence[Sequence[Union[int, float]]],
    width: int,
    height: int,
) -> List[List[int]]:
    """Convert pixel coordinates to Gemini normalized [0, 1000] integer coordinates.

    Args:
        pixel_points: Sequence of [x, y] coordinates in pixel space.
        width: Image width in pixels (> 0).
        height: Image height in pixels (> 0).

    Returns:
        List of [x, y] integer coordinates normalized to [0, 1000].

    Raises:
        ValueError: If width/height <= 0, points list contains < 3 points.
        TypeError: If point coordinates have invalid types.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={width}, height={height} (both must be positive integers)"
        )

    if width * height > SAFE_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions {width}x{height} ({width * height} pixels) exceed safe limit of {SAFE_MAX_IMAGE_PIXELS} pixels"
        )

    if not isinstance(pixel_points, (list, tuple)):
        raise TypeError(f"Contour must be a list or tuple of points, got {type(pixel_points).__name__}")

    if len(pixel_points) > MAX_CONTOUR_VERTICES:
        raise ValueError(
            f"Contour vertex count ({len(pixel_points)}) exceeds maximum allowed limit ({MAX_CONTOUR_VERTICES})"
        )

    if len(pixel_points) < 3:
        raise ValueError(
            f"Contour must contain at least 3 points, got {len(pixel_points)}"
        )

    max_x_px = float(width - 1)
    max_y_px = float(height - 1)
    normalized: List[List[int]] = []

    for idx, pt in enumerate(pixel_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"Contour point at index {idx} must be a 2-element sequence [x, y], got {pt!r}")
        x, y = pt[0], pt[1]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError(f"Contour point at index {idx} coordinates must be numeric, got {[x, y]}")
        x_f, y_f = float(x), float(y)
        if math.isnan(x_f) or math.isnan(y_f) or math.isinf(x_f) or math.isinf(y_f):
            raise ValueError(f"Contour point at index {idx} coordinates cannot be NaN or Inf: {[x, y]}")

        clamped_x = max(0.0, min(max_x_px, x_f))
        clamped_y = max(0.0, min(max_y_px, y_f))

        nx = int(round((clamped_x / max_x_px) * NORMALIZATION_MAX)) if max_x_px > 0 else 0
        ny = int(round((clamped_y / max_y_px) * NORMALIZATION_MAX)) if max_y_px > 0 else 0

        nx = max(0, min(int(NORMALIZATION_MAX), nx))
        ny = max(0, min(int(NORMALIZATION_MAX), ny))
        normalized.append([nx, ny])

    return normalized


def contour_to_bounding_box(
    pixel_points: List[Tuple[float, float]],
    width: int,
    height: int,
    allow_zero_area: bool = False,
) -> Tuple[int, int, int, int]:
    """Compute tight integer [xmin, ymin, xmax, ymax] clamped within [0, width-1] and [0, height-1].

    Args:
        pixel_points: List of (x, y) coordinates in pixel space.
        width: Image width in pixels (> 0).
        height: Image height in pixels (> 0).
        allow_zero_area: If False, raises ValueError when the resulting box has zero width or height.

    Returns:
        Tight inclusive integer bounding box (xmin, ymin, xmax, ymax) satisfying
        0 <= xmin <= xmax < width and 0 <= ymin <= ymax < height.

    Raises:
        ValueError: If width/height <= 0, points list contains < 3 points,
                    or points produce a bounding box of 0 area (when allow_zero_area is False).
        TypeError: If points are not formatted correctly.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={width}, height={height} (both must be positive integers)"
        )

    if width * height > SAFE_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions {width}x{height} ({width * height} pixels) exceed safe limit of {SAFE_MAX_IMAGE_PIXELS} pixels"
        )

    if not isinstance(pixel_points, (list, tuple)):
        raise TypeError(f"Points must be a list or tuple of points, got {type(pixel_points).__name__}")

    if len(pixel_points) > MAX_CONTOUR_VERTICES:
        raise ValueError(
            f"Point count ({len(pixel_points)}) exceeds maximum allowed limit ({MAX_CONTOUR_VERTICES})"
        )

    if len(pixel_points) < 3:
        raise ValueError(
            f"Contour must contain at least 3 points, got {len(pixel_points)}"
        )

    xs: List[float] = []
    ys: List[float] = []
    for idx, pt in enumerate(pixel_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"Point at index {idx} must be (x, y), got {pt!r}")
        x, y = pt[0], pt[1]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError(f"Point at index {idx} coordinates must be numeric, got {[x, y]}")
        x_f, y_f = float(x), float(y)
        if math.isnan(x_f) or math.isnan(y_f) or math.isinf(x_f) or math.isinf(y_f):
            raise ValueError(f"Point at index {idx} coordinates cannot be NaN or Inf: {[x, y]}")
        xs.append(x_f)
        ys.append(y_f)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if not allow_zero_area and (min_x == max_x or min_y == max_y):
        raise ValueError(
            f"Contour produces a bounding box of 0 area: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}"
        )

    max_x_bound = width - 1
    max_y_bound = height - 1

    xmin = max(0, min(max_x_bound, int(round(min_x))))
    ymin = max(0, min(max_y_bound, int(round(min_y))))
    xmax = max(0, min(max_x_bound, int(round(max_x))))
    ymax = max(0, min(max_y_bound, int(round(max_y))))

    if xmin > xmax or ymin > ymax:
        raise ValueError(
            f"Invalid bounding box computed: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
        )

    if not allow_zero_area and (xmin == xmax or ymin == ymax):
        raise ValueError(
            f"Bounding box has zero area after integer clamping: {[xmin, ymin, xmax, ymax]}"
        )

    return xmin, ymin, xmax, ymax


def rasterize_polygon_to_mask(
    pixel_points: List[Tuple[float, float]],
    width: int,
    height: int,
) -> Image.Image:
    """Render polygon onto an 8-bit mode 'L' PIL Image (0 for background, 1 for foreground).

    Args:
        pixel_points: List of (x, y) coordinates in pixel space.
        width: Image width in pixels (> 0).
        height: Image height in pixels (> 0).

    Returns:
        PIL.Image.Image of mode 'L' and size (width, height), where 0 represents
        background and 1 represents foreground.

    Raises:
        ValueError: If dimensions <= 0, points list contains < 3 points,
                    or points are collinear/enclosing zero area.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={width}, height={height} (both must be positive integers)"
        )

    if width * height > SAFE_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions {width}x{height} ({width * height} pixels) exceed safe limit of {SAFE_MAX_IMAGE_PIXELS} pixels"
        )

    if not isinstance(pixel_points, (list, tuple)):
        raise TypeError(f"Polygon must be a list or tuple of points, got {type(pixel_points).__name__}")

    if len(pixel_points) > MAX_CONTOUR_VERTICES:
        raise ValueError(
            f"Polygon vertex count ({len(pixel_points)}) exceeds maximum allowed limit ({MAX_CONTOUR_VERTICES})"
        )

    if len(pixel_points) < 3:
        raise ValueError(
            f"Polygon must contain at least 3 points to rasterize, got {len(pixel_points)}"
        )

    area = calculate_polygon_area(pixel_points)
    if area < 1e-5:
        raise ValueError(f"Degenerate polygon has zero area ({area}); cannot rasterize")

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # Convert coordinates to tuples of floats for PIL ImageDraw
    pts_tuples = [(float(pt[0]), float(pt[1])) for pt in pixel_points]
    draw.polygon(pts_tuples, fill=1)

    return mask_img


def mask_to_cvat_flat_list(
    mask_image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[List[int]]:
    """Extract crop and flatten with trailing [xmin, ymin, xmax, ymax].

    According to the CVAT native mask format specification:
    - Tight bounding box is [xmin, ymin, xmax, ymax] (all inclusive integer coordinates,
      0 <= xmin <= xmax < W, 0 <= ymin <= ymax < H).
    - Crop binary mask to the bounding box: shape is (ymax - ymin + 1, xmax - xmin + 1).
    - Flatten the crop in row-major order into a 1D list of 0 and 1 integers:
      length = (ymax - ymin + 1) * (xmax - xmin + 1).
    - Append [xmin, ymin, xmax, ymax] as the last 4 elements of the list:
      [p0, p1, p2, ..., xmin, ymin, xmax, ymax].

    Args:
        mask_image: PIL Image containing binary mask (mode 'L', '1', etc.).
        bbox: Optional precomputed tight bounding box (xmin, ymin, xmax, ymax).
              If None, tight bounding box is computed from non-zero pixels.

    Returns:
        List of integers [p0, p1, ..., xmin, ymin, xmax, ymax], or None if the mask
        is empty (no foreground pixels) or degenerate.
    """
    if not isinstance(mask_image, Image.Image):
        raise TypeError(f"mask_image must be a PIL.Image.Image, got {type(mask_image).__name__}")

    w, h = mask_image.size
    if w <= 0 or h <= 0:
        return None

    if bbox is None:
        img_bbox = mask_image.getbbox()
        if img_bbox is None:
            # Mask is completely blank, no foreground pixels
            return None
        left, top, right, bottom = img_bbox
        xmin, ymin = left, top
        xmax, ymax = right - 1, bottom - 1
    else:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"bbox must be a 4-element sequence [xmin, ymin, xmax, ymax], got {bbox!r}")
        xmin, ymin, xmax, ymax = (int(b) for b in bbox)
        if not (0 <= xmin <= xmax < w and 0 <= ymin <= ymax < h):
            raise ValueError(
                f"Invalid bounding box: {[xmin, ymin, xmax, ymax]} for image size ({w}, {h})"
            )

    crop_w = xmax - xmin + 1
    crop_h = ymax - ymin + 1
    if crop_w <= 0 or crop_h <= 0:
        return None

    # In PIL crop, box is (left, top, right, bottom) with right and bottom exclusive
    crop_img = mask_image.crop((xmin, ymin, xmax + 1, ymax + 1))
    if crop_img.mode != "L":
        crop_img = crop_img.convert("L")

    raw_bytes = crop_img.tobytes()
    pixels = [1 if b > 0 else 0 for b in raw_bytes]

    # If the crop contains zero foreground pixels, treat as empty mask
    if not any(pixels):
        return None

    return pixels + [int(xmin), int(ymin), int(xmax), int(ymax)]


def polygon_to_cvat_mask(
    contour: List[List[Union[int, float]]],
    width: int,
    height: int,
    label: Optional[str] = None,
    confidence: Optional[Union[float, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Full pipeline: denormalize contour -> rasterize -> crop & flatten -> CVAT shape dict.

    Gracefully handles edge cases (empty contour, < 3 points, collinear vertices,
    zero-area bounding box, out-of-bounds coordinates) by returning None.

    Args:
        contour: Normalized [0, 1000] polygon coordinates in [x, y] format.
        width: Image width in pixels (> 0).
        height: Image height in pixels (> 0).
        label: Optional CVAT class label string.
        confidence: Optional detection confidence score.

    Returns:
        Dictionary containing:
        {
            "mask": [p0, p1, ..., xmin, ymin, xmax, ymax],
            "bbox": [xmin, ymin, xmax, ymax],
            # If label is provided:
            "label": label,
            "type": "mask",
            # If confidence is provided:
            "confidence": str(confidence)
        }
        or None if contour is invalid, empty, degenerate, collinear, or outside image.
    """
    if width <= 0 or height <= 0:
        return None

    if not isinstance(contour, (list, tuple)) or len(contour) < 3:
        return None

    try:
        pixel_points = denormalize_contour(contour, width=width, height=height)
    except (ValueError, TypeError):
        return None

    # Reject degenerate / collinear contours enclosing zero area
    area = calculate_polygon_area(pixel_points)
    if area < 0.5:
        return None

    try:
        mask_img = rasterize_polygon_to_mask(pixel_points, width=width, height=height)
    except (ValueError, TypeError):
        return None

    flat_mask = mask_to_cvat_flat_list(mask_img, bbox=None)
    if flat_mask is None:
        return None

    tight_bbox = flat_mask[-4:]

    res: Dict[str, Any] = {
        "mask": flat_mask,
        "bbox": [int(b) for b in tight_bbox],
        "pixel_polygon": pixel_points,
    }

    if label is not None:
        res["label"] = str(label)
        res["type"] = "mask"

    if confidence is not None:
        if isinstance(confidence, (int, float)):
            res["confidence"] = str(round(float(confidence), 2))
        else:
            res["confidence"] = str(confidence)

    return res


def cvat_mask_to_binary_image(
    cvat_mask: List[int],
    width: int,
    height: int,
) -> Image.Image:
    """Reverse operation: unpacks flat list and bbox, reconstructs full (width, height) binary image.

    Essential for round-trip testing, verification, and visualization.

    Args:
        cvat_mask: CVAT flat mask list [p0, p1, ..., xmin, ymin, xmax, ymax].
        width: Target full image width in pixels (> 0).
        height: Target full image height in pixels (> 0).

    Returns:
        PIL.Image.Image in mode 'L' of size (width, height) with 0 for background
        and 1 for foreground.

    Raises:
        ValueError: If width/height <= 0, mask list is too short (< 5 elements),
                    bbox coordinates are out of bounds or invalid, or pixel count
                    does not match (xmax - xmin + 1) * (ymax - ymin + 1).
        TypeError: If cvat_mask is not a list or sequence of integers.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={width}, height={height} (both must be positive integers)"
        )

    if width * height > SAFE_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions {width}x{height} ({width * height} pixels) exceed safe limit of {SAFE_MAX_IMAGE_PIXELS} pixels"
        )

    if not isinstance(cvat_mask, (list, tuple)):
        raise TypeError(f"cvat_mask must be a list or tuple of integers, got {type(cvat_mask).__name__}")

    if len(cvat_mask) < 5:
        raise ValueError(
            f"cvat_mask is too short: expected at least 5 elements (1 pixel + 4 bbox values), got {len(cvat_mask)}"
        )

    xmin, ymin, xmax, ymax = cvat_mask[-4:]
    crop_pixels = cvat_mask[:-4]

    for val, name in [(xmin, "xmin"), (ymin, "ymin"), (xmax, "xmax"), (ymax, "ymax")]:
        if not isinstance(val, int) and not (isinstance(val, float) and val.is_integer()):
            raise ValueError(f"Bbox coordinate {name} must be an integer, got {val!r}")

    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    if not (0 <= xmin <= xmax < width and 0 <= ymin <= ymax < height):
        raise ValueError(
            f"Bbox coordinates {[xmin, ymin, xmax, ymax]} are out of bounds for image size ({width}, {height})"
        )

    crop_w = xmax - xmin + 1
    crop_h = ymax - ymin + 1
    expected_pixels = crop_w * crop_h

    if len(crop_pixels) != expected_pixels:
        raise ValueError(
            f"Pixel count mismatch: bbox {[xmin, ymin, xmax, ymax]} specifies {crop_w}x{crop_h}={expected_pixels} "
            f"pixels, but mask list contains {len(crop_pixels)} pixels"
        )

    full_img = Image.new("L", (width, height), 0)
    if expected_pixels > 0:
        crop_bytes = bytes(1 if p > 0 else 0 for p in crop_pixels)
        crop_img = Image.frombytes("L", (crop_w, crop_h), crop_bytes)
        full_img.paste(crop_img, (xmin, ymin))

    return full_img


def overlay_mask_on_image(
    image: Image.Image,
    mask_image: Image.Image,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a binary mask onto an RGB image with semi-transparent color tint.

    Args:
        image: Original PIL Image (converted to RGB).
        mask_image: Binary mask image (mode 'L' or '1').
        color: RGB tuple for mask color (default green: (0, 255, 0)).
        alpha: Blending opacity in [0.0, 1.0] (default 0.4).

    Returns:
        New RGB PIL Image with mask tinted.
    """
    rgb = image.convert("RGB")
    mask_l = mask_image.convert("L")

    tint = Image.new("RGB", rgb.size, color)
    blended = Image.blend(rgb, tint, max(0.0, min(1.0, float(alpha))))

    stencil = mask_l.point(lambda p: 255 if p > 0 else 0)
    return Image.composite(blended, rgb, stencil)
