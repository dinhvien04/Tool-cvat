"""Image operations for CVAT x 9Router AI Annotation.

Provides image loading, dimension inspection, aspect-ratio preserving resizing,
base64 encoding, bounding box drawing on original images, and saving.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

# Pre-defined high-contrast distinct color palette for common autonomous driving labels
LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pedestrian": (255, 140, 0),      # Dark Orange
    "person": (255, 165, 0),          # Orange
    "rider": (154, 205, 50),          # Yellow Green
    "car": (30, 144, 255),            # Dodger Blue
    "truck": (70, 130, 180),          # Steel Blue
    "bus": (0, 139, 139),             # Dark Cyan
    "train": (199, 21, 133),          # Medium Violet Red
    "motorcycle": (153, 50, 204),     # Dark Orchid
    "bicycle": (50, 205, 50),         # Lime Green
    "traffic light": (255, 69, 0),    # Orange Red
    "traffic_light": (255, 99, 71),   # Tomato
    "traffic sign": (218, 165, 32),   # Goldenrod
    "traffic_sign": (240, 230, 140),  # Khaki
    "building": (128, 128, 128),      # Gray
    "road": (105, 105, 105),          # Dim Gray
    "sidewalk": (169, 169, 169),      # Dark Gray
    "vegetation": (34, 139, 34),      # Forest Green
    "sky": (135, 206, 235),           # Sky Blue
}


# Maximum image pixels to prevent decompression bomb attacks (approx 89 megapixels)
SAFE_MAX_IMAGE_PIXELS = 89_478_485
Image.MAX_IMAGE_PIXELS = SAFE_MAX_IMAGE_PIXELS


def get_label_color(label: str) -> Tuple[int, int, int]:
    """Get a consistent distinct RGB color for a given label string."""
    clean_label = label.strip().lower()
    if clean_label in LABEL_COLORS:
        return LABEL_COLORS[clean_label]

    # Deterministic color generation via MD5 hashing
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)

    # Avoid overly dark colors for visibility
    r = max(50, min(230, r))
    g = max(50, min(230, g))
    b = max(50, min(230, b))
    return (r, g, b)


def load_image(image_source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """Load an image and ensure RGB mode and correct EXIF orientation.

    Applies memory safety checks (preventing decompression bombs, corrupted data,
    and leaking file descriptors).

    Args:
        image_source: File path, raw bytes, or existing PIL Image.

    Returns:
        PIL Image in RGB format.
    """
    try:
        if isinstance(image_source, Image.Image):
            img = image_source.copy()
        elif isinstance(image_source, bytes):
            with Image.open(io.BytesIO(image_source)) as opened:
                img = opened.copy()
                img.load()
        elif isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Image file not found: {p}")
            with Image.open(p) as opened:
                img = opened.copy()
                img.load()
        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")
    except Image.DecompressionBombError as e:
        raise ValueError(f"Image rejected: potential decompression bomb (exceeds pixel limit): {e}") from e
    except Image.UnidentifiedImageError as e:
        raise ValueError(f"Cannot identify image file or unsupported image format: {e}") from e
    except OSError as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Failed to read or decode image: {e}") from e

    # Apply EXIF rotation if present (e.g. smartphone photos)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")

    if img.width <= 0 or img.height <= 0:
        raise ValueError(f"Invalid image dimensions: {img.width}x{img.height}")

    return img


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Return (width, height) of PIL Image."""
    return image.size


def resize_image_if_needed(
    image: Image.Image,
    max_size: int = 1600,
) -> Tuple[Image.Image, bool, Tuple[int, int]]:
    """Create resized copy for API sending respecting max_size preserving aspect ratio.

    Args:
        image: Original PIL Image.
        max_size: Maximum allowed width or height in pixels.

    Returns:
        Tuple of (resulting_image, was_resized, (new_width, new_height)).
    """
    orig_w, orig_h = image.size
    max_dim = max(orig_w, orig_h)

    if max_size <= 0 or max_dim <= max_size:
        return image.copy(), False, (orig_w, orig_h)

    scale = float(max_size) / float(max_dim)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized, True, (new_w, new_h)


def image_to_base64(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 90,
) -> str:
    """Encode PIL image to base64 string."""
    buf = io.BytesIO()
    save_format = format.upper()
    if save_format in ("JPG", "JPEG"):
        image.save(buf, format="JPEG", quality=quality, optimize=True)
    elif save_format == "PNG":
        image.save(buf, format="PNG", optimize=True)
    else:
        image.save(buf, format=save_format)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_data_url(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 90,
) -> str:
    """Encode PIL image to base64 Data URL (e.g. data:image/jpeg;base64,...)."""
    b64 = image_to_base64(image, format=format, quality=quality)
    mime = "jpeg" if format.lower() in ("jpg", "jpeg") else format.lower()
    return f"data:image/{mime};base64,{b64}"


# Aliases for convenience
encode_image_to_base64 = image_to_base64
encode_image_to_data_url = image_to_data_url


def _get_font(font_size: int) -> ImageFont.ImageFont:
    """Attempt to load TrueType font or fallback to default."""
    try:
        # Common Windows fonts
        for font_name in ["arial.ttf", "segoeui.ttf", "tahoma.ttf", "DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(font_name, font_size)
            except OSError:
                continue
    except Exception:
        pass
    return ImageFont.load_default()


def draw_bounding_boxes(
    original_image: Image.Image,
    detections: List[Any],
    line_width: Optional[int] = None,
    font_size: Optional[int] = None,
) -> Image.Image:
    """Draw bounding boxes with distinct colors and labels on the ORIGINAL image.

    Args:
        original_image: Original full-resolution image.
        detections: List of detection objects. Each can be:
            - An object with .label and .box_2d ([ymin, xmin, ymax, xmax] in [0, 1000])
            - A dict with 'label' and 'box_2d' or 'points' / 'pixel_box'
        line_width: Custom box border width. If None, scales adaptively.
        font_size: Custom font size. If None, scales adaptively.

    Returns:
        New PIL Image with rendered annotations.
    """
    img = original_image.copy()
    width, height = img.size
    if width <= 0 or height <= 0:
        return img
    draw = ImageDraw.Draw(img)

    # Adaptive scaling based on image size
    min_dim = min(width, height)
    box_width = line_width or max(2, int(min_dim / 350))
    f_size = font_size or max(12, int(min_dim / 70))
    font = _get_font(f_size)

    # Phase 3: Check if any items have polygon masks to render as semi-transparent overlays
    has_masks = False
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for item in detections:
        poly_pts = None
        if hasattr(item, "pixel_polygon") and getattr(item, "pixel_polygon"):
            poly_pts = getattr(item, "pixel_polygon")
        elif hasattr(item, "mask") and getattr(item, "mask"):
            try:
                from core.geometry import denormalize_contour
                poly_pts = denormalize_contour(getattr(item, "mask"), width=width, height=height)
            except Exception:
                poly_pts = None
        elif isinstance(item, dict):
            if "pixel_polygon" in item and item["pixel_polygon"]:
                poly_pts = item["pixel_polygon"]
            elif "mask" in item and isinstance(item["mask"], list) and len(item["mask"]) >= 3:
                first_pt = item["mask"][0]
                if isinstance(first_pt, (list, tuple)) and len(first_pt) == 2:
                    try:
                        from core.geometry import denormalize_contour
                        poly_pts = denormalize_contour(item["mask"], width=width, height=height)
                    except Exception:
                        poly_pts = None

        if poly_pts and len(poly_pts) >= 3:
            has_masks = True
            lbl = getattr(item, "label", None) if hasattr(item, "label") else item.get("label", "")
            c = get_label_color(lbl)
            fill_rgba = (c[0], c[1], c[2], 85)  # semi-transparent tint
            outline_rgba = (c[0], c[1], c[2], 220)
            pts_tuples = [(float(p[0]), float(p[1])) for p in poly_pts]
            overlay_draw.polygon(pts_tuples, fill=fill_rgba, outline=outline_rgba)

    if has_masks:
        img_rgba = img.convert("RGBA")
        img = Image.alpha_composite(img_rgba, overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

    for item in detections:
        # Extract label and coordinates
        label = ""
        box_coords: Optional[Tuple[float, float, float, float]] = None

        if hasattr(item, "label") and hasattr(item, "box_2d"):
            label = getattr(item, "label")
            box_2d = getattr(item, "box_2d")
            ymin, xmin, ymax, xmax = [float(c) for c in box_2d]
            x1 = (xmin / 1000.0) * width
            y1 = (ymin / 1000.0) * height
            x2 = (xmax / 1000.0) * width
            y2 = (ymax / 1000.0) * height
            box_coords = (x1, y1, x2, y2)
        elif isinstance(item, dict):
            label = item.get("label", "")
            if "pixel_box" in item:
                x1, y1, x2, y2 = item["pixel_box"]
                box_coords = (float(x1), float(y1), float(x2), float(y2))
            elif "box_2d" in item:
                ymin, xmin, ymax, xmax = [float(c) for c in item["box_2d"]]
                x1 = (xmin / 1000.0) * width
                y1 = (ymin / 1000.0) * height
                x2 = (xmax / 1000.0) * width
                y2 = (ymax / 1000.0) * height
                box_coords = (x1, y1, x2, y2)
            elif "points" in item and len(item["points"]) == 4:
                x1, y1, x2, y2 = item["points"]
                box_coords = (float(x1), float(y1), float(x2), float(y2))

        if box_coords is None:
            continue

        x1, y1, x2, y2 = box_coords

        # Clamp to image boundaries
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        # Check degenerate box
        if x2 <= x1 or y2 <= y1:
            continue

        color = get_label_color(label)

        # Draw outer rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        # Draw label badge
        badge_text = f" {label} "
        try:
            bbox = draw.textbbox((x1, y1), badge_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(badge_text) * (f_size * 0.6)
            text_h = f_size + 4

        # Position label badge above box, or inside if at the very top
        badge_y1 = y1 - text_h - 4
        badge_y2 = y1
        if badge_y1 < 0:
            badge_y1 = y1
            badge_y2 = min(float(height), y1 + text_h + 4)
        else:
            badge_y1 = max(0.0, badge_y1)

        badge_x1 = max(0.0, x1)
        badge_x2 = min(float(width), x1 + text_w + 4)

        # Draw badge background if valid
        if badge_x2 > badge_x1 and badge_y2 > badge_y1:
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)

            # Choose white or black text based on color brightness
            luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255.0
            text_color = (0, 0, 0) if luminance > 0.6 else (255, 255, 255)

            draw.text((badge_x1 + 2, badge_y1 + 1), badge_text, fill=text_color, font=font)

    return img


def save_annotated_image(
    image: Image.Image,
    output_path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """Save annotated image to target path ensuring parent directory exists.

    Args:
        image: PIL Image to save.
        output_path: Destination file path.
        quality: JPEG compression quality.

    Returns:
        Resolved Path to saved image.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    suffix = dest.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        image.save(dest, format="JPEG", quality=quality, optimize=True)
    elif suffix == ".png":
        image.save(dest, format="PNG", optimize=True)
    else:
        image.save(dest)

    return dest
