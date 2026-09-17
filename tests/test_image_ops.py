"""Tests for app.image_ops module."""

import io
from pathlib import Path
from PIL import Image

from app.image_ops import (
    draw_bounding_boxes,
    get_image_dimensions,
    get_label_color,
    image_to_base64,
    image_to_data_url,
    load_image,
    resize_image_if_needed,
    save_annotated_image,
)


def test_load_image_rgb_conversion():
    """Verify loading converts RGBA/grayscale to RGB."""
    # Test RGBA
    rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    loaded = load_image(rgba_img)
    assert loaded.mode == "RGB"
    assert loaded.size == (100, 100)

    # Test raw bytes
    buf = io.BytesIO()
    rgba_img.save(buf, format="PNG")
    loaded_from_bytes = load_image(buf.getvalue())
    assert loaded_from_bytes.mode == "RGB"


def test_resize_image_aspect_ratio_preservation():
    """Verify resize_image_if_needed respects max_size and aspect ratio."""
    # Image within limit: 800x600, max_size=1600 -> not resized
    small_img = Image.new("RGB", (800, 600), color="blue")
    res_img, was_resized, (new_w, new_h) = resize_image_if_needed(small_img, max_size=1600)
    assert not was_resized
    assert (new_w, new_h) == (800, 600)

    # Large image: 3200x1600 (aspect ratio 2:1), max_size=1600 -> 1600x800
    large_img = Image.new("RGB", (3200, 1600), color="green")
    res_img, was_resized, (new_w, new_h) = resize_image_if_needed(large_img, max_size=1600)
    assert was_resized
    assert new_w == 1600
    assert new_h == 800

    # Large vertical image: 1000x4000 (aspect ratio 1:4), max_size=1600 -> 400x1600
    tall_img = Image.new("RGB", (1000, 4000), color="yellow")
    res_img, was_resized, (new_w, new_h) = resize_image_if_needed(tall_img, max_size=1600)
    assert was_resized
    assert new_h == 1600
    assert new_w == 400


def test_image_base64_encoding():
    """Verify base64 string and data URL formatting."""
    img = Image.new("RGB", (50, 50), color="red")
    b64 = image_to_base64(img, format="JPEG")
    assert isinstance(b64, str)
    assert len(b64) > 0

    data_url = image_to_data_url(img, format="JPEG")
    assert data_url.startswith("data:image/jpeg;base64,")


def test_draw_bounding_boxes_on_original_image():
    """Verify drawing bounding boxes on original image."""
    orig_img = Image.new("RGB", (1000, 500), color="white")

    detections = [
        {"label": "car", "box_2d": [100, 100, 400, 500]},  # ymin=100, xmin=100, ymax=400, xmax=500
        {"label": "pedestrian", "box_2d": [200, 600, 800, 750]},
    ]

    annotated = draw_bounding_boxes(orig_img, detections)
    assert annotated.size == (1000, 500)
    # Ensure drawing altered pixels
    assert annotated.tobytes() != orig_img.tobytes()


def test_get_label_color_consistency():
    """Verify distinct and deterministic colors."""
    car_color = get_label_color("car")
    ped_color = get_label_color("pedestrian")
    assert car_color != ped_color

    # Case insensitivity
    assert get_label_color("Car") == car_color
    assert get_label_color("  car  ") == car_color


def test_save_annotated_image(tmp_path):
    """Verify saving image to disk creating directories as necessary."""
    img = Image.new("RGB", (100, 100), color="purple")
    out_file = tmp_path / "subfolder" / "result_bbox.jpg"
    saved = save_annotated_image(img, out_file)

    assert saved.exists()
    assert saved.is_file()
    assert saved.stat().st_size > 0
