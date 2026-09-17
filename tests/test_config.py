"""Tests for app.config module."""

from pathlib import Path

from app.config import AppConfig, LabelConfig


def test_label_config_loading():
    """Verify loading labels.yaml into LabelConfig."""
    yaml_path = Path("D:/tool-cvat/config/labels.yaml")
    cfg = LabelConfig.from_yaml(yaml_path)

    assert len(cfg.all_labels) == 31
    assert len(cfg.bbox_labels) == 13
    assert len(cfg.non_bbox_labels) == 18

    assert "car" in cfg.bbox_labels
    assert "pedestrian" in cfg.bbox_labels
    assert "person" in cfg.bbox_labels
    assert "traffic light" in cfg.bbox_labels
    assert "traffic_light" in cfg.bbox_labels


def test_app_config_defaults():
    """Verify default values in AppConfig."""
    cfg = AppConfig.load()
    assert cfg.ninerouter_url.startswith("http://")
    assert cfg.max_image_size == 1600
    assert cfg.vision_model == "ag/gemini-3.8-flash-high"
    assert len(cfg.labels.bbox_labels) == 13


def test_app_config_overrides():
    """Verify explicit overrides take precedence."""
    cfg = AppConfig.load(
        url_override="http://custom-host:9999/",
        key_override="test-key-123",
        model_override="custom-vision-model",
        output_dir_override="custom_output",
        max_image_size_override=1200,
    )

    assert cfg.ninerouter_url == "http://custom-host:9999"  # Trailing slash stripped
    assert cfg.ninerouter_key == "test-key-123"
    assert cfg.vision_model == "custom-vision-model"
    assert str(cfg.output_dir) == "custom_output"
    assert cfg.max_image_size == 1200
