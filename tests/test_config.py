"""Tests for app.config module."""

from pathlib import Path

from app.config import DEFAULT_VISION_MODEL, AppConfig, LabelConfig


def test_label_config_loading():
    """Verify loading labels.yaml into LabelConfig."""
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "labels.yaml"
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
    assert cfg.vision_model == DEFAULT_VISION_MODEL
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


def test_app_config_timeout_and_container(monkeypatch):
    """Verify NINEROUTER_TIMEOUT and container URL defaulting."""
    from app.config import (
        DEFAULT_NINEROUTER_URL_CONTAINER,
        DEFAULT_NINEROUTER_URL_HOST,
        get_default_ninerouter_url,
        is_running_in_container,
    )

    # In container simulation
    monkeypatch.setenv("NUCLIO_FUNCTION_NAME", "cvat-vision-annotator")
    assert is_running_in_container() is True
    assert get_default_ninerouter_url() == DEFAULT_NINEROUTER_URL_CONTAINER

    # On host simulation
    monkeypatch.delenv("NUCLIO_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
    # Check default on host
    if not Path("/.dockerenv").exists():
        assert get_default_ninerouter_url() == DEFAULT_NINEROUTER_URL_HOST

    # Test timeout override and env var
    monkeypatch.setenv("NINEROUTER_TIMEOUT", "45.5")
    cfg = AppConfig.load()
    assert cfg.ninerouter_timeout == 45.5

    cfg_override = AppConfig.load(timeout_override=15.0)
    assert cfg_override.ninerouter_timeout == 15.0


def test_api_key_masking_in_config():
    """Verify API key is never exposed in AppConfig representation."""
    cfg = AppConfig.load(key_override="sk-123456789abcdef")
    repr_str = repr(cfg)
    assert "sk-123456789abcdef" not in repr_str
    assert "sk-...def" in repr_str
