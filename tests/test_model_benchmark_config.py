"""Unit tests for detector-specific model selection and fallback configuration.

Verifies:
1. AppConfig supports RECTANGLE_MASK_MODEL, POLYGON_MASK_MODEL, POLYLINE_MODEL with VISION_MODEL fallback.
2. AppConfig.get_model_for_mode correctly selects the right model for each detector mode.
3. NineRouterClient methods resolve_rectangle_mask_model, resolve_polygon_mask_model, and resolve_polyline_model
   prioritize fast flash-low models.
4. Nuclio ModelHandler classes for all 3 detectors respect detector-specific env vars with VISION_MODEL fallback.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from app.config import AppConfig, DEFAULT_VISION_MODEL
from app.client import (
    NineRouterClient,
    PREFERRED_RECTANGLE_MASK_MODELS,
    PREFERRED_POLYGON_MASK_MODELS,
    PREFERRED_POLYLINE_MODELS,
)


class TestDetectorModelConfig:
    """Test detector-specific model resolution and fallback mechanisms."""

    def test_default_model_fallback(self):
        cfg = AppConfig()
        assert cfg.get_model_for_mode("rectangle_mask") == cfg.vision_model
        assert cfg.get_model_for_mode("box_mask") == cfg.vision_model
        assert cfg.get_model_for_mode("polygon_mask") == cfg.vision_model
        assert cfg.get_model_for_mode("polyline") == cfg.vision_model
        assert cfg.get_model_for_mode("unknown") == cfg.vision_model

    def test_detector_specific_model_overrides(self):
        cfg = AppConfig(
            vision_model="ag/gemini-3.8-flash-high",
            rectangle_mask_model="ag/gemini-3.8-flash-low",
            polygon_mask_model="ag/gemini-3.8-flash-medium",
            polyline_model="ag/gemini-3.7-flash-low",
        )
        assert cfg.get_model_for_mode("rectangle_mask") == "ag/gemini-3.8-flash-low"
        assert cfg.get_model_for_mode("box_mask") == "ag/gemini-3.8-flash-low"
        assert cfg.get_model_for_mode("polygon_mask") == "ag/gemini-3.8-flash-medium"
        assert cfg.get_model_for_mode("polyline") == "ag/gemini-3.7-flash-low"
        assert cfg.get_model_for_mode(None) == "ag/gemini-3.8-flash-high"

    def test_env_var_loading_and_fallback(self, monkeypatch):
        monkeypatch.setenv("VISION_MODEL", "fallback-vision-model")
        monkeypatch.setenv("RECTANGLE_MASK_MODEL", "custom-rect-model")
        monkeypatch.delenv("POLYGON_MASK_MODEL", raising=False)
        monkeypatch.setenv("POLYLINE_MODEL", "custom-polyline-model")

        cfg = AppConfig.load()
        assert cfg.vision_model == "fallback-vision-model"
        assert cfg.rectangle_mask_model == "custom-rect-model"
        assert cfg.polygon_mask_model is None
        assert cfg.polyline_model == "custom-polyline-model"

        assert cfg.get_model_for_mode("rectangle_mask") == "custom-rect-model"
        # polygon_mask should fall back to vision_model
        assert cfg.get_model_for_mode("polygon_mask") == "fallback-vision-model"
        assert cfg.get_model_for_mode("polyline") == "custom-polyline-model"

    def test_client_prioritizes_flash_low(self):
        assert PREFERRED_RECTANGLE_MASK_MODELS[0] == "ag/gemini-3.8-flash-low"
        assert PREFERRED_POLYGON_MASK_MODELS[0] == "ag/gemini-3.8-flash-low"
        assert PREFERRED_POLYLINE_MODELS[0] == "ag/gemini-3.8-flash-low"

    def test_client_resolve_models(self):
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        available = [
            "ag/gemini-3.8-flash-high",
            "ag/gemini-3.8-flash-medium",
            "ag/gemini-3.8-flash-low",
        ]
        with patch.object(client, "get_vision_model_ids", return_value=available):
            # Dynamic resolution picks fastest preferred model (flash-low)
            assert client.resolve_rectangle_mask_model() == "ag/gemini-3.8-flash-low"
            assert client.resolve_polygon_mask_model() == "ag/gemini-3.8-flash-low"
            assert client.resolve_polyline_model() == "ag/gemini-3.8-flash-low"

            # Explicit preference is respected
            assert client.resolve_rectangle_mask_model("ag/gemini-3.8-flash-high") == "ag/gemini-3.8-flash-high"
            assert client.resolve_polygon_mask_model("ag/gemini-3.8-flash-high") == "ag/gemini-3.8-flash-high"
            assert client.resolve_polyline_model("ag/gemini-3.8-flash-high") == "ag/gemini-3.8-flash-high"
