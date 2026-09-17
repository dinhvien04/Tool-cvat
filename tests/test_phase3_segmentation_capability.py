"""Tests for Phase 3 segmentation model capability probing and resolution."""

import json
from unittest.mock import MagicMock, patch
import pytest

from app.client import (
    _SEGMENTATION_CAPABILITY_CACHE,
    NineRouterClient,
    NineRouterError,
    PREFERRED_SEGMENTATION_MODELS,
    VisionResponse,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear capability cache before and after each test."""
    _SEGMENTATION_CAPABILITY_CACHE.clear()
    yield
    _SEGMENTATION_CAPABILITY_CACHE.clear()


@pytest.fixture
def mock_client():
    client = NineRouterClient(base_url="http://127.0.0.1:20128", api_key="dummy")
    client.get_vision_model_ids = MagicMock(return_value=[
        "ag/gemini-3.8-flash-high",
        "ag/gemini-3.7-flash-high",
        "custom/generic-vision",
    ])
    return client


def test_resolve_segmentation_model_prioritizes_gemini_flash(mock_client):
    """Verify resolve_segmentation_model prioritizes proven segmentation models."""
    resolved = mock_client.resolve_segmentation_model()
    assert resolved == "ag/gemini-3.8-flash-high"
    assert resolved in PREFERRED_SEGMENTATION_MODELS


def test_resolve_segmentation_model_respects_explicit_model(mock_client):
    """Verify explicit model request is respected if available in 9Router."""
    resolved = mock_client.resolve_segmentation_model(preferred_model="custom/generic-vision")
    assert resolved == "custom/generic-vision"


def test_resolve_segmentation_model_fails_on_unavailable_model(mock_client):
    """Verify requesting an unavailable model raises NineRouterError."""
    with pytest.raises(NineRouterError) as exc_info:
        mock_client.resolve_segmentation_model(preferred_model="nonexistent-model")
    assert "not available in 9Router" in str(exc_info.value)


def test_probe_segmentation_capability_validates_label_box_and_mask(mock_client):
    """Verify probe succeeds when model returns valid label, box_2d, and polygon mask."""
    mock_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [100, 100, 400, 500],
                    "mask": [[100, 100], [500, 100], [500, 400], [100, 400]],
                }
            ]
        }),
        raw_response={},
        duration_seconds=0.2,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request = MagicMock(return_value=mock_resp)

    ok, msg = mock_client.probe_segmentation_capability("ag/gemini-3.8-flash-high")
    assert ok is True
    assert "successfully verified" in msg
    assert _SEGMENTATION_CAPABILITY_CACHE.get("ag/gemini-3.8-flash-high") is True


def test_probe_segmentation_capability_fails_when_mask_omitted(mock_client):
    """Verify probe fails when model omits segmentation mask (bbox-only model)."""
    mock_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [100, 100, 400, 500],
                    # No mask
                }
            ]
        }),
        raw_response={},
        duration_seconds=0.2,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request = MagicMock(return_value=mock_resp)

    ok, msg = mock_client.probe_segmentation_capability("ag/gemini-3.8-flash-high")
    assert ok is False
    assert "missing or degenerate mask" in msg
    assert _SEGMENTATION_CAPABILITY_CACHE.get("ag/gemini-3.8-flash-high") is False


def test_probe_segmentation_capability_fails_when_mask_degenerate(mock_client):
    """Verify probe fails when model returns fewer than 3 vertices."""
    mock_resp = VisionResponse(
        content=json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [100, 100, 400, 500],
                    "mask": [[100, 100], [500, 100]],  # Only 2 points, not a polygon
                }
            ]
        }),
        raw_response={},
        duration_seconds=0.2,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )
    mock_client.send_vision_request = MagicMock(return_value=mock_resp)

    ok, msg = mock_client.probe_segmentation_capability("ag/gemini-3.8-flash-high")
    assert ok is False
    assert "missing or degenerate mask" in msg


def test_probe_segmentation_capability_caching(mock_client):
    """Verify capability probe uses cached result and avoids repeated network requests."""
    _SEGMENTATION_CAPABILITY_CACHE["cached-model"] = True
    mock_client.get_vision_model_ids.return_value.append("cached-model")

    mock_client.send_vision_request = MagicMock()
    ok, msg = mock_client.probe_segmentation_capability("cached-model")

    assert ok is True
    assert "Cached" in msg
    mock_client.send_vision_request.assert_not_called()


def test_resolve_segmentation_model_with_probe_true(mock_client):
    """Verify resolve_segmentation_model with probe=True iterates candidates until one passes."""
    def fake_probe(model, timeout=None):
        if model == "ag/gemini-3.8-flash-high":
            return True, "verified"
        return False, "failed"

    mock_client.probe_segmentation_capability = MagicMock(side_effect=fake_probe)
    resolved = mock_client.resolve_segmentation_model(probe=True)
    assert resolved == "ag/gemini-3.8-flash-high"
