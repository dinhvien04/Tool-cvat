"""Tests for Phase 2 configuration extensions."""

import os
from pathlib import Path
import pytest

from app.config import (
    DEFAULT_FALLBACK_CONFIDENCE,
    DEFAULT_NINEROUTER_TIMEOUT,
    DEFAULT_NINEROUTER_URL_CONTAINER,
    DEFAULT_NINEROUTER_URL_HOST,
    AppConfig,
    get_default_ninerouter_url,
    is_running_in_container,
)


def test_container_detection(monkeypatch):
    """Verify is_running_in_container detects container environments."""
    # When NUCLIO_FUNCTION_NAME is set
    monkeypatch.setenv("NUCLIO_FUNCTION_NAME", "ninerouter-vision")
    assert is_running_in_container() is True

    # When DOCKER_CONTAINER is set
    monkeypatch.delenv("NUCLIO_FUNCTION_NAME", raising=False)
    monkeypatch.setenv("DOCKER_CONTAINER", "true")
    assert is_running_in_container() is True

    # Clean environment
    monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
    if not Path("/.dockerenv").exists():
        assert is_running_in_container() is False


def test_default_ninerouter_url_selection(monkeypatch):
    """Verify get_default_ninerouter_url returns host.docker.internal in container."""
    monkeypatch.setenv("NUCLIO_FUNCTION_NAME", "ninerouter-vision")
    assert get_default_ninerouter_url() == DEFAULT_NINEROUTER_URL_CONTAINER

    monkeypatch.delenv("NUCLIO_FUNCTION_NAME", raising=False)
    if not Path("/.dockerenv").exists():
        assert get_default_ninerouter_url() == DEFAULT_NINEROUTER_URL_HOST


def test_fallback_confidence_constant():
    """Verify DEFAULT_FALLBACK_CONFIDENCE is within [0.0, 1.0]."""
    assert 0.0 <= DEFAULT_FALLBACK_CONFIDENCE <= 1.0
    assert DEFAULT_FALLBACK_CONFIDENCE == 1.0


def test_app_config_phase2_defaults():
    """Verify AppConfig load respects Phase 2 defaults."""
    cfg = AppConfig.load()
    assert cfg.ninerouter_timeout == DEFAULT_NINEROUTER_TIMEOUT
    assert cfg.vision_model == "ag/gemini-3.8-flash-high"
