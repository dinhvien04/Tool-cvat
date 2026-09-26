"""Tests for Buddha Model Resolution and Fallback Policy (core/buddha_contract.py).

Validates:
- Resolving Claude Opus 5.5 as the canonical vision model.
- Strict rejection when Opus 5.5 is missing and fallback is disallowed.
- Authorized fallback to Opus 4.6 when allow_fallback=True or BUDDHA_ALLOW_MODEL_FALLBACK=1.
- Explicit preferred model resolution and validation against available models.
- Clean error messaging enumerating discovered models.
"""

from __future__ import annotations

import os
from unittest import mock
import pytest

from app.client import NineRouterClient, NineRouterError
from core.buddha_contract import resolve_buddha_model


class DummyMockClient:
    """Mock client returning configurable model lists."""

    def __init__(self, available_models: list[str]):
        self._available_models = available_models

    def list_models(self) -> list[str]:
        return list(self._available_models)


class TestBuddhaModelResolution:
    """Validate Buddha Multi-Limb model resolution rules."""

    def test_resolve_opus_55_when_available(self):
        client = DummyMockClient(["claude-opus-5-5", "claude-sonnet-4-6", "ag/claude-opus-4-6-thinking"])
        resolved = resolve_buddha_model(client, allow_fallback=False)
        assert resolved == "claude-opus-5-5"

    def test_resolve_ag_opus_55_alias(self):
        client = DummyMockClient(["ag/claude-opus-5-5", "claude-sonnet-4-6"])
        resolved = resolve_buddha_model(client, allow_fallback=False)
        assert resolved == "ag/claude-opus-5-5"

    def test_strict_mode_rejects_when_opus_55_missing(self):
        client = DummyMockClient(["ag/claude-opus-4-6-thinking", "claude-sonnet-4-6"])
        with pytest.raises(NineRouterError) as exc_info:
            resolve_buddha_model(client, allow_fallback=False)
        assert "Claude Opus 5.5 vision model" in str(exc_info.value)
        assert "ag/claude-opus-4-6-thinking" in str(exc_info.value)

    def test_fallback_to_opus_46_when_allowed(self):
        client = DummyMockClient(["ag/claude-opus-4-6-thinking", "claude-sonnet-4-6"])
        resolved = resolve_buddha_model(client, allow_fallback=True)
        assert resolved == "ag/claude-opus-4-6-thinking"

    def test_fallback_via_env_var(self):
        client = DummyMockClient(["ag/claude-opus-4-6-thinking", "claude-sonnet-4-6"])
        with mock.patch.dict(os.environ, {"BUDDHA_ALLOW_MODEL_FALLBACK": "1"}):
            resolved = resolve_buddha_model(client)
            assert resolved == "ag/claude-opus-4-6-thinking"

    def test_explicit_preferred_model_success(self):
        client = DummyMockClient(["custom-deity-vlm-v1", "claude-opus-5-5"])
        resolved = resolve_buddha_model(client, preferred_model="custom-deity-vlm-v1")
        assert resolved == "custom-deity-vlm-v1"

    def test_explicit_preferred_model_not_found(self):
        client = DummyMockClient(["claude-opus-5-5", "claude-sonnet-4-6"])
        with pytest.raises(NineRouterError) as exc_info:
            resolve_buddha_model(client, preferred_model="nonexistent-model")
        assert "Preferred vision model 'nonexistent-model' not found" in str(exc_info.value)

    def test_buddha_model_env_var_override(self):
        client = DummyMockClient(["my-env-vlm", "claude-opus-5-5"])
        with mock.patch.dict(os.environ, {"BUDDHA_MODEL": "my-env-vlm"}):
            resolved = resolve_buddha_model(client)
            assert resolved == "my-env-vlm"
