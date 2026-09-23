"""Tests for Week-2 AI Detector Runtime Status Auditor (scripts/week2_runtime_status.py).

Validates:
1. Dual registry query distinction:
   - Nuclio Dashboard API (:8070/api/functions/{fn_name})
   - CVAT Lambda Registry API (:18080/api/lambda/functions)
2. Secret protection and sanitization:
   - URL basic auth credential stripping.
   - Safe environment variable allowlist enforcement.
   - Sensitive key regex pattern masking.
3. Build SHA tracking and strict verification:
   - Matching git HEAD vs deployed container SHA.
   - Detection of missing or mismatched SHA.
   - Strict mode drift escalation.
4. Spec fingerprinting separation:
   - Semantic schema hash vs canonical CVAT spec hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from core.week2_schema import (
    POSE_CVAT_SPEC_HASH,
    POSE_SCHEMA_HASH,
    VF50_CVAT_SPEC_HASH,
    VF50_SCHEMA_HASH,
    compute_spec_fingerprint,
    get_canonical_cvat_spec,
)
from scripts.week2_runtime_status import (
    SAFE_ENV_ALLOWLIST,
    SENSITIVE_KEY_PATTERN,
    analyze_function_runtime,
    get_cvat_lambda_spec,
    get_nuclio_dashboard_spec,
    get_repo_spec,
    sanitize_url,
)


class TestSecretSanitizationAndSafety:
    """Validate that secrets and credentials can never be leaked or logged."""

    def test_sanitize_url_strips_basic_auth(self):
        url_with_auth = "http://admin:supersecretpassword@127.0.0.1:8080/api/v2"
        clean = sanitize_url(url_with_auth)
        assert "supersecretpassword" not in clean
        assert "admin" not in clean
        assert clean == "http://127.0.0.1:8080/api/v2"

    def test_sanitize_url_handles_clean_urls(self):
        clean_url = "http://127.0.0.1:8070/api/functions/my-fn"
        assert sanitize_url(clean_url) == clean_url

    def test_sensitive_key_pattern_detects_secrets(self):
        sensitive_keys = [
            "API_KEY",
            "NINEROUTER_KEY",
            "SECRET_TOKEN",
            "CVAT_WEBHOOK_SECRET",
            "AUTH_PASSWORD",
            "USER_CREDENTIALS",
            "DB_PWD",
            "AWS_SECRET_ACCESS_KEY",
        ]
        for key in sensitive_keys:
            assert SENSITIVE_KEY_PATTERN.search(key) is not None, f"Expected {key} to match sensitive pattern"

    def test_safe_env_allowlist_contains_only_benign_vars(self):
        for var in SAFE_ENV_ALLOWLIST:
            assert SENSITIVE_KEY_PATTERN.search(var) is None, f"{var} in SAFE_ENV_ALLOWLIST must not match sensitive pattern"
            assert "SECRET" not in var
            assert "KEY" not in var
            assert "TOKEN" not in var
            assert "PASSWORD" not in var


class TestDualRegistryQueries:
    """Validate that Nuclio Dashboard (:8070) and CVAT Lambda (:18080) registries are queried distinctly."""

    def test_get_nuclio_dashboard_spec_parses_response(self):
        dummy_spec = [{"name": "person", "type": "skeleton"}]
        mock_response_data = {
            "metadata": {
                "name": "ninerouter-human-pose-17",
                "annotations": {
                    "spec": json.dumps(dummy_spec),
                },
            }
        }
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with mock.patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = get_nuclio_dashboard_spec("ninerouter-human-pose-17", nuclio_url="http://127.0.0.1:8070")
            assert result == dummy_spec
            assert mock_urlopen.called
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://127.0.0.1:8070/api/functions/ninerouter-human-pose-17"

    def test_get_cvat_lambda_spec_parses_response_with_token(self):
        dummy_spec = [{"name": "person", "type": "skeleton"}]
        mock_response_data = [
            {
                "id": "cvat.ninerouter-human-pose-17",
                "name": "ninerouter-human-pose-17",
                "spec": json.dumps(dummy_spec),
            }
        ]
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with mock.patch.dict("os.environ", {"CVAT_TOKEN": "test-cvat-token-123"}):
            with mock.patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                result = get_cvat_lambda_spec("ninerouter-human-pose-17", cvat_url="http://127.0.0.1:18080")
                assert result == dummy_spec
                assert mock_urlopen.called
                req = mock_urlopen.call_args[0][0]
                assert req.full_url == "http://127.0.0.1:18080/api/lambda/functions"
                assert req.headers.get("Authorization") == "Token test-cvat-token-123"


class TestAnalyzeFunctionRuntime:
    """Validate runtime status, spec drift, and build SHA auditing."""

    @pytest.fixture
    def mock_pose17_cfg(self):
        repo_root = Path(__file__).resolve().parent.parent
        return {
            "container_name": "nuclio-nuclio-ninerouter-human-pose-17",
            "function_name": "ninerouter-human-pose-17",
            "yaml_path": repo_root / "serverless" / "ninerouter-human-pose-17" / "nuclio" / "function.yaml",
            "expected_labels": 1,
        }

    def test_not_deployed_analysis(self, mock_pose17_cfg):
        with mock.patch("scripts.week2_runtime_status.get_container_inspect", return_value=None), \
             mock.patch("scripts.week2_runtime_status.get_git_head_sha", return_value="abc1234567890"), \
             mock.patch("scripts.week2_runtime_status.get_nuclio_dashboard_spec", return_value=None), \
             mock.patch("scripts.week2_runtime_status.get_cvat_lambda_spec", return_value=None):

            report = analyze_function_runtime("pose17", mock_pose17_cfg)
            assert report["deployed"] is False
            assert report["status"] == "NOT_FOUND"
            assert report["ready"] is False
            assert report["fingerprints"]["canonical_spec"] == POSE_CVAT_SPEC_HASH
            assert report["fingerprints"]["semantic_schema"] == POSE_SCHEMA_HASH

    def test_deployed_matching_spec_and_sha(self, mock_pose17_cfg):
        canonical_spec = get_canonical_cvat_spec("pose17")
        test_sha = "1234567890abcdef1234567890abcdef12345678"

        mock_inspect = {
            "State": {
                "Status": "running",
                "Health": {"Status": "healthy"},
            },
            "NetworkSettings": {
                "Ports": {
                    "8080/tcp": [{"HostPort": "32768"}],
                },
            },
            "Config": {
                "Env": [
                    f"TOOL_CVAT_BUILD_SHA={test_sha}",
                    "VISION_MODEL=ag/gemini-3.8-flash-low",
                    "SECRET_KEY=should_be_stripped",
                ],
                "Labels": {
                    "nuclio.io/annotations": json.dumps({
                        "spec": json.dumps(canonical_spec),
                    }),
                },
            },
        }

        with mock.patch("scripts.week2_runtime_status.get_container_inspect", return_value=mock_inspect), \
             mock.patch("scripts.week2_runtime_status.get_git_head_sha", return_value=test_sha), \
             mock.patch("scripts.week2_runtime_status.get_nuclio_dashboard_spec", return_value=canonical_spec), \
             mock.patch("scripts.week2_runtime_status.get_cvat_lambda_spec", return_value=canonical_spec):

            report = analyze_function_runtime("pose17", mock_pose17_cfg, strict=True)
            assert report["deployed"] is True
            assert report["status"] == "running"
            assert report["health"] == "healthy"
            assert report["build_sha"] == test_sha
            assert report["build_sha_match"] is True
            assert report["deployed_spec_type"] == "CANONICAL_POSE17_17KPS"
            assert report["spec_drift"] is False
            assert report["ready"] is True
            assert "SECRET_KEY" not in report["env"]
            assert report["env"]["TOOL_CVAT_BUILD_SHA"] == test_sha
            assert report["fingerprints"]["canonical_spec"] == POSE_CVAT_SPEC_HASH
            assert report["fingerprints"]["deployed_spec"] == POSE_CVAT_SPEC_HASH
            assert report["fingerprints"]["nuclio_dashboard"] == POSE_CVAT_SPEC_HASH
            assert report["fingerprints"]["cvat_lambda"] == POSE_CVAT_SPEC_HASH

    def test_deployed_build_sha_mismatch_strict_mode(self, mock_pose17_cfg):
        canonical_spec = get_canonical_cvat_spec("pose17")
        deployed_sha = "1111111111111111111111111111111111111111"
        repo_sha = "2222222222222222222222222222222222222222"

        mock_inspect = {
            "State": {
                "Status": "running",
                "Health": {"Status": "healthy"},
            },
            "NetworkSettings": {"Ports": {}},
            "Config": {
                "Env": [f"TOOL_CVAT_BUILD_SHA={deployed_sha}"],
                "Labels": {
                    "nuclio.io/annotations": json.dumps({
                        "spec": json.dumps(canonical_spec),
                    }),
                },
            },
        }

        with mock.patch("scripts.week2_runtime_status.get_container_inspect", return_value=mock_inspect), \
             mock.patch("scripts.week2_runtime_status.get_git_head_sha", return_value=repo_sha), \
             mock.patch("scripts.week2_runtime_status.get_nuclio_dashboard_spec", return_value=canonical_spec), \
             mock.patch("scripts.week2_runtime_status.get_cvat_lambda_spec", return_value=canonical_spec):

            # Non-strict mode: build_sha_match is False, but spec_drift is False
            non_strict = analyze_function_runtime("pose17", mock_pose17_cfg, strict=False)
            assert non_strict["build_sha_match"] is False
            assert non_strict["spec_drift"] is False
            assert non_strict["ready"] is True

            # Strict mode: build_sha_match is False, spec_drift is set to True, ready is False
            strict_rep = analyze_function_runtime("pose17", mock_pose17_cfg, strict=True)
            assert strict_rep["build_sha_match"] is False
            assert strict_rep["spec_drift"] is True
            assert strict_rep["ready"] is False
            assert any("BUILD SHA MISMATCH" in d for d in strict_rep["drift_details"])

    def test_deployed_build_sha_missing_strict_mode(self, mock_pose17_cfg):
        canonical_spec = get_canonical_cvat_spec("pose17")
        repo_sha = "2222222222222222222222222222222222222222"

        mock_inspect = {
            "State": {
                "Status": "running",
                "Health": {"Status": "healthy"},
            },
            "NetworkSettings": {"Ports": {}},
            "Config": {
                "Env": ["VISION_MODEL=ag/gemini-3.8-flash-low"],
                "Labels": {
                    "nuclio.io/annotations": json.dumps({
                        "spec": json.dumps(canonical_spec),
                    }),
                },
            },
        }

        with mock.patch("scripts.week2_runtime_status.get_container_inspect", return_value=mock_inspect), \
             mock.patch("scripts.week2_runtime_status.get_git_head_sha", return_value=repo_sha), \
             mock.patch("scripts.week2_runtime_status.get_nuclio_dashboard_spec", return_value=canonical_spec), \
             mock.patch("scripts.week2_runtime_status.get_cvat_lambda_spec", return_value=canonical_spec):

            # Non-strict mode: build_sha_match is False, spec_drift is False, ready is True
            non_strict = analyze_function_runtime("pose17", mock_pose17_cfg, strict=False)
            assert non_strict["build_sha"] is None
            assert non_strict["build_sha_match"] is False
            assert non_strict["spec_drift"] is False
            assert non_strict["ready"] is True
            assert any("BUILD SHA MISSING" in d for d in non_strict["drift_details"])

            # Strict mode: build_sha missing escalates to spec_drift True and ready False
            strict_rep = analyze_function_runtime("pose17", mock_pose17_cfg, strict=True)
            assert strict_rep["build_sha"] is None
            assert strict_rep["build_sha_match"] is False
            assert strict_rep["spec_drift"] is True
            assert strict_rep["ready"] is False
            assert any("BUILD SHA MISSING" in d for d in strict_rep["drift_details"])

    def test_legacy_monolithic_face_detection(self):
        repo_root = Path(__file__).resolve().parent.parent
        vf50_cfg = {
            "container_name": "nuclio-nuclio-ninerouter-face-vf50",
            "function_name": "ninerouter-face-vf50",
            "yaml_path": repo_root / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml",
            "expected_labels": 7,
        }

        legacy_spec = [{"name": "face", "type": "skeleton", "sublabels": [{"id": i, "name": str(i)} for i in range(50)]}]
        mock_inspect = {
            "State": {
                "Status": "running",
                "Health": {"Status": "healthy"},
            },
            "NetworkSettings": {"Ports": {}},
            "Config": {
                "Env": [],
                "Labels": {
                    "nuclio.io/annotations": json.dumps({
                        "spec": json.dumps(legacy_spec),
                    }),
                },
            },
        }

        with mock.patch("scripts.week2_runtime_status.get_container_inspect", return_value=mock_inspect), \
             mock.patch("scripts.week2_runtime_status.get_git_head_sha", return_value="abc"), \
             mock.patch("scripts.week2_runtime_status.get_nuclio_dashboard_spec", return_value=legacy_spec), \
             mock.patch("scripts.week2_runtime_status.get_cvat_lambda_spec", return_value=legacy_spec):

            report = analyze_function_runtime("vf50", vf50_cfg)
            assert report["deployed_spec_type"] == "LEGACY_MONOLITHIC_FACE"
            assert report["spec_drift"] is True
            assert report["ready"] is False
            assert any("CRITICAL SPEC DRIFT" in d for d in report["drift_details"])
