"""Tests for app.client module covering all HTTP failure modes and edge cases."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.client import (
    NineRouterAuthError,
    NineRouterClient,
    NineRouterConnectionError,
    NineRouterError,
    NineRouterRequestError,
)


def test_client_headers_without_key():
    """Verify headers without API key."""
    client = NineRouterClient(base_url="http://127.0.0.1:20128")
    headers = client._get_headers()
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_client_headers_with_key():
    """Verify headers with API key."""
    client = NineRouterClient(base_url="http://127.0.0.1:20128", api_key="secret-key")
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer secret-key"


def test_client_health_success():
    """Test health check returns expected status."""
    client = NineRouterClient(base_url="http://127.0.0.1:20128")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}

    with patch.object(client.session, "get", return_value=mock_resp):
        res = client.get_health()
        assert res == {"ok": True}


def test_client_health_connection_error():
    """Test connection failure raises NineRouterConnectionError."""
    client = NineRouterClient(base_url="http://127.0.0.1:99999")
    with patch.object(client.session, "get", side_effect=requests.exceptions.ConnectionError("Failed")):
        with pytest.raises(NineRouterConnectionError):
            client.get_health()


def test_client_health_non_200():
    """Test health check returning 500 raises NineRouterRequestError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    with patch.object(client.session, "get", return_value=mock_resp):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.get_health()
        assert exc_info.value.status_code == 500


def test_client_health_timeout():
    """Test timeout in health check raises NineRouterError."""
    client = NineRouterClient()
    with patch.object(client.session, "get", side_effect=requests.exceptions.Timeout("Timed out")):
        with pytest.raises(NineRouterError, match="Health check failed"):
            client.get_health()


def test_get_vision_models_fallback_filtering():
    """Test discovery when /v1/models/image-to-text is empty and fallback filters /v1/models."""
    client = NineRouterClient()

    resp_empty = MagicMock()
    resp_empty.status_code = 200
    resp_empty.json.return_value = {"data": []}

    resp_models = MagicMock()
    resp_models.status_code = 200
    resp_models.json.return_value = {
        "data": [
            {
                "id": "text-only-model",
                "capabilities": {"vision": False},
            },
            {
                "id": "ag/gemini-3.8-flash-high",
                "capabilities": {"vision": True},
            },
            {
                "id": "custom-claude-vl",
                "capabilities": {},
            },
        ]
    }

    def mock_get(url, *args, **kwargs):
        if "image-to-text" in url:
            return resp_empty
        return resp_models

    with patch.object(client.session, "get", side_effect=mock_get):
        vision_models = client.get_vision_models()
        assert len(vision_models) == 2
        ids = [m["id"] for m in vision_models]
        assert "ag/gemini-3.8-flash-high" in ids
        assert "custom-claude-vl" in ids
        assert "text-only-model" not in ids


def test_get_vision_models_connection_error():
    """Test connection error in get_vision_models raises NineRouterConnectionError."""
    client = NineRouterClient()
    with patch.object(client.session, "get", side_effect=requests.exceptions.ConnectionError("Unreachable")):
        with pytest.raises(NineRouterConnectionError):
            client.get_vision_models()


def test_get_vision_models_http_error():
    """Test HTTP 500 error in get_vision_models raises NineRouterRequestError."""
    client = NineRouterClient()
    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.text = "Server Error"

    # image-to-text fails with RequestException, /v1/models returns 500
    def mock_get(url, *args, **kwargs):
        if "image-to-text" in url:
            raise requests.exceptions.RequestException("404")
        return resp_500

    with patch.object(client.session, "get", side_effect=mock_get):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.get_vision_models()
        assert exc_info.value.status_code == 500


def test_get_vision_model_ids_convenience():
    """Test get_vision_model_ids convenience method."""
    client = NineRouterClient()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}

    with patch.object(client.session, "get", return_value=resp):
        ids = client.get_vision_model_ids()
        assert ids == ["model-1", "model-2"]


def test_send_vision_request_success():
    """Test sending vision request generates proper payload and processes response."""
    client = NineRouterClient()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "ag/gemini-3.8-flash-high",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"objects": [{"label": "car", "box_2d": [100, 200, 300, 400]}]}',
                }
            }
        ],
        "usage": {"total_tokens": 120},
    }

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        res = client.send_vision_request(
            model="ag/gemini-3.8-flash-high",
            image_bytes_or_b64="data:image/jpeg;base64,dGVzdA==",
            allowed_labels=["car", "pedestrian"],
        )

        assert res.model == "ag/gemini-3.8-flash-high"
        assert res.status_code == 200
        assert "objects" in res.content
        assert res.duration_seconds >= 0.0

        # Verify posted payload
        called_payload = mock_post.call_args[1]["json"]
        assert called_payload["model"] == "ag/gemini-3.8-flash-high"
        assert called_payload["stream"] is False
        assert called_payload["response_format"] == {"type": "json_object"}
        assert len(called_payload["messages"]) == 2


def test_send_vision_request_with_raw_bytes():
    """Test sending vision request accepting raw bytes."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"objects": []}'}}],
    }

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        client.send_vision_request(
            model="ag/gemini-3.8-flash-high",
            image_bytes_or_b64=b"fake_image_bytes",
        )
        called_payload = mock_post.call_args[1]["json"]
        image_url = called_payload["messages"][1]["content"][1]["image_url"]["url"]
        assert image_url.startswith("data:image/jpeg;base64,")


def test_send_vision_request_invalid_input_type():
    """Test sending vision request with invalid image type raises ValueError."""
    client = NineRouterClient()
    with pytest.raises(ValueError, match="image_bytes_or_b64 must be bytes or str"):
        client.send_vision_request(
            model="test-model",
            image_bytes_or_b64=12345,  # type: ignore
        )


# =====================================================================
# HTTP Failure Modes
# =====================================================================

def test_send_vision_request_connection_timeout():
    """Test connection timeout raises NineRouterRequestError with 408."""
    client = NineRouterClient()
    with patch.object(client.session, "post", side_effect=requests.exceptions.ConnectTimeout("Connect timeout")):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )
        assert exc_info.value.status_code == 408


def test_send_vision_request_read_timeout():
    """Test read timeout raises NineRouterRequestError with 408."""
    client = NineRouterClient()
    with patch.object(client.session, "post", side_effect=requests.exceptions.ReadTimeout("Read timeout")):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )
        assert exc_info.value.status_code == 408


def test_send_vision_request_401_unauthorized():
    """Test 401 Unauthorized raises NineRouterAuthError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized: Invalid API key"

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterAuthError, match="401"):
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )


def test_send_vision_request_403_forbidden():
    """Test 403 Forbidden raises NineRouterAuthError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden: Access denied"

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterAuthError, match="403"):
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )


def test_send_vision_request_500_internal_error():
    """Test 500 Internal Server Error raises NineRouterRequestError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )
        assert exc_info.value.status_code == 500
        assert exc_info.value.response_body == "Internal Server Error"


def test_send_vision_request_502_bad_gateway():
    """Test 502 Bad Gateway raises NineRouterRequestError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.text = "502 Bad Gateway"

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )
        assert exc_info.value.status_code == 502


def test_send_vision_request_503_service_unavailable():
    """Test 503 Service Unavailable raises NineRouterRequestError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "503 Service Unavailable"

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterRequestError) as exc_info:
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )
        assert exc_info.value.status_code == 503


def test_send_vision_request_network_unreachable():
    """Test network connection failure raises NineRouterConnectionError."""
    client = NineRouterClient()
    with patch.object(client.session, "post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(NineRouterConnectionError):
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )


def test_send_vision_request_invalid_json_body():
    """Test 200 response with invalid JSON body raises NineRouterError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON")
    mock_resp.text = "Not JSON"

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterError, match="Failed to decode"):
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )


def test_send_vision_request_empty_choices():
    """Test 200 response with empty choices raises NineRouterError."""
    client = NineRouterClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(NineRouterError, match="No choices returned"):
            client.send_vision_request(
                model="test-model",
                image_bytes_or_b64="dGVzdA==",
            )
