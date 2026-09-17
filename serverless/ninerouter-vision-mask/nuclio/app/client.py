"""9Router HTTP Client for Vision Model Inference.

Handles health checks, vision model discovery, and sending multimodal
chat completion requests to 9Router.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from core.vision_contract import (
    DEFAULT_BBOX_LABELS,
    MODE_BOX,
    SYSTEM_PROMPT,
    build_openai_vision_payload,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


class NineRouterError(Exception):
    """Base exception for 9Router client errors."""
    pass


class NineRouterConnectionError(NineRouterError):
    """Raised when connecting to 9Router fails."""
    pass


class NineRouterAuthError(NineRouterError):
    """Raised when authentication fails (401/403)."""
    pass


class NineRouterRequestError(NineRouterError):
    """Raised when 9Router API returns an error status code."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class VisionResponse:
    """Represents a response from 9Router vision chat completion."""
    content: str
    raw_response: Dict[str, Any]
    duration_seconds: float
    model: str
    status_code: int
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def raw_content(self) -> str:
        """Alias for content to support both naming conventions."""
        return self.content


class NineRouterClient:
    """Client for interacting with 9Router's OpenAI-compatible and management endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:20128",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        try:
            self.timeout = float(timeout) if (timeout is not None and float(timeout) > 0) else 60.0
        except (ValueError, TypeError):
            self.timeout = 60.0
        self.session = requests.Session()

    def _sanitize_error_text(self, text: Optional[str]) -> str:
        """Sanitize sensitive credentials from error messages and response texts."""
        if not text:
            return ""
        if self.api_key and self.api_key in text:
            return text.replace(self.api_key, "***")
        return text

    def __repr__(self) -> str:
        """Safe string representation masking API key."""
        if not self.api_key:
            masked = "None"
        elif len(self.api_key) <= 6:
            masked = "***"
        else:
            masked = f"{self.api_key[:3]}...{self.api_key[-3:]}"
        return f"NineRouterClient(base_url={self.base_url!r}, api_key={masked!r}, timeout={self.timeout})"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CVAT-9Router-Client/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_health(self) -> Dict[str, Any]:
        """Check 9Router health status via GET /api/health.

        Returns:
            Dict containing health status (e.g. {"ok": True}).
        """
        url = f"{self.base_url}/api/health"
        health_timeout = min(5.0, self.timeout) if self.timeout > 0 else 5.0
        try:
            resp = self.session.get(url, headers=self._get_headers(), timeout=health_timeout)
            if resp.status_code in (401, 403):
                raise NineRouterAuthError(
                    f"Health check authentication failed with status {resp.status_code}: {self._sanitize_error_text(resp.text)}"
                )
            if resp.status_code != 200:
                raise NineRouterRequestError(
                    f"Health check returned status {resp.status_code}: {self._sanitize_error_text(resp.text)}",
                    status_code=resp.status_code,
                    response_body=self._sanitize_error_text(resp.text),
                )
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise NineRouterConnectionError(
                f"Could not connect to 9Router at {self.base_url}. Is 9Router running?"
            ) from e
        except requests.exceptions.RequestException as e:
            raise NineRouterError(f"Health check failed: {e}") from e

    def get_vision_models(self) -> List[Dict[str, Any]]:
        """Discover available vision models from 9Router.

        Queries GET /v1/models/image-to-text first. If empty or unsupported,
        queries GET /v1/models and filters for models with vision capability.

        Returns:
            List of model dictionaries containing at least 'id'.
        """
        headers = self._get_headers()

        # Step 1: Query GET /v1/models/image-to-text
        try:
            url = f"{self.base_url}/v1/models/image-to-text"
            resp = self.session.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", []) if isinstance(data, dict) else data
                if isinstance(models, list) and len(models) > 0:
                    return models
        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.debug(f"/v1/models/image-to-text query failed: {e}, falling back to /v1/models")

        # Step 2: Query GET /v1/models and filter for vision capabilities
        try:
            url = f"{self.base_url}/v1/models"
            resp = self.session.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code in (401, 403):
                raise NineRouterAuthError(
                    f"Models query authentication failed with status {resp.status_code}: {self._sanitize_error_text(resp.text)}"
                )
            if resp.status_code != 200:
                raise NineRouterRequestError(
                    f"Models query returned status {resp.status_code}: {self._sanitize_error_text(resp.text)}",
                    status_code=resp.status_code,
                    response_body=self._sanitize_error_text(resp.text),
                )
            data = resp.json()
            all_models = data.get("data", []) if isinstance(data, dict) else data
            if not isinstance(all_models, list):
                return []

            vision_models: List[Dict[str, Any]] = []
            for m in all_models:
                if not isinstance(m, dict):
                    continue
                # Check capability flags
                caps = m.get("capabilities", {})
                if isinstance(caps, dict) and caps.get("vision") is True:
                    vision_models.append(m)
                    continue

                # Fallback: check model ID keywords
                model_id = str(m.get("id", "")).lower()
                if any(k in model_id for k in ["vision", "vl", "gemini", "claude", "gpt-4o", "4o-mini", "qwen-vl"]):
                    # If capabilities dict exists and explicitly sets vision=False, skip
                    if isinstance(caps, dict) and caps.get("vision") is False:
                        continue
                    vision_models.append(m)

            return vision_models
        except requests.exceptions.ConnectionError as e:
            raise NineRouterConnectionError(
                f"Could not connect to 9Router at {self.base_url} to fetch models."
            ) from e
        except requests.exceptions.RequestException as e:
            raise NineRouterError(f"Failed to fetch models from 9Router: {e}") from e

    def get_vision_model_ids(self) -> List[str]:
        """Convenience method returning list of vision model IDs."""
        models = self.get_vision_models()
        return [str(m["id"]) for m in models if isinstance(m, dict) and "id" in m]

    def resolve_vision_model(self, preferred_model: Optional[str] = None) -> str:
        """Resolve a real vision model available on 9Router.

        - If preferred_model is explicitly supplied and exists in 9Router's vision models, use it.
        - Otherwise query the actual local 9Router vision model list and return the first available model.
        - Never fabricates model IDs.
        - Raises NineRouterError if no vision models are available or if the requested model is not found.

        Args:
            preferred_model: Optional model ID requested by caller or environment.

        Returns:
            Resolved valid model ID string.

        Raises:
            NineRouterError: If no vision models are available, or requested model does not exist.
        """
        available_ids = self.get_vision_model_ids()
        if not available_ids:
            raise NineRouterError(
                f"No vision-capable models found in 9Router at {self.base_url}."
            )

        if preferred_model and preferred_model.strip():
            target = preferred_model.strip()
            if target in available_ids:
                return target
            available_str = ", ".join(available_ids)
            raise NineRouterError(
                f"Requested vision model '{target}' is not available in 9Router. "
                f"Available vision models: {available_str}"
            )

        return available_ids[0]

    def send_vision_request(
        self,
        model: str,
        image_bytes_or_b64: Union[bytes, str],
        prompt: Optional[str] = None,
        allowed_labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        mode: str = MODE_BOX,
    ) -> VisionResponse:
        """Send a multimodal chat completion request to 9Router.

        Args:
            model: Model identifier (e.g. 'ag/gemini-3.8-flash-high').
            image_bytes_or_b64: Raw image bytes or base64 string (with or without data URL).
            prompt: Custom user prompt text. If None, built from allowed_labels.
            allowed_labels: List of allowed labels to detect.
            system_prompt: Optional custom system prompt.
            temperature: Sampling temperature (default 0.0).
            max_tokens: Maximum tokens to generate (default 4096).
            timeout: Request timeout in seconds.

        Returns:
            VisionResponse with parsed content, raw response, and duration.
        """
        # Format base64 image data URL
        if isinstance(image_bytes_or_b64, bytes):
            b64_str = base64.b64encode(image_bytes_or_b64).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64_str}"
        elif isinstance(image_bytes_or_b64, str):
            if image_bytes_or_b64.startswith("data:image/"):
                image_url = image_bytes_or_b64
            else:
                image_url = f"data:image/jpeg;base64,{image_bytes_or_b64}"
        else:
            raise ValueError(f"image_bytes_or_b64 must be bytes or str, got {type(image_bytes_or_b64)}")

        # Construct prompts
        sys_prompt = system_prompt or SYSTEM_PROMPT
        if prompt is not None:
            user_text = prompt
        else:
            labels = allowed_labels if allowed_labels is not None else list(DEFAULT_BBOX_LABELS)
            user_text = build_user_prompt(allowed_labels=labels, mode=mode)

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        }

        try:
            req_timeout = float(timeout) if (timeout is not None and float(timeout) > 0) else self.timeout
        except (ValueError, TypeError):
            req_timeout = self.timeout
        url = f"{self.base_url}/v1/chat/completions"

        # Safe logging: log metadata only, NEVER log API keys or base64 payload
        logger.info(
            f"Sending vision request: model={model}, image_url_len={len(image_url)}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

        start_time = time.perf_counter()
        try:
            resp = self.session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=req_timeout,
            )
        except requests.exceptions.Timeout as e:
            raise NineRouterRequestError(
                f"Request to 9Router timed out after {req_timeout}s",
                status_code=408,
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise NineRouterConnectionError(
                f"Connection error when communicating with 9Router at {self.base_url}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise NineRouterError(f"Request failed: {e}") from e

        duration = time.perf_counter() - start_time

        if resp.status_code == 401 or resp.status_code == 403:
            raise NineRouterAuthError(
                f"Authentication failed with status {resp.status_code}: {self._sanitize_error_text(resp.text)}"
            )

        if resp.status_code != 200:
            raise NineRouterRequestError(
                f"9Router returned status {resp.status_code}: {self._sanitize_error_text(resp.text)}",
                status_code=resp.status_code,
                response_body=self._sanitize_error_text(resp.text),
            )

        try:
            resp_data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise NineRouterError(f"Failed to decode 9Router response as JSON: {self._sanitize_error_text(resp.text)}") from e

        # Extract content
        choices = resp_data.get("choices", [])
        if not choices:
            raise NineRouterError(f"No choices returned in 9Router response: {resp_data}")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")

        return VisionResponse(
            content=content,
            raw_response=resp_data,
            duration_seconds=duration,
            model=resp_data.get("model", model),
            status_code=resp.status_code,
            usage=resp_data.get("usage", {}),
        )
