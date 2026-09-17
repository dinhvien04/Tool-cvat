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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from core.vision_contract import (
    DEFAULT_BBOX_LABELS,
    MODE_BOX,
    MODE_BOX_AND_MASK,
    MODE_MASK,
    SYSTEM_PROMPT,
    build_openai_vision_payload,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

# Prioritized segmentation-capable models available via 9Router
PREFERRED_SEGMENTATION_MODELS: Tuple[str, ...] = (
    "ag/gemini-3.8-flash-high",
    "ag/gemini-3.8-flash",
    "ag/gemini-3.8-flash-medium",
    "ag/gemini-3.8-flash-low",
    "ag/gemini-3.7-flash-high",
    "ag/gemini-3.7-flash",
    "ag/gemini-3.7-flash-medium",
    "ag/gemini-3.7-flash-low",
    "ag/gemini-3.6-flash-high",
    "ag/gemini-3.5-flash-high",
    "ag/claude-sonnet-4-6",
)

# In-memory capability cache to prevent redundant API calls during runtime inference
_SEGMENTATION_CAPABILITY_CACHE: Dict[str, bool] = {}


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

    def probe_segmentation_capability(
        self,
        model: str,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Probe model for actual instance segmentation capability using ONE minimal request.

        Validates that the model returns valid JSON with:
        1. 'label' (string matching allowed labels)
        2. 'box_2d' ([ymin, xmin, ymax, xmax] in [0, 1000])
        3. 'mask' (polygon contour with at least 3 vertices [[x, y], ...] in [0, 1000])

        Args:
            model: Model identifier to probe.
            timeout: Optional request timeout in seconds (defaults to 15.0s).

        Returns:
            Tuple of (is_capable: bool, details: str).
        """
        if model in _SEGMENTATION_CAPABILITY_CACHE:
            cached_val = _SEGMENTATION_CAPABILITY_CACHE[model]
            return cached_val, f"Cached segmentation capability: {cached_val}"

        # 1. Verify model is available in 9Router
        available_ids = self.get_vision_model_ids()
        if model not in available_ids:
            msg = f"Model '{model}' is not in available vision models ({available_ids})"
            _SEGMENTATION_CAPABILITY_CACHE[model] = False
            return False, msg

        # 2. Prepare minimal probe image (< 10KB)
        import io
        from PIL import Image, ImageDraw

        repo_root = Path(__file__).resolve().parent.parent
        test_img_path = repo_root / "test.jpg"
        if not test_img_path.exists():
            test_img_path = Path("test.jpg")

        probe_labels = ["car", "truck", "bus", "pedestrian"]
        if test_img_path.exists():
            try:
                with Image.open(test_img_path) as im:
                    thumb = im.copy()
                    thumb.thumbnail((256, 256))
                    buf = io.BytesIO()
                    thumb.save(buf, format="JPEG", quality=80)
                    img_bytes = buf.getvalue()
            except Exception:
                test_img_path = None

        if not test_img_path or not test_img_path.exists():
            # Draw synthetic car silhouette
            img = Image.new("RGB", (256, 256), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            # Body
            draw.rectangle([30, 100, 226, 170], fill=(20, 20, 20))
            # Cabin
            draw.polygon([(60, 100), (90, 50), (160, 50), (190, 100)], fill=(20, 20, 20))
            # Wheels
            draw.ellipse([50, 150, 90, 190], fill=(50, 50, 50))
            draw.ellipse([160, 150, 200, 190], fill=(50, 50, 50))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            img_bytes = buf.getvalue()
            probe_labels = ["car", "truck", "bus", "pedestrian"]

        # 3. Construct minimal probe prompt for car / vehicle / object
        prompt = build_user_prompt(allowed_labels=probe_labels, mode=MODE_BOX_AND_MASK)

        # 4. Send probe request
        probe_timeout = timeout or 15.0
        try:
            resp = self.send_vision_request(
                model=model,
                image_bytes_or_b64=img_bytes,
                prompt=prompt,
                timeout=probe_timeout,
                mode=MODE_BOX_AND_MASK,
            )
        except Exception as e:
            msg = f"Probe request failed: {e}"
            _SEGMENTATION_CAPABILITY_CACHE[model] = False
            return False, msg

        # 5. Parse and validate segmentation outputs
        from app.parser import clean_json_string
        try:
            cleaned = clean_json_string(resp.content)
            data = json.loads(cleaned)
        except Exception as e:
            msg = f"Probe failed to parse JSON response: {e}. Raw: {resp.content[:150]!r}"
            _SEGMENTATION_CAPABILITY_CACHE[model] = False
            return False, msg

        raw_objects = data.get("objects", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not raw_objects:
            msg = f"Model '{model}' returned empty objects list on probe image"
            _SEGMENTATION_CAPABILITY_CACHE[model] = False
            return False, msg

        # Inspect first detection for valid label, box_2d, and polygon mask
        has_valid_seg = False
        reasons = []
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            lbl = obj.get("label")
            box = obj.get("box_2d")
            msk = obj.get("mask") or obj.get("polygon")

            if not lbl or not isinstance(lbl, str):
                reasons.append(f"missing or invalid label: {lbl!r}")
                continue

            if not isinstance(box, (list, tuple)) or len(box) != 4:
                reasons.append(f"invalid box_2d: {box!r}")
                continue

            try:
                ymin, xmin, ymax, xmax = [int(round(float(c))) for c in box]
                if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
                    reasons.append(f"out of bounds box_2d: {box!r}")
                    continue
            except (ValueError, TypeError):
                reasons.append(f"non-numeric box_2d: {box!r}")
                continue

            if not isinstance(msk, (list, tuple)) or len(msk) < 3:
                reasons.append(f"missing or degenerate mask contour: len={len(msk) if isinstance(msk, list) else 0}")
                continue

            valid_pts = True
            for pt in msk:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    valid_pts = False
                    break
                try:
                    px, py = float(pt[0]), float(pt[1])
                    if not (0 <= px <= 1000 and 0 <= py <= 1000):
                        valid_pts = False
                        break
                except (ValueError, TypeError):
                    valid_pts = False
                    break

            if not valid_pts:
                reasons.append("invalid vertex coordinates in mask contour")
                continue

            has_valid_seg = True
            break

        if has_valid_seg:
            _SEGMENTATION_CAPABILITY_CACHE[model] = True
            return True, f"Model '{model}' successfully verified: returned label, box_2d, and polygon mask"

        msg = f"Model '{model}' did not return valid segmentation: {'; '.join(reasons)}"
        _SEGMENTATION_CAPABILITY_CACHE[model] = False
        return False, msg

    def resolve_segmentation_model(
        self,
        preferred_model: Optional[str] = None,
        probe: bool = False,
    ) -> str:
        """Resolve a model capable of instance segmentation (polygon masks) from 9Router.

        - Respects explicit preferred_model / VISION_MODEL if supplied (verifies existence).
        - If probe=True (e.g. during preflight / deployment / smoke test), actively verifies
          that the model returns label, box_2d, and mask polygon.
        - If probe=False (e.g. during runtime image inference), uses cached or prioritized
          segmentation models without consuming extra API quota.
        - Never fabricates model IDs.
        - Fails clearly if no segmentation-capable model is available.

        Args:
            preferred_model: Optional explicit model ID.
            probe: If True, executes a real minimal capability probe against the candidate.

        Returns:
            Resolved valid model ID string.
        """
        available_ids = self.get_vision_model_ids()
        if not available_ids:
            raise NineRouterError(
                f"No vision models found in 9Router at {self.base_url}."
            )

        if preferred_model and preferred_model.strip():
            target = preferred_model.strip()
            if target not in available_ids:
                available_str = ", ".join(available_ids)
                raise NineRouterError(
                    f"Requested segmentation model '{target}' is not available in 9Router. "
                    f"Available vision models: {available_str}"
                )
            if probe:
                ok, reason = self.probe_segmentation_capability(target)
                if not ok:
                    raise NineRouterError(
                        f"Requested model '{target}' failed segmentation capability validation: {reason}"
                    )
            return target

        # Candidate selection: prioritize PREFERRED_SEGMENTATION_MODELS
        candidates: List[str] = []
        for pref in PREFERRED_SEGMENTATION_MODELS:
            if pref in available_ids and pref not in candidates:
                candidates.append(pref)

        # Append remaining vision models (excluding explicit thinking models)
        for mid in available_ids:
            if mid not in candidates and "thinking" not in mid.lower():
                candidates.append(mid)

        if not candidates:
            raise NineRouterError(
                f"No suitable segmentation vision models found in 9Router at {self.base_url}."
            )

        if probe:
            for cand in candidates:
                ok, reason = self.probe_segmentation_capability(cand)
                if ok:
                    return cand
            raise NineRouterError(
                f"None of the available vision models ({candidates}) passed segmentation capability validation."
            )

        # In non-probing runtime mode: return top prioritized candidate
        return candidates[0]

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
        visual_examples: Optional[List[Dict[str, Any]]] = None,
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
            mode: Detection output mode.
            visual_examples: Optional list of multimodal few-shot correction examples with crops.

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

        # Construct multimodal user content (interleaved visual few-shot if provided)
        user_content: List[Dict[str, Any]] = []
        if visual_examples:
            for idx, ex in enumerate(visual_examples):
                crop_url = ex.get("crop_data_url")
                if not crop_url:
                    continue
                ex_desc = ex.get("description", f"Example #{idx + 1}")
                ex_json = ex.get("expected_output_json") or json.dumps(ex.get("expected_output", {}), indent=2)
                user_content.append({
                    "type": "text",
                    "text": f"--- Corrected Example #{idx + 1} ({ex_desc}) ---\nReviewer crop:",
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": crop_url,
                        "detail": "high",
                    },
                })
                user_content.append({
                    "type": "text",
                    "text": f"Expected structured annotation for Example #{idx + 1}:\n{ex_json}\n",
                })

            user_content.append({
                "type": "text",
                "text": f"--- Target Image to Annotate ---\n{user_text}",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high",
                },
            })
        else:
            user_content = [
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
            ]

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
                    "content": user_content,
                },
            ],
        }

        try:
            req_timeout = float(timeout) if (timeout is not None and float(timeout) > 0) else self.timeout
        except (ValueError, TypeError):
            req_timeout = self.timeout
        url = f"{self.base_url}/v1/chat/completions"

        # Safe logging: log metadata only, NEVER log API keys or base64 payload
        vis_count = len(visual_examples) if visual_examples else 0
        logger.info(
            f"Sending vision request: model={model}, target_image_url_len={len(image_url)}, "
            f"visual_examples={vis_count}, max_tokens={max_tokens}, temperature={temperature}"
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

        content_type = resp.headers.get("content-type", "")
        if "event-stream" in content_type:
            parts = []
            for line in resp.text.splitlines():
                line = line.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            parts.append(delta["content"])
                    except Exception:
                        pass
            content = "".join(parts)
            resp_data = {"choices": [{"message": {"content": content}}], "model": model}
        else:
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
