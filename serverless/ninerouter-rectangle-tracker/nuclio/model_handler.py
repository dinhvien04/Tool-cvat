"""Experimental 9Router rectangle tracker for CVAT.

CVAT initializes the tracker with a rectangle on frame N. We store a compact
appearance crop plus the previous rectangle in the signed tracker state.
For frame N+1, only a local search crop around the previous box is sent to
9Router together with the previous target crop. This is intentionally a small
prototype: rectangle only, no mask, no re-identification across long occlusion.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image


TRACKER_SYSTEM_PROMPT = """You are a single-object visual tracker.
The first image is a reference crop showing the exact target object from the previous frame.
The second image is the current-frame search region around the target's previous position.
Find the SAME physical object in the search region. Do not switch to a similar nearby object.
Return ONLY JSON:
{"visible": true, "box_2d": [ymin, xmin, ymax, xmax], "confidence": 0.95}
Coordinates are normalized integers from 0 to 1000 relative to the CURRENT SEARCH REGION.
box_2d order is [ymin, xmin, ymax, xmax].
If the same target cannot be located reliably, return:
{"visible": false, "confidence": 0.0}
No markdown and no explanation."""


class ModelHandler:
    def __init__(self) -> None:
        self.base_url = os.getenv("NINEROUTER_URL", "http://host.docker.internal:20128").rstrip("/")
        self.model = (
            os.getenv("RECTANGLE_TRACKER_MODEL")
            or os.getenv("VISION_MODEL")
            or "ag/gemini-3.8-flash-low"
        ).strip()
        self.api_key = os.getenv("NINEROUTER_KEY", "").strip()
        self.timeout = self._env_float("NINEROUTER_TIMEOUT", 12.0, minimum=1.0, maximum=30.0)
        self.search_scale = self._env_float("TRACKER_SEARCH_SCALE", 3.0, minimum=1.5, maximum=8.0)
        self.template_size = self._env_int("TRACKER_TEMPLATE_SIZE", 224, minimum=96, maximum=512)
        self.max_search_size = self._env_int("TRACKER_MAX_SEARCH_SIZE", 640, minimum=256, maximum=1280)
        self.max_tokens = self._env_int("TRACKER_MAX_TOKENS", 192, minimum=64, maximum=512)
        self.update_template_confidence = self._env_float(
            "TRACKER_UPDATE_TEMPLATE_CONFIDENCE", 0.65, minimum=0.0, maximum=1.0
        )
        self.session = requests.Session()

    @staticmethod
    def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _normalize_shape(shape: Any) -> Dict[str, Any]:
        if isinstance(shape, dict):
            points = shape.get("points")
        else:
            points = shape
        if not isinstance(points, (list, tuple)) or len(points) < 4:
            raise ValueError("rectangle shape must contain [x1, y1, x2, y2]")
        x1, y1, x2, y2 = map(float, points[:4])
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if right - left < 2 or bottom - top < 2:
            raise ValueError("rectangle is degenerate")
        return {"type": "rectangle", "points": [left, top, right, bottom]}

    @staticmethod
    def _clamp_box(box: Sequence[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = map(float, box)
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(x1 + 1.0, min(float(width), x2))
        y2 = max(y1 + 1.0, min(float(height), y2))
        return [x1, y1, x2, y2]

    @staticmethod
    def _resize_copy(image: Image.Image, max_size: int) -> Image.Image:
        out = image.copy()
        if max(out.size) > max_size:
            scale = float(max_size) / float(max(out.size))
            size = (max(1, round(out.width * scale)), max(1, round(out.height * scale)))
            out = out.resize(size, Image.Resampling.LANCZOS)
        return out

    @staticmethod
    def _to_data_url(image: Image.Image, *, quality: int = 84) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _target_crop(self, image: Image.Image, box: Sequence[float]) -> str:
        x1, y1, x2, y2 = self._clamp_box(box, image.width, image.height)
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = 0.08 * w, 0.08 * h
        crop_box = self._clamp_box(
            [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y],
            image.width,
            image.height,
        )
        crop = image.crop(tuple(map(int, crop_box)))
        crop = self._resize_copy(crop, self.template_size)
        return self._to_data_url(crop, quality=86)

    def _search_box(self, previous_box: Sequence[float], width: int, height: int, failures: int) -> List[float]:
        x1, y1, x2, y2 = self._clamp_box(previous_box, width, height)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Expand after a failed frame so the target has a chance to re-enter the search window.
        scale = min(6.0, self.search_scale * (1.0 + 0.35 * max(0, failures)))
        sw = max(96.0, bw * scale)
        sh = max(96.0, bh * scale)
        return self._clamp_box(
            [cx - sw / 2.0, cy - sh / 2.0, cx + sw / 2.0, cy + sh / 2.0],
            width,
            height,
        )

    @staticmethod
    def _clean_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise
            data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("tracker response must be a JSON object")
        return data

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call_router(self, reference_url: str, search_url: str) -> Tuple[Optional[List[float]], float, bool]:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": TRACKER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reference target from previous frame:"},
                        {"type": "image_url", "image_url": {"url": reference_url, "detail": "high"}},
                        {"type": "text", "text": "Current-frame search region. Locate the same target:"},
                        {"type": "image_url", "image_url": {"url": search_url, "detail": "high"}},
                    ],
                },
            ],
        }
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("9Router returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        data = self._clean_json(content)
        # Be tolerant to providers that wrap the single target in an "objects"
        # array despite the tracker-specific schema.
        if "box_2d" not in data and isinstance(data.get("objects"), list) and data["objects"]:
            first = data["objects"][0]
            if isinstance(first, dict):
                data = first

        raw_visible = data.get("visible", True)
        if isinstance(raw_visible, str):
            visible = raw_visible.strip().lower() not in {"false", "0", "no", "off"}
        else:
            visible = bool(raw_visible)
        confidence = float(data.get("confidence", 0.0) or 0.0)
        box = data.get("box_2d")
        if not visible or not isinstance(box, (list, tuple)) or len(box) != 4:
            return None, confidence, False
        ymin, xmin, ymax, xmax = map(float, box)
        if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
            return None, confidence, False
        return [ymin, xmin, ymax, xmax], confidence, True

    @staticmethod
    def _map_normalized_box(box_2d: Sequence[float], roi: Sequence[float]) -> List[float]:
        ymin, xmin, ymax, xmax = map(float, box_2d)
        rx1, ry1, rx2, ry2 = map(float, roi)
        rw, rh = rx2 - rx1, ry2 - ry1
        return [
            rx1 + (xmin / 1000.0) * rw,
            ry1 + (ymin / 1000.0) * rh,
            rx1 + (xmax / 1000.0) * rw,
            ry1 + (ymax / 1000.0) * rh,
        ]

    @staticmethod
    def _plausible(previous_box: Sequence[float], new_box: Sequence[float]) -> bool:
        px1, py1, px2, py2 = map(float, previous_box)
        nx1, ny1, nx2, ny2 = map(float, new_box)
        prev_area = max(1.0, (px2 - px1) * (py2 - py1))
        new_area = max(1.0, (nx2 - nx1) * (ny2 - ny1))
        ratio = new_area / prev_area
        if ratio < 0.18 or ratio > 5.5:
            return False
        # Reject catastrophic jumps relative to the object's own size.
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        ncx, ncy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
        diag = max(8.0, math.hypot(px2 - px1, py2 - py1))
        return math.hypot(ncx - pcx, ncy - pcy) <= 4.5 * diag

    def infer(self, image: Image.Image, shape: Any, state: Optional[Dict[str, Any]]):
        if state is None:
            normalized = self._normalize_shape(shape)
            box = self._clamp_box(normalized["points"], image.width, image.height)
            state = {
                "box": box,
                "template": self._target_crop(image, box),
                "failures": 0,
                "model": self.model,
            }
            return {"type": "rectangle", "points": box}, state

        previous_box = self._clamp_box(state.get("box") or [], image.width, image.height)
        template = state.get("template")
        failures = int(state.get("failures", 0) or 0)
        if not template:
            # State from an incompatible/older tracker version: reinitialize from shape if possible.
            if shape is None:
                return {"type": "rectangle", "points": previous_box}, state
            normalized = self._normalize_shape(shape)
            previous_box = self._clamp_box(normalized["points"], image.width, image.height)
            template = self._target_crop(image, previous_box)

        roi = self._search_box(previous_box, image.width, image.height, failures)
        search_crop = image.crop(tuple(map(int, roi)))
        search_crop = self._resize_copy(search_crop, self.max_search_size)
        search_url = self._to_data_url(search_crop, quality=84)

        try:
            normalized_box, confidence, visible = self._call_router(template, search_url)
            if not visible or normalized_box is None:
                raise ValueError("target not visible")
            candidate = self._clamp_box(
                self._map_normalized_box(normalized_box, roi),
                image.width,
                image.height,
            )
            if not self._plausible(previous_box, candidate):
                raise ValueError("implausible tracker jump")

            new_state = dict(state)
            new_state["box"] = candidate
            new_state["failures"] = 0
            new_state["confidence"] = round(float(confidence), 3)
            if confidence >= self.update_template_confidence:
                new_state["template"] = self._target_crop(image, candidate)
            return {"type": "rectangle", "points": candidate}, new_state

        except Exception as exc:
            # Keep the track alive for the prototype. The next frame gets a larger
            # search region. We never fabricate a new unrelated box.
            new_state = dict(state)
            new_state["box"] = previous_box
            new_state["failures"] = min(8, failures + 1)
            new_state["last_error"] = type(exc).__name__
            return {"type": "rectangle", "points": previous_box}, new_state
