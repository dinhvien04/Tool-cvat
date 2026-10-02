"""ModelHandler for the 9Router Box 3D detector.

Returns CVAT cuboids for the eight labels shown in the 3D task.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from PIL import Image

current_dir = Path(__file__).resolve().parent
if (current_dir / "app").exists():
    sys.path.insert(0, str(current_dir))
else:
    project_root = current_dir.parent.parent.parent
    if (project_root / "app").exists():
        sys.path.insert(0, str(project_root))

from app.client import NineRouterClient
from app.config import (
    DEFAULT_NINEROUTER_TIMEOUT,
    DEFAULT_NINEROUTER_URL_CONTAINER,
    DEFAULT_VISION_MODEL,
    mask_api_key,
)
from app.parser import clean_json_string
from cuboid_geometry import cuboid_from_front_face, normalized_box_to_pixels

logger = logging.getLogger("cvat.nuclio.ninerouter.box3d")

BOX3D_LABELS: Sequence[str] = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
)
DEFAULT_BOX3D_MAX_IMAGE_SIZE = 1280
DEFAULT_BOX3D_MAX_TOKENS = 1600
SYSTEM_PROMPT = (
    "You estimate image-plane cuboids for road users. "
    "Return one JSON object and nothing else."
)


def load_spec_labels_from_function_yaml(yaml_path: Optional[Path | str] = None) -> List[str]:
    target_path = Path(yaml_path) if yaml_path else current_dir / "function.yaml"
    if not target_path.exists():
        return list(BOX3D_LABELS)
    try:
        with open(target_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw_spec = data.get("metadata", {}).get("annotations", {}).get("spec", "")
        if isinstance(raw_spec, str) and raw_spec.strip():
            spec_list = json.loads(raw_spec)
            labels = [item["name"] for item in spec_list if isinstance(item, dict) and "name" in item]
            if labels:
                return labels
    except Exception as exc:
        logger.warning("Could not parse labels from %s: %s", target_path, exc)
    return list(BOX3D_LABELS)


def build_box3d_prompt(labels: Sequence[str]) -> str:
    label_lines = "\n".join(f"- {label}" for label in labels)
    return f"""Detect every visible road user that matches the allowed labels and describe it as a 3D cuboid seen in the image.

Allowed labels (choose ONLY from this list, exact spelling):
{label_lines}

Output schema:
{{
  "objects": [
    {{
      "label": "<exact_allowed_label>",
      "box_2d": [ymin, xmin, ymax, xmax],
      "depth_ratio": 0.18,
      "side": "right",
      "confidence": 0.9
    }}
  ]
}}

Rules:
- box_2d is the front face of the object, normalized integers in [0, 1000], format [ymin, xmin, ymax, xmax].
- depth_ratio is how deep the box is relative to the front face, between 0.08 and 0.45.
- side is "right" when the receding face goes to the viewer's right, otherwise "left".
- Do not return masks, polygons, or 2D-only rectangles.
- If nothing matches, return {{"objects": []}}.
- Output raw JSON only.
"""


def cuboid_shapes_only(
    shapes: Sequence[Dict[str, Any]],
    allowed_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = set(allowed_labels)
    kept: List[Dict[str, Any]] = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("type") != "cuboid":
            continue
        if shape.get("label") not in allowed:
            continue
        points = shape.get("points")
        if not isinstance(points, (list, tuple)) or len(points) != 16:
            continue
        kept.append(dict(shape))
    return kept


def _resize_limit(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


def objects_to_cuboids(
    objects: Sequence[Dict[str, Any]],
    allowed_labels: Sequence[str],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    allowed = set(allowed_labels)
    shapes: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        box = obj.get("box_2d")
        if label not in allowed or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            xtl, ytl, xbr, ybr = normalized_box_to_pixels(box, width, height)
            if xbr - xtl < 2 or ybr - ytl < 2:
                continue
            depth = float(obj.get("depth_ratio", 0.18))
            side = str(obj.get("side", "right"))
            points = [round(value, 2) for value in cuboid_from_front_face(xtl, ytl, xbr, ybr, depth, side)]
        except (TypeError, ValueError):
            continue
        if len(points) != 16:
            continue
        shape: Dict[str, Any] = {"label": label, "type": "cuboid", "points": points}
        if obj.get("confidence") is not None:
            try:
                shape["confidence"] = str(round(float(obj["confidence"]), 2))
            except (TypeError, ValueError):
                pass
        shapes.append(shape)
    return shapes


class ModelHandler:
    """Runs cuboid inference for the eight 3D labels."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        candidate_labels: Optional[List[str]] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("BOX3D_MODEL") or os.getenv("VISION_MODEL") or DEFAULT_VISION_MODEL
        timeout_env = os.getenv("NINEROUTER_TIMEOUT")
        try:
            self.timeout = float(timeout if timeout is not None else (timeout_env or DEFAULT_NINEROUTER_TIMEOUT))
            if self.timeout <= 0:
                self.timeout = DEFAULT_NINEROUTER_TIMEOUT
        except (ValueError, TypeError):
            self.timeout = DEFAULT_NINEROUTER_TIMEOUT
        max_size_env = os.getenv("MAX_IMAGE_SIZE")
        try:
            self.max_image_size = int(max_image_size if max_image_size is not None else (max_size_env or DEFAULT_BOX3D_MAX_IMAGE_SIZE))
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_BOX3D_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_BOX3D_MAX_IMAGE_SIZE
        max_tokens_env = os.getenv("MAX_TOKENS")
        try:
            self.max_tokens = int(max_tokens if max_tokens is not None else (max_tokens_env or DEFAULT_BOX3D_MAX_TOKENS))
            if self.max_tokens <= 0:
                self.max_tokens = DEFAULT_BOX3D_MAX_TOKENS
        except (ValueError, TypeError):
            self.max_tokens = DEFAULT_BOX3D_MAX_TOKENS
        self.candidate_labels = candidate_labels or load_spec_labels_from_function_yaml()
        self.client = NineRouterClient(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        self.active_model = self.client.resolve_vision_model(self.requested_model)
        logger.info(
            "Initialized ModelHandler (Box 3D): base_url=%r, key=%s, model=%r, labels=%d",
            self.base_url,
            mask_api_key(self.api_key),
            self.active_model,
            len(self.candidate_labels),
        )

    def infer(self, image_bytes: bytes, threshold: float = 0.5, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        if not image_bytes:
            raise ValueError("Empty image payload received")
        started = time.perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            resized = _resize_limit(rgb, self.max_image_size)
            width, height = resized.size
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=90)
            payload = buffer.getvalue()

        response = self.client.send_vision_request(
            model=self.active_model,
            image_bytes_or_b64=payload,
            prompt=build_box3d_prompt(self.candidate_labels),
            system_prompt=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        parsed = json.loads(clean_json_string(response.content))
        objects = parsed.get("objects", []) if isinstance(parsed, dict) else []
        if threshold > 0:
            filtered = []
            for obj in objects:
                if not isinstance(obj, dict) or obj.get("confidence") is None:
                    filtered.append(obj)
                    continue
                try:
                    if float(obj["confidence"]) >= threshold:
                        filtered.append(obj)
                except (TypeError, ValueError):
                    filtered.append(obj)
            objects = filtered
        shapes = cuboid_shapes_only(objects_to_cuboids(objects, self.candidate_labels, width, height), self.candidate_labels)
        logger.info(
            "Box 3D inference complete: model=%s total_s=%.2f shapes=%d",
            self.active_model,
            time.perf_counter() - started,
            len(shapes),
        )
        return shapes
