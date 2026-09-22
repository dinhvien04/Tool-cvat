"""ModelHandler for 9Router Human Pose 17 Nuclio Detector.

Bridges CVAT's serverless detector invocation with the local 9Router vision model.
Enforces COCO 17-Keypoint Human Pose Topology:
- Exactly 17 canonical keypoints
- Emits native CVAT skeleton shapes
- Clamps coordinates to image bounds and maps visibility/occlusion states
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from PIL import Image

# Ensure project modules (app, core, config) are importable
current_dir = Path(__file__).resolve().parent
if (current_dir / "app").exists():
    sys.path.insert(0, str(current_dir))
else:
    project_root = current_dir.parent.parent.parent
    if (project_root / "app").exists():
        sys.path.insert(0, str(project_root))

from app.client import NineRouterClient, NineRouterConnectionError, NineRouterError
from app.config import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NINEROUTER_TIMEOUT,
    DEFAULT_NINEROUTER_URL_CONTAINER,
    DEFAULT_VISION_MODEL,
    mask_api_key,
)
from app.parser import clean_json_string
from core.pose_face_schema import (
    POSE17_KEYPOINTS,
    POSE17_EDGES,
    parse_and_sanitize_pose17_instance,
)

logger = logging.getLogger("cvat.nuclio.ninerouter.human_pose_17")

DEFAULT_POSE17_MAX_TOKENS = 2000
DEFAULT_POSE17_MAX_IMAGE_SIZE = 1280


def build_pose17_prompt(keypoints: Sequence[str]) -> str:
    """Build strict, deterministic prompt for 9Router Pose 17 vision model."""
    kps_str = ", ".join(f'"{k}"' for k in keypoints)
    return (
        "Detect all persons and estimate their 17 COCO keypoints in this image.\n"
        f"Keypoints to detect: [{kps_str}].\n"
        "Coordinate convention:\n"
        "- Coordinates must be normalized integers [x, y] in range [0, 1000] relative to image width and height.\n"
        "- Visibility flag:\n"
        "    0 = outside image frame or unobserved\n"
        "    1 = present but occluded\n"
        "    2 = clearly visible\n"
        "Return STRICT JSON only, matching this structure:\n"
        "{\n"
        '  "people": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "label": "person",\n'
        '      "confidence": 0.95,\n'
        '      "keypoints": {\n'
        '        "nose": [x, y, 2],\n'
        "        ...\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        'If no persons are found, return {"people": []}.'
    )


class ModelHandler:
    """Manages the 9Router vision client and runs Human Pose 17 inference for CVAT."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("POSE17_MODEL") or os.getenv("VISION_MODEL")

        timeout_env = os.getenv("NINEROUTER_TIMEOUT")
        try:
            val_timeout = timeout if timeout is not None else (timeout_env or DEFAULT_NINEROUTER_TIMEOUT)
            self.timeout = float(val_timeout)
            if self.timeout <= 0:
                self.timeout = DEFAULT_NINEROUTER_TIMEOUT
        except (ValueError, TypeError):
            self.timeout = DEFAULT_NINEROUTER_TIMEOUT

        max_size_env = os.getenv("MAX_IMAGE_SIZE")
        try:
            val_size = max_image_size if max_image_size is not None else (max_size_env or DEFAULT_POSE17_MAX_IMAGE_SIZE)
            self.max_image_size = int(val_size)
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_POSE17_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_POSE17_MAX_IMAGE_SIZE

        max_tokens_env = os.getenv("MAX_TOKENS")
        try:
            val_tokens = max_tokens if max_tokens is not None else (max_tokens_env or DEFAULT_POSE17_MAX_TOKENS)
            self.max_tokens = int(val_tokens)
            if self.max_tokens <= 0:
                self.max_tokens = DEFAULT_POSE17_MAX_TOKENS
        except (ValueError, TypeError):
            self.max_tokens = DEFAULT_POSE17_MAX_TOKENS

        # Initialize client
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Dynamic vision model resolution
        try:
            self.active_model: str = self.client.resolve_vision_model(self.requested_model)
        except Exception as e:
            logger.error(f"Failed to resolve vision model {self.requested_model!r} from 9Router: {e}")
            raise

        logger.info(
            f"Initialized ModelHandler (Human Pose 17): base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)}, model={self.active_model!r}, "
            f"timeout={self.timeout}s"
        )

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        return (
            f"ModelHandler(human_pose_17, base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)!r}, "
            f"model={self.active_model!r}, "
            f"timeout={self.timeout})"
        )

    def infer(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
        mode: Optional[str] = None,
        roi: Optional[Sequence[Union[int, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Process incoming image bytes and return CVAT native skeleton shapes.

        Args:
            image_bytes: Raw JPEG/PNG image binary data.
            threshold: Minimum confidence score [0.0, 1.0].
            mode: Optional detection mode override.
            roi: Optional sub-region of interest [x1, y1, x2, y2].

        Returns:
            List of CVAT skeleton shape dictionaries.
        """
        if not image_bytes:
            raise ValueError("Empty image payload received")

        t0 = time.perf_counter()

        # Load image and obtain dimensions
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size
            working_img = img.convert("RGB")

            # Apply ROI if specified
            if roi is not None and len(roi) == 4:
                rx1 = max(0, min(orig_w - 1, int(round(float(roi[0])))))
                ry1 = max(0, min(orig_h - 1, int(round(float(roi[1])))))
                rx2 = max(rx1 + 1, min(orig_w, int(round(float(roi[2])))))
                ry2 = max(ry1 + 1, min(orig_h, int(round(float(roi[3])))))
                working_img = working_img.crop((rx1, ry1, rx2, ry2))

            # Resize if exceeding max size
            curr_w, curr_h = working_img.size
            if max(curr_w, curr_h) > self.max_image_size:
                ratio = self.max_image_size / float(max(curr_w, curr_h))
                new_w = max(1, int(round(curr_w * ratio)))
                new_h = max(1, int(round(curr_h * ratio)))
                working_img = working_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            working_img.save(buf, format="JPEG", quality=90)
            send_bytes = buf.getvalue()

        # Build prompt
        prompt = build_pose17_prompt(POSE17_KEYPOINTS)

        # Execute 9Router vision inference
        resp = self.client.send_vision_request(
            model=self.active_model,
            image_bytes_or_b64=send_bytes,
            prompt=prompt,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Parse JSON
        cleaned = clean_json_string(resp.content)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse JSON from 9Router response: {e}. Content: {resp.content[:200]!r}")
            return []

        # Parse candidate person instances
        raw_people: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            if "people" in parsed and isinstance(parsed["people"], list):
                raw_people = [p for p in parsed["people"] if isinstance(p, dict)]
            elif "persons" in parsed and isinstance(parsed["persons"], list):
                raw_people = [p for p in parsed["persons"] if isinstance(p, dict)]
            elif "keypoints" in parsed:
                raw_people = [parsed]
        elif isinstance(parsed, list):
            raw_people = [p for p in parsed if isinstance(p, dict)]

        shapes: List[Dict[str, Any]] = []
        for person_dict in raw_people:
            conf = float(person_dict.get("confidence", 1.0))
            if conf < threshold:
                continue

            instance = parse_and_sanitize_pose17_instance(
                person_dict,
                img_width=orig_w,
                img_height=orig_h,
                coord_range=1000.0,
            )
            if instance and instance.elements:
                shapes.append(instance.to_cvat_dict())

        elapsed = time.perf_counter() - t0
        logger.info(f"Human Pose 17 inference completed in {elapsed:.2f}s: detected {len(shapes)} person skeleton(s)")
        return shapes
