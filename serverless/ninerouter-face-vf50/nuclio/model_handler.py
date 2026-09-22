"""ModelHandler for 9Router Face VF50 Nuclio Detector.

Bridges CVAT's serverless detector invocation with the local 9Router vision model.
Enforces VinAI 50-Landmark Facial Topology:
- Exactly 50 canonical landmarks
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
from core.skeleton_contract import (
    VF50_LANDMARKS,
    VF50_COMPONENT_NAMES,
    build_vf50_prompt,
    parse_vf50_response,
    faces_to_cvat_skeletons,
    assess_vf50_quality,
)

logger = logging.getLogger("cvat.nuclio.ninerouter.face_vf50")

DEFAULT_VF50_MAX_TOKENS = 2500
DEFAULT_VF50_MAX_IMAGE_SIZE = 1280


def build_vf50_prompt(landmarks: Sequence[str]) -> str:
    """Build strict, deterministic prompt for 9Router VF50 facial landmark vision model."""
    return (
        "Detect all faces and estimate their 50 VinAI facial landmarks in this image.\n"
        "The 50 landmarks are partitioned into components:\n"
        "- longmaytrai (00..04): left eyebrow (viewer's left / image left)\n"
        "- longmayphai (05..09): right eyebrow (viewer's right / image right)\n"
        "- songmui (10..13): nose bridge\n"
        "- mattrai (14..21): left eye\n"
        "- matphai (22..29): right eye\n"
        "- moingoai (30..41): outer lip contour\n"
        "- moitrong (42..49): inner lip contour\n"
        "Coordinate convention:\n"
        "- Coordinates must be normalized integers [x, y] in range [0, 1000] relative to image width and height.\n"
        "- Visibility flag:\n"
        "    0 = outside image frame or unobserved\n"
        "    1 = present but occluded\n"
        "    2 = clearly visible\n"
        "Return STRICT JSON only, matching this structure:\n"
        "{\n"
        '  "faces": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "label": "face",\n'
        '      "confidence": 0.98,\n'
        '      "landmarks": {\n'
        '        "longmaytrai_00": [x, y, 2],\n'
        "        ...\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        'If no faces are found, return {"faces": []}.'
    )


class ModelHandler:
    """Manages the 9Router vision client and runs Face VF50 inference for CVAT."""

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
        self.requested_model = model or os.getenv("VF50_MODEL") or os.getenv("VISION_MODEL")

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
            val_size = max_image_size if max_image_size is not None else (max_size_env or DEFAULT_VF50_MAX_IMAGE_SIZE)
            self.max_image_size = int(val_size)
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_VF50_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_VF50_MAX_IMAGE_SIZE

        max_tokens_env = os.getenv("MAX_TOKENS")
        try:
            val_tokens = max_tokens if max_tokens is not None else (max_tokens_env or DEFAULT_VF50_MAX_TOKENS)
            self.max_tokens = int(val_tokens)
            if self.max_tokens <= 0:
                self.max_tokens = DEFAULT_VF50_MAX_TOKENS
        except (ValueError, TypeError):
            self.max_tokens = DEFAULT_VF50_MAX_TOKENS

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
            f"Initialized ModelHandler (Face VF50): base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)}, model={self.active_model!r}, "
            f"timeout={self.timeout}s"
        )

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        return (
            f"ModelHandler(face_vf50, base_url={self.base_url!r}, "
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
            crop_box_norm = None
            if roi is not None and len(roi) == 4:
                rx1 = max(0, min(orig_w - 1, int(round(float(roi[0])))))
                ry1 = max(0, min(orig_h - 1, int(round(float(roi[1])))))
                rx2 = max(rx1 + 1, min(orig_w, int(round(float(roi[2])))))
                ry2 = max(ry1 + 1, min(orig_h, int(round(float(roi[3])))))
                working_img = working_img.crop((rx1, ry1, rx2, ry2))
                crop_box_norm = [
                    int(round(ry1 / float(orig_h) * 1000.0)),
                    int(round(rx1 / float(orig_w) * 1000.0)),
                    int(round(ry2 / float(orig_h) * 1000.0)),
                    int(round(rx2 / float(orig_w) * 1000.0)),
                ]

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
        prompt = build_vf50_prompt()

        # Execute 9Router vision inference
        resp = self.client.send_vision_request(
            model=self.active_model,
            image_bytes_or_b64=send_bytes,
            prompt=prompt,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Parse VF-50 faces with multi-schema parsing, crop coordinate transformation
        parsed_faces = parse_vf50_response(
            resp.content,
            crop_box=crop_box_norm,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        # Filter by confidence threshold
        active_faces = [f for f in parsed_faces if f.confidence >= threshold]

        # Convert to CVAT component skeletons with quality gate validation and multi-face group_id
        as_components = True
        if mode in ("single", "parent"):
            as_components = False

        shapes = faces_to_cvat_skeletons(
            active_faces,
            width=orig_w,
            height=orig_h,
            as_components=as_components,
            filter_corrupt=True,
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"Face VF50 inference completed in {elapsed:.2f}s: detected {len(shapes)} face skeleton(s)")
        return shapes
