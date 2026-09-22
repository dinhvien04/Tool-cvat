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
    VF50_COMPONENT_NAMES,
    VF50Face,
    build_vf50_prompt as contract_build_vf50_prompt,
    build_vf50_crop_prompt,
    expand_face_bbox,
    parse_vf50_response,
    faces_to_cvat_skeletons,
    assess_vf50_quality,
)

logger = logging.getLogger("cvat.nuclio.ninerouter.face_vf50")

DEFAULT_VF50_MAX_TOKENS = 2500
DEFAULT_VF50_MAX_IMAGE_SIZE = 1280


def build_vf50_prompt(landmarks: Optional[Sequence[str]] = None) -> str:
    """Build strict, deterministic prompt for 9Router VF50 facial landmark vision model."""
    return contract_build_vf50_prompt()


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
        self.refine_model = os.getenv("VF50_REFINE_MODEL") or os.getenv("QUALITY_REFINE_MODEL")

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
            if hasattr(self.client, "resolve_vf50_model"):
                self.active_model: str = self.client.resolve_vf50_model(self.requested_model)
            else:
                self.active_model = self.client.resolve_vision_model(self.requested_model)
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
        refine_crops: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Process incoming image bytes and return CVAT native skeleton shapes.

        Args:
            image_bytes: Raw JPEG/PNG image binary data.
            threshold: Minimum confidence score [0.0, 1.0].
            mode: Optional detection mode override.
            roi: Optional sub-region of interest [x1, y1, x2, y2].
            refine_crops: Whether to run two-pass crop refinement (defaults to VF50_TWO_PASS_REFINE env var or True).

        Returns:
            List of CVAT skeleton shape dictionaries.
        """
        if not image_bytes:
            raise ValueError("Empty image payload received")

        t0 = time.perf_counter()

        # Load image and obtain dimensions
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size
            base_pil_img = img.convert("RGB")
            working_img = base_pil_img.copy()

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

        # Execute 9Router vision inference (Pass 1)
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

        # Two-pass crop refinement (Pass 2)
        # If face bounding box is small (< 100px) or quality score is low, crop face + 20% margin
        enable_two_pass = (
            refine_crops if refine_crops is not None
            else os.getenv("VF50_TWO_PASS_REFINE", "1").lower() in ("1", "true", "yes")
        )

        if enable_two_pass and active_faces:
            refined_faces: List[VF50Face] = []
            for face in active_faces:
                bbox = face.box_2d or face.derive_bbox(pad_ratio=0.05)
                needs_refine = False
                if bbox:
                    w_px = ((bbox[3] - bbox[1]) / 1000.0) * float(orig_w)
                    h_px = ((bbox[2] - bbox[0]) / 1000.0) * float(orig_h)
                    face_size = max(w_px, h_px)
                    if face_size < 100.0:
                        needs_refine = True
                    else:
                        q_rep = assess_vf50_quality(face, width=orig_w, height=orig_h)
                        if not q_rep.is_valid or q_rep.score < 0.70 or len(q_rep.anomalies) > 0:
                            needs_refine = True

                if needs_refine and bbox:
                    refined = self._refine_face_crop(base_pil_img, face, bbox, orig_w, orig_h)
                    if refined:
                        refined_faces.append(refined)
                    else:
                        refined_faces.append(face)
                else:
                    refined_faces.append(face)
            active_faces = refined_faces

        # Convert to CVAT component skeletons with quality gate validation and multi-face group_id
        shapes = faces_to_cvat_skeletons(
            active_faces,
            width=orig_w,
            height=orig_h,
            as_components=True,
            filter_corrupt=True,
            fallback_on_corrupt=True,
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"Face VF50 inference completed in {elapsed:.2f}s: detected {len(shapes)} face skeleton(s)")
        return shapes

    def _refine_face_crop(
        self,
        full_img: Image.Image,
        face: VF50Face,
        bbox: Sequence[Union[int, float]],
        orig_w: int,
        orig_h: int,
    ) -> Optional[VF50Face]:
        """Execute second-pass crop refinement with targeted 50-point prompt.

        1. Expands face bounding box by 20% margin.
        2. Crops the high-resolution face patch from original image.
        3. Invokes 9Router with targeted 50-point prompt.
        4. Transforms coordinates from crop-normalized space back to full image space.
        5. Compares quality score against initial face and returns the superior candidate.
        """
        try:
            crop_norm = expand_face_bbox(bbox, margin=0.20)
            c_ymin, c_xmin, c_ymax, c_xmax = crop_norm

            px1 = max(0, int(round((c_xmin / 1000.0) * orig_w)))
            py1 = max(0, int(round((c_ymin / 1000.0) * orig_h)))
            px2 = min(orig_w, int(round((c_xmax / 1000.0) * orig_w)))
            py2 = min(orig_h, int(round((c_ymax / 1000.0) * orig_h)))

            if (px2 - px1) < 10 or (py2 - py1) < 10:
                return None

            actual_crop_box = [
                int(round((py1 / float(orig_h)) * 1000.0)),
                int(round((px1 / float(orig_w)) * 1000.0)),
                int(round((py2 / float(orig_h)) * 1000.0)),
                int(round((px2 / float(orig_w)) * 1000.0)),
            ]

            crop_patch = full_img.crop((px1, py1, px2, py2))
            if max(crop_patch.size) > self.max_image_size:
                ratio = self.max_image_size / float(max(crop_patch.size))
                crop_patch = crop_patch.resize(
                    (max(1, int(round(crop_patch.width * ratio))), max(1, int(round(crop_patch.height * ratio)))),
                    Image.Resampling.LANCZOS,
                )

            buf = io.BytesIO()
            crop_patch.save(buf, format="JPEG", quality=92)
            crop_bytes = buf.getvalue()

            prompt = build_vf50_crop_prompt()
            resp = self.client.send_vision_request(
                model=self.active_model,
                image_bytes_or_b64=crop_bytes,
                prompt=prompt,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

            refined_faces = parse_vf50_response(
                resp.content,
                crop_box=actual_crop_box,
                orig_w=orig_w,
                orig_h=orig_h,
            )
            if refined_faces:
                refined_face = refined_faces[0]
                refined_face.face_id = face.face_id
                q_orig = assess_vf50_quality(face, width=orig_w, height=orig_h)
                q_new = assess_vf50_quality(refined_face, width=orig_w, height=orig_h)
                if q_new.score >= q_orig.score:
                    logger.info(
                        f"Crop refinement improved face {face.face_id} quality from {q_orig.score:.2f} to {q_new.score:.2f}"
                    )
                    return refined_face
                else:
                    logger.debug(
                        f"Initial face {face.face_id} retained: orig score {q_orig.score:.2f} >= crop score {q_new.score:.2f}"
                    )
            return None
        except Exception as e:
            logger.warning(f"Second-pass face crop refinement failed for face {face.face_id}: {e}")
            return None
