"""ModelHandler for 9Router Vision Box+Mask Nuclio Detector.

Bridges CVAT's serverless detector invocation with the local 9Router vision model.
Returns BOTH CVAT rectangle shapes and native mask shapes with shared group_id
for each detected instance.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Ensure project modules (app, core) are importable
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
from app.service import AnnotationResult, annotate_image
from core.vision_contract import DEFAULT_BBOX_LABELS, MODE_BOX, MODE_BOX_AND_MASK, MODE_MASK

logger = logging.getLogger("cvat.nuclio.ninerouter.boxmask")


def load_spec_labels_from_function_yaml(yaml_path: Optional[Path | str] = None) -> List[str]:
    """Extract label names declared in function.yaml metadata annotations spec."""
    target_path = Path(yaml_path) if yaml_path else current_dir / "function.yaml"
    if not target_path.exists():
        return list(DEFAULT_BBOX_LABELS)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        annotations = data.get("metadata", {}).get("annotations", {})
        raw_spec = annotations.get("spec", "")
        if isinstance(raw_spec, str) and raw_spec.strip():
            spec_list = json.loads(raw_spec)
            labels = [item["name"] for item in spec_list if "name" in item]
            if labels:
                return labels
    except Exception as e:
        logger.warning(f"Could not parse labels from {target_path}: {e}. Using default bbox labels.")

    return list(DEFAULT_BBOX_LABELS)


class ModelHandler:
    """Manages the 9Router vision client and runs unified Box+Mask inference for CVAT."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        candidate_labels: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("VISION_MODEL")
        env_mode = os.getenv("DETECTION_MODE", MODE_BOX_AND_MASK)
        self.default_mode = (mode or env_mode or MODE_BOX_AND_MASK).strip().lower()
        if self.default_mode not in (MODE_BOX, MODE_MASK, MODE_BOX_AND_MASK):
            self.default_mode = MODE_BOX_AND_MASK

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
            val_size = max_image_size if max_image_size is not None else (max_size_env or DEFAULT_MAX_IMAGE_SIZE)
            self.max_image_size = int(val_size)
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_MAX_IMAGE_SIZE

        self.candidate_labels = candidate_labels or load_spec_labels_from_function_yaml()

        # Initialize client
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Dynamic segmentation model resolution
        try:
            self.active_model: str = self.client.resolve_segmentation_model(self.requested_model)
        except Exception as e:
            logger.error(f"Failed to resolve segmentation model {self.requested_model!r} from 9Router: {e}")
            raise

        logger.info(
            f"Initialized ModelHandler (Box+Mask): base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)}, model={self.active_model!r}, "
            f"timeout={self.timeout}s, labels_count={len(self.candidate_labels)}"
        )

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        return (
            f"ModelHandler(box+mask, base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)!r}, "
            f"model={self.active_model!r}, "
            f"timeout={self.timeout})"
        )

    def verify_readiness(self) -> Dict[str, Any]:
        """Verify health check and model availability with 9Router."""
        health = self.client.get_health()
        return health

    def infer(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Process incoming image bytes and return both CVAT rectangle and mask shapes.

        Args:
            image_bytes: Raw JPEG/PNG image binary data.
            threshold: Minimum confidence score [0.0, 1.0].
            mode: Optional detection mode override ('box', 'mask', or 'box_and_mask').
                  Defaults to self.default_mode (configured via DETECTION_MODE env).

        Returns:
            List of CVAT shape dictionaries containing both rectangle and mask shapes
            with matching group_id for paired instances:
            [
                {
                    "confidence": "0.95",
                    "label": "car",
                    "points": [xtl, ytl, xbr, ybr],
                    "type": "rectangle",
                    "group_id": 1
                },
                {
                    "confidence": "0.95",
                    "label": "car",
                    "mask": [p0, p1, ..., xmin, ymin, xmax, ymax],
                    "points": [x1, y1, x2, y2, ...],
                    "type": "mask",
                    "group_id": 1
                }
            ]
        """
        if not image_bytes:
            raise ValueError("Empty image payload received")

        active_mode = (mode or self.default_mode).strip().lower()
        if active_mode not in (MODE_BOX, MODE_MASK, MODE_BOX_AND_MASK):
            raise ValueError(
                f"Invalid detection mode {active_mode!r}. Must be one of: "
                f"'{MODE_BOX}', '{MODE_MASK}', '{MODE_BOX_AND_MASK}'"
            )

        result: AnnotationResult = annotate_image(
            image_source=image_bytes,
            client=self.client,
            model=self.active_model,
            candidate_labels=self.candidate_labels,
            threshold=threshold,
            max_size=self.max_image_size,
            strict=False,
            mode=active_mode,
        )

        logger.info(
            f"Box+Mask inference complete: model={self.active_model}, mode={active_mode}, "
            f"image={result.original_dimensions[0]}x{result.original_dimensions[1]}, "
            f"api_latency={result.api_duration_seconds:.2f}s, "
            f"shapes_returned={len(result.shapes)} (threshold={threshold})"
        )

        if active_mode == MODE_MASK:
            return result.to_cvat_masks()
        elif active_mode == MODE_BOX:
            return result.to_cvat_rectangles()
        return result.shapes
