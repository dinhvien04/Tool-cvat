"""ModelHandler for 9Router Vision Nuclio Detector.

Bridges CVAT's serverless detector invocation with the local 9Router vision model.
Converts normalized [0, 1000] integer bounding boxes from remote LLM vision models
into CVAT pixel coordinates [xtl, ytl, xbr, ybr].
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
from core.vision_contract import DEFAULT_BBOX_LABELS

logger = logging.getLogger("cvat.nuclio.ninerouter")


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
    """Manages the 9Router vision client and runs inference for CVAT detector requests."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        candidate_labels: Optional[List[str]] = None,
    ):
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL)

        timeout_env = os.getenv("NINEROUTER_TIMEOUT")
        self.timeout = float(timeout if timeout is not None else (timeout_env or DEFAULT_NINEROUTER_TIMEOUT))

        max_size_env = os.getenv("MAX_IMAGE_SIZE")
        self.max_image_size = int(max_image_size if max_image_size is not None else (max_size_env or DEFAULT_MAX_IMAGE_SIZE))

        self.candidate_labels = candidate_labels or load_spec_labels_from_function_yaml()

        # Initialize client
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        self.active_model: str = self.requested_model
        logger.info(
            f"Initialized ModelHandler: base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)}, model={self.requested_model!r}, "
            f"timeout={self.timeout}s, labels_count={len(self.candidate_labels)}"
        )

    def verify_readiness(self) -> Dict[str, Any]:
        """Verify health check and model availability with 9Router."""
        health = self.client.get_health()
        return health

    def infer(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Process incoming image bytes and return CVAT-compatible rectangle shapes.

        Args:
            image_bytes: Raw JPEG/PNG image binary data.
            threshold: Minimum confidence score [0.0, 1.0].

        Returns:
            List of CVAT rectangle dictionaries:
            [
                {
                    "confidence": "0.95",
                    "label": "car",
                    "points": [x1, y1, x2, y2],
                    "type": "rectangle"
                }
            ]
        """
        if not image_bytes:
            raise ValueError("Empty image payload received")

        result: AnnotationResult = annotate_image(
            image_source=image_bytes,
            client=self.client,
            model=self.active_model,
            candidate_labels=self.candidate_labels,
            threshold=threshold,
            max_size=self.max_image_size,
            strict=False,
        )

        logger.info(
            f"Inference complete: model={self.active_model}, "
            f"image={result.original_dimensions[0]}x{result.original_dimensions[1]}, "
            f"api_latency={result.api_duration_seconds:.2f}s, "
            f"detections_returned={len(result.shapes)} (threshold={threshold})"
        )

        return result.shapes
