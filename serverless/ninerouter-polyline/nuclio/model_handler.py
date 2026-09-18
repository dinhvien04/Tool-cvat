"""ModelHandler for 9Router Polyline Nuclio Detector.

Bridges CVAT's serverless detector invocation with the local 9Router vision model.
Enforces Policy C:
- Exactly 7 lane demarcation labels
- Emits polyline ONLY with direct linear coordinate projection
- No mask, polygon, box, or group ID
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

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
    DEFAULT_POLYLINE_MAX_IMAGE_SIZE,
    DEFAULT_POLYLINE_MAX_TOKENS,
    DEFAULT_POLYLINE_VISION_MODEL,
    DEFAULT_VISION_MODEL,
    mask_api_key,
)
from app.service import AnnotationResult, annotate_image
from core.taxonomy import POLYLINE_LABELS
from core.vision_contract import MODE_POLYLINE

logger = logging.getLogger("cvat.nuclio.ninerouter.polyline")


def load_spec_labels_from_function_yaml(yaml_path: Optional[Path | str] = None) -> List[str]:
    """Extract label names declared in function.yaml metadata annotations spec."""
    target_path = Path(yaml_path) if yaml_path else current_dir / "function.yaml"
    if not target_path.exists():
        return list(POLYLINE_LABELS)

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
        logger.warning(f"Could not parse labels from {target_path}: {e}. Using default POLYLINE_LABELS.")

    return list(POLYLINE_LABELS)


class ModelHandler:
    """Manages the 9Router vision client and runs Policy C (Polyline) inference for CVAT."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        candidate_labels: Optional[List[str]] = None,
        mode: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("POLYLINE_MODEL") or os.getenv("VISION_MODEL")
        env_mode = os.getenv("DETECTION_MODE", MODE_POLYLINE)
        self.default_mode = (mode or env_mode or MODE_POLYLINE).strip().lower()

        timeout_env = os.getenv("NINEROUTER_TIMEOUT")
        try:
            val_timeout = timeout if timeout is not None else (timeout_env or DEFAULT_NINEROUTER_TIMEOUT)
            self.timeout = float(val_timeout)
            if self.timeout <= 0:
                self.timeout = DEFAULT_NINEROUTER_TIMEOUT
        except (ValueError, TypeError):
            self.timeout = DEFAULT_NINEROUTER_TIMEOUT

        max_size_env = os.getenv("POLYLINE_MAX_IMAGE_SIZE") or os.getenv("MAX_IMAGE_SIZE")
        try:
            val_size = max_image_size if max_image_size is not None else (max_size_env or DEFAULT_POLYLINE_MAX_IMAGE_SIZE)
            self.max_image_size = int(val_size)
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_POLYLINE_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_POLYLINE_MAX_IMAGE_SIZE

        max_tokens_env = os.getenv("POLYLINE_MAX_TOKENS")
        try:
            val_tokens = max_tokens if max_tokens is not None else (max_tokens_env or DEFAULT_POLYLINE_MAX_TOKENS)
            self.max_tokens = int(val_tokens)
            if self.max_tokens <= 0:
                self.max_tokens = DEFAULT_POLYLINE_MAX_TOKENS
        except (ValueError, TypeError):
            self.max_tokens = DEFAULT_POLYLINE_MAX_TOKENS

        self.candidate_labels = candidate_labels or load_spec_labels_from_function_yaml()

        # Initialize client
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Dynamic vision model resolution (prioritize fast polyline model)
        try:
            self.active_model: str = self.client.resolve_polyline_model(self.requested_model)
        except Exception as e:
            logger.error(f"Failed to resolve vision model {self.requested_model!r} from 9Router: {e}")
            raise

        logger.info(
            f"Initialized ModelHandler (Polyline): base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)}, model={self.active_model!r}, "
            f"timeout={self.timeout}s, labels_count={len(self.candidate_labels)}"
        )

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        return (
            f"ModelHandler(polyline, base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)!r}, "
            f"model={self.active_model!r}, "
            f"timeout={self.timeout})"
        )

    def verify_readiness(self) -> Dict[str, Any]:
        """Verify health check and model availability with 9Router."""
        return self.client.get_health()

    def infer(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
        mode: Optional[str] = None,
        roi: Optional[Sequence[Union[int, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Process incoming image bytes and return Policy C CVAT shapes (polyline only).

        Args:
            image_bytes: Raw JPEG/PNG image binary data.
            threshold: Minimum confidence score [0.0, 1.0].
            mode: Optional detection mode override. Defaults to self.default_mode ('polyline').
            roi: Optional sub-region of interest [x1, y1, x2, y2] to crop and annotate.

        Returns:
            List of CVAT shape dictionaries routed according to Policy C (polyline only, no masks/boxes).
        """
        if not image_bytes:
            raise ValueError("Empty image payload received")

        active_mode = (mode or self.default_mode).strip().lower()

        t_infer_start = time.perf_counter()
        result: AnnotationResult = annotate_image(
            image_source=image_bytes,
            client=self.client,
            model=self.active_model,
            candidate_labels=self.candidate_labels,
            threshold=threshold,
            max_size=self.max_image_size,
            max_tokens=self.max_tokens,
            strict=False,
            mode=active_mode,
            roi=roi,
        )

        total_s = time.perf_counter() - t_infer_start
        router_s = result.api_duration_seconds
        local_ms = max(0.0, (total_s - router_s) * 1000.0)

        logger.info(
            f"PERF detector={active_mode} router_s={router_s:.2f} local_ms={local_ms:.1f} "
            f"total_s={total_s:.2f} shapes={len(result.shapes)}"
        )

        logger.info(
            f"Polyline inference complete: model={self.active_model}, mode={active_mode}, "
            f"image={result.original_dimensions[0]}x{result.original_dimensions[1]}, "
            f"api_latency={result.api_duration_seconds:.2f}s, "
            f"shapes_returned={len(result.shapes)} (threshold={threshold})"
        )

        if result.warnings:
            for w in result.warnings[:10]:
                logger.warning(f"Inference warning: {w}")

        return result.shapes
