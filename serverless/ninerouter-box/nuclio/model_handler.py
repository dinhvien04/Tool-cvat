"""ModelHandler for the 9Router Box detector.

Returns rectangle shapes only for the 14 foreground instance labels.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

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
from app.service import AnnotationResult, annotate_image
from core.taxonomy import BOX_MASK_LABELS
from core.vision_contract import MODE_BOX

logger = logging.getLogger("cvat.nuclio.ninerouter.box")

BOX_LABELS: Sequence[str] = BOX_MASK_LABELS
DEFAULT_BOX_MAX_IMAGE_SIZE = 1280
DEFAULT_BOX_MAX_TOKENS = 1200


def load_spec_labels_from_function_yaml(yaml_path: Optional[Path | str] = None) -> List[str]:
    """Extract label names declared in function.yaml metadata annotations spec."""
    target_path = Path(yaml_path) if yaml_path else current_dir / "function.yaml"
    if not target_path.exists():
        return list(BOX_LABELS)

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
        logger.warning("Could not parse labels from %s: %s. Using the 14 box labels.", target_path, exc)

    return list(BOX_LABELS)


def rectangle_shapes_only(
    shapes: Sequence[Dict[str, Any]],
    allowed_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    """Keep rectangle shapes whose label is one of the 14 box labels."""
    allowed = set(allowed_labels)
    kept: List[Dict[str, Any]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("type") != "rectangle":
            continue
        if shape.get("label") not in allowed:
            continue
        points = shape.get("points")
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            continue
        kept.append(dict(shape))
    return kept


class ModelHandler:
    """Runs box-only inference for the 14 foreground labels."""

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
        self.requested_model = model or os.getenv("BOX_MODEL") or os.getenv("VISION_MODEL") or DEFAULT_VISION_MODEL
        self.default_mode = MODE_BOX

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
            val_size = max_image_size if max_image_size is not None else (max_size_env or DEFAULT_BOX_MAX_IMAGE_SIZE)
            self.max_image_size = int(val_size)
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_BOX_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_BOX_MAX_IMAGE_SIZE

        max_tokens_env = os.getenv("MAX_TOKENS")
        try:
            val_tokens = max_tokens if max_tokens is not None else (max_tokens_env or DEFAULT_BOX_MAX_TOKENS)
            self.max_tokens = int(val_tokens)
            if self.max_tokens <= 0:
                self.max_tokens = DEFAULT_BOX_MAX_TOKENS
        except (ValueError, TypeError):
            self.max_tokens = DEFAULT_BOX_MAX_TOKENS

        self.candidate_labels = candidate_labels or load_spec_labels_from_function_yaml()
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.active_model = self.client.resolve_vision_model(self.requested_model)
        logger.info(
            "Initialized ModelHandler (Box): base_url=%r, key=%s, model=%r, labels=%d",
            self.base_url,
            mask_api_key(self.api_key),
            self.active_model,
            len(self.candidate_labels),
        )

    def infer(self, image_bytes: bytes, threshold: float = 0.5, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return rectangle shapes. Any requested mode other than box is ignored."""
        if not image_bytes:
            raise ValueError("Empty image payload received")
        if mode and mode.strip().lower() not in (MODE_BOX, ""):
            logger.info("Ignoring requested mode %r; box detector emits rectangles only.", mode)

        started = time.perf_counter()
        result: AnnotationResult = annotate_image(
            image_source=image_bytes,
            client=self.client,
            model=self.active_model,
            candidate_labels=self.candidate_labels,
            threshold=threshold,
            max_size=self.max_image_size,
            max_tokens=self.max_tokens,
            strict=False,
            mode=MODE_BOX,
            enable_feedback=False,
        )
        shapes = rectangle_shapes_only(result.shapes, self.candidate_labels)
        logger.info(
            "Box inference complete: model=%s total_s=%.2f shapes=%d",
            self.active_model,
            time.perf_counter() - started,
            len(shapes),
        )
        return shapes
