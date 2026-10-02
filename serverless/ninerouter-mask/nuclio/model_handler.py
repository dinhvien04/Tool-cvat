"""Mask-only detector for the 14 foreground instance labels."""

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
from app.config import DEFAULT_NINEROUTER_TIMEOUT, DEFAULT_NINEROUTER_URL_CONTAINER, DEFAULT_VISION_MODEL, mask_api_key
from app.service import annotate_image
from core.taxonomy import BOX_MASK_LABELS
from core.vision_contract import MODE_MASK

logger = logging.getLogger("cvat.nuclio.ninerouter.mask")


def load_spec_labels(yaml_path: Optional[Path | str] = None) -> List[str]:
    target = Path(yaml_path) if yaml_path else current_dir / "function.yaml"
    if not target.exists():
        return list(BOX_MASK_LABELS)
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        raw = data.get("metadata", {}).get("annotations", {}).get("spec", "")
        if isinstance(raw, str) and raw.strip():
            labels = [item["name"] for item in json.loads(raw) if isinstance(item, dict) and "name" in item]
            if labels:
                return labels
    except Exception as exc:
        logger.warning("Could not parse mask labels: %s", exc)
    return list(BOX_MASK_LABELS)


def mask_shapes_only(shapes: Sequence[Dict[str, Any]], allowed: Sequence[str]) -> List[Dict[str, Any]]:
    names = set(allowed)
    kept = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("type") != "mask":
            continue
        if shape.get("label") not in names:
            continue
        mask = shape.get("mask")
        if not isinstance(mask, (list, tuple)) or len(mask) < 5:
            continue
        kept.append({"label": shape["label"], "type": "mask", "mask": list(mask), **({"confidence": shape["confidence"]} if "confidence" in shape else {})})
    return kept


class ModelHandler:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=None, max_image_size=None, max_tokens=None, candidate_labels=None):
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("MASK_MODEL") or os.getenv("VISION_MODEL") or DEFAULT_VISION_MODEL
        self.timeout = float(timeout or os.getenv("NINEROUTER_TIMEOUT") or DEFAULT_NINEROUTER_TIMEOUT)
        self.max_image_size = int(max_image_size or os.getenv("MAX_IMAGE_SIZE") or 1280)
        self.max_tokens = int(max_tokens or os.getenv("MAX_TOKENS") or 2000)
        self.candidate_labels = candidate_labels or load_spec_labels()
        self.client = NineRouterClient(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        self.active_model = self.client.resolve_segmentation_model(self.requested_model)
        logger.info("Initialized Mask handler model=%r key=%s", self.active_model, mask_api_key(self.api_key))

    def infer(self, image_bytes: bytes, threshold: float = 0.5, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        if not image_bytes:
            raise ValueError("Empty image payload received")
        started = time.perf_counter()
        result = annotate_image(
            image_source=image_bytes,
            client=self.client,
            model=self.active_model,
            candidate_labels=self.candidate_labels,
            threshold=threshold,
            max_size=self.max_image_size,
            max_tokens=self.max_tokens,
            strict=False,
            mode=MODE_MASK,
            enable_feedback=False,
        )
        shapes = mask_shapes_only(result.shapes, self.candidate_labels)
        logger.info("Mask inference total_s=%.2f shapes=%d", time.perf_counter() - started, len(shapes))
        return shapes
