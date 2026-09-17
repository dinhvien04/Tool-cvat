"""Shared in-memory annotation service for CVAT x 9Router AI Annotation.

This service is decoupled from filesystem operations and provides a unified,
in-memory pipeline used by:
1. Phase 1 CLI (main.py / app.pipeline) — runs inference and writes disk artifacts.
2. Phase 2 Nuclio function (serverless/main.py) — runs inference on incoming CVAT
   image bytes and returns CVAT rectangle shape dictionaries.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from app.client import NineRouterClient, NineRouterError
from app.config import DEFAULT_MAX_IMAGE_SIZE, DEFAULT_VISION_MODEL
from app.image_ops import image_to_data_url, load_image, resize_image_if_needed
from app.parser import ParsedObject, ParseResult, parse_and_validate
from core.vision_contract import DEFAULT_BBOX_LABELS, build_user_prompt


@dataclass
class AnnotationResult:
    """Result of annotating an image via 9Router vision models."""
    shapes: List[Dict[str, Any]]
    parsed_objects: List[ParsedObject]
    raw_response: str
    original_dimensions: Tuple[int, int]
    resized_dimensions: Tuple[int, int]
    was_resized: bool
    api_duration_seconds: float
    total_duration_seconds: float
    model_used: str
    warnings: List[str] = field(default_factory=list)

    def to_cvat_rectangles(self) -> List[Dict[str, Any]]:
        """Return pure CVAT detector response shapes."""
        return self.shapes


def annotate_image(
    image_source: Union[bytes, str, Path, Image.Image],
    client: NineRouterClient,
    model: Optional[str] = None,
    candidate_labels: Optional[List[str]] = None,
    threshold: float = 0.0,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
    strict: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    fallback_confidence: Optional[float] = None,
) -> AnnotationResult:
    """Run end-to-end in-memory detection on an image.

    Args:
        image_source: Raw image bytes, file path, or PIL.Image.Image.
        client: Configured NineRouterClient instance.
        model: Vision model identifier (e.g. 'ag/gemini-3.8-flash-high'). If None, resolved dynamically.
        candidate_labels: Allowed labels for detection (defaults to 13 bbox labels).
        threshold: Minimum confidence threshold [0.0, 1.0] for filtering detections.
        max_size: Maximum dimension for the API-transmitted copy.
        strict: If True, raises on invalid model outputs; if False, skips invalid boxes.
        temperature: Sampling temperature for model completion.
        max_tokens: Maximum completion tokens.
        fallback_confidence: Default confidence to assign when model omits confidence (defaults to None; never fake 1.0).

    Returns:
        AnnotationResult containing CVAT rectangle shapes and parsed objects.
    """
    total_start = time.perf_counter()
    warnings: List[str] = []

    # 1. Resolve model dynamically if not explicitly specified
    active_model = model or client.resolve_vision_model()

    # 1. Resolve candidate labels
    labels = list(candidate_labels) if candidate_labels else list(DEFAULT_BBOX_LABELS)

    # 2. Load original image
    pil_image = load_image(image_source)
    orig_w, orig_h = pil_image.size

    # 3. Create aspect-ratio preserving copy for transmission
    send_image, was_resized, (send_w, send_h) = resize_image_if_needed(
        pil_image, max_size=max_size
    )

    # 4. Prepare Base64 Data URL and vision prompt
    data_url = image_to_data_url(send_image, format="JPEG", quality=90)
    prompt = build_user_prompt(allowed_labels=labels)

    # 5. Send multimodal request to 9Router
    vision_resp = client.send_vision_request(
        model=active_model,
        image_bytes_or_b64=data_url,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 6. Parse and validate response
    raw_text = getattr(vision_resp, "content", getattr(vision_resp, "raw_content", ""))
    parse_result: ParseResult = parse_and_validate(
        raw_response=raw_text,
        allowed_labels=labels,
        image_width=orig_w,
        image_height=orig_h,
        strict=strict,
        fallback_confidence=fallback_confidence,
    )
    warnings.extend(parse_result.warnings)

    # 7. Apply confidence policy and format CVAT shapes
    # Policy:
    # - Model-reported confidence: validated in [0.0, 1.0] and filtered against threshold.
    # - Missing confidence: NEVER represented as fake 100% certainty (no "1.0").
    #   When threshold > 0.0, unrated detections cannot satisfy the threshold (unless fallback is set).
    #   When threshold == 0.0, unrated detections are retained without a fake confidence score.
    filtered_objects: List[ParsedObject] = []
    cvat_shapes: List[Dict[str, Any]] = []

    for obj in parse_result.objects:
        pts = obj.pixel_box if obj.pixel_box is not None else [0.0, 0.0, 0.0, 0.0]
        shape_dict: Dict[str, Any] = {
            "label": obj.label,
            "points": [round(float(p), 2) for p in pts],
            "type": "rectangle",
        }

        if obj.confidence is not None:
            # Real model-reported confidence
            if threshold > 0.0 and obj.confidence < threshold:
                continue
            shape_dict["confidence"] = str(round(float(obj.confidence), 2))
            filtered_objects.append(obj)
            cvat_shapes.append(shape_dict)
        elif fallback_confidence is not None:
            # Explicitly configured fallback
            if threshold > 0.0 and fallback_confidence < threshold:
                continue
            shape_dict["confidence"] = str(round(float(fallback_confidence), 2))
            filtered_objects.append(obj)
            cvat_shapes.append(shape_dict)
        else:
            # Omitted confidence: do NOT assign fake 1.0!
            if threshold > 0.0:
                continue
            filtered_objects.append(obj)
            cvat_shapes.append(shape_dict)

    total_duration = time.perf_counter() - total_start

    return AnnotationResult(
        shapes=cvat_shapes,
        parsed_objects=filtered_objects,
        raw_response=raw_text,
        original_dimensions=(orig_w, orig_h),
        resized_dimensions=(send_w, send_h),
        was_resized=was_resized,
        api_duration_seconds=vision_resp.duration_seconds,
        total_duration_seconds=round(total_duration, 3),
        model_used=active_model,
        warnings=warnings,
    )
