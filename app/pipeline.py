"""End-to-end 2D object detection annotation pipeline.

Coordinates image loading, preprocessing, 9Router vision inference,
schema validation, coordinate denormalization, bounding box rendering,
and artifact generation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from app.client import NineRouterClient, NineRouterError, VisionResponse
from app.config import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NINEROUTER_URL,
    DEFAULT_OUTPUT_DIR,
    AppConfig,
    LabelConfig,
    mask_api_key,
)
from app.image_ops import (
    draw_bounding_boxes,
    encode_image_to_data_url,
    get_image_dimensions,
    image_to_data_url,
    load_image,
    resize_image_if_needed,
    save_annotated_image,
)
from app.parser import ParseResult, VisionParseError, parse_and_validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cvat.pipeline")

PREFERRED_MODELS = [
    "ag/gemini-3.8-flash-high",
    "ag/gemini-3.8-flash-medium",
    "ag/gemini-3.8-flash",
    "ag/gemini-3.7-flash-high",
    "ag/gemini-3.6-flash-high",
    "ag/gemini-3.5-flash-high",
    "ag/claude-sonnet-4-6",
]


@dataclass
class PipelineOptions:
    """Options configuring pipeline execution."""
    image_path: Path
    model: Optional[str] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE
    labels_config: Path = DEFAULT_LABELS_PATH
    ninerouter_url: str = DEFAULT_NINEROUTER_URL
    ninerouter_key: Optional[str] = None
    strict: bool = False
    temperature: float = 0.0
    max_tokens: int = 4096

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        masked_key = mask_api_key(self.ninerouter_key) if self.ninerouter_key else None
        return (
            f"PipelineOptions(image_path={self.image_path!r}, "
            f"model={self.model!r}, "
            f"output_dir={self.output_dir!r}, "
            f"max_image_size={self.max_image_size}, "
            f"labels_config={self.labels_config!r}, "
            f"ninerouter_url={self.ninerouter_url!r}, "
            f"ninerouter_key={masked_key!r}, "
            f"strict={self.strict}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )


@dataclass
class PipelineResult:
    """Complete results from pipeline execution."""
    success: bool
    image_path: str
    model_used: str
    original_dimensions: Tuple[int, int]
    resized_dimensions: Tuple[int, int]
    was_resized: bool
    request_duration_seconds: float
    total_duration_seconds: float
    detections_count: int
    label_distribution: Dict[str, int]
    output_files: Dict[str, str]
    parse_result: Optional[ParseResult] = None
    error_message: Optional[str] = None


def select_best_vision_model(client: NineRouterClient, requested_model: Optional[str] = None) -> str:
    """Select the requested model or discover and choose the best available vision model."""
    if requested_model and requested_model.strip():
        return requested_model.strip()

    available_models = client.get_vision_models()
    if not available_models:
        raise NineRouterError("No vision models available in 9Router.")

    available_ids = [str(m.get("id")) for m in available_models if "id" in m]

    # Pick first matching preferred model
    for pref in PREFERRED_MODELS:
        if pref in available_ids:
            logger.info(f"Auto-selected preferred vision model: {pref}")
            return pref

    # Otherwise pick first available vision model
    first_choice = available_ids[0]
    logger.info(f"Auto-selected first available vision model: {first_choice}")
    return first_choice


def resolve_image_path(provided_path: Optional[Union[str, Path]]) -> Path:
    """Find the target image path or test.jpg/test.png fallback."""
    if provided_path:
        p = Path(provided_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Specified image not found: {provided_path}")

    # Fallback search order
    fallbacks = [
        Path("test.jpg"),
        Path("test.png"),
        Path("test.jpeg"),
        Path("sample.jpg"),
        Path("sample.png"),
    ]
    for fb in fallbacks:
        if fb.exists() and fb.is_file():
            return fb

    raise FileNotFoundError(
        "No image specified and default test images (test.jpg, test.png) were not found. "
        "Please specify an image with --image <path>."
    )


def run_pipeline(options: PipelineOptions) -> PipelineResult:
    """Execute end-to-end 2D detection and annotation pipeline.

    Produces:
    - {output_dir}/result_bbox.jpg
    - {output_dir}/predictions.json
    - {output_dir}/raw_response.txt
    - {output_dir}/run_report.json
    """
    total_start_time = time.perf_counter()

    # Step 1: Resolve image path
    resolved_img_path = resolve_image_path(options.image_path)
    logger.info(f"Input image: {resolved_img_path.resolve()}")

    # Step 2: Load and validate labels configuration
    logger.info(f"Loading label configuration from: {options.labels_config}")
    labels_cfg = LabelConfig.from_yaml(options.labels_config)
    target_labels = labels_cfg.bbox_labels if labels_cfg.bbox_labels else labels_cfg.all_labels
    if not target_labels:
        raise ValueError(f"No candidate labels found in {options.labels_config}")
    logger.info(f"Active candidate labels ({len(target_labels)}): {', '.join(target_labels)}")

    # Step 3: Initialize 9Router client & check health
    client = NineRouterClient(
        base_url=options.ninerouter_url,
        api_key=options.ninerouter_key,
    )
    health = client.get_health()
    logger.info(f"9Router health status: {health}")

    # Step 4: Resolve model
    model = select_best_vision_model(client, requested_model=options.model)
    logger.info(f"Using vision model: {model}")

    # Step 5: Load original image and inspect dimensions
    orig_image = load_image(resolved_img_path)
    orig_w, orig_h = get_image_dimensions(orig_image)
    logger.info(f"Original image dimensions: {orig_w}x{orig_h}")

    # Step 6: Preprocess & resize if needed for API transmission
    resized_img, was_resized, (new_w, new_h) = resize_image_if_needed(
        orig_image, max_size=options.max_image_size
    )
    if was_resized:
        logger.info(f"Resized image for API transmission: {new_w}x{new_h} (max_size={options.max_image_size})")
    else:
        logger.info(f"Image within max_size ({options.max_image_size}px), sending at original resolution.")

    # Step 7: Encode image to base64 Data URL
    image_data_url = image_to_data_url(resized_img, format="JPEG", quality=90)

    # Step 8: Send vision chat completion request
    logger.info("Sending multimodal chat completion request to 9Router...")
    vision_resp: VisionResponse = client.send_vision_request(
        model=model,
        image_bytes_or_b64=image_data_url,
        allowed_labels=target_labels,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )
    logger.info(
        f"Received model response in {vision_resp.duration_seconds:.2f}s "
        f"(status={vision_resp.status_code})"
    )

    # Step 9: Parse and validate model response
    logger.info("Parsing and validating model output against schema and label contract...")
    parse_result: ParseResult = parse_and_validate(
        raw_response=vision_resp.content,
        allowed_labels=target_labels,
        image_width=orig_w,
        image_height=orig_h,
        strict=options.strict,
    )

    detections = parse_result.objects
    label_counts = Counter(obj.label for obj in detections)
    logger.info(
        f"Detected {len(detections)} objects across {len(label_counts)} distinct labels: "
        f"{dict(label_counts)}"
    )
    if parse_result.warnings:
        for w in parse_result.warnings:
            logger.warning(f"Parse warning: {w}")

    # Step 10: Render bounding boxes on ORIGINAL image
    logger.info("Drawing bounding boxes and label badges on original image...")
    annotated_image = draw_bounding_boxes(
        original_image=orig_image,
        detections=detections,
    )

    # Step 11: Save all output artifacts
    output_dir = Path(options.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result_bbox_path = output_dir / "result_bbox.jpg"
    predictions_path = output_dir / "predictions.json"
    raw_response_path = output_dir / "raw_response.txt"
    run_report_path = output_dir / "run_report.json"

    # Save annotated image
    save_annotated_image(annotated_image, result_bbox_path)
    logger.info(f"Saved annotated image: {result_bbox_path}")

    # Save predictions.json
    predictions_payload = {
        "image": {
            "file": str(resolved_img_path.name),
            "width": orig_w,
            "height": orig_h,
        },
        "model": model,
        "objects": [obj.to_dict() for obj in detections],
        "cvat_annotations": parse_result.to_cvat_annotations(),
    }
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=2)
    logger.info(f"Saved structured predictions: {predictions_path}")

    # Save raw_response.txt
    with open(raw_response_path, "w", encoding="utf-8") as f:
        f.write(vision_resp.content)
    logger.info(f"Saved raw response text: {raw_response_path}")

    total_duration = time.perf_counter() - total_start_time

    # Save run_report.json
    run_report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "success": True,
        "model": model,
        "timing": {
            "request_duration_seconds": round(vision_resp.duration_seconds, 3),
            "total_duration_seconds": round(total_duration, 3),
        },
        "image": {
            "path": str(resolved_img_path.resolve()),
            "original_width": orig_w,
            "original_height": orig_h,
            "api_sent_width": new_w,
            "api_sent_height": new_h,
            "resized": was_resized,
        },
        "detections": {
            "total_count": len(detections),
            "label_distribution": dict(label_counts),
        },
        "artifacts": {
            "result_bbox": str(result_bbox_path),
            "predictions": str(predictions_path),
            "raw_response": str(raw_response_path),
            "run_report": str(run_report_path),
        },
        "token_usage": vision_resp.usage,
        "warnings": parse_result.warnings,
    }
    with open(run_report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    logger.info(f"Saved execution report: {run_report_path}")

    return PipelineResult(
        success=True,
        image_path=str(resolved_img_path),
        model_used=model,
        original_dimensions=(orig_w, orig_h),
        resized_dimensions=(new_w, new_h),
        was_resized=was_resized,
        request_duration_seconds=vision_resp.duration_seconds,
        total_duration_seconds=total_duration,
        detections_count=len(detections),
        label_distribution=dict(label_counts),
        output_files={
            "result_bbox": str(result_bbox_path),
            "predictions": str(predictions_path),
            "raw_response": str(raw_response_path),
            "run_report": str(run_report_path),
        },
        parse_result=parse_result,
    )
