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
from core.line_geometry import lane_shape_pipeline
from core.taxonomy import (
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    Taxonomy,
    validate_cvat_output_shapes,
)
from core.vision_contract import (
    ALL_31_LABELS,
    DEFAULT_BBOX_LABELS,
    MODE_BOX,
    MODE_BOX_AND_MASK,
    MODE_FULL_31,
    MODE_MASK,
    build_user_prompt,
)

MAX_OBJECTS: int = 100
MAX_REGIONS: int = 50
MAX_LANES: int = 50


def route_phase3b_shapes(
    raw_shapes: List[Dict[str, Any]],
    max_objects: int = MAX_OBJECTS,
    max_regions: int = MAX_REGIONS,
    max_lanes: int = MAX_LANES,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Enforce Phase 3B 3-policy routing and limits on shapes.

    - Policy A: Box + Mask with shared group_id (14 instance labels)
    - Policy B: Polygon + Mask with shared group_id (10 region labels)
    - Policy C: Polyline only (7 lane labels)
    """
    taxonomy = Taxonomy()
    routed: List[Dict[str, Any]] = []
    warnings: List[str] = []

    inst_count = 0
    reg_count = 0
    lane_count = 0

    for s in raw_shapes:
        lbl = s.get("label", "")
        policy = taxonomy.get_policy(lbl)
        stype = s.get("type")

        if policy == POLICY_BOX_MASK:
            if inst_count >= max_objects:
                warnings.append(f"limit_exceeded: Exceeded MAX_OBJECTS ({max_objects}); dropped {lbl}")
                continue
            routed.append(s)
            if stype == "rectangle":
                inst_count += 1

        elif policy == POLICY_POLYGON_MASK:
            if reg_count >= max_regions:
                warnings.append(f"limit_exceeded: Exceeded MAX_REGIONS ({max_regions}); dropped {lbl}")
                continue
            if stype == "rectangle":
                warnings.append(f"region_box_suppressed: Suppressed box for semantic region '{lbl}'")
                continue
            routed.append(s)
            if stype == "polygon":
                reg_count += 1

        elif policy == POLICY_POLYLINE:
            if lane_count >= max_lanes:
                warnings.append(f"limit_exceeded: Exceeded MAX_LANES ({max_lanes}); dropped {lbl}")
                continue
            if stype != "polyline":
                warnings.append(f"lane_non_polyline_suppressed: Suppressed non-polyline shape '{stype}' for lane '{lbl}'")
                continue
            routed.append(s)
            lane_count += 1

    validated, val_warnings = validate_cvat_output_shapes(routed, taxonomy=taxonomy)
    warnings.extend(val_warnings)
    return validated, warnings


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
    mode: str = MODE_BOX
    warnings: List[str] = field(default_factory=list)
    image_hash: Optional[str] = None
    rules_injected: List[str] = field(default_factory=list)
    visual_examples_used: int = 0
    text_rules_used: int = 0

    def to_cvat_rectangles(self) -> List[Dict[str, Any]]:
        """Return pure CVAT detector response shapes."""
        return [s for s in self.shapes if s.get("type") == "rectangle"]

    def to_cvat_masks(self) -> List[Dict[str, Any]]:
        """Return pure CVAT mask response shapes."""
        return [s for s in self.shapes if s.get("type") == "mask"]


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
    mode: Optional[str] = None,
    output_mode: Optional[str] = None,
    roi: Optional[Sequence[Union[int, float]]] = None,
    feedback_db: Optional[Any] = None,
    enable_feedback: bool = True,
    task_id: Optional[int] = None,
    job_id: Optional[int] = None,
    frame_index: Optional[int] = None,
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
        mode: Detection output mode: 'box' (default), 'mask', 'box_and_mask', or 'full_31'. Alias for output_mode.
        output_mode: Detection output mode: 'box' (default), 'mask', 'box_and_mask', or 'full_31'.
        roi: Optional sub-region of interest [x1, y1, x2, y2] to crop and annotate.

    Returns:
        AnnotationResult containing CVAT shapes and parsed objects.
    """
    total_start = time.perf_counter()
    warnings: List[str] = []
    active_mode = (output_mode or mode or MODE_BOX).strip().lower()

    # 1. Resolve model dynamically if not explicitly specified
    if active_mode in (MODE_MASK, MODE_BOX_AND_MASK, MODE_FULL_31):
        active_model = model or client.resolve_segmentation_model()
    else:
        active_model = model or client.resolve_vision_model()

    # 2. Resolve candidate labels
    if candidate_labels:
        labels = list(candidate_labels)
    elif active_mode == MODE_FULL_31:
        labels = list(ALL_31_LABELS)
    else:
        labels = list(DEFAULT_BBOX_LABELS)

    # 3. Load original image and apply ROI crop if specified
    pil_image = load_image(image_source)
    orig_w, orig_h = pil_image.size

    roi_offset_x = 0
    roi_offset_y = 0
    if roi is not None and len(roi) == 4:
        rx1 = max(0, min(orig_w - 1, int(round(float(roi[0])))))
        ry1 = max(0, min(orig_h - 1, int(round(float(roi[1])))))
        rx2 = max(rx1 + 1, min(orig_w, int(round(float(roi[2])))))
        ry2 = max(ry1 + 1, min(orig_h, int(round(float(roi[3])))))
        crop_image = pil_image.crop((rx1, ry1, rx2, ry2))
        roi_offset_x = rx1
        roi_offset_y = ry1
        curr_w, curr_h = crop_image.size
    else:
        crop_image = pil_image
        curr_w, curr_h = orig_w, orig_h

    # 4. Create aspect-ratio preserving copy for transmission
    send_image, was_resized, (send_w, send_h) = resize_image_if_needed(
        crop_image, max_size=max_size
    )

    # 5. Prepare Base64 Data URL and vision prompt with requested mode
    data_url = image_to_data_url(send_image, format="JPEG", quality=90)
    prompt = build_user_prompt(allowed_labels=labels, mode=active_mode)

    # 5b. Compute image fingerprint and inject correction memory rules
    from app.feedback import FeedbackDatabase, compute_image_hash, compute_perceptual_hash
    from app.retrieval import CorrectionRetrievalEngine

    try:
        image_hash = compute_image_hash(pil_image)
    except Exception:
        image_hash = None

    try:
        perceptual_hash = compute_perceptual_hash(pil_image)
    except Exception:
        perceptual_hash = None

    rules_injected: List[str] = []
    visual_examples: List[Dict[str, Any]] = []
    visual_examples_used: int = 0
    text_rules_used: int = 0

    f_db: Optional[FeedbackDatabase] = None
    if enable_feedback:
        try:
            f_db = feedback_db if feedback_db is not None else FeedbackDatabase()
            if not f_db.is_enabled():
                f_db = None
        except Exception as e:
            logger.warning(f"Could not initialize feedback database: {e}")
            warnings.append(f"feedback_init_warning: {e}")
            f_db = None

    if f_db is not None:
        try:
            retrieval_engine = CorrectionRetrievalEngine(db=f_db)
            retrieval_res = retrieval_engine.retrieve(candidate_labels=labels)
            if retrieval_res.prompt_extension:
                prompt += "\n\n" + retrieval_res.prompt_extension
            rules_injected = retrieval_res.rules
            text_rules_used = len(rules_injected)
            visual_examples = retrieval_res.visual_examples
            visual_examples_used = len(visual_examples)
        except Exception as e:
            logger.warning(f"Feedback retrieval warning: {e}")
            warnings.append(f"feedback_retrieval_warning: {e}")

    # 6. Send multimodal request to 9Router
    vision_resp = client.send_vision_request(
        model=active_model,
        image_bytes_or_b64=data_url,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        visual_examples=visual_examples if visual_examples else None,
    )

    # 7. Parse and validate response
    raw_text = getattr(vision_resp, "content", getattr(vision_resp, "raw_content", ""))
    parse_result: ParseResult = parse_and_validate(
        raw_response=raw_text,
        allowed_labels=labels,
        image_width=curr_w,
        image_height=curr_h,
        strict=strict,
        fallback_confidence=fallback_confidence,
    )
    warnings.extend(parse_result.warnings)

    # 8. Apply confidence policy and format CVAT shapes according to mode
    filtered_objects: List[ParsedObject] = []
    cvat_shapes: List[Dict[str, Any]] = []

    taxonomy = Taxonomy() if active_mode == MODE_FULL_31 else None
    inst_count = 0
    reg_count = 0
    lane_count = 0

    items_to_process = parse_result.all_items if hasattr(parse_result, "all_items") else parse_result.objects

    existing_gids = [
        o.group_id for o in items_to_process
        if getattr(o, "group_id", None) is not None and isinstance(o.group_id, int)
    ]
    next_group_id = (max(existing_gids) + 1) if existing_gids else 1

    for obj in items_to_process:
        effective_conf: Optional[float] = None
        if obj.confidence is not None:
            if threshold > 0.0 and obj.confidence < threshold:
                continue
            effective_conf = obj.confidence
        elif fallback_confidence is not None:
            if threshold > 0.0 and fallback_confidence < threshold:
                continue
            effective_conf = fallback_confidence
        else:
            if threshold > 0.0:
                continue

        # Adjust coordinates for ROI offset if needed
        if roi_offset_x > 0 or roi_offset_y > 0:
            if obj.pixel_box is not None:
                obj.pixel_box = [
                    round(obj.pixel_box[0] + roi_offset_x, 2),
                    round(obj.pixel_box[1] + roi_offset_y, 2),
                    round(obj.pixel_box[2] + roi_offset_x, 2),
                    round(obj.pixel_box[3] + roi_offset_y, 2),
                ]
            if obj.pixel_polygon is not None:
                obj.pixel_polygon = [
                    (round(p[0] + roi_offset_x, 2), round(p[1] + roi_offset_y, 2))
                    for p in obj.pixel_polygon
                ]
            if obj.cvat_mask is not None and len(obj.cvat_mask) >= 4:
                crop_pixels = obj.cvat_mask[:-4]
                xm1, ym1, xm2, ym2 = obj.cvat_mask[-4:]
                obj.cvat_mask = crop_pixels + [
                    xm1 + roi_offset_x,
                    ym1 + roi_offset_y,
                    xm2 + roi_offset_x,
                    ym2 + roi_offset_y,
                ]

        filtered_objects.append(obj)
        conf_str = str(round(float(effective_conf), 2)) if effective_conf is not None else None

        has_valid_box = (
            obj.pixel_box is not None
            and len(obj.pixel_box) == 4
            and (obj.pixel_box[2] > obj.pixel_box[0])
            and (obj.pixel_box[3] > obj.pixel_box[1])
        )
        has_valid_mask = obj.cvat_mask is not None and len(obj.cvat_mask) > 4

        if active_mode == MODE_BOX:
            if has_valid_box:
                box_shape: Dict[str, Any] = {
                    "label": obj.label,
                    "points": [round(float(p), 2) for p in obj.pixel_box],
                    "type": "rectangle",
                }
                if conf_str is not None:
                    box_shape["confidence"] = conf_str
                cvat_shapes.append(box_shape)

        elif active_mode == MODE_MASK:
            if has_valid_mask:
                mask_shape: Dict[str, Any] = {
                    "label": obj.label,
                    "type": "mask",
                    "mask": obj.cvat_mask,
                }
                if obj.pixel_polygon:
                    mask_shape["points"] = [round(float(c), 2) for pt in obj.pixel_polygon for c in pt]
                elif obj.mask:
                    from core.geometry import denormalize_contour
                    pts_px = denormalize_contour(obj.mask, width=orig_w, height=orig_h)
                    mask_shape["points"] = [round(float(c), 2) for pt in pts_px for c in pt]
                if conf_str is not None:
                    mask_shape["confidence"] = conf_str
                cvat_shapes.append(mask_shape)
            else:
                warnings.append(
                    f"mask_missing: Detection '{obj.label}' has missing or invalid mask; annotation rejected in mask mode"
                )

        elif active_mode == MODE_BOX_AND_MASK:
            if has_valid_box:
                box_shape = {
                    "label": obj.label,
                    "points": [round(float(p), 2) for p in obj.pixel_box],
                    "type": "rectangle",
                }
                if conf_str is not None:
                    box_shape["confidence"] = conf_str
                if obj.group_id is not None:
                    box_shape["group_id"] = obj.group_id
                cvat_shapes.append(box_shape)
            else:
                warnings.append(
                    f"box_missing: Detection '{obj.label}' has missing or invalid bounding box; emitted mask only in box_and_mask mode"
                )

            if has_valid_mask:
                mask_shape = {
                    "label": obj.label,
                    "type": "mask",
                    "mask": obj.cvat_mask,
                }
                if obj.pixel_polygon:
                    mask_shape["points"] = [round(float(c), 2) for pt in obj.pixel_polygon for c in pt]
                elif obj.mask:
                    from core.geometry import denormalize_contour
                    pts_px = denormalize_contour(obj.mask, width=orig_w, height=orig_h)
                    mask_shape["points"] = [round(float(c), 2) for pt in pts_px for c in pt]
                if conf_str is not None:
                    mask_shape["confidence"] = conf_str
                if obj.group_id is not None:
                    mask_shape["group_id"] = obj.group_id
                cvat_shapes.append(mask_shape)
            else:
                warnings.append(
                    f"mask_missing: Detection '{obj.label}' has missing or invalid mask; emitted rectangle only in box_and_mask mode"
                )

        elif active_mode == MODE_FULL_31:
            policy = taxonomy.get_policy(obj.label) if taxonomy else POLICY_BOX_MASK

            # POLICY A: RECTANGLE + MASK (14 instance labels)
            # For every accepted detection, emit BOTH a rectangle and a mask sharing the exact same group_id.
            # If either shape is missing or invalid, drop the entire annotation. Never emit box-only or mask-only.
            if policy == POLICY_BOX_MASK:
                if inst_count >= MAX_OBJECTS:
                    warnings.append(f"limit_exceeded: Exceeded MAX_OBJECTS ({MAX_OBJECTS}); dropped {obj.label}")
                    continue

                if not (has_valid_box and has_valid_mask):
                    missing_parts = []
                    if not has_valid_box:
                        missing_parts.append("rectangle")
                    if not has_valid_mask:
                        missing_parts.append("mask")
                    warnings.append(
                        f"policy_a_incomplete: Instance detection '{obj.label}' missing required {(' and '.join(missing_parts))}; annotation dropped per Policy A"
                    )
                    continue

                inst_count += 1
                group_id = obj.group_id if (obj.group_id is not None and isinstance(obj.group_id, int)) else next_group_id
                next_group_id = max(next_group_id + 1, group_id + 1)

                box_shape = {
                    "label": obj.label,
                    "points": [round(float(p), 2) for p in obj.pixel_box],
                    "type": "rectangle",
                    "group_id": group_id,
                }
                if conf_str is not None:
                    box_shape["confidence"] = conf_str

                mask_shape = {
                    "label": obj.label,
                    "type": "mask",
                    "mask": obj.cvat_mask,
                    "group_id": group_id,
                }
                if obj.pixel_polygon:
                    mask_shape["points"] = [round(float(c), 2) for pt in obj.pixel_polygon for c in pt]
                elif obj.mask:
                    from core.geometry import denormalize_contour
                    pts_px = denormalize_contour(obj.mask, width=orig_w, height=orig_h)
                    mask_shape["points"] = [round(float(c), 2) for pt in pts_px for c in pt]
                else:
                    mask_shape["points"] = [round(float(p), 2) for p in obj.pixel_box]

                if conf_str is not None:
                    mask_shape["confidence"] = conf_str

                cvat_shapes.append(box_shape)
                cvat_shapes.append(mask_shape)

            # POLICY B: POLYGON + MASK (10 semantic region labels)
            # For every accepted region, emit BOTH a polygon and a mask derived from the exact same contour,
            # sharing the same group_id. Never emit bounding boxes, mask-only, or polygon-only.
            elif policy == POLICY_POLYGON_MASK:
                if reg_count >= MAX_REGIONS:
                    warnings.append(f"limit_exceeded: Exceeded MAX_REGIONS ({MAX_REGIONS}); dropped {obj.label}")
                    continue

                if has_valid_box:
                    warnings.append(f"region_box_suppressed: Suppressed box for semantic region '{obj.label}'")

                from core.geometry import calculate_polygon_area, denormalize_contour, polygon_to_cvat_mask

                pts_px: Optional[List[Tuple[float, float]]] = None
                if obj.pixel_polygon and len(obj.pixel_polygon) >= 3:
                    pts_px = obj.pixel_polygon
                elif obj.mask and len(obj.mask) >= 3:
                    try:
                        pts_px = denormalize_contour(obj.mask, width=orig_w, height=orig_h, min_points=3)
                    except Exception:
                        pts_px = None

                if not pts_px or len(pts_px) < 3 or calculate_polygon_area(pts_px) < 0.5:
                    warnings.append(
                        f"policy_b_incomplete: Semantic region '{obj.label}' has invalid/degenerate contour; annotation dropped per Policy B"
                    )
                    continue

                cvat_mask_data = obj.cvat_mask
                if not cvat_mask_data and obj.mask:
                    mask_dict = polygon_to_cvat_mask(obj.mask, width=orig_w, height=orig_h)
                    if mask_dict and "mask" in mask_dict:
                        cvat_mask_data = mask_dict["mask"]

                if not cvat_mask_data:
                    warnings.append(
                        f"policy_b_incomplete: Semantic region '{obj.label}' failed raster mask derivation; annotation dropped per Policy B"
                    )
                    continue

                reg_count += 1
                group_id = obj.group_id if (obj.group_id is not None and isinstance(obj.group_id, int)) else next_group_id
                next_group_id = max(next_group_id + 1, group_id + 1)

                flat_pts = [round(float(c), 2) for pt in pts_px for c in pt]
                poly_shape = {
                    "label": obj.label,
                    "type": "polygon",
                    "points": flat_pts,
                    "group_id": group_id,
                }
                if conf_str is not None:
                    poly_shape["confidence"] = conf_str

                mask_shape = {
                    "label": obj.label,
                    "type": "mask",
                    "mask": cvat_mask_data,
                    "points": flat_pts,
                    "group_id": group_id,
                }
                if conf_str is not None:
                    mask_shape["confidence"] = conf_str

                cvat_shapes.append(poly_shape)
                cvat_shapes.append(mask_shape)

            # POLICY C: POLYLINE ONLY (7 lane labels)
            # All 7 lane labels must emit POLYLINE ONLY (including lane/crosswalk).
            # Remove all auto polygon/mask fallbacks. If polyline extraction fails,
            # drop the annotation and emit warning lane_polyline_failed.
            elif policy == POLICY_POLYLINE:
                if lane_count >= MAX_LANES:
                    warnings.append(f"limit_exceeded: Exceeded MAX_LANES ({MAX_LANES}); dropped {obj.label}")
                    continue

                if has_valid_box:
                    warnings.append(f"lane_box_suppressed: Suppressed box for lane marking '{obj.label}'")

                if not obj.mask:
                    warnings.append(
                        f"lane_polyline_failed: Lane marking '{obj.label}' missing contour for polyline extraction; dropped per Policy C"
                    )
                    continue

                lane_shape = lane_shape_pipeline(
                    label=obj.label,
                    contour=obj.mask,
                    width=orig_w,
                    height=orig_h,
                    confidence=effective_conf,
                    preferred_geometry="polyline",
                    allow_fallback=False,
                )

                if lane_shape is not None and lane_shape.get("type") == "polyline":
                    lane_count += 1
                    cvat_shapes.append(lane_shape)
                else:
                    warnings.append(
                        f"lane_polyline_failed: Lane marking '{obj.label}' polyline centerline extraction failed; dropped per Policy C"
                    )
                    continue

    # 8b. Strict Runtime Output Validation
    if active_mode == MODE_FULL_31 and taxonomy is not None:
        cvat_shapes, val_warnings = validate_cvat_output_shapes(cvat_shapes, taxonomy=taxonomy)
        warnings.extend(val_warnings)

    # 9. Store AI prediction baseline for future human correction reconciliation
    if enable_feedback and f_db is not None and f_db.is_enabled() and image_hash:
        try:
            f_db.save_prediction_baseline(
                image_hash=image_hash,
                shapes=cvat_shapes,
                model=active_model,
                mode=active_mode,
                task_id=task_id,
                job_id=job_id,
                frame_index=frame_index,
                perceptual_hash=perceptual_hash,
            )
        except Exception as e:
            warnings.append(f"feedback_baseline_warning: {e}")

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
        mode=active_mode,
        warnings=warnings,
        image_hash=image_hash,
        rules_injected=rules_injected,
        visual_examples_used=visual_examples_used,
        text_rules_used=text_rules_used,
    )
