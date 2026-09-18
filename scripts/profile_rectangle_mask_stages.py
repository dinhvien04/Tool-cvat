"""Detailed Profiling Script for Detector #1: 9Router Rectangle + Mask (Policy A).

Measures and breaks down the exact timing of each stage in annotate_image() for rectangle_mask:
1. Image loading and preprocessing
2. Feedback DB query / retrieval
3. Model resolution / capability checks
4. 9Router API call latency
5. JSON parsing
6. Shape pairing / geometry processing (bounding box + mask pairing with group_id)
7. CVAT output validation
8. Feedback storage
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.client import NineRouterClient, VisionResponse
from app.config import DEFAULT_MAX_IMAGE_SIZE
from app.feedback import FeedbackDatabase, compute_image_hash, compute_perceptual_hash
from app.image_ops import image_to_data_url, load_image, resize_image_if_needed
from app.parser import ParsedObject, ParseResult, clean_json_string, parse_and_validate
from app.retrieval import CorrectionRetrievalEngine
from core.geometry import polygon_to_cvat_mask
from core.taxonomy import (
    BOX_MASK_LABELS,
    POLICY_BOX_MASK,
    Taxonomy,
    validate_detector_a_shapes,
)
from core.vision_contract import (
    MODE_RECTANGLE_MASK,
    build_user_prompt,
)


def profile_stage1_image_preprocessing(
    image_source: Any,
    labels: List[str],
    mode: str,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> Tuple[Dict[str, float], Image.Image, Image.Image, str, str]:
    """Breakdown of Stage 1: Image loading and preprocessing."""
    t_start = time.perf_counter()

    # 1.1 Load image
    t0 = time.perf_counter()
    pil_image = load_image(image_source)
    orig_w, orig_h = pil_image.size
    t_load = time.perf_counter() - t0

    # 1.2 Resize if needed
    t0 = time.perf_counter()
    send_image, was_resized, (send_w, send_h) = resize_image_if_needed(
        pil_image, max_size=max_size
    )
    t_resize = time.perf_counter() - t0

    # 1.3 Data URL encoding (JPEG quality 90, optimize=True, Base64)
    t0 = time.perf_counter()
    data_url = image_to_data_url(send_image, format="JPEG", quality=90)
    t_encode = time.perf_counter() - t0

    # 1.4 Prompt building
    t0 = time.perf_counter()
    prompt = build_user_prompt(allowed_labels=labels, mode=mode)
    t_prompt = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    timings = {
        "load_image_sec": t_load,
        "resize_sec": t_resize,
        "data_url_encode_sec": t_encode,
        "build_prompt_sec": t_prompt,
        "total_preprocessing_sec": t_total,
    }
    return timings, pil_image, send_image, data_url, prompt


def profile_stage2_feedback_retrieval(
    pil_image: Image.Image,
    labels: List[str],
    policy: str,
) -> Tuple[Dict[str, float], Optional[str], Optional[str], List[str], List[Dict[str, Any]], FeedbackDatabase]:
    """Breakdown of Stage 2: Feedback DB query / retrieval."""
    t_start = time.perf_counter()

    # 2.1 SHA-256 raw RGB image hash
    t0 = time.perf_counter()
    try:
        image_hash = compute_image_hash(pil_image)
    except Exception:
        image_hash = None
    t_img_hash = time.perf_counter() - t0

    # 2.2 Perceptual dHash
    t0 = time.perf_counter()
    try:
        perceptual_hash = compute_perceptual_hash(pil_image)
    except Exception:
        perceptual_hash = None
    t_p_hash = time.perf_counter() - t0

    # 2.3 FeedbackDatabase instantiation & schema init
    t0 = time.perf_counter()
    f_db = FeedbackDatabase()
    t_db_init = time.perf_counter() - t0

    # 2.4 Retrieval engine (rules + few-shot selection + visual crop loading)
    t0 = time.perf_counter()
    rules_injected: List[str] = []
    visual_examples: List[Dict[str, Any]] = []
    if f_db.is_enabled():
        retrieval_engine = CorrectionRetrievalEngine(db=f_db)
        retrieval_res = retrieval_engine.retrieve(candidate_labels=labels, policy=policy)
        rules_injected = retrieval_res.rules
        visual_examples = retrieval_res.visual_examples
    t_retrieval = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    timings = {
        "image_hash_sha256_sec": t_img_hash,
        "perceptual_hash_dhash_sec": t_p_hash,
        "feedback_db_init_sec": t_db_init,
        "retrieval_query_sec": t_retrieval,
        "total_feedback_retrieval_sec": t_total,
    }
    return timings, image_hash, perceptual_hash, rules_injected, visual_examples, f_db


def profile_stage3_model_resolution(
    client: NineRouterClient,
    explicit_model: Optional[str] = None,
) -> Dict[str, float]:
    """Breakdown of Stage 3: Model resolution / capability checks."""
    timings = {}

    # 3.1 Resolving with explicit model (cached / existence check)
    if explicit_model:
        t0 = time.perf_counter()
        _ = client.resolve_segmentation_model(explicit_model, probe=False)
        timings["resolve_with_explicit_model_sec"] = time.perf_counter() - t0

    # 3.2 Resolving without model (dynamic discovery via HTTP GET /v1/models)
    t0 = time.perf_counter()
    _ = client.resolve_segmentation_model(None, probe=False)
    timings["resolve_dynamic_no_model_sec"] = time.perf_counter() - t0

    # 3.3 Raw get_vision_models call
    t0 = time.perf_counter()
    _ = client.get_vision_models()
    timings["get_vision_models_http_sec"] = time.perf_counter() - t0

    return timings


def profile_stage4_api_call(
    client: NineRouterClient,
    model: str,
    data_url: str,
    prompt: str,
    mode: str,
    visual_examples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, float], VisionResponse]:
    """Stage 4: 9Router API call latency."""
    t0 = time.perf_counter()
    vision_resp = client.send_vision_request(
        model=model,
        image_bytes_or_b64=data_url,
        prompt=prompt,
        temperature=0.0,
        max_tokens=4096,
        visual_examples=visual_examples if visual_examples else None,
        mode=mode,
        timeout=120.0,
    )
    t_api = time.perf_counter() - t0

    usage = vision_resp.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    tps = (completion_tokens / t_api) if (t_api > 0 and completion_tokens > 0) else 0.0

    timings = {
        "api_call_latency_sec": t_api,
        "reported_duration_sec": vision_resp.duration_seconds,
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "tokens_per_sec": tps,
    }
    return timings, vision_resp


def profile_stage5_json_parsing(
    raw_response_text: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Stage 5: JSON parsing."""
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    cleaned = clean_json_string(raw_response_text)
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    data = json.loads(cleaned)
    t_loads = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    timings = {
        "clean_json_regex_sec": t_clean,
        "json_loads_sec": t_loads,
        "total_json_parsing_sec": t_total,
    }
    return timings, data


def profile_stage6_geometry_and_shape_pairing(
    raw_response_text: str,
    labels: List[str],
    image_width: int,
    image_height: int,
    mode: str,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[ParsedObject]]:
    """Stage 6: Shape pairing / geometry processing (bounding box + mask pairing with group_id)."""
    t_start = time.perf_counter()

    # 6.1 Parse and validate into ParsedObjects (including polygon_to_cvat_mask)
    t0 = time.perf_counter()
    parse_result: ParseResult = parse_and_validate(
        raw_response=raw_response_text,
        allowed_labels=labels,
        image_width=image_width,
        image_height=image_height,
        strict=False,
        mode=mode,
    )
    t_parse_validate = time.perf_counter() - t0

    # 6.2 Measure raster mask generation specifically
    t_raster_total = 0.0
    items_to_process = parse_result.all_items if hasattr(parse_result, "all_items") else parse_result.objects
    for obj in items_to_process:
        if obj.instance_mask or obj._legacy_mask:
            c = obj.instance_mask or obj._legacy_mask
            t_m0 = time.perf_counter()
            _ = polygon_to_cvat_mask(c, width=image_width, height=image_height)
            t_raster_total += (time.perf_counter() - t_m0)

    # 6.3 Policy A Shape Pairing (Rectangle + Mask with shared group_id)
    t0 = time.perf_counter()
    cvat_shapes: List[Dict[str, Any]] = []
    filtered_objects: List[ParsedObject] = []
    inst_count = 0
    next_group_id = 1
    used_group_ids = set()

    for obj in items_to_process:
        if obj.confidence is not None and threshold > 0.0 and obj.confidence < threshold:
            continue

        has_valid_box = (
            obj.pixel_box is not None
            and len(obj.pixel_box) == 4
            and (obj.pixel_box[2] > obj.pixel_box[0])
            and (obj.pixel_box[3] > obj.pixel_box[1])
        )
        has_valid_mask = obj.cvat_mask is not None and len(obj.cvat_mask) > 4

        # Policy A strictly drops incomplete detections
        if not (has_valid_box and has_valid_mask):
            continue

        inst_count += 1
        gid = next_group_id
        next_group_id += 1
        used_group_ids.add(gid)

        conf_str = str(round(float(obj.confidence), 2)) if obj.confidence is not None else None

        box_shape = {
            "label": obj.label,
            "points": [round(float(p), 2) for p in obj.pixel_box],
            "type": "rectangle",
            "group_id": gid,
        }
        if conf_str is not None:
            box_shape["confidence"] = conf_str

        mask_shape = {
            "label": obj.label,
            "type": "mask",
            "mask": obj.cvat_mask,
            "group_id": gid,
        }
        if obj.pixel_polygon:
            mask_shape["points"] = [round(float(c), 2) for pt in obj.pixel_polygon for c in pt]
        else:
            mask_shape["points"] = [round(float(p), 2) for p in obj.pixel_box]

        if conf_str is not None:
            mask_shape["confidence"] = conf_str

        cvat_shapes.append(box_shape)
        cvat_shapes.append(mask_shape)
        filtered_objects.append(obj)

    t_pairing = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    timings = {
        "parse_and_validate_sec": t_parse_validate,
        "mask_raster_generation_sec": t_raster_total,
        "pairing_and_grouping_sec": t_pairing,
        "total_geometry_pairing_sec": t_total,
    }
    return timings, cvat_shapes, filtered_objects


def profile_stage7_cvat_output_validation(
    cvat_shapes: List[Dict[str, Any]],
    taxonomy: Taxonomy,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[str]]:
    """Stage 7: CVAT output validation."""
    t0 = time.perf_counter()
    validated_shapes, warnings = validate_detector_a_shapes(
        cvat_shapes, taxonomy=taxonomy
    )
    t_val = time.perf_counter() - t0

    timings = {
        "cvat_output_validation_sec": t_val,
    }
    return timings, validated_shapes, warnings


def profile_stage8_feedback_storage(
    f_db: FeedbackDatabase,
    image_hash: str,
    perceptual_hash: Optional[str],
    shapes: List[Dict[str, Any]],
    model: str,
    mode: str,
) -> Dict[str, float]:
    """Stage 8: Feedback baseline storage."""
    t0 = time.perf_counter()
    _ = f_db.save_prediction_baseline(
        image_hash=image_hash,
        shapes=shapes,
        model=model,
        mode=mode,
        perceptual_hash=perceptual_hash,
    )
    t_storage = time.perf_counter() - t0

    timings = {
        "feedback_sqlite_storage_sec": t_storage,
    }
    return timings


def run_full_pipeline_profile(
    image_source: Any,
    explicit_model: str = "ag/gemini-3.8-flash-high",
    base_url: str = "http://127.0.0.1:20128",
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Run comprehensive profiling across all 8 stages."""
    labels = list(BOX_MASK_LABELS)
    mode = MODE_RECTANGLE_MASK
    policy = POLICY_BOX_MASK
    taxonomy = Taxonomy()

    client = NineRouterClient(base_url=base_url)

    print("=" * 70)
    print(f"PROFILING DETECTOR #1: 9Router Rectangle + Mask ({mode})")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")
    print(f"Base URL: {base_url}")
    print(f"Model: {explicit_model}")
    print("=" * 70)

    # 1. Check model resolution latency in isolation
    print("\n[Stage 3: Model Resolution / Capability Checks]")
    model_res_timings = profile_stage3_model_resolution(client, explicit_model=explicit_model)
    for k, v in model_res_timings.items():
        print(f"  {k}: {v * 1000:.2f} ms")

    all_run_timings = []
    sample_response = None

    for run_idx in range(num_runs):
        print(f"\n>>> Running End-to-End Profile Iteration {run_idx + 1}/{num_runs} <<<")
        run_record: Dict[str, Any] = {}

        # Stage 1: Image loading and preprocessing
        s1_timings, pil_image, send_image, data_url, prompt = profile_stage1_image_preprocessing(
            image_source=image_source,
            labels=labels,
            mode=mode,
            max_size=DEFAULT_MAX_IMAGE_SIZE,
        )
        run_record["stage1"] = s1_timings
        orig_w, orig_h = pil_image.size
        send_w, send_h = send_image.size
        print(f"Stage 1 (Image Loading & Preprocessing): {s1_timings['total_preprocessing_sec']*1000:.2f} ms")
        print(f"  - load_image: {s1_timings['load_image_sec']*1000:.2f} ms")
        print(f"  - resize ({orig_w}x{orig_h} -> {send_w}x{send_h}): {s1_timings['resize_sec']*1000:.2f} ms")
        print(f"  - JPEG/Base64 encode: {s1_timings['data_url_encode_sec']*1000:.2f} ms (payload len: {len(data_url)} chars)")
        print(f"  - build_prompt: {s1_timings['build_prompt_sec']*1000:.2f} ms")

        # Stage 2: Feedback DB query / retrieval
        s2_timings, img_hash, p_hash, rules, visual_examples, f_db = profile_stage2_feedback_retrieval(
            pil_image=pil_image,
            labels=labels,
            policy=policy,
        )
        run_record["stage2"] = s2_timings
        print(f"Stage 2 (Feedback DB Query & Retrieval): {s2_timings['total_feedback_retrieval_sec']*1000:.2f} ms")
        print(f"  - compute_image_hash (SHA-256): {s2_timings['image_hash_sha256_sec']*1000:.2f} ms")
        print(f"  - compute_perceptual_hash (dHash): {s2_timings['perceptual_hash_dhash_sec']*1000:.2f} ms")
        print(f"  - FeedbackDatabase __init__ & _init_db: {s2_timings['feedback_db_init_sec']*1000:.2f} ms")
        print(f"  - retrieval engine queries: {s2_timings['retrieval_query_sec']*1000:.2f} ms (rules={len(rules)}, visual_crops={len(visual_examples)})")

        # Stage 4: 9Router API call latency
        s4_timings, vision_resp = profile_stage4_api_call(
            client=client,
            model=explicit_model,
            data_url=data_url,
            prompt=prompt,
            mode=mode,
            visual_examples=visual_examples,
        )
        run_record["stage4"] = s4_timings
        sample_response = vision_resp.content
        print(f"Stage 4 (9Router API Call Latency): {s4_timings['api_call_latency_sec']:.3f} s")
        print(f"  - Tokens: Prompt={s4_timings['prompt_tokens']:.0f}, Completion={s4_timings['completion_tokens']:.0f}, Total={s4_timings['total_tokens']:.0f}")
        print(f"  - Gen speed: {s4_timings['tokens_per_sec']:.2f} tokens/sec")
        print(f"  - Raw response bytes: {len(sample_response)} chars")

        # Stage 5: JSON parsing
        s5_timings, parsed_json = profile_stage5_json_parsing(sample_response)
        run_record["stage5"] = s5_timings
        print(f"Stage 5 (JSON Parsing): {s5_timings['total_json_parsing_sec']*1000:.2f} ms")
        print(f"  - clean_json_string: {s5_timings['clean_json_regex_sec']*1000:.2f} ms")
        print(f"  - json.loads: {s5_timings['json_loads_sec']*1000:.2f} ms")

        # Stage 6: Geometry & shape pairing
        s6_timings, cvat_shapes, filtered_objects = profile_stage6_geometry_and_shape_pairing(
            raw_response_text=sample_response,
            labels=labels,
            image_width=orig_w,
            image_height=orig_h,
            mode=mode,
            threshold=0.3,
        )
        run_record["stage6"] = s6_timings
        print(f"Stage 6 (Geometry & Shape Pairing): {s6_timings['total_geometry_pairing_sec']*1000:.2f} ms")
        print(f"  - parse_and_validate: {s6_timings['parse_and_validate_sec']*1000:.2f} ms")
        print(f"  - mask rasterization & flattening: {s6_timings['mask_raster_generation_sec']*1000:.2f} ms")
        print(f"  - pairing & group_id assignment: {s6_timings['pairing_and_grouping_sec']*1000:.2f} ms")
        print(f"  - output pairs: {len(cvat_shapes)//2} instances ({len(cvat_shapes)} shapes: {len([s for s in cvat_shapes if s['type']=='rectangle'])} rects, {len([s for s in cvat_shapes if s['type']=='mask'])} masks)")

        # Stage 7: CVAT output validation
        s7_timings, validated_shapes, val_warnings = profile_stage7_cvat_output_validation(
            cvat_shapes=cvat_shapes,
            taxonomy=taxonomy,
        )
        run_record["stage7"] = s7_timings
        print(f"Stage 7 (CVAT Output Validation): {s7_timings['cvat_output_validation_sec']*1000:.2f} ms")
        print(f"  - validated shapes: {len(validated_shapes)}, warnings: {len(val_warnings)}")

        # Stage 8: Feedback storage
        s8_timings = profile_stage8_feedback_storage(
            f_db=f_db,
            image_hash=img_hash or "test_hash",
            perceptual_hash=p_hash,
            shapes=validated_shapes,
            model=explicit_model,
            mode=mode,
        )
        run_record["stage8"] = s8_timings
        print(f"Stage 8 (Feedback Baseline Storage): {s8_timings['feedback_sqlite_storage_sec']*1000:.2f} ms")

        # Total pipeline latency
        total_time = (
            s1_timings["total_preprocessing_sec"]
            + s2_timings["total_feedback_retrieval_sec"]
            + s4_timings["api_call_latency_sec"]
            + s5_timings["total_json_parsing_sec"]
            + s6_timings["total_geometry_pairing_sec"]
            + s7_timings["cvat_output_validation_sec"]
            + s8_timings["feedback_sqlite_storage_sec"]
        )
        run_record["total_sec"] = total_time
        print(f"--> Total Pipeline Latency: {total_time:.3f} s ({total_time*1000:.1f} ms) <--")

        all_run_timings.append(run_record)

    # Compute averages across runs
    avg_profile = {
        "stage1_preprocessing_ms": sum(r["stage1"]["total_preprocessing_sec"] for r in all_run_timings) / num_runs * 1000,
        "stage2_feedback_retrieval_ms": sum(r["stage2"]["total_feedback_retrieval_sec"] for r in all_run_timings) / num_runs * 1000,
        "stage3_model_res_explicit_ms": model_res_timings.get("resolve_with_explicit_model_sec", 0.0) * 1000,
        "stage3_model_res_dynamic_ms": model_res_timings.get("resolve_dynamic_no_model_sec", 0.0) * 1000,
        "stage4_api_latency_s": sum(r["stage4"]["api_call_latency_sec"] for r in all_run_timings) / num_runs,
        "stage5_json_parsing_ms": sum(r["stage5"]["total_json_parsing_sec"] for r in all_run_timings) / num_runs * 1000,
        "stage6_geometry_pairing_ms": sum(r["stage6"]["total_geometry_pairing_sec"] for r in all_run_timings) / num_runs * 1000,
        "stage7_validation_ms": sum(r["stage7"]["cvat_output_validation_sec"] for r in all_run_timings) / num_runs * 1000,
        "stage8_feedback_storage_ms": sum(r["stage8"]["feedback_sqlite_storage_sec"] for r in all_run_timings) / num_runs * 1000,
        "total_latency_s": sum(r["total_sec"] for r in all_run_timings) / num_runs,
    }

    print("\n" + "=" * 70)
    print("AVERAGE TIMING SUMMARY ACROSS ALL RUNS")
    print("=" * 70)
    for k, v in avg_profile.items():
        if k.endswith("_s"):
            print(f"  {k:<35}: {v:.3f} s")
        else:
            print(f"  {k:<35}: {v:.2f} ms")

    return {
        "averages": avg_profile,
        "runs": all_run_timings,
        "stage3_isolation": model_res_timings,
    }


if __name__ == "__main__":
    test_img = Path("test.jpg")
    if not test_img.exists():
        # Create a synthetic image
        img = Image.new("RGB", (1280, 720), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        test_source = buf.getvalue()
    else:
        test_source = test_img.read_bytes()

    res = run_full_pipeline_profile(image_source=test_source, num_runs=2)
