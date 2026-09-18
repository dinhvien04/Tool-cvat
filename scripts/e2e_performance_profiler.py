"""Comprehensive End-to-End Performance Profiler for CVAT x 9Router Detectors.

Profiles and benchmarks:
1. Full CVAT request path (Client -> Traefik -> cvat_server Nginx -> Uvicorn -> Nuclio -> 9Router -> Remote LLM -> Response)
2. Direct Nuclio request path (Client -> Nuclio -> 9Router -> Remote LLM -> Response)
3. Direct 9Router request path (Client -> 9Router -> Remote LLM -> Response)
4. Exact stage latencies inside annotate_image() pipeline:
   - Image loading, decoding, resizing, JPEG encoding, Base64 data URL
   - Feedback DB lookup, hashing, and visual crop retrieval
   - Prompt building and token metrics
   - 9Router network transmission and remote LLM generation
   - JSON parsing, cleaning, and coordinate scaling
   - Geometry generation (mask rasterization vs linear polyline scaling)
   - CVAT shape pairing, validation, and feedback baseline storage
   - CVAT server frame provider and DetectionResultConverter overhead
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

import requests
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.client import NineRouterClient, VisionResponse
from app.config import DEFAULT_MAX_IMAGE_SIZE
from app.feedback import FeedbackDatabase, compute_image_hash, compute_perceptual_hash
from app.image_ops import image_to_data_url, load_image, resize_image_if_needed
from app.parser import (
    ParsedObject,
    ParseResult,
    clean_json_string,
    parse_and_validate,
)
from app.retrieval import CorrectionRetrievalEngine
from core.geometry import calculate_polygon_area, denormalize_contour, polygon_to_cvat_mask
from core.line_geometry import (
    denormalize_polyline,
    polyline_to_cvat_polyline,
)
from core.taxonomy import (
    BOX_MASK_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    Taxonomy,
    validate_cvat_output_shapes,
    validate_detector_a_shapes,
    validate_detector_b_shapes,
    validate_detector_c_shapes,
)
from core.vision_contract import (
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
    build_user_prompt,
)

IMAGE_PATH = "D:/tool-cvat/test.jpg"
CVAT_URL = "http://localhost:18080"
CVAT_TOKEN = "984527795df646f41ee018bab7b3e424dfaae443"
NINEROUTER_URL = "http://127.0.0.1:20128"
MODEL_NAME = "ag/gemini-3.8-flash-high"
JOB_ID = 15
FRAME_ID = 0

DETECTOR_CONFIGS = [
    {
        "name": "Polyline (Policy C)",
        "mode": MODE_POLYLINE,
        "policy": POLICY_POLYLINE,
        "cvat_id": "ninerouter-polyline",
        "port": 14313,
        "labels": list(POLYLINE_LABELS),
    },
    {
        "name": "Rectangle + Mask (Policy A)",
        "mode": MODE_RECTANGLE_MASK,
        "policy": POLICY_BOX_MASK,
        "cvat_id": "ninerouter-rectangle-mask",
        "port": 4659,
        "labels": list(BOX_MASK_LABELS),
    },
    {
        "name": "Polygon + Mask (Policy B)",
        "mode": MODE_POLYGON_MASK,
        "policy": POLICY_POLYGON_MASK,
        "cvat_id": "ninerouter-polygon-mask",
        "port": 4183,
        "labels": list(POLYGON_MASK_LABELS),
    },
]


def benchmark_cvat_endpoint(func_id: str) -> Dict[str, Any]:
    """Benchmark full request path via CVAT Traefik reverse proxy."""
    url = f"{CVAT_URL}/api/lambda/functions/{func_id}"
    headers = {
        "Authorization": f"Token {CVAT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"job": JOB_ID, "frame": FRAME_ID}

    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    elapsed = time.perf_counter() - t0

    result = {
        "status_code": resp.status_code,
        "elapsed_sec": elapsed,
        "shapes_count": 0,
        "shape_types": {},
    }
    if resp.status_code == 200:
        data = resp.json()
        shapes = data.get("shapes", [])
        result["shapes_count"] = len(shapes)
        for s in shapes:
            t = s.get("type", "unknown")
            result["shape_types"][t] = result["shape_types"].get(t, 0) + 1
    else:
        result["error"] = resp.text[:300]
    return result


def benchmark_nuclio_direct(port: int, image_b64: str) -> Dict[str, Any]:
    """Benchmark direct Nuclio container endpoint."""
    url = f"http://localhost:{port}"
    headers = {"Content-Type": "application/json"}
    payload = {"image": image_b64}

    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    elapsed = time.perf_counter() - t0

    result = {
        "status_code": resp.status_code,
        "elapsed_sec": elapsed,
        "shapes_count": 0,
        "shape_types": {},
    }
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            result["shapes_count"] = len(data)
            for s in data:
                t = s.get("type", "unknown")
                result["shape_types"][t] = result["shape_types"].get(t, 0) + 1
        elif isinstance(data, dict):
            shapes = data.get("shapes", [])
            result["shapes_count"] = len(shapes)
            for s in shapes:
                t = s.get("type", "unknown")
                result["shape_types"][t] = result["shape_types"].get(t, 0) + 1
    else:
        result["error"] = resp.text[:300]
    return result


def profile_isolated_stages(
    image_path: str,
    cfg: Dict[str, Any],
    client: NineRouterClient,
    f_db: FeedbackDatabase,
    taxonomy: Taxonomy,
) -> Dict[str, Any]:
    """Profile each sub-stage of inference in isolation."""
    mode = cfg["mode"]
    policy = cfg["policy"]
    labels = cfg["labels"]

    timings: Dict[str, float] = {}

    # Stage 1: Client Image Decoding, Resizing, JPEG Encoding, Base64 Data URL
    t0 = time.perf_counter()
    pil_image = load_image(image_path)
    orig_w, orig_h = pil_image.size
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    send_image, was_resized, (send_w, send_h) = resize_image_if_needed(
        pil_image, max_size=DEFAULT_MAX_IMAGE_SIZE
    )
    t_resize = time.perf_counter() - t0

    t0 = time.perf_counter()
    data_url = image_to_data_url(send_image, format="JPEG", quality=90)
    t_encode = time.perf_counter() - t0

    timings["stage1_image_decode_sec"] = t_load
    timings["stage1_image_resize_sec"] = t_resize
    timings["stage1_jpeg_base64_encode_sec"] = t_encode
    timings["stage1_total_preprocessing_sec"] = t_load + t_resize + t_encode

    # Stage 2: Feedback DB Lookup, Hashing, and Crop Retrieval
    t0 = time.perf_counter()
    img_hash = compute_image_hash(pil_image)
    t_sha256 = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_hash = compute_perceptual_hash(pil_image)
    t_dhash = time.perf_counter() - t0

    t0 = time.perf_counter()
    rules_injected: List[str] = []
    visual_examples: List[Dict[str, Any]] = []
    if f_db.is_enabled():
        retrieval_engine = CorrectionRetrievalEngine(db=f_db)
        retrieval_res = retrieval_engine.retrieve(candidate_labels=labels, policy=policy)
        rules_injected = retrieval_res.rules
        visual_examples = retrieval_res.visual_examples
    t_retrieval = time.perf_counter() - t0

    timings["stage2_sha256_hash_sec"] = t_sha256
    timings["stage2_perceptual_dhash_sec"] = t_dhash
    timings["stage2_db_retrieval_sec"] = t_retrieval
    timings["stage2_total_feedback_retrieval_sec"] = t_sha256 + t_dhash + t_retrieval

    # Stage 3: Prompt Building & Token Estimation
    t0 = time.perf_counter()
    prompt = build_user_prompt(allowed_labels=labels, mode=mode)
    if rules_injected:
        prompt += "\n\n" + "\n".join(rules_injected)
    t_prompt = time.perf_counter() - t0
    timings["stage3_prompt_build_sec"] = t_prompt

    # Stage 4: 9Router Network Transmission & Remote LLM Generation Time
    t0 = time.perf_counter()
    vision_resp: VisionResponse = client.send_vision_request(
        model=MODEL_NAME,
        image_bytes_or_b64=data_url,
        prompt=prompt,
        temperature=0.0,
        max_tokens=4096,
        visual_examples=visual_examples if visual_examples else None,
        mode=mode,
        timeout=180.0,
    )
    t_api = time.perf_counter() - t0
    timings["stage4_9router_api_call_sec"] = t_api
    timings["stage4_reported_duration_sec"] = vision_resp.duration_seconds

    usage = vision_resp.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    details = usage.get("completion_tokens_details", {}) or {}
    reasoning_tokens = details.get("reasoning_tokens", 0)
    gen_tokens = completion_tokens - reasoning_tokens if completion_tokens >= reasoning_tokens else completion_tokens
    tps = (completion_tokens / t_api) if (t_api > 0 and completion_tokens > 0) else 0.0

    token_metrics = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "generation_tokens": gen_tokens,
        "total_tokens": total_tokens,
        "generation_rate_tps": round(tps, 2),
    }

    raw_text = getattr(vision_resp, "content", getattr(vision_resp, "raw_content", ""))

    # Stage 5: JSON Parsing & Coordinate Scaling
    t0 = time.perf_counter()
    cleaned_json = clean_json_string(raw_text)
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        raw_parsed_data = json.loads(cleaned_json)
    except Exception:
        raw_parsed_data = {}
    t_loads = time.perf_counter() - t0

    t0 = time.perf_counter()
    parse_result: ParseResult = parse_and_validate(
        raw_response=raw_text,
        allowed_labels=labels,
        image_width=orig_w,
        image_height=orig_h,
        strict=False,
        mode=mode,
    )
    t_parse_val = time.perf_counter() - t0

    timings["stage5_json_regex_clean_sec"] = t_clean
    timings["stage5_json_loads_sec"] = t_loads
    timings["stage5_parse_validate_scaling_sec"] = t_parse_val
    timings["stage5_total_parsing_scaling_sec"] = t_clean + t_loads + t_parse_val

    # Stage 6: Geometry Generation & Shape Pairing
    items_to_process = parse_result.all_items if hasattr(parse_result, "all_items") else parse_result.objects
    t0 = time.perf_counter()
    t_raster_total = 0.0
    t_polyline_scaling_total = 0.0

    cvat_shapes: List[Dict[str, Any]] = []
    next_group_id = 1

    for obj in items_to_process:
        if mode == MODE_RECTANGLE_MASK:
            has_valid_box = (
                obj.pixel_box is not None
                and len(obj.pixel_box) == 4
                and (obj.pixel_box[2] > obj.pixel_box[0])
                and (obj.pixel_box[3] > obj.pixel_box[1])
            )
            has_valid_mask = obj.cvat_mask is not None and len(obj.cvat_mask) > 4
            if not (has_valid_box and has_valid_mask):
                continue
            gid = next_group_id
            next_group_id += 1

            # Measure raster mask generation
            t_m0 = time.perf_counter()
            _ = polygon_to_cvat_mask(obj.mask or obj.pixel_polygon, width=orig_w, height=orig_h)
            t_raster_total += (time.perf_counter() - t_m0)

            box_shape = {
                "label": obj.label,
                "points": [round(float(p), 2) for p in obj.pixel_box],
                "type": "rectangle",
                "group_id": gid,
            }
            mask_shape = {
                "label": obj.label,
                "type": "mask",
                "mask": obj.cvat_mask,
                "points": [round(float(p), 2) for p in obj.pixel_box],
                "group_id": gid,
            }
            cvat_shapes.extend([box_shape, mask_shape])

        elif mode == MODE_POLYGON_MASK:
            pts_px = obj.pixel_polygon
            if not pts_px or len(pts_px) < 3 or calculate_polygon_area(pts_px) < 0.5:
                continue
            t_m0 = time.perf_counter()
            mask_dict = polygon_to_cvat_mask(obj.mask or pts_px, width=orig_w, height=orig_h)
            t_raster_total += (time.perf_counter() - t_m0)
            if not mask_dict or "mask" not in mask_dict:
                continue

            gid = next_group_id
            next_group_id += 1
            flat_pts = [round(float(c), 2) for pt in pts_px for c in pt]
            poly_shape = {
                "label": obj.label,
                "type": "polygon",
                "points": flat_pts,
                "group_id": gid,
            }
            mask_shape = {
                "label": obj.label,
                "type": "mask",
                "mask": mask_dict["mask"],
                "points": flat_pts,
                "group_id": gid,
            }
            cvat_shapes.extend([poly_shape, mask_shape])

        elif mode == MODE_POLYLINE:
            t_l0 = time.perf_counter()
            if getattr(obj, "pixel_polyline", None) and len(obj.pixel_polyline) >= 2:
                flat_pts = [round(float(c), 2) for pt in obj.pixel_polyline for c in pt]
                lane_shape = {
                    "type": "polyline",
                    "label": obj.label,
                    "points": flat_pts,
                }
            else:
                polyline_pts = getattr(obj, "lane_polyline", None) or getattr(obj, "polyline", None) or obj.mask
                lane_shape = polyline_to_cvat_polyline(
                    polyline=polyline_pts,
                    width=orig_w,
                    height=orig_h,
                    label=obj.label,
                    confidence=obj.confidence,
                )
            t_polyline_scaling_total += (time.perf_counter() - t_l0)
            if lane_shape and lane_shape.get("type") == "polyline":
                cvat_shapes.append(lane_shape)

    t_geom_total = time.perf_counter() - t0
    timings["stage6_mask_rasterization_sec"] = t_raster_total
    timings["stage6_polyline_scaling_sec"] = t_polyline_scaling_total
    timings["stage6_total_geometry_pairing_sec"] = t_geom_total

    # Stage 7: CVAT Output Validation & Baseline Feedback Storage
    t0 = time.perf_counter()
    validated_shapes, warnings = validate_cvat_output_shapes(
        cvat_shapes, taxonomy=taxonomy, policy=policy
    )
    t_val = time.perf_counter() - t0

    t0 = time.perf_counter()
    if f_db.is_enabled() and img_hash:
        f_db.save_prediction_baseline(
            image_hash=img_hash,
            shapes=validated_shapes,
            model=MODEL_NAME,
            mode=mode,
            perceptual_hash=p_hash,
        )
    t_storage = time.perf_counter() - t0

    timings["stage7_taxonomy_validation_sec"] = t_val
    timings["stage7_feedback_sqlite_storage_sec"] = t_storage
    timings["stage7_total_validation_storage_sec"] = t_val + t_storage

    # Stage 8: Serialization to JSON
    t0 = time.perf_counter()
    resp_json_bytes = json.dumps(validated_shapes).encode("utf-8")
    t_json_serial = time.perf_counter() - t0
    timings["stage8_json_serialization_sec"] = t_json_serial
    timings["stage8_response_bytes"] = len(resp_json_bytes)

    # Compute Total Internal annotate_image() Latency
    internal_total = (
        timings["stage1_total_preprocessing_sec"]
        + timings["stage2_total_feedback_retrieval_sec"]
        + timings["stage3_prompt_build_sec"]
        + timings["stage4_9router_api_call_sec"]
        + timings["stage5_total_parsing_scaling_sec"]
        + timings["stage6_total_geometry_pairing_sec"]
        + timings["stage7_total_validation_storage_sec"]
        + timings["stage8_json_serialization_sec"]
    )
    timings["total_internal_service_sec"] = internal_total

    shape_counts = {}
    for s in validated_shapes:
        st = s.get("type", "unknown")
        shape_counts[st] = shape_counts.get(st, 0) + 1

    return {
        "mode": mode,
        "policy": policy,
        "timings": timings,
        "token_metrics": token_metrics,
        "shapes_count": len(validated_shapes),
        "shape_types": shape_counts,
        "raw_response_len": len(raw_text),
        "warnings_count": len(warnings),
    }


def main():
    print("=" * 80)
    print("END-TO-END PERFORMANCE PROFILING: TOOL-CVAT x 9ROUTER")
    print(f"CVAT URL: {CVAT_URL} (Job {JOB_ID}, Frame {FRAME_ID})")
    print(f"9Router URL: {NINEROUTER_URL} (Model: {MODEL_NAME})")
    print(f"Test Image: {IMAGE_PATH} (1280x720 RGB)")
    print("=" * 80)

    # Prepare base64 test image for direct Nuclio benchmarking
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = NineRouterClient(base_url=NINEROUTER_URL, timeout=180.0)
    f_db = FeedbackDatabase()
    taxonomy = Taxonomy()

    full_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image": {
            "path": IMAGE_PATH,
            "width": 1280,
            "height": 720,
            "size_bytes": os.path.getsize(IMAGE_PATH),
        },
        "detectors": {},
    }

    for cfg in DETECTOR_CONFIGS:
        name = cfg["name"]
        print(f"\n{'#' * 80}")
        print(f"BENCHMARKING: {name}")
        print(f"Mode: {cfg['mode']}, Policy: {cfg['policy']}, CVAT ID: {cfg['cvat_id']}, Port: {cfg['port']}")
        print(f"{'#' * 80}")

        # 1. Full CVAT Endpoint
        print("\n[1/3] Benchmarking Full Request Path via CVAT Traefik (http://localhost:18080)...", flush=True)
        cvat_res = benchmark_cvat_endpoint(cfg["cvat_id"])
        print(f"  CVAT Full Path Latency: {cvat_res['elapsed_sec']:.2f} s | Status: {cvat_res['status_code']} | Shapes: {cvat_res['shapes_count']} {cvat_res['shape_types']}")

        # 2. Direct Nuclio Endpoint
        print(f"\n[2/3] Benchmarking Direct Nuclio Endpoint (http://localhost:{cfg['port']})...", flush=True)
        nuclio_res = benchmark_nuclio_direct(cfg["port"], img_b64)
        print(f"  Direct Nuclio Latency: {nuclio_res['elapsed_sec']:.2f} s | Status: {nuclio_res['status_code']} | Shapes: {nuclio_res['shapes_count']} {nuclio_res['shape_types']}")

        # 3. Isolated Stages Profiling
        print("\n[3/3] Running Deep Micro-Stage Profiling...", flush=True)
        stage_res = profile_isolated_stages(IMAGE_PATH, cfg, client, f_db, taxonomy)

        # Print stage timings summary
        t = stage_res["timings"]
        tm = stage_res["token_metrics"]
        print("\n  --- Micro-Stage Latencies ---")
        print(f"  1. Preprocessing (decode+resize+encode): {t['stage1_total_preprocessing_sec']*1000:.2f} ms")
        print(f"     - Image decode (PIL):                 {t['stage1_image_decode_sec']*1000:.2f} ms")
        print(f"     - Aspect resize check:               {t['stage1_image_resize_sec']*1000:.2f} ms")
        print(f"     - JPEG quality 90 + Base64 encode:   {t['stage1_jpeg_base64_encode_sec']*1000:.2f} ms")
        print(f"  2. Feedback Retrieval:                  {t['stage2_total_feedback_retrieval_sec']*1000:.2f} ms")
        print(f"     - SHA-256 fingerprint:               {t['stage2_sha256_hash_sec']*1000:.2f} ms")
        print(f"     - Perceptual dHash:                  {t['stage2_perceptual_dhash_sec']*1000:.2f} ms")
        print(f"     - SQLite Rules & Crops lookup:       {t['stage2_db_retrieval_sec']*1000:.2f} ms")
        print(f"  3. Prompt Assembly:                     {t['stage3_prompt_build_sec']*1000:.2f} ms")
        print(f"  4. 9Router Network & Remote LLM:        {t['stage4_9router_api_call_sec']:.2f} s")
        print(f"     - Prompt tokens:                     {tm['prompt_tokens']}")
        print(f"     - Completion tokens:                 {tm['completion_tokens']}")
        print(f"     - Thinking / Reasoning tokens:       {tm['reasoning_tokens']}")
        print(f"     - Generation tokens (JSON text):     {tm['generation_tokens']}")
        print(f"     - Generation speed:                  {tm['generation_rate_tps']} tokens/s")
        print(f"  5. Parsing & Coordinate Scaling:        {t['stage5_total_parsing_scaling_sec']*1000:.2f} ms")
        print(f"     - JSON regex strip & loads:          {(t['stage5_json_regex_clean_sec'] + t['stage5_json_loads_sec'])*1000:.2f} ms")
        print(f"     - Normalization to pixel scaling:    {t['stage5_parse_validate_scaling_sec']*1000:.2f} ms")
        print(f"  6. Geometry Generation:                 {t['stage6_total_geometry_pairing_sec']*1000:.2f} ms")
        print(f"     - Mask rasterization (RLE/PIL):      {t['stage6_mask_rasterization_sec']*1000:.2f} ms")
        print(f"     - Linear polyline scaling:           {t['stage6_polyline_scaling_sec']*1000:.2f} ms")
        print(f"  7. Strict Validation & Feedback Store:  {t['stage7_total_validation_storage_sec']*1000:.2f} ms")
        print(f"     - Taxonomy output validation:        {t['stage7_taxonomy_validation_sec']*1000:.2f} ms")
        print(f"     - SQLite prediction baseline store:  {t['stage7_feedback_sqlite_storage_sec']*1000:.2f} ms")
        print(f"  8. JSON Shape Serialization:            {t['stage8_json_serialization_sec']*1000:.2f} ms ({t['stage8_response_bytes']} bytes)")
        print(f"  TOTAL Internal Pipeline Latency:        {t['total_internal_service_sec']:.2f} s")

        # Network and Proxy Overhead Calculations
        cvat_overhead = max(0.0, cvat_res["elapsed_sec"] - nuclio_res["elapsed_sec"])
        nuclio_wrapper_overhead = max(0.0, nuclio_res["elapsed_sec"] - t["total_internal_service_sec"])

        print("\n  --- Request Hop Overhead Analysis ---")
        print(f"  Traefik + CVAT Nginx + Uvicorn + Frame Overhead: {cvat_overhead:.3f} s")
        print(f"  Nuclio Go Processor Event Wrapper Overhead:     {nuclio_wrapper_overhead:.3f} s")
        print(f"  9Router Remote LLM Generation Fraction:          {(t['stage4_9router_api_call_sec'] / cvat_res['elapsed_sec']) * 100:.1f}% of total CVAT request time")

        full_results["detectors"][cfg["cvat_id"]] = {
            "name": name,
            "cvat_benchmark": cvat_res,
            "nuclio_benchmark": nuclio_res,
            "stage_breakdown": stage_res,
            "overhead": {
                "cvat_proxy_and_frame_overhead_sec": round(cvat_overhead, 3),
                "nuclio_wrapper_overhead_sec": round(nuclio_wrapper_overhead, 3),
                "remote_llm_percentage": round((t["stage4_9router_api_call_sec"] / cvat_res["elapsed_sec"]) * 100, 1),
            }
        }

    out_file = Path("D:/tool-cvat/output/e2e_performance_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
