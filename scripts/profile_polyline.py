"""Comprehensive Profiling Script for 9Router Detectors with focus on Polyline.

Profiles each stage of annotate_image() in isolation:
1. Image Preprocessing
2. Feedback Retrieval
3. 9Router Vision Call Latency (including token usage and tokens/sec)
4. Coordinate Extraction & Validation
5. CVAT Output Validation (including baseline storage)
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any, Dict, List, Tuple
from PIL import Image

from app.client import NineRouterClient
from app.config import DEFAULT_MAX_IMAGE_SIZE
from app.feedback import FeedbackDatabase, compute_image_hash, compute_perceptual_hash
from app.image_ops import image_to_data_url, load_image, resize_image_if_needed
from app.parser import parse_and_validate
from app.retrieval import CorrectionRetrievalEngine
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
    MODE_BOX_MASK,
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
    build_user_prompt,
)


def profile_detector_pipeline(
    image_path: str,
    mode: str,
    labels: List[str],
    policy: str,
    runs: int = 3,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"PROFILING DETECTOR: mode={mode}, policy={policy}, labels_count={len(labels)}")
    print(f"=======================================================")

    client = NineRouterClient(base_url="http://127.0.0.1:20128", timeout=120.0)
    model = client.resolve_vision_model()
    print(f"Active Model: {model}")

    taxonomy = Taxonomy()
    f_db = FeedbackDatabase()

    results = []

    for run_idx in range(runs):
        print(f"\n--- Run {run_idx + 1}/{runs} ---")
        timings = {}

        # -------------------------------------------------------------
        # Stage 1: Image Preprocessing
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        pil_image = load_image(image_path)
        orig_w, orig_h = pil_image.size

        send_image, was_resized, (send_w, send_h) = resize_image_if_needed(
            pil_image, max_size=DEFAULT_MAX_IMAGE_SIZE
        )
        data_url = image_to_data_url(send_image, format="JPEG", quality=90)
        user_prompt = build_user_prompt(allowed_labels=labels, mode=mode)
        t1 = time.perf_counter()
        timings["stage1_preprocessing_sec"] = t1 - t0
        print(f"Stage 1 (Image Preprocessing): {timings['stage1_preprocessing_sec']*1000:.2f} ms")

        # -------------------------------------------------------------
        # Stage 2: Feedback Retrieval
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            image_hash = compute_image_hash(pil_image)
        except Exception:
            image_hash = None
        try:
            perceptual_hash = compute_perceptual_hash(pil_image)
        except Exception:
            perceptual_hash = None

        rules_injected = []
        visual_examples = []
        if f_db.is_enabled():
            retrieval_engine = CorrectionRetrievalEngine(db=f_db)
            retrieval_res = retrieval_engine.retrieve(candidate_labels=labels, policy=policy)
            if retrieval_res.prompt_extension:
                user_prompt += "\n\n" + retrieval_res.prompt_extension
            rules_injected = retrieval_res.rules
            visual_examples = retrieval_res.visual_examples
        t1 = time.perf_counter()
        timings["stage2_feedback_retrieval_sec"] = t1 - t0
        print(f"Stage 2 (Feedback Retrieval): {timings['stage2_feedback_retrieval_sec']*1000:.2f} ms")

        # -------------------------------------------------------------
        # Stage 3: 9Router Vision Call Latency
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        vision_resp = client.send_vision_request(
            model=model,
            image_bytes_or_b64=data_url,
            prompt=user_prompt,
            temperature=0.0,
            max_tokens=4096,
            visual_examples=visual_examples if visual_examples else None,
            mode=mode,
        )
        t1 = time.perf_counter()
        timings["stage3_vision_call_sec"] = t1 - t0
        raw_text = vision_resp.content
        usage = vision_resp.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        gen_speed = (completion_tokens / timings["stage3_vision_call_sec"]) if (timings["stage3_vision_call_sec"] > 0 and completion_tokens > 0) else 0.0

        print(f"Stage 3 (Vision Call Latency): {timings['stage3_vision_call_sec']:.3f} s")
        print(f"  Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
        print(f"  Generation speed: {gen_speed:.2f} tokens/s")
        print(f"  Raw response length: {len(raw_text)} chars")

        # -------------------------------------------------------------
        # Stage 4: Coordinate Extraction & Validation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        parse_result = parse_and_validate(
            raw_response=raw_text,
            allowed_labels=labels,
            image_width=orig_w,
            image_height=orig_h,
            strict=False,
            mode=mode,
        )

        # Mode-specific object conversion to CVAT shapes
        cvat_shapes = []
        items_to_process = parse_result.all_items if hasattr(parse_result, "all_items") else parse_result.objects

        if mode == MODE_POLYLINE:
            for obj in items_to_process:
                if getattr(obj, "pixel_polyline", None) and len(obj.pixel_polyline) >= 2:
                    flat_pts = [round(float(c), 2) for pt in obj.pixel_polyline for c in pt]
                    cvat_shapes.append({
                        "type": "polyline",
                        "label": obj.label,
                        "points": flat_pts,
                        "confidence": str(round(float(obj.confidence), 2)) if obj.confidence is not None else "0.90",
                    })
        elif mode in (MODE_RECTANGLE_MASK, MODE_BOX_MASK):
            gid = 1
            for obj in items_to_process:
                if obj.pixel_box and obj.cvat_mask:
                    cvat_shapes.append({
                        "type": "rectangle",
                        "label": obj.label,
                        "points": [round(float(p), 2) for p in obj.pixel_box],
                        "group_id": gid,
                        "confidence": str(round(float(obj.confidence), 2)) if obj.confidence is not None else "0.90",
                    })
                    cvat_shapes.append({
                        "type": "mask",
                        "label": obj.label,
                        "mask": obj.cvat_mask,
                        "points": [round(float(c), 2) for pt in (obj.pixel_polygon or []) for c in pt] if obj.pixel_polygon else [round(float(p), 2) for p in obj.pixel_box],
                        "group_id": gid,
                        "confidence": str(round(float(obj.confidence), 2)) if obj.confidence is not None else "0.90",
                    })
                    gid += 1
        elif mode == MODE_POLYGON_MASK:
            gid = 1
            for obj in items_to_process:
                if obj.pixel_polygon and obj.cvat_mask:
                    flat_pts = [round(float(c), 2) for pt in obj.pixel_polygon for c in pt]
                    cvat_shapes.append({
                        "type": "polygon",
                        "label": obj.label,
                        "points": flat_pts,
                        "group_id": gid,
                        "confidence": str(round(float(obj.confidence), 2)) if obj.confidence is not None else "0.90",
                    })
                    cvat_shapes.append({
                        "type": "mask",
                        "label": obj.label,
                        "mask": obj.cvat_mask,
                        "points": flat_pts,
                        "group_id": gid,
                        "confidence": str(round(float(obj.confidence), 2)) if obj.confidence is not None else "0.90",
                    })
                    gid += 1

        t1 = time.perf_counter()
        timings["stage4_parsing_extraction_sec"] = t1 - t0
        print(f"Stage 4 (Parsing & Extraction): {timings['stage4_parsing_extraction_sec']*1000:.2f} ms")
        print(f"  Parsed items: {len(items_to_process)}, Formatted shapes: {len(cvat_shapes)}")

        # -------------------------------------------------------------
        # Stage 5: CVAT Output Validation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        validated_shapes, val_warnings = validate_cvat_output_shapes(
            cvat_shapes, taxonomy=taxonomy, policy=policy
        )
        # Store AI prediction baseline
        if f_db.is_enabled() and image_hash:
            try:
                f_db.save_prediction_baseline(
                    image_hash=image_hash,
                    shapes=validated_shapes,
                    model=model,
                    mode=mode,
                    perceptual_hash=perceptual_hash,
                )
            except Exception as e:
                val_warnings.append(f"baseline_err: {e}")
        t1 = time.perf_counter()
        timings["stage5_cvat_validation_sec"] = t1 - t0
        print(f"Stage 5 (CVAT Output Validation & Baseline): {timings['stage5_cvat_validation_sec']*1000:.2f} ms")
        print(f"  Validated shapes: {len(validated_shapes)}, Warnings: {len(val_warnings)}")

        total_e2e = sum(timings.values())
        timings["total_e2e_sec"] = total_e2e
        print(f"Total Pipeline Duration: {total_e2e:.3f} s")

        run_data = {
            "timings": timings,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "generation_speed_tps": gen_speed,
            },
            "shapes_count": len(validated_shapes),
            "raw_response_len": len(raw_text),
            "warnings_count": len(val_warnings),
        }
        results.append(run_data)

    # Compute averages across runs
    avg_timings = {}
    for k in results[0]["timings"]:
        avg_timings[k] = sum(r["timings"][k] for r in results) / len(results)

    avg_tokens = {
        "prompt_tokens": sum(r["tokens"]["prompt_tokens"] for r in results) / len(results),
        "completion_tokens": sum(r["tokens"]["completion_tokens"] for r in results) / len(results),
        "total_tokens": sum(r["tokens"]["total_tokens"] for r in results) / len(results),
        "generation_speed_tps": sum(r["tokens"]["generation_speed_tps"] for r in results) / len(results),
    }

    return {
        "mode": mode,
        "policy": policy,
        "runs": len(results),
        "avg_timings": avg_timings,
        "avg_tokens": avg_tokens,
        "avg_shapes": sum(r["shapes_count"] for r in results) / len(results),
        "avg_raw_response_len": sum(r["raw_response_len"] for r in results) / len(results),
        "individual_runs": results,
    }


if __name__ == "__main__":
    image_path = "test.jpg"

    # Profile Polyline (Detector 3)
    poly_res = profile_detector_pipeline(
        image_path=image_path,
        mode=MODE_POLYLINE,
        labels=list(POLYLINE_LABELS),
        policy=POLICY_POLYLINE,
        runs=3,
    )

    # Profile Rectangle + Mask (Detector 1)
    rect_res = profile_detector_pipeline(
        image_path=image_path,
        mode=MODE_RECTANGLE_MASK,
        labels=list(BOX_MASK_LABELS),
        policy=POLICY_BOX_MASK,
        runs=2,
    )

    # Profile Polygon + Mask (Detector 2)
    poly_mask_res = profile_detector_pipeline(
        image_path=image_path,
        mode=MODE_POLYGON_MASK,
        labels=list(POLYGON_MASK_LABELS),
        policy=POLICY_POLYGON_MASK,
        runs=2,
    )

    comparison = {
        "polyline": poly_res,
        "rectangle_mask": rect_res,
        "polygon_mask": poly_mask_res,
    }

    print("\n\n" + "=" * 70)
    print("SUMMARY OF PROFILING RESULTS ACROSS 3 DETECTORS")
    print("=" * 70)
    for name, data in comparison.items():
        t = data["avg_timings"]
        tok = data["avg_tokens"]
        print(f"\nDetector: {name.upper()}")
        print(f"  Stage 1 (Image Preprocessing):           {t['stage1_preprocessing_sec']*1000:.2f} ms")
        print(f"  Stage 2 (Feedback Retrieval):            {t['stage2_feedback_retrieval_sec']*1000:.2f} ms")
        print(f"  Stage 3 (9Router Vision Call Latency):    {t['stage3_vision_call_sec']:.3f} s ({t['stage3_vision_call_sec']/t['total_e2e_sec']*100:.1f}% of total)")
        print(f"  Stage 4 (Coordinate Parsing & Geometry): {t['stage4_parsing_extraction_sec']*1000:.2f} ms")
        print(f"  Stage 5 (CVAT Output Validation):        {t['stage5_cvat_validation_sec']*1000:.2f} ms")
        print(f"  Total E2E Pipeline Duration:             {t['total_e2e_sec']:.3f} s")
        print(f"  Tokens: Prompt={tok['prompt_tokens']:.0f}, Completion={tok['completion_tokens']:.0f}, Total={tok['total_tokens']:.0f}")
        print(f"  Generation Speed:                        {tok['generation_speed_tps']:.2f} tokens/s")
        print(f"  Avg Shapes Emitted:                      {data['avg_shapes']:.1f}")
        print(f"  Avg Raw Response Length:                 {data['avg_raw_response_len']:.0f} chars")

    # Output JSON summary for machine verification
    with open("output/profiling_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("\nSaved full profiling comparison to output/profiling_comparison.json")
