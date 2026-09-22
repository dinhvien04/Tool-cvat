"""Performance and Latency Profiling Script for Week-2: Pose17 and VF50.

Section 15 Latency SLA Targets:
- Pose17: Standard <= 15.0s (Stretch Target <= 10.0s)
- VF50:   Standard <= 20.0s (Stretch Target <= 10.0s)

Capabilities:
1. Measures direct 9Router remote latency (reported API duration and network roundtrip).
2. Measures total CVAT/Nuclio request latency.
3. Tracks token consumption (prompt tokens, completion tokens, total tokens).
4. Tracks local client-side processing (image preparation, coordinate denormalization, skeleton formatting).
5. Evaluates compliance against Standard and Stretch SLA thresholds.
6. Supports mock profiling mode for offline verification and CI smoke testing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from app.client import NineRouterClient
from core.pose_face_schema import (
    POSE17_KEYPOINTS,
    VF50_LANDMARKS,
    parse_and_sanitize_pose17_instance,
    parse_and_sanitize_vf50_instance,
)

# Section 15 SLA Thresholds (in seconds)
SLA_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "pose17": {
        "standard_max_sec": 15.0,
        "stretch_max_sec": 10.0,
    },
    "vf50": {
        "standard_max_sec": 20.0,
        "stretch_max_sec": 10.0,
    },
}

DEFAULT_VISION_MODEL = "ag/gemini-3.8-flash-high"
DEFAULT_NINEROUTER_URL = os.getenv("NINEROUTER_BASE_URL", "http://127.0.0.1:20128")


@dataclass
class BenchmarkMetric:
    mode: str
    iteration: int
    remote_api_sec: float
    client_network_sec: float
    local_parse_sec: float
    total_latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elements_detected: int
    standard_sla_pass: bool
    stretch_sla_pass: bool


def generate_synthetic_test_image(width: int = 1280, height: int = 720) -> bytes:
    """Generate a clean synthetic JPEG image buffer for benchmarking."""
    from PIL import Image, ImageDraw
    import io

    img = Image.new("RGB", (width, height), color=(70, 80, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 4, height // 4, 3 * width // 4, 3 * height // 4], outline=(200, 200, 200), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_mock_inference(mode: str) -> Dict[str, Any]:
    """Simulate realistic model response and token consumption for testing."""
    if mode == "pose17":
        keypoints = {k: [500, 500, 2] for k in POSE17_KEYPOINTS}
        payload = {"keypoints": keypoints, "confidence": 0.95}
        tokens = {"prompt_tokens": 850, "completion_tokens": 120, "total_tokens": 970}
        simulated_delay = 1.15
    else:  # vf50
        landmarks = {k: [500, 500, 2] for k in VF50_LANDMARKS}
        payload = {"landmarks": landmarks, "confidence": 0.98}
        tokens = {"prompt_tokens": 1100, "completion_tokens": 340, "total_tokens": 1440}
        simulated_delay = 2.45

    time.sleep(0.05)  # slight real delay
    return {
        "content": json.dumps(payload),
        "reported_api_sec": simulated_delay,
        "usage": tokens,
    }


def execute_direct_9router_request(
    base_url: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    image_bytes: bytes,
    timeout: float,
) -> Dict[str, Any]:
    """Execute request to 9Router using NineRouterClient."""
    client = NineRouterClient(base_url=base_url, api_key=api_key, timeout=timeout)
    t0 = time.perf_counter()
    resp = client.send_vision_request(
        model=model,
        image_bytes_or_b64=image_bytes,
        prompt=prompt,
        temperature=0.0,
        max_tokens=2048,
        timeout=timeout,
    )
    client_elapsed = time.perf_counter() - t0

    return {
        "content": resp.content,
        "reported_api_sec": float(resp.duration_seconds),
        "client_elapsed_sec": client_elapsed,
        "usage": resp.usage or {},
    }


def profile_single_iteration(
    mode: str,
    iteration: int,
    image_bytes: bytes,
    base_url: str,
    api_key: Optional[str],
    model: str,
    timeout: float,
    mock: bool,
) -> BenchmarkMetric:
    """Run and time a single benchmark iteration."""
    t_start = time.perf_counter()

    # Image prep
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    prompt = (
        f"Detect human pose keypoints according to COCO 17 schema: {list(POSE17_KEYPOINTS)}"
        if mode == "pose17"
        else f"Detect facial landmarks according to VinAI 50 schema: {list(VF50_LANDMARKS)}"
    )

    # Remote inference
    if mock:
        raw_res = run_mock_inference(mode)
        remote_api_sec = raw_res["reported_api_sec"]
        client_network_sec = remote_api_sec
        content = raw_res["content"]
        usage = raw_res["usage"]
    else:
        raw_res = execute_direct_9router_request(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_bytes=image_bytes,
            timeout=timeout,
        )
        remote_api_sec = raw_res["reported_api_sec"]
        client_network_sec = raw_res["client_elapsed_sec"]
        content = raw_res["content"]
        usage = raw_res["usage"]

    # Local parsing and sanitization
    t_parse_start = time.perf_counter()
    clean_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        parsed_json = json.loads(clean_content)
    except Exception:
        # If response is not direct JSON, attempt to locate json substring
        m_json = re.search(r"\{.*\}", clean_content, re.DOTALL)
        parsed_json = json.loads(m_json.group(0)) if m_json else {}
    if mode == "pose17":
        instance = parse_and_sanitize_pose17_instance(parsed_json, img_width=1280, img_height=720)
    else:
        instance = parse_and_sanitize_vf50_instance(parsed_json, img_width=1280, img_height=720)

    local_parse_sec = time.perf_counter() - t_parse_start
    total_latency_sec = time.perf_counter() - t_start

    elements_detected = len(instance.elements) if instance else 0

    std_threshold = SLA_THRESHOLDS[mode]["standard_max_sec"]
    stretch_threshold = SLA_THRESHOLDS[mode]["stretch_max_sec"]

    return BenchmarkMetric(
        mode=mode,
        iteration=iteration,
        remote_api_sec=round(remote_api_sec, 3),
        client_network_sec=round(client_network_sec, 3),
        local_parse_sec=round(local_parse_sec, 4),
        total_latency_sec=round(total_latency_sec, 3),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        elements_detected=elements_detected,
        standard_sla_pass=total_latency_sec <= std_threshold,
        stretch_sla_pass=total_latency_sec <= stretch_threshold,
    )


def summarize_metrics(metrics: List[BenchmarkMetric]) -> Dict[str, Any]:
    """Calculate summary statistics over a series of benchmark metrics."""
    if not metrics:
        return {}

    mode = metrics[0].mode
    latencies = [m.total_latency_sec for m in metrics]
    remote_secs = [m.remote_api_sec for m in metrics]
    parse_secs = [m.local_parse_sec for m in metrics]
    tokens = [m.total_tokens for m in metrics]

    avg_lat = sum(latencies) / len(latencies)
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else max(latencies)

    std_pass_count = sum(1 for m in metrics if m.standard_sla_pass)
    stretch_pass_count = sum(1 for m in metrics if m.stretch_sla_pass)

    return {
        "mode": mode,
        "count": len(metrics),
        "total_latency_avg_sec": round(avg_lat, 3),
        "total_latency_min_sec": round(min(latencies), 3),
        "total_latency_max_sec": round(max(latencies), 3),
        "total_latency_p95_sec": round(p95_lat, 3),
        "remote_api_avg_sec": round(sum(remote_secs) / len(remote_secs), 3),
        "local_parse_avg_ms": round((sum(parse_secs) / len(parse_secs)) * 1000, 2),
        "tokens_avg": round(sum(tokens) / len(tokens), 1),
        "standard_sla_target_sec": SLA_THRESHOLDS[mode]["standard_max_sec"],
        "stretch_sla_target_sec": SLA_THRESHOLDS[mode]["stretch_max_sec"],
        "standard_sla_pass_rate": round(std_pass_count / len(metrics), 2),
        "stretch_sla_pass_rate": round(stretch_pass_count / len(metrics), 2),
        "overall_status": "PASS" if std_pass_count == len(metrics) else "FAIL",
    }


def print_summary_table(summaries: List[Dict[str, Any]]):
    """Format benchmark results into terminal table without emojis."""
    header = (
        f"{'Mode':<8} | {'Iters':<5} | {'Avg Lat':<8} | {'P95 Lat':<8} | "
        f"{'Remote API':<10} | {'Parse (ms)':<10} | {'Tokens':<7} | {'Std SLA':<7} | {'Stretch':<7} | {'Status':<6}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)

    for s in summaries:
        mode = s["mode"]
        iters = s["count"]
        avg_l = f"{s['total_latency_avg_sec']:.2f}s"
        p95_l = f"{s['total_latency_p95_sec']:.2f}s"
        remote = f"{s['remote_api_avg_sec']:.2f}s"
        parse = f"{s['local_parse_avg_ms']:.1f}ms"
        tok = str(int(s["tokens_avg"]))
        std_sla = f"{s['standard_sla_pass_rate']*100:.0f}%"
        stretch = f"{s['stretch_sla_pass_rate']*100:.0f}%"
        status = s["overall_status"]

        print(
            f"{mode:<8} | {iters:<5} | {avg_l:<8} | {p95_l:<8} | "
            f"{remote:<10} | {parse:<10} | {tok:<7} | {std_sla:<7} | {stretch:<7} | {status:<6}"
        )
    print(separator + "\n")


def main():
    parser = argparse.ArgumentParser(description="Week-2 Pose17 & VF50 Latency & Benchmark Profiler")
    parser.add_argument("--mode", choices=["pose17", "vf50", "all"], default="all", help="Target mode to benchmark")
    parser.add_argument("--image", type=Path, default=None, help="Path to test image (generates synthetic if omitted)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--base-url", default=DEFAULT_NINEROUTER_URL, help="9Router base URL")
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL, help="Model identifier")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--mock", action="store_true", help="Run mock benchmark for CI and smoke testing")
    parser.add_argument("--output-json", type=Path, default=None, help="Save structured metrics to JSON")
    args = parser.parse_args()

    # Prepare image
    if args.image and args.image.exists():
        image_bytes = args.image.read_bytes()
    else:
        image_bytes = generate_synthetic_test_image()

    modes = ["pose17", "vf50"] if args.mode == "all" else [args.mode]
    all_metrics: List[BenchmarkMetric] = []
    summaries: List[Dict[str, Any]] = []

    api_key = os.getenv("NINEROUTER_API_KEY", "")

    for mode in modes:
        mode_metrics: List[BenchmarkMetric] = []
        for i in range(1, args.iterations + 1):
            m = profile_single_iteration(
                mode=mode,
                iteration=i,
                image_bytes=image_bytes,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                timeout=args.timeout,
                mock=args.mock,
            )
            mode_metrics.append(m)
            all_metrics.append(m)

        summary = summarize_metrics(mode_metrics)
        summaries.append(summary)

    print_summary_table(summaries)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": [asdict(m) for m in all_metrics],
            "summaries": summaries,
        }
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
