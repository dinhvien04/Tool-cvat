"""Production-aligned E2E profiler for the three active Tool-cvat detectors.

Security:
- No credentials are embedded in source.
- CVAT_ACCESS_TOKEN (PAT, Bearer) is preferred.
- Legacy CVAT_TOKEN or .tool-cvat/cvat_token.txt is supported only for local backwards compatibility.

The profiler intentionally uses the SAME model, max image size, max token budget,
prompt builder, and default timeout as production. This prevents benchmark drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.client import NineRouterClient
from app.config import (
    DEFAULT_NINEROUTER_TIMEOUT,
    DEFAULT_POLYGON_MASK_MAX_IMAGE_SIZE,
    DEFAULT_POLYGON_MASK_MAX_TOKENS,
    DEFAULT_POLYGON_MASK_VISION_MODEL,
    DEFAULT_POLYLINE_MAX_IMAGE_SIZE,
    DEFAULT_POLYLINE_MAX_TOKENS,
    DEFAULT_POLYLINE_VISION_MODEL,
    DEFAULT_RECTANGLE_MASK_MAX_IMAGE_SIZE,
    DEFAULT_RECTANGLE_MASK_MAX_TOKENS,
    DEFAULT_RECTANGLE_MASK_VISION_MODEL,
)
from app.image_ops import image_to_data_url, load_image, resize_image_if_needed
from core.taxonomy import BOX_MASK_LABELS, POLYGON_MASK_LABELS, POLYLINE_LABELS
from core.vision_contract import (
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
    build_user_prompt,
)

DETECTORS = {
    "rectangle-mask": {
        "function": "ninerouter-rectangle-mask",
        "mode": MODE_RECTANGLE_MASK,
        "labels": list(BOX_MASK_LABELS),
        "model_env": "RECTANGLE_MASK_MODEL",
        "default_model": DEFAULT_RECTANGLE_MASK_VISION_MODEL,
        "max_size": DEFAULT_RECTANGLE_MASK_MAX_IMAGE_SIZE,
        "max_tokens": DEFAULT_RECTANGLE_MASK_MAX_TOKENS,
        "nuclio_env": "RECTANGLE_MASK_NUCLIO_URL",
    },
    "polygon-mask": {
        "function": "ninerouter-polygon-mask",
        "mode": MODE_POLYGON_MASK,
        "labels": list(POLYGON_MASK_LABELS),
        "model_env": "POLYGON_MASK_MODEL",
        "default_model": DEFAULT_POLYGON_MASK_VISION_MODEL,
        "max_size": DEFAULT_POLYGON_MASK_MAX_IMAGE_SIZE,
        "max_tokens": DEFAULT_POLYGON_MASK_MAX_TOKENS,
        "nuclio_env": "POLYGON_MASK_NUCLIO_URL",
    },
    "polyline": {
        "function": "ninerouter-polyline",
        "mode": MODE_POLYLINE,
        "labels": list(POLYLINE_LABELS),
        "model_env": "POLYLINE_MODEL",
        "default_model": DEFAULT_POLYLINE_VISION_MODEL,
        "max_size": DEFAULT_POLYLINE_MAX_IMAGE_SIZE,
        "max_tokens": DEFAULT_POLYLINE_MAX_TOKENS,
        "nuclio_env": "POLYLINE_NUCLIO_URL",
    },
}


def _cvat_headers() -> Optional[Dict[str, str]]:
    pat = os.getenv("CVAT_ACCESS_TOKEN", "").strip()
    legacy = os.getenv("CVAT_TOKEN", "").strip()
    legacy_file = ROOT / ".tool-cvat" / "cvat_token.txt"

    if pat:
        auth = f"Bearer {pat}"
    elif legacy:
        auth = f"Token {legacy}"
    elif legacy_file.exists():
        auth = f"Token {legacy_file.read_text(encoding='utf-8').strip()}"
    else:
        return None

    return {
        "Authorization": auth,
        "Accept": "application/vnd.cvat+json",
        "Content-Type": "application/json",
    }


def _shape_summary(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        shapes = payload.get("shapes", [])
    elif isinstance(payload, list):
        shapes = payload
    else:
        shapes = []
    by_type: Dict[str, int] = {}
    for shape in shapes:
        if isinstance(shape, dict):
            stype = str(shape.get("type", "unknown"))
            by_type[stype] = by_type.get(stype, 0) + 1
    return {"shapes": len(shapes), "shape_types": by_type}


def benchmark_direct_9router(
    image_path: Path,
    cfg: Dict[str, Any],
    base_url: str,
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    model = os.getenv(cfg["model_env"], "").strip() or cfg["default_model"]
    image = load_image(image_path)
    resized, was_resized, dims = resize_image_if_needed(image, max_size=cfg["max_size"])
    data_url = image_to_data_url(resized, format="JPEG", quality=90)
    prompt = build_user_prompt(allowed_labels=cfg["labels"], mode=cfg["mode"])
    client = NineRouterClient(base_url=base_url, api_key=api_key, timeout=timeout)

    t0 = time.perf_counter()
    response = client.send_vision_request(
        model=model,
        image_bytes_or_b64=data_url,
        prompt=prompt,
        temperature=0.0,
        max_tokens=cfg["max_tokens"],
        mode=cfg["mode"],
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    usage = response.usage or {}
    return {
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "reported_api_sec": round(response.duration_seconds, 3),
        "image_size": list(dims),
        "was_resized": was_resized,
        "max_tokens": cfg["max_tokens"],
        "prompt_chars": len(prompt),
        "response_chars": len(response.content),
        "usage": usage,
    }


def benchmark_nuclio(image_path: Path, cfg: Dict[str, Any], url: str, timeout: float) -> Dict[str, Any]:
    import base64

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    t0 = time.perf_counter()
    response = requests.post(url, json={"image": image_b64}, timeout=timeout)
    elapsed = time.perf_counter() - t0
    out: Dict[str, Any] = {"status": response.status_code, "elapsed_sec": round(elapsed, 3)}
    if response.ok:
        try:
            out.update(_shape_summary(response.json()))
        except ValueError:
            out["error"] = "non-json response"
    else:
        out["error"] = response.text[:300]
    return out


def benchmark_cvat(
    cfg: Dict[str, Any],
    cvat_url: str,
    headers: Dict[str, str],
    job_id: int,
    frame_id: int,
    timeout: float,
) -> Dict[str, Any]:
    url = f"{cvat_url}/api/lambda/functions/{cfg['function']}"
    t0 = time.perf_counter()
    response = requests.post(
        url,
        headers=headers,
        json={"job": job_id, "frame": frame_id},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    out: Dict[str, Any] = {"status": response.status_code, "elapsed_sec": round(elapsed, 3)}
    if response.ok:
        try:
            out.update(_shape_summary(response.json()))
        except ValueError:
            out["error"] = "non-json response"
    else:
        out["error"] = response.text[:300]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.getenv("TOOL_CVAT_PROFILE_IMAGE", str(ROOT / "test.jpg")))
    parser.add_argument("--detector", choices=[*DETECTORS.keys(), "all"], default="all")
    parser.add_argument("--cvat-url", default=os.getenv("CVAT_URL", "http://localhost:18080"))
    parser.add_argument("--job-id", type=int, default=int(os.getenv("CVAT_JOB_ID", "0") or 0))
    parser.add_argument("--frame-id", type=int, default=int(os.getenv("CVAT_FRAME_ID", "0") or 0))
    parser.add_argument("--router-url", default=os.getenv("NINEROUTER_URL", "http://127.0.0.1:20128"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("NINEROUTER_TIMEOUT", DEFAULT_NINEROUTER_TIMEOUT)))
    parser.add_argument("--output", default=str(ROOT / "output" / "e2e_performance_report.json"))
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    selected = DETECTORS if args.detector == "all" else {args.detector: DETECTORS[args.detector]}
    headers = _cvat_headers()
    cvat_url = args.cvat_url.rstrip("/")
    api_key = os.getenv("NINEROUTER_KEY") or None

    report: Dict[str, Any] = {
        "production_aligned": True,
        "timeout_sec": args.timeout,
        "detectors": {},
    }

    for name, cfg in selected.items():
        item: Dict[str, Any] = {}
        try:
            item["direct_9router"] = benchmark_direct_9router(
                image_path, cfg, args.router_url.rstrip("/"), api_key, args.timeout
            )
        except Exception as exc:
            item["direct_9router"] = {"error": f"{type(exc).__name__}: {exc}"}

        nuclio_url = os.getenv(cfg["nuclio_env"], "").strip()
        if nuclio_url:
            try:
                item["direct_nuclio"] = benchmark_nuclio(image_path, cfg, nuclio_url, args.timeout)
            except Exception as exc:
                item["direct_nuclio"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            item["direct_nuclio"] = {"skipped": f"set {cfg['nuclio_env']} to enable"}

        if headers and args.job_id > 0:
            try:
                item["via_cvat"] = benchmark_cvat(
                    cfg, cvat_url, headers, args.job_id, args.frame_id, args.timeout + 15
                )
            except Exception as exc:
                item["via_cvat"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            item["via_cvat"] = {
                "skipped": "set CVAT_ACCESS_TOKEN and CVAT_JOB_ID to enable authenticated CVAT profiling"
            }

        report["detectors"][name] = item

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
