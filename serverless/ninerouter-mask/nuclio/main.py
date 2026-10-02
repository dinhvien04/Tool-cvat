"""Nuclio entry point for the mask-only detector."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from model_handler import ModelHandler

MAX_REQUEST_BODY_SIZE = 33554432


def init_context(context):
    context.logger.info("Initializing 9Router Mask Nuclio function...")
    context.user_data.model_handler = ModelHandler()
    context.logger.info("9Router Mask context initialized.")


def _response(context, payload, status_code):
    return context.Response(body=json.dumps(payload), headers={}, content_type="application/json", status_code=status_code)


def handler(context, event):
    raw_body = event.body
    if isinstance(raw_body, (bytes, bytearray, str)) and len(raw_body) > MAX_REQUEST_BODY_SIZE:
        return _response(context, {"error": "Request body exceeds 32MB"}, 413)
    data = raw_body
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data.decode("utf-8"))
    elif isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict) or not isinstance(data.get("image"), str):
        return _response(context, {"error": "Missing mandatory 'image' field"}, 400)
    image_b64 = data["image"]
    if "," in image_b64 and image_b64[:30].startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64)
    try:
        threshold = max(0.0, min(1.0, float(data.get("threshold", 0.5))))
    except (TypeError, ValueError):
        threshold = 0.5
    try:
        return _response(context, context.user_data.model_handler.infer(image_bytes, threshold), 200)
    except ValueError as exc:
        return _response(context, {"error": f"Validation error: {exc}"}, 400)
    except Exception as exc:
        err = str(exc)
        key = getattr(context.user_data.model_handler, "api_key", None)
        if isinstance(key, str) and key and key in err:
            err = err.replace(key, "***")
        return _response(context, {"error": f"Inference failed: {err}"}, 500)
