"""Nuclio entry point for the 9Router Box detector.

Returns CVAT rectangle annotations for the 14 foreground instance labels.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from model_handler import ModelHandler

logger = logging.getLogger("cvat.nuclio.ninerouter.box")
MAX_REQUEST_BODY_SIZE = 33554432


def init_context(context):
    """Initialize the box detector when the container starts."""
    context.logger.info("Initializing 9Router Box Nuclio function...")
    context.user_data.model_handler = ModelHandler()
    context.logger.info("9Router Box context initialized.")


def _response(context, payload, status_code):
    return context.Response(
        body=json.dumps(payload),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )


def handler(context, event):
    """Handle a CVAT detector request and return rectangle shapes."""
    raw_body = event.body
    if isinstance(raw_body, (bytes, bytearray, str)) and len(raw_body) > MAX_REQUEST_BODY_SIZE:
        return _response(
            context,
            {"error": f"Request body exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes (32MB)"},
            413,
        )

    data = raw_body
    if isinstance(data, (bytes, bytearray)):
        try:
            data = json.loads(data.decode("utf-8"))
        except Exception as exc:
            return _response(context, {"error": f"Invalid JSON body: {exc}"}, 400)
    elif isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as exc:
            return _response(context, {"error": f"Invalid JSON body: {exc}"}, 400)

    if not isinstance(data, dict):
        return _response(context, {"error": "Expected JSON object in request body"}, 400)

    image_b64 = data.get("image")
    if not image_b64 or not isinstance(image_b64, str):
        return _response(context, {"error": "Missing mandatory 'image' field in request body"}, 400)
    if len(image_b64) > MAX_REQUEST_BODY_SIZE:
        return _response(
            context,
            {"error": f"Image payload exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes (32MB)"},
            413,
        )

    try:
        if "," in image_b64 and "data:" in image_b64[:30]:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        return _response(context, {"error": f"Failed to decode Base64 image: {exc}"}, 400)

    try:
        threshold = float(data.get("threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.5

    try:
        handler_instance: ModelHandler = context.user_data.model_handler
        shapes = handler_instance.infer(image_bytes=image_bytes, threshold=threshold, mode="box")
        return _response(context, shapes, 200)
    except ValueError as exc:
        context.logger.error("Box validation error: %s", exc)
        return _response(context, {"error": f"Validation error: {exc}"}, 400)
    except Exception as exc:
        err_msg = str(exc)
        handler_inst = getattr(getattr(context, "user_data", None), "model_handler", None)
        key = getattr(handler_inst, "api_key", None)
        if isinstance(key, str) and key and key in err_msg:
            err_msg = err_msg.replace(key, "***")
        context.logger.error("Box inference failed: %s", err_msg)
        return _response(context, {"error": f"Inference failed: {err_msg}"}, 500)
