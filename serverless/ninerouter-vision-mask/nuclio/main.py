"""Nuclio Function Entry Point for 9Router Vision Mask CVAT Detector.

Receives CVAT inference requests, forwards images to local 9Router vision models,
and returns CVAT native mask shape annotations (Instance Segmentation).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure function directory is in sys.path
_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from model_handler import ModelHandler

logger = logging.getLogger("cvat.nuclio.ninerouter.mask")

# 32MB maximum request body size limit (matches Nuclio function.yaml maxRequestBodySize)
MAX_REQUEST_BODY_SIZE = 33554432  # 32 * 1024 * 1024 bytes


def init_context(context):
    """Initialize function context when container starts."""
    context.logger.info("Initializing 9Router Vision Mask Nuclio function context...")
    try:
        handler_instance = ModelHandler()
        context.user_data.model_handler = handler_instance
        context.logger.info("9Router Vision Mask context initialized successfully.")
    except Exception as e:
        context.logger.error(f"Failed to initialize 9Router Vision Mask context: {e}")
        raise


def handler(context, event):
    """Handle incoming mask detector event from CVAT."""
    context.logger.info("Handling CVAT mask detector request...")

    # Guard against excessively large request bodies (DoS / memory exhaustion prevention)
    raw_body = event.body
    if isinstance(raw_body, (bytes, bytearray, str)) and len(raw_body) > MAX_REQUEST_BODY_SIZE:
        return context.Response(
            body=json.dumps({"error": f"Request body exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes (32MB)"}),
            headers={},
            content_type="application/json",
            status_code=413,
        )

    # Parse request payload
    data = event.body
    if isinstance(data, (bytes, bytearray)):
        try:
            data = json.loads(data.decode("utf-8"))
        except Exception as e:
            return context.Response(
                body=json.dumps({"error": f"Invalid JSON body: {str(e)}"}),
                headers={},
                content_type="application/json",
                status_code=400,
            )
    elif isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return context.Response(
                body=json.dumps({"error": f"Invalid JSON body: {str(e)}"}),
                headers={},
                content_type="application/json",
                status_code=400,
            )

    if not isinstance(data, dict):
        return context.Response(
            body=json.dumps({"error": "Expected JSON object in request body"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    # Extract base64 image
    image_b64 = data.get("image")
    if not image_b64:
        return context.Response(
            body=json.dumps({"error": "Missing mandatory 'image' field in request body"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    if not isinstance(image_b64, str):
        return context.Response(
            body=json.dumps({"error": f"Field 'image' must be a Base64 string, got {type(image_b64).__name__}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    if len(image_b64) > MAX_REQUEST_BODY_SIZE:
        return context.Response(
            body=json.dumps({"error": f"Image payload exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes (32MB)"}),
            headers={},
            content_type="application/json",
            status_code=413,
        )

    # Decode base64 image (stripping data: URL prefix if present)
    try:
        if "," in image_b64 and "data:" in image_b64[:30]:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return context.Response(
            body=json.dumps({"error": f"Failed to decode Base64 image: {str(e)}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    if len(image_bytes) > MAX_REQUEST_BODY_SIZE:
        return context.Response(
            body=json.dumps({"error": f"Decoded image exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes (32MB)"}),
            headers={},
            content_type="application/json",
            status_code=413,
        )

    # Extract threshold (defaults to 0.5)
    try:
        raw_thresh = data.get("threshold", 0.5)
        threshold = float(raw_thresh) if raw_thresh is not None else 0.5
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.5

    # Extract optional mode override ('box', 'mask', 'box_and_mask')
    req_mode = data.get("mode")
    if req_mode is not None and isinstance(req_mode, str):
        req_mode = req_mode.strip().lower()
    else:
        req_mode = None

    # Run inference
    try:
        handler_instance: ModelHandler = context.user_data.model_handler
        shapes = handler_instance.infer(
            image_bytes=image_bytes,
            threshold=threshold,
            mode=req_mode,
        )

        return context.Response(
            body=json.dumps(shapes),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except ValueError as e:
        context.logger.error(f"Image or validation error during mask inference: {e}")
        return context.Response(
            body=json.dumps({"error": f"Validation error: {str(e)}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
    except Exception as e:
        err_msg = str(e)
        handler_inst = getattr(getattr(context, "user_data", None), "model_handler", None)
        key = getattr(handler_inst, "api_key", None)
        if isinstance(key, str) and key and key in err_msg:
            err_msg = err_msg.replace(key, "***")
        context.logger.error(f"Error during 9Router vision mask inference: {err_msg}")
        return context.Response(
            body=json.dumps({"error": f"Inference failed: {err_msg}"}),
            headers={},
            content_type="application/json",
            status_code=502,
        )
