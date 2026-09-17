"""Nuclio Function Entry Point for 9Router Vision CVAT Detector.

Receives CVAT inference requests, forwards images to local 9Router vision models,
and returns CVAT rectangle shape annotations.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict

from model_handler import ModelHandler

logger = logging.getLogger("cvat.nuclio.ninerouter")


def init_context(context):
    """Initialize function context when container starts."""
    context.logger.info("Initializing 9Router Vision Nuclio function context...")
    try:
        handler_instance = ModelHandler()
        context.user_data.model_handler = handler_instance
        context.logger.info("9Router Vision context initialized successfully.")
    except Exception as e:
        context.logger.error(f"Failed to initialize 9Router Vision context: {e}")
        raise


def handler(context, event):
    """Handle incoming detector event from CVAT."""
    context.logger.info("Handling CVAT detector request...")

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

    # Decode base64 image (stripping data: URL prefix if present)
    try:
        if isinstance(image_b64, str) and "," in image_b64 and "data:" in image_b64[:30]:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return context.Response(
            body=json.dumps({"error": f"Failed to decode Base64 image: {str(e)}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    # Extract threshold (defaults to 0.5)
    try:
        raw_thresh = data.get("threshold", 0.5)
        threshold = float(raw_thresh) if raw_thresh is not None else 0.5
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.5

    # Run inference
    try:
        handler_instance: ModelHandler = context.user_data.model_handler
        shapes = handler_instance.infer(image_bytes=image_bytes, threshold=threshold)

        return context.Response(
            body=json.dumps(shapes),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except ValueError as e:
        context.logger.error(f"Image or validation error during inference: {e}")
        return context.Response(
            body=json.dumps({"error": f"Validation error: {str(e)}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
    except Exception as e:
        err_msg = str(e)
        context.logger.error(f"Error during 9Router vision inference: {err_msg}")
        return context.Response(
            body=json.dumps({"error": f"Inference failed: {err_msg}"}),
            headers={},
            content_type="application/json",
            status_code=502,
        )
