"""Nuclio Function Entry Point for 9Router Polygon + Mask CVAT Detector.

Receives CVAT inference requests, forwards images to local 9Router vision models,
and returns CVAT shapes routed according to Policy B:
- Exactly 10 semantic surface and infrastructure region labels
- Emits paired polygon + native CVAT mask derived from the same contour sharing identical integer group_id
- Strictly suppresses bounding boxes
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure function directory is in sys.path
_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from model_handler import ModelHandler

logger = logging.getLogger("cvat.nuclio.ninerouter.polygon_mask")

# 32MB maximum request body size limit (matches Nuclio function.yaml maxRequestBodySize)
MAX_REQUEST_BODY_SIZE = 33554432  # 32 * 1024 * 1024 bytes


def init_context(context):
    """Initialize function context when container starts."""
    context.logger.info("Initializing 9Router Polygon + Mask Nuclio function context...")
    try:
        handler_instance = ModelHandler()
        context.user_data.model_handler = handler_instance
        context.logger.info("9Router Polygon + Mask context initialized successfully.")
    except Exception as e:
        context.logger.error(f"Failed to initialize 9Router Polygon + Mask context: {e}")
        raise


def handler(context, event):
    """Handle incoming detector event from CVAT."""
    context.logger.info("Handling CVAT Polygon + Mask detector request...")

    # Retain raw request bytes before JSON parsing (bytes, bytearray, or str)
    raw_body = event.body
    raw_bytes: Optional[bytes] = None
    if isinstance(raw_body, (bytes, bytearray)):
        raw_bytes = bytes(raw_body)
    elif isinstance(raw_body, str):
        raw_bytes = raw_body.encode("utf-8")

    # Guard against excessively large request bodies (DoS / memory exhaustion prevention)
    if raw_bytes is not None and len(raw_bytes) > MAX_REQUEST_BODY_SIZE:
        return context.Response(
            body=json.dumps({"error": f"Request body exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes (32MB)"}),
            headers={},
            content_type="application/json",
            status_code=413,
        )

    # Parse request payload
    data = None
    if raw_bytes is not None:
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            return context.Response(
                body=json.dumps({"error": f"Invalid JSON body: {str(e)}"}),
                headers={},
                content_type="application/json",
                status_code=400,
            )
    elif isinstance(raw_body, dict):
        data = raw_body
    else:
        return context.Response(
            body=json.dumps({"error": "Expected JSON object or raw bytes in request body"}),
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

    # -------------------------------------------------------------------------
    # Dual-Dispatch: Check if incoming request is a CVAT Webhook event
    # -------------------------------------------------------------------------
    headers = getattr(event, "headers", {}) or {}
    from app.cvat_sync import (
        extract_cvat_event_header,
        extract_cvat_signature_header,
        verify_cvat_webhook_signature,
    )
    event_type = data.get("event") or extract_cvat_event_header(headers)
    if event_type:
        context.logger.info(f"Received CVAT Webhook event: {event_type}")
        webhook_secret = os.getenv("CVAT_WEBHOOK_SECRET")
        sig_header = extract_cvat_signature_header(headers)

        if webhook_secret:
            if raw_bytes is None or not sig_header:
                context.logger.warning("CVAT Webhook signature verification failed: missing signature or raw bytes.")
                return context.Response(
                    body=json.dumps({"error": "Invalid webhook signature"}),
                    headers={},
                    content_type="application/json",
                    status_code=401,
                )

            sig = sig_header.strip()
            if sig.startswith("sha256="):
                sig = sig[7:].strip()

            expected_sig = hmac.new(
                webhook_secret.encode("utf-8"),
                raw_bytes,
                hashlib.sha256,
            ).hexdigest()

            if not hmac.compare_digest(expected_sig.lower(), sig.lower()):
                context.logger.warning("CVAT Webhook signature verification failed: signature mismatch.")
                return context.Response(
                    body=json.dumps({"error": "Invalid webhook signature"}),
                    headers={},
                    content_type="application/json",
                    status_code=401,
                )
        else:
            context.logger.warning(
                "CVAT_WEBHOOK_SECRET is not configured; running webhook dispatch in local dev mode without signature verification."
            )

        # Process job/task completion events
        if "job" in event_type or "task" in event_type:
            job_info = data.get("job", {})
            job_id = job_info.get("id") or data.get("job_id")
            state = job_info.get("state") or data.get("state")
            stage = job_info.get("stage") or data.get("stage")

            if job_id and (state == "completed" or stage in ("acceptance", "validation")):
                try:
                    from app.cvat_sync import sync_job_feedback
                    cvat_url = os.getenv("CVAT_URL", "http://cvat_server:8080")
                    token = os.getenv("CVAT_TOKEN")
                    report = sync_job_feedback(job_id=int(job_id), cvat_url=cvat_url, token=token)
                    return context.Response(
                        body=json.dumps({"status": "synced", "report": report.to_dict()}),
                        headers={},
                        content_type="application/json",
                        status_code=200,
                    )
                except Exception as e:
                    context.logger.error(f"Error during CVAT feedback sync: {e}")
                    return context.Response(
                        body=json.dumps({"error": f"Feedback sync failed: {str(e)}"}),
                        headers={},
                        content_type="application/json",
                        status_code=500,
                    )

        return context.Response(
            body=json.dumps({"status": "ignored", "event": event_type}),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    # -------------------------------------------------------------------------
    # Detector Inference Flow
    # -------------------------------------------------------------------------
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

    try:
        raw_thresh = data.get("threshold", 0.5)
        threshold = float(raw_thresh) if raw_thresh is not None else 0.5
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.5

    roi = None
    raw_roi = data.get("roi") or data.get("pos_tracker") or data.get("box")
    if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
        try:
            roi = [float(v) for v in raw_roi]
        except (ValueError, TypeError):
            roi = None

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
            roi=roi,
        )

        return context.Response(
            body=json.dumps(shapes),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except ValueError as e:
        context.logger.error(f"Image or validation error during Polygon + Mask inference: {e}")
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
        context.logger.error(f"Error during 9Router Polygon + Mask inference: {err_msg}")
        return context.Response(
            body=json.dumps({"error": f"Inference failed: {err_msg}"}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
