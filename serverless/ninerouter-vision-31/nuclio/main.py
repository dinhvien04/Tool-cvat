"""Nuclio Function Entry Point for 9Router Vision 31-Label Multi-Shape CVAT Detector.

Receives CVAT inference requests, forwards images to local 9Router vision models,
and returns CVAT shapes routed according to the 31-label autonomous driving taxonomy:
- Instance Objects (14 labels): Paired rectangle + mask (group_id)
- Semantic Regions (10 labels): Semantic mask with contour points (ungrouped)
- Lane Markings (7 labels): Ribbon mask / polyline / polygon (ungrouped)
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

logger = logging.getLogger("cvat.nuclio.ninerouter.full31")

# 32MB maximum request body size limit (matches Nuclio function.yaml maxRequestBodySize)
MAX_REQUEST_BODY_SIZE = 33554432  # 32 * 1024 * 1024 bytes


def init_context(context):
    """Initialize function context when container starts."""
    context.logger.info("Initializing 9Router Vision 31-Label Nuclio function context...")
    try:
        handler_instance = ModelHandler()
        context.user_data.model_handler = handler_instance
        context.logger.info("9Router Vision 31-Label context initialized successfully.")
    except Exception as e:
        context.logger.error(f"Failed to initialize 9Router Vision 31-Label context: {e}")
        raise


def handler(context, event):
    """Handle incoming detector event from CVAT."""
    context.logger.info("Handling CVAT 31-Label detector request...")

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
        # Webhook signature verification if secret is configured
        webhook_secret = os.getenv("CVAT_WEBHOOK_SECRET")
        sig_header = extract_cvat_signature_header(headers)

        if webhook_secret:
            # Strictly verify HMAC-SHA256 directly on the exact raw body bytes:
            # - Disallow fallback to str(parsed_dict).encode('utf-8')
            # - Support case-insensitive headers: X-Signature-256, x-signature-256, X-CVAT-Signature, x-cvat-signature, X-Hub-Signature-256
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

        # Process job/task/annotation events
        event_lower = (event_type or "").lower()
        if any(k in event_lower for k in ("job", "task", "annotation")):
            job_info = data.get("job", {}) if isinstance(data.get("job"), dict) else {}
            job_id = job_info.get("id") or data.get("job_id")
            state = str(job_info.get("state") or data.get("state") or "").lower()
            stage = str(job_info.get("stage") or data.get("stage") or "").lower()

            # Trigger sync immediately on Save, update, completion, or active annotation states
            should_sync = bool(
                job_id
                and (
                    state in ("completed", "in progress", "annotation", "validation", "acceptance")
                    or stage in ("acceptance", "validation", "annotation", "in progress")
                    or "save" in event_lower
                    or "update" in event_lower
                )
            )

            if should_sync:
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

    # Extract optional ROI (region of interest) coordinates [xtl, ytl, xbr, ybr]
    roi = None
    raw_roi = data.get("roi") or data.get("pos_tracker") or data.get("box")
    if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
        try:
            roi = [float(v) for v in raw_roi]
        except (ValueError, TypeError):
            roi = None

    # Extract optional mode override ('full_31', 'box', 'mask', 'box_and_mask')
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
        context.logger.error(f"Image or validation error during 31-Label inference: {e}")
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
        context.logger.error(f"Error during 9Router vision 31-Label inference: {err_msg}")
        return context.Response(
            body=json.dumps({"error": f"Inference failed: {err_msg}"}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
