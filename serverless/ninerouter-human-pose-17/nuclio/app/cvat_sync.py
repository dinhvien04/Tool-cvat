"""CVAT Job Annotations Synchronization and Webhook Processing Engine.

Synchronizes human-reviewed annotations from CVAT into Tool-cvat correction memory:
1. CVAT REST API Client (and internal Docker Django ORM fallback).
2. Webhook HMAC-SHA256 signature verification (X-CVAT-Signature).
3. Image hashing, AI baseline reconciliation, and CorrectionDiffEngine dispatch.
4. Local privacy-preserving crop extraction and rule derivation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import subprocess
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image

from app.feedback import (
    CorrectionDiffEngine,
    CorrectionDiffItem,
    FeedbackDatabase,
    compute_image_hash,
)

logger = logging.getLogger(__name__)


def verify_cvat_webhook_signature(
    payload_bytes: Union[bytes, bytearray, str],
    signature_header: Optional[str],
    secret: str,
) -> bool:
    """Verify CVAT webhook HMAC-SHA256 signature against original request bytes.

    Args:
        payload_bytes: Raw HTTP request body bytes (NOT re-serialized JSON).
        signature_header: Contents of 'X-Signature-256' or 'X-CVAT-Signature' header.
        secret: Configured shared secret string.

    Returns:
        True if signature matches securely; False otherwise.
    """
    if not signature_header or not secret:
        return False

    if isinstance(payload_bytes, (bytes, bytearray)):
        raw_bytes = bytes(payload_bytes)
    elif isinstance(payload_bytes, str):
        raw_bytes = payload_bytes.encode("utf-8")
    else:
        # Strictly disallow fallback to str(parsed_dict).encode('utf-8')
        return False

    sig = signature_header.strip()
    if sig.startswith("sha256="):
        sig = sig[7:].strip()

    expected_sig = hmac.new(
        secret.encode("utf-8"),
        raw_bytes,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_sig.lower(), sig.lower())


def extract_cvat_signature_header(headers: Dict[str, Any]) -> Optional[str]:
    """Case-insensitive extraction of CVAT webhook signature header.

    Supports 'X-Signature-256', 'x-signature-256', 'X-CVAT-Signature',
    'x-cvat-signature', and 'X-Hub-Signature-256'.
    """
    if not isinstance(headers, dict):
        return None
    lower_map = {str(k).lower(): str(v) for k, v in headers.items()}
    for key in ("x-signature-256", "x-cvat-signature", "x-hub-signature-256"):
        if key in lower_map:
            return lower_map[key]
    return None


def extract_cvat_event_header(headers: Dict[str, Any]) -> Optional[str]:
    """Case-insensitive extraction of CVAT webhook event header."""
    if not isinstance(headers, dict):
        return None
    lower_map = {str(k).lower(): str(v) for k, v in headers.items()}
    for key in ("x-cvat-event", "x-event"):
        if key in lower_map:
            return lower_map[key]
    return None


@dataclass
class SyncReport:
    """Report summarizing CVAT synchronization run."""
    job_id: int
    task_id: Optional[int] = None
    frames_checked: int = 0
    frames_reconciled: int = 0
    total_human_shapes: int = 0
    corrections_found: Dict[str, int] = field(default_factory=dict)
    rules_derived: int = 0
    success: bool = True
    message: str = ""

    @property
    def total_corrections(self) -> int:
        if isinstance(self.corrections_found, dict):
            return sum(self.corrections_found.values())
        try:
            return int(self.corrections_found)
        except (TypeError, ValueError):
            return 0

    @property
    def corrections_recorded(self) -> int:
        return self.total_corrections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "frames_checked": self.frames_checked,
            "frames_reconciled": self.frames_reconciled,
            "total_human_shapes": self.total_human_shapes,
            "corrections_found": self.corrections_found,
            "corrections_recorded": self.corrections_recorded,
            "rules_derived": self.rules_derived,
            "success": self.success,
            "message": self.message,
        }


class CVATSyncClient:
    """Client for querying CVAT REST API with Docker Django ORM fallback."""

    def __init__(
        self,
        base_url: str = "http://localhost:18080",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.cvat+json"})
        if token:
            self.session.headers.update({"Authorization": f"Token {token}"})

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Fetch job details from CVAT REST API."""
        url = f"{self.base_url}/api/jobs/{job_id}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("Failed GET %s: %s", url, e)
        return None

    def get_labels(self, task_id: int) -> Dict[int, str]:
        """Fetch task labels mapping {label_id: label_name}."""
        url = f"{self.base_url}/api/labels?task_id={task_id}&page_size=100"
        mapping: Dict[int, str] = {}
        try:
            while url:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    break
                data = resp.json()
                results = data.get("results", data if isinstance(data, list) else [])
                for item in results:
                    lid = item.get("id")
                    name = item.get("name")
                    if lid is not None and name:
                        mapping[int(lid)] = str(name)
                url = data.get("next") if isinstance(data, dict) else None
        except Exception as e:
            logger.debug("Failed GET %s: %s", url, e)
        return mapping

    def get_annotations(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Fetch all annotations for a given job."""
        url = f"{self.base_url}/api/jobs/{job_id}/annotations"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("Failed GET %s: %s", url, e)
        return None

    def get_frame_image(self, job_id: int, frame: int) -> Optional[bytes]:
        """Fetch image bytes for a specific frame from CVAT job."""
        url = f"{self.base_url}/api/jobs/{job_id}/data?type=frame&number={frame}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.content
        except Exception as e:
            logger.debug("Failed GET %s: %s", url, e)
        return None

    def list_webhooks(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch all webhooks registered in CVAT, optionally filtered by project_id."""
        url = f"{self.base_url}/api/webhooks"
        params: Dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = int(project_id)
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", data if isinstance(data, list) else [])
        except Exception as e:
            logger.debug("Failed GET %s: %s", url, e)
        return []

    def create_webhook(
        self,
        target_url: str,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        description: str = "tool-cvat-feedback",
        project_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Register a new webhook in CVAT."""
        url = f"{self.base_url}/api/webhooks"
        payload: Dict[str, Any] = {
            "target_url": target_url,
            "description": description,
            "events": events or ["update:job", "create:job"],
            "is_active": True,
        }
        if secret:
            payload["secret"] = secret
        if project_id is not None:
            payload["project_id"] = int(project_id)
            payload["type"] = "project"
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code in (200, 201):
                return resp.json()
            logger.warning("Failed POST %s (status %d): %s", url, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("Failed POST %s: %s", url, e)
        return None

    def update_webhook(
        self,
        webhook_id: int,
        target_url: Optional[str] = None,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        description: Optional[str] = None,
        project_id: Optional[int] = None,
        is_active: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing CVAT webhook."""
        url = f"{self.base_url}/api/webhooks/{webhook_id}"
        payload: Dict[str, Any] = {"is_active": is_active}
        if target_url:
            payload["target_url"] = target_url
        if events:
            payload["events"] = events
        if secret:
            payload["secret"] = secret
        if description:
            payload["description"] = description
        if project_id is not None:
            payload["project_id"] = int(project_id)
            payload["type"] = "project"
        try:
            resp = self.session.patch(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Failed PATCH %s (status %d): %s", url, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("Failed PATCH %s: %s", url, e)
        return None

    def setup_webhook(
        self,
        target_url: str,
        project_id: Optional[int] = None,
        secret: Optional[str] = None,
        description: str = "tool-cvat-feedback",
        events: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Safely inspect, create, or update the feedback webhook.

        Idempotent: matches by project_id, target_url, or description to update
        existing webhooks rather than creating duplicates.
        """
        # Guard against positional secret if passed as second argument
        if isinstance(project_id, str) and not project_id.isdigit():
            secret = project_id
            project_id = None
        elif project_id is not None:
            try:
                project_id = int(project_id)
            except (ValueError, TypeError):
                pass

        existing = self.list_webhooks()
        matching = None

        # Priority 1: Match project_id (if project_id provided)
        if project_id is not None:
            for wh in existing:
                wh_proj = wh.get("project_id")
                if wh_proj is not None:
                    try:
                        if int(wh_proj) == int(project_id):
                            matching = wh
                            break
                    except (ValueError, TypeError):
                        pass

        # Priority 2: Match target_url
        if matching is None and target_url:
            for wh in existing:
                if wh.get("target_url") == target_url:
                    matching = wh
                    break

        # Priority 3: Match description
        if matching is None and description:
            for wh in existing:
                if wh.get("description") == description:
                    matching = wh
                    break

        if matching:
            wh_id = matching["id"]
            updated = self.update_webhook(
                webhook_id=wh_id,
                target_url=target_url,
                events=events or ["update:job", "create:job"],
                secret=secret,
                description=description,
                project_id=project_id,
            )
            return {"action": "updated", "webhook": updated or matching, "id": wh_id}
        else:
            created = self.create_webhook(
                target_url=target_url,
                events=events or ["update:job", "create:job"],
                secret=secret,
                description=description,
                project_id=project_id,
            )
            return {"action": "created", "webhook": created, "id": created.get("id") if created else None}


def sync_job_feedback(
    job_id: int,
    cvat_client: Optional[CVATSyncClient] = None,
    db: Optional[FeedbackDatabase] = None,
    diff_engine: Optional[CorrectionDiffEngine] = None,
    cvat_url: str = "http://localhost:18080",
    token: Optional[str] = None,
    min_rule_samples: int = 3,
) -> SyncReport:
    """Synchronize annotations from a CVAT job and update correction memory.

    Reconciles AI prediction baselines with final human shapes:
    1. Fetches shapes and task labels for the job.
    2. Groups human shapes by frame.
    3. Hashes frame images to retrieve corresponding AI baseline prediction.
    4. Executes bipartite matching CorrectionDiffEngine.
    5. Records corrections, generates crops, and derives updated rules.

    Args:
        job_id: ID of the CVAT job to synchronize.
        cvat_client: Optional configured CVATSyncClient.
        db: Optional FeedbackDatabase instance.
        diff_engine: Optional CorrectionDiffEngine instance.
        cvat_url: CVAT base URL (default http://localhost:18080).
        token: Optional CVAT user authentication token.
        min_rule_samples: Minimum samples required to synthesize a rule.

    Returns:
        SyncReport with detailed breakdown of reconciled frames and corrections.
    """
    client = cvat_client or CVATSyncClient(base_url=cvat_url, token=token)
    database = db or FeedbackDatabase()
    engine = diff_engine or CorrectionDiffEngine()

    report = SyncReport(job_id=job_id)

    # 1. Fetch Job Metadata
    job_data = client.get_job(job_id)
    task_id = job_data.get("task_id") if job_data else None
    report.task_id = task_id

    # 2. Fetch Labels
    label_map: Dict[int, str] = {}
    if task_id:
        fetched_map = client.get_labels(task_id)
        if isinstance(fetched_map, dict):
            label_map = fetched_map

    # 3. Fetch Job Annotations
    anno_data = client.get_annotations(job_id)
    if not anno_data or "shapes" not in anno_data:
        report.success = False
        report.message = f"No annotation shapes found for job #{job_id} or API unreachable."
        return report

    shapes = anno_data.get("shapes", [])
    report.total_human_shapes = len(shapes)

    # 4. Group Human Shapes by Frame Index
    frames_dict: Dict[int, List[Dict[str, Any]]] = {}
    for s in shapes:
        frame_idx = int(s.get("frame", 0))
        label_id = s.get("label_id")
        raw_label = s.get("label")
        if label_id is not None and label_id in label_map:
            label_name = str(label_map[label_id])
        elif raw_label is not None:
            label_name = str(raw_label)
        else:
            label_name = str(label_id) if label_id is not None else ""

        converted_shape = {
            "id": s.get("id"),
            "type": s.get("type", "rectangle"),
            "label": label_name,
            "points": s.get("points", []),
            "group_id": s.get("group"),
        }
        if "mask" in s:
            converted_shape["mask"] = s["mask"]

        frames_dict.setdefault(frame_idx, []).append(converted_shape)

    report.frames_checked = len(frames_dict)
    correction_counts: Dict[str, int] = {}

    # 5. Process each frame
    for frame_idx, human_shapes in frames_dict.items():
        # Fetch frame image to compute hash
        img_bytes = client.get_frame_image(job_id, frame_idx)
        if not img_bytes:
            continue

        try:
            image_hash = compute_image_hash(img_bytes)
        except Exception:
            continue

        try:
            pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_w, img_h = pil_image.width, pil_image.height
        except Exception:
            pil_image = None
            img_w, img_h = 1280, 720

        # Look up baseline prediction
        baseline = database.get_prediction_baseline(image_hash)
        if not baseline:
            # If no direct hash match, check if predictions were recorded with job_id and frame_index
            with database._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT shapes_json FROM predictions WHERE job_id = ? AND frame_index = ? ORDER BY id DESC LIMIT 1",
                    (job_id, frame_idx),
                )
                row = cur.fetchone()
                if row:
                    baseline = {"shapes": json.loads(row["shapes_json"])}

        # Save human annotations record
        database.save_human_annotations(
            image_hash=image_hash,
            shapes=human_shapes,
            task_id=task_id,
            job_id=job_id,
            frame_index=frame_idx,
            quality_tier="completed",
        )

        if baseline and "shapes" in baseline:
            ai_shapes = baseline["shapes"]
            diff_items = engine.diff(
                ai_shapes=ai_shapes,
                human_shapes=human_shapes,
                image_width=img_w,
                image_height=img_h,
            )

            # Record corrections into SQLite database
            database.record_corrections(
                corrections=diff_items,
                image_hash=image_hash,
                image=pil_image,
                task_id=task_id,
                job_id=job_id,
                frame_index=frame_idx,
            )

            report.frames_reconciled += 1
            for item in diff_items:
                correction_counts[item.correction_type] = correction_counts.get(item.correction_type, 0) + 1

    # 6. Derive Updated Rules
    new_rules = database.derive_rules(min_samples=min_rule_samples)
    report.rules_derived = len(new_rules)
    report.corrections_found = correction_counts
    report.success = True
    report.message = f"Synchronized job #{job_id}: {report.frames_reconciled}/{report.frames_checked} frames reconciled with baseline."

    return report


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="CVAT Feedback Synchronization and Webhook Setup")
    parser.add_argument("--job-id", type=int, help="CVAT job ID to synchronize")
    parser.add_argument("--cvat-url", type=str, default=os.getenv("CVAT_URL", "http://localhost:18080"), help="CVAT server URL")
    parser.add_argument("--token", type=str, default=os.getenv("CVAT_TOKEN"), help="CVAT authentication token")
    parser.add_argument("--setup-webhook", action="store_true", help="Inspect or register CVAT feedback webhook")
    parser.add_argument("--target-url", type=str, default="http://nuclio-nuclio-ninerouter-vision-31:8080", help="Webhook destination URL")
    parser.add_argument("--webhook-secret", type=str, default=os.getenv("CVAT_WEBHOOK_SECRET"), help="Webhook HMAC shared secret")
    parser.add_argument("--project-id", type=int, default=None, help="CVAT project ID to scope webhook to")
    parser.add_argument("--list-webhooks", action="store_true", help="List registered webhooks")

    args = parser.parse_args()
    sync_client = CVATSyncClient(base_url=args.cvat_url, token=args.token)

    if args.list_webhooks:
        hooks = sync_client.list_webhooks(project_id=args.project_id)
        print(f"Registered Webhooks ({len(hooks)}):")
        for h in hooks:
            proj_str = f" | Project: {h.get('project_id')}" if h.get('project_id') is not None else ""
            print(f"  ID: {h.get('id')} | URL: {h.get('target_url')} | Desc: {h.get('description')}{proj_str}")
    elif args.setup_webhook:
        res = sync_client.setup_webhook(
            target_url=args.target_url,
            project_id=args.project_id,
            secret=args.webhook_secret,
        )
        print(f"Webhook setup result: {res['action']} (ID: {res.get('id')})")
    elif args.job_id:
        sync_report = sync_job_feedback(job_id=args.job_id, cvat_client=sync_client)
        print(json.dumps(sync_report.to_dict(), indent=2))
    else:
        parser.print_help()

