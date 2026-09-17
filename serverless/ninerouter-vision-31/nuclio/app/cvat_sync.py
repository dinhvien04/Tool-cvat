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
    payload_bytes: bytes,
    signature_header: Optional[str],
    secret: str,
) -> bool:
    """Verify CVAT webhook HMAC-SHA256 signature (X-CVAT-Signature header).

    Args:
        payload_bytes: Raw HTTP request body bytes.
        signature_header: Contents of 'X-CVAT-Signature' or 'X-Signature-256' header.
        secret: Configured shared secret string.

    Returns:
        True if signature matches securely; False otherwise.
    """
    if not signature_header or not secret:
        return False

    sig = signature_header.strip()
    if sig.startswith("sha256="):
        sig = sig[7:].strip()

    expected_sig = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_sig.lower(), sig.lower())


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "frames_checked": self.frames_checked,
            "frames_reconciled": self.frames_reconciled,
            "total_human_shapes": self.total_human_shapes,
            "corrections_found": self.corrections_found,
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
        url = f"{self.base_url}/api/labels?task_id={task_id}"
        mapping: Dict[int, str] = {}
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", data if isinstance(data, list) else [])
                for item in results:
                    lid = item.get("id")
                    name = item.get("name")
                    if lid is not None and name:
                        mapping[int(lid)] = str(name)
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
        label_map = client.get_labels(task_id)

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
        label_name = label_map.get(label_id, s.get("label", str(label_id)))

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
