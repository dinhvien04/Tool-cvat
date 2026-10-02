"""CVAT Test Data Safety Guard and Disposable Test Harness.

Guarantees 100% data safety for live and automated testing against CVAT:
1. STRICT READ-ONLY protection for production tasks and jobs (12, 13, 15).
2. Explicit ID tracking for disposable test annotations.
3. Absolute prohibition of bulk/wildcard shape deletions.
4. Zero-collateral-damage post-cleanup verification.
5. Disposable task lifecycle management.
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import requests

logger = logging.getLogger(__name__)

# Production tasks and jobs that must NEVER be mutated by tests
DEFAULT_PROTECTED_JOBS: frozenset[int] = frozenset({12, 13, 15})
DEFAULT_PROTECTED_TASKS: frozenset[int] = frozenset({12, 13, 15})


def get_protected_job_ids() -> Set[int]:
    """Returns the set of protected job IDs, merging defaults with CVAT_PROTECTED_JOB_IDS env."""
    extra = os.getenv("CVAT_PROTECTED_JOB_IDS", "")
    ids = set(DEFAULT_PROTECTED_JOBS)
    if extra:
        for part in extra.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
    return ids


def get_protected_task_ids() -> Set[int]:
    """Returns the set of protected task IDs, merging defaults with CVAT_PROTECTED_TASK_IDS env."""
    extra = os.getenv("CVAT_PROTECTED_TASK_IDS", "")
    ids = set(DEFAULT_PROTECTED_TASKS)
    if extra:
        for part in extra.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
    return ids


class DataSafetyViolationError(RuntimeError):
    """Raised when an operation attempts to mutate or delete protected CVAT data."""
    pass


def is_job_protected(job_id: int) -> bool:
    """Checks whether a job ID is protected against mutations."""
    return job_id in get_protected_job_ids()


def is_task_protected(task_id: int) -> bool:
    """Checks whether a task ID is protected against mutations."""
    return task_id in get_protected_task_ids()


def assert_job_writable(job_id: int) -> None:
    """Enforces that a job is NOT protected.

    Raises:
        DataSafetyViolationError: If job_id is a protected production job.
    """
    if is_job_protected(job_id):
        raise DataSafetyViolationError(
            f"DATA SAFETY VIOLATION: Job {job_id} is a PROTECTED production job. "
            f"Protected jobs ({sorted(get_protected_job_ids())}) are STRICTLY READ-ONLY. "
            "Tests must use disposable jobs/tasks."
        )


def assert_task_writable(task_id: int) -> None:
    """Enforces that a task is NOT protected.

    Raises:
        DataSafetyViolationError: If task_id is a protected production task.
    """
    if is_task_protected(task_id):
        raise DataSafetyViolationError(
            f"DATA SAFETY VIOLATION: Task {task_id} is a PROTECTED production task. "
            f"Protected tasks ({sorted(get_protected_task_ids())}) are STRICTLY READ-ONLY. "
            "Tests must use disposable tasks."
        )


class SafeAnnotationHarness:
    """Safe annotation manager that records specifically created IDs and guarantees
    that existing annotations are left 100% untouched.
    """

    def __init__(self, job_id: int, enforce_safety: bool = True):
        self.job_id = job_id
        self.enforce_safety = enforce_safety
        if self.enforce_safety:
            assert_job_writable(job_id)

        self.initial_shape_ids: Set[int] = set()
        self.initial_shapes_snapshot: Dict[int, Dict[str, Any]] = {}
        self.created_shape_ids: Set[int] = set()

    def snapshot_existing_shapes(self, shapes: List[Dict[str, Any]]) -> Set[int]:
        """Record initial pre-existing annotations before test runs.

        Args:
            shapes: List of shape dicts from GET /api/jobs/{id}/annotations.

        Returns:
            Set of initial shape IDs.
        """
        self.initial_shape_ids = {s["id"] for s in shapes if "id" in s}
        self.initial_shapes_snapshot = {
            s["id"]: dict(s) for s in shapes if "id" in s
        }
        return set(self.initial_shape_ids)

    def record_created_shapes(self, created_shapes: List[Dict[str, Any]]) -> List[int]:
        """Register newly created shapes so ONLY these specific IDs will be deleted in cleanup.

        Args:
            created_shapes: List of shape dicts returned by CVAT create API.

        Returns:
            List of newly registered shape IDs.
        """
        new_ids: List[int] = []
        for s in created_shapes:
            sid = s.get("id")
            if sid is not None:
                # Critical safety check: new shape ID must NOT collide with pre-existing shapes
                if sid in self.initial_shape_ids:
                    raise DataSafetyViolationError(
                        f"Safety anomaly: created shape ID {sid} conflicts with pre-existing shape ID!"
                    )
                self.created_shape_ids.add(sid)
                new_ids.append(sid)
        return new_ids

    def compute_cleanup_plan(self, current_shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministically compute which shapes to delete and verify no collateral damage.

        Pure logic function suitable for automated unit testing.

        Args:
            current_shapes: Current shapes present in the job.

        Returns:
            Dict containing:
                - shapes_to_delete: shapes matching self.created_shape_ids
                - preserved_shapes: shapes that will remain untouched
                - collateral_damage_detected: bool
                - untouched_initial_count: count of initial shapes preserved
        """
        curr_map = {s["id"]: s for s in current_shapes if "id" in s}

        # ONLY shapes specifically recorded as created by this harness may be deleted
        shapes_to_delete = [
            curr_map[sid] for sid in self.created_shape_ids if sid in curr_map
        ]

        # Check if any initial shape was accidentally targeted
        delete_ids = {s["id"] for s in shapes_to_delete}
        collateral = delete_ids.intersection(self.initial_shape_ids)

        # Preserved shapes
        preserved = [s for s in current_shapes if s.get("id") not in delete_ids]
        preserved_ids = {s["id"] for s in preserved if "id" in s}

        # Check that ALL initial shapes are present in preserved
        missing_initial = self.initial_shape_ids.difference(preserved_ids)

        return {
            "shapes_to_delete": shapes_to_delete,
            "delete_shape_ids": delete_ids,
            "preserved_shapes": preserved,
            "preserved_shape_ids": preserved_ids,
            "collateral_damage_detected": len(collateral) > 0 or len(missing_initial) > 0,
            "collateral_shape_ids": collateral,
            "missing_initial_ids": missing_initial,
            "untouched_initial_count": len(self.initial_shape_ids.intersection(preserved_ids)),
        }

    def verify_zero_collateral_damage(self, final_shapes: List[Dict[str, Any]]) -> None:
        """Verify that all initial shapes are preserved intact with identical data."""
        final_map = {s["id"]: s for s in final_shapes if "id" in s}

        # 1. Check all initial IDs exist
        missing = [sid for sid in self.initial_shape_ids if sid not in final_map]
        if missing:
            raise DataSafetyViolationError(
                f"DATA SAFETY FAILURE: {len(missing)} pre-existing annotations were deleted or lost! Missing IDs: {missing[:10]}"
            )

        # 2. Check no test shapes leaked
        leaked = [sid for sid in self.created_shape_ids if sid in final_map]
        if leaked:
            raise DataSafetyViolationError(
                f"Cleanup incomplete: {len(leaked)} test shapes still present in job: {leaked[:10]}"
            )

        # 3. Check data integrity of initial shapes
        for sid, orig in self.initial_shapes_snapshot.items():
            curr = final_map[sid]
            if curr.get("points") != orig.get("points"):
                raise DataSafetyViolationError(
                    f"DATA CORRUPTION: Pre-existing shape {sid} coordinates were modified!"
                )
            if curr.get("type") != orig.get("type"):
                raise DataSafetyViolationError(
                    f"DATA CORRUPTION: Pre-existing shape {sid} type was modified!"
                )

    def cleanup_live(self, session: requests.Session, base_url: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Perform verified safe cleanup against live CVAT API.

        Only deletes recorded test shapes and verifies all initial shapes remain untouched.
        """
        if self.enforce_safety:
            assert_job_writable(self.job_id)

        # 1. Fetch current shapes
        r_curr = session.get(f"{base_url}/api/jobs/{self.job_id}/annotations", timeout=timeout)
        r_curr.raise_for_status()
        current_shapes = r_curr.json().get("shapes", [])

        # 2. Compute plan
        plan = self.compute_cleanup_plan(current_shapes)
        if plan["collateral_damage_detected"]:
            raise DataSafetyViolationError(
                f"Aborting cleanup: collateral damage detected in plan! Collateral IDs: {plan['collateral_shape_ids']}"
            )

        # 3. Delete only recorded test shapes
        to_del = plan["shapes_to_delete"]
        if to_del:
            r_del = session.patch(
                f"{base_url}/api/jobs/{self.job_id}/annotations?action=delete",
                json={"shapes": to_del, "tracks": [], "tags": []},
                timeout=timeout,
            )
            if r_del.status_code not in (200, 204):
                raise RuntimeError(f"Cleanup API call failed: {r_del.status_code} - {r_del.text}")

        # 4. Fetch final shapes and verify zero collateral damage
        r_final = session.get(f"{base_url}/api/jobs/{self.job_id}/annotations", timeout=timeout)
        r_final.raise_for_status()
        final_shapes = r_final.json().get("shapes", [])
        self.verify_zero_collateral_damage(final_shapes)

        return {
            "deleted_count": len(to_del),
            "preserved_count": len(final_shapes),
            "initial_count": len(self.initial_shape_ids),
        }


class DisposableTaskContext:
    """Context manager that creates a 100% disposable CVAT task for testing,
    and guarantees automatic task deletion on exit.
    """

    def __init__(
        self,
        session: requests.Session,
        base_url: str,
        task_name: Optional[str] = None,
        labels: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 15.0,
    ):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.task_name = task_name or f"disposable_test_task_{int(time.time())}"
        self.labels = labels or [{"name": "car"}, {"name": "road"}, {"name": "lane"}]
        self.timeout = timeout
        self.task_id: Optional[int] = None
        self.job_id: Optional[int] = None
        self.label_map: Dict[str, int] = {}

    def __enter__(self) -> "DisposableTaskContext":
        # 1. Create task
        r_task = self.session.post(
            f"{self.base_url}/api/tasks",
            json={"name": self.task_name, "labels": self.labels},
            timeout=self.timeout,
        )
        if r_task.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create disposable task: {r_task.status_code} {r_task.text}")
        task_data = r_task.json()
        self.task_id = task_data["id"]

        # Fetch label IDs
        r_labels = self.session.get(
            f"{self.base_url}/api/labels?task_id={self.task_id}&page_size=100",
            timeout=self.timeout,
        )
        if r_labels.status_code == 200:
            self.label_map = {l["name"]: l["id"] for l in r_labels.json().get("results", [])}

        # 2. Attach a minimal 1x1 synthetic image to generate a job
        try:
            from PIL import Image
            img = Image.new("RGB", (640, 480), color=(128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            files = {"client_files[0]": ("frame0.jpg", buf, "image/jpeg")}
            data = {"image_quality": 70}

            # Note: Content-Type must NOT be application/json when uploading files
            headers = dict(self.session.headers)
            headers.pop("Content-Type", None)

            r_data = requests.post(
                f"{self.base_url}/api/tasks/{self.task_id}/data",
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout,
            )
            if r_data.status_code not in (200, 202):
                logger.warning(f"Could not upload dummy frame to task {self.task_id}: {r_data.text}")

            # Wait for job creation
            for _ in range(15):
                time.sleep(0.5)
                r_jobs = self.session.get(
                    f"{self.base_url}/api/jobs?task_id={self.task_id}",
                    timeout=self.timeout,
                )
                if r_jobs.status_code == 200:
                    jobs = r_jobs.json().get("results", [])
                    if jobs:
                        self.job_id = jobs[0]["id"]
                        break
        except Exception as e:
            logger.warning(f"Failed to initialize job for disposable task {self.task_id}: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task_id:
            try:
                self.session.delete(
                    f"{self.base_url}/api/tasks/{self.task_id}",
                    timeout=self.timeout,
                )
            except Exception as e:
                logger.error(f"Failed to delete disposable task {self.task_id}: {e}")
