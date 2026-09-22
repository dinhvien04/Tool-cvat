"""Tests verifying Data Safety, Production Resource Protection, and Safe Test Harness.

Specifically satisfies requirement:
- Production tasks/jobs (12, 13, 15) must be STRICTLY READ-ONLY.
- Tests must record specifically created IDs and ONLY delete those IDs.
- Given 100 existing annotations, creating 1 test annotation and running cleanup leaves all 100 existing annotations untouched.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List
import pytest

from core.cvat_safety import (
    DEFAULT_PROTECTED_JOBS,
    DEFAULT_PROTECTED_TASKS,
    DataSafetyViolationError,
    SafeAnnotationHarness,
    assert_job_writable,
    assert_task_writable,
    is_job_protected,
    is_task_protected,
)


def generate_100_mock_annotations() -> List[Dict[str, Any]]:
    """Generates 100 diverse realistic CVAT annotations with IDs 1 to 100."""
    shapes = []
    for i in range(1, 101):
        if i % 4 == 1:
            # Rectangle
            shapes.append({
                "id": i,
                "type": "rectangle",
                "label_id": 10 + (i % 5),
                "frame": i % 10,
                "group": i,
                "points": [10.0 + i, 20.0 + i, 100.0 + i, 150.0 + i],
                "occluded": False,
                "z_order": 0,
            })
        elif i % 4 == 2:
            # Mask
            shapes.append({
                "id": i,
                "type": "mask",
                "label_id": 10 + (i % 5),
                "frame": i % 10,
                "group": i - 1,
                "points": [0, 50, 10.0 + i, 20.0 + i, 100.0 + i, 150.0 + i],
                "occluded": False,
                "z_order": 0,
            })
        elif i % 4 == 3:
            # Polygon
            shapes.append({
                "id": i,
                "type": "polygon",
                "label_id": 20 + (i % 5),
                "frame": i % 10,
                "group": 0,
                "points": [10.0 + i, 10.0, 50.0 + i, 10.0, 50.0 + i, 50.0, 10.0 + i, 50.0],
                "occluded": False,
                "z_order": 0,
            })
        else:
            # Polyline
            shapes.append({
                "id": i,
                "type": "polyline",
                "label_id": 30 + (i % 5),
                "frame": i % 10,
                "group": None,
                "points": [100.0 + i, 200.0, 150.0 + i, 250.0, 200.0 + i, 300.0],
                "occluded": False,
                "z_order": 0,
            })
    return shapes


class TestProductionResourceProtection:
    """Verify production tasks and jobs (12, 13, 15) are strictly READ-ONLY."""

    @pytest.mark.parametrize("job_id", [12, 13, 15])
    def test_protected_jobs_reject_write_access(self, job_id: int):
        """Jobs 12, 13, and 15 must be flagged protected and reject mutation assertion."""
        assert is_job_protected(job_id) is True
        with pytest.raises(DataSafetyViolationError) as exc_info:
            assert_job_writable(job_id)
        assert f"Job {job_id} is a PROTECTED production job" in str(exc_info.value)
        assert "STRICTLY READ-ONLY" in str(exc_info.value)

    @pytest.mark.parametrize("task_id", [12, 13, 15])
    def test_protected_tasks_reject_write_access(self, task_id: int):
        """Tasks 12, 13, and 15 must be flagged protected and reject mutation assertion."""
        assert is_task_protected(task_id) is True
        with pytest.raises(DataSafetyViolationError) as exc_info:
            assert_task_writable(task_id)
        assert f"Task {task_id} is a PROTECTED production task" in str(exc_info.value)
        assert "STRICTLY READ-ONLY" in str(exc_info.value)

    @pytest.mark.parametrize("job_id", [12, 13, 15])
    def test_harness_instantiation_blocks_protected_jobs(self, job_id: int):
        """SafeAnnotationHarness must immediately refuse instantiation on protected jobs."""
        with pytest.raises(DataSafetyViolationError):
            SafeAnnotationHarness(job_id=job_id, enforce_safety=True)

    def test_unprotected_disposable_job_allowed(self):
        """Disposable jobs (e.g. 9999 or temporary IDs) must be writable."""
        disposable_job_id = 9999
        assert is_job_protected(disposable_job_id) is False
        assert_job_writable(disposable_job_id)  # Does not raise
        harness = SafeAnnotationHarness(job_id=disposable_job_id, enforce_safety=True)
        assert harness.job_id == disposable_job_id


class TestSafeAnnotationHarnessZeroCollateralDamage:
    """Verify that given 100 existing annotations, creating 1 test annotation
    and running cleanup leaves all 100 existing annotations untouched.
    """

    def test_100_existing_annotations_untouched_after_test_annotation_cleanup(self):
        """CORE SAFETY CONTRACT:
        Given: 100 pre-existing annotations in a job.
        Action: 1 test annotation is created (ID 101).
        Action: Cleanup is performed via SafeAnnotationHarness.
        Assert:
        - Exactly 1 shape (ID 101) is targeted for deletion.
        - Zero of the 100 existing annotations are marked for deletion.
        - All 100 existing annotations remain completely untouched with exact coordinates preserved.
        """
        disposable_job_id = 8888
        harness = SafeAnnotationHarness(job_id=disposable_job_id)

        # 1. Snapshot the 100 existing annotations
        existing_100 = generate_100_mock_annotations()
        assert len(existing_100) == 100
        existing_ids = {s["id"] for s in existing_100}
        assert len(existing_ids) == 100
        assert existing_ids == set(range(1, 101))

        initial_snapshot = harness.snapshot_existing_shapes(existing_100)
        assert initial_snapshot == existing_ids

        # 2. Simulate creating 1 test annotation (ID 101)
        created_test_shape = {
            "id": 101,
            "type": "rectangle",
            "label_id": 99,
            "frame": 0,
            "group": 0,
            "points": [50.0, 50.0, 150.0, 150.0],
            "occluded": False,
            "z_order": 0,
        }
        registered_ids = harness.record_created_shapes([created_test_shape])
        assert registered_ids == [101]
        assert harness.created_shape_ids == {101}

        # 3. Simulate job state containing all 101 shapes (100 existing + 1 created)
        current_job_shapes = copy.deepcopy(existing_100) + [copy.deepcopy(created_test_shape)]
        assert len(current_job_shapes) == 101

        # 4. Compute cleanup plan
        plan = harness.compute_cleanup_plan(current_job_shapes)

        # 5. Assert cleanup plan invariants
        assert plan["collateral_damage_detected"] is False
        assert len(plan["collateral_shape_ids"]) == 0
        assert len(plan["missing_initial_ids"]) == 0

        # Shapes to delete MUST contain ONLY ID 101
        assert len(plan["shapes_to_delete"]) == 1
        assert plan["shapes_to_delete"][0]["id"] == 101
        assert plan["delete_shape_ids"] == {101}

        # Preserved shapes MUST contain ALL 100 initial shapes
        assert len(plan["preserved_shapes"]) == 100
        assert plan["preserved_shape_ids"] == existing_ids
        assert plan["untouched_initial_count"] == 100

        # Simulate applying deletion of only shapes_to_delete
        delete_ids = plan["delete_shape_ids"]
        final_job_shapes = [s for s in current_job_shapes if s["id"] not in delete_ids]

        # 6. Verify zero collateral damage against post-cleanup state
        harness.verify_zero_collateral_damage(final_job_shapes)

        # 7. Exact coordinate and property preservation check on all 100 shapes
        final_map = {s["id"]: s for s in final_job_shapes}
        assert len(final_map) == 100
        for orig in existing_100:
            oid = orig["id"]
            assert oid in final_map, f"Existing shape {oid} was lost!"
            curr = final_map[oid]
            assert curr["points"] == orig["points"], f"Shape {oid} points were modified!"
            assert curr["type"] == orig["type"], f"Shape {oid} type was modified!"
            assert curr["label_id"] == orig["label_id"], f"Shape {oid} label_id was modified!"
            assert curr["group"] == orig["group"], f"Shape {oid} group was modified!"

    def test_detection_of_collateral_damage_prevents_deletion(self):
        """If a buggy cleanup attempts to delete pre-existing IDs, the harness MUST detect and abort."""
        harness = SafeAnnotationHarness(job_id=8888)
        existing_100 = generate_100_mock_annotations()
        harness.snapshot_existing_shapes(existing_100)

        # Maliciously or buggily inject an existing ID into created_shape_ids
        harness.created_shape_ids = {50, 101}  # 50 is an existing shape!

        current_shapes = copy.deepcopy(existing_100) + [{"id": 101, "type": "rectangle"}]
        plan = harness.compute_cleanup_plan(current_shapes)

        # Plan MUST detect collateral damage
        assert plan["collateral_damage_detected"] is True
        assert 50 in plan["collateral_shape_ids"]

    def test_verify_zero_collateral_damage_raises_on_missing_shape(self):
        """verify_zero_collateral_damage must raise DataSafetyViolationError if even 1 shape is missing."""
        harness = SafeAnnotationHarness(job_id=8888)
        existing_100 = generate_100_mock_annotations()
        harness.snapshot_existing_shapes(existing_100)
        harness.record_created_shapes([{"id": 101, "type": "rectangle"}])

        # Accidentally omit shape 42 from final shapes
        corrupted_final = [s for s in existing_100 if s["id"] != 42]
        with pytest.raises(DataSafetyViolationError) as exc_info:
            harness.verify_zero_collateral_damage(corrupted_final)
        assert "pre-existing annotations were deleted or lost" in str(exc_info.value)
        assert "42" in str(exc_info.value)

    def test_verify_zero_collateral_damage_raises_on_modified_coordinates(self):
        """verify_zero_collateral_damage must raise DataSafetyViolationError if shape coordinates changed."""
        harness = SafeAnnotationHarness(job_id=8888)
        existing_100 = generate_100_mock_annotations()
        harness.snapshot_existing_shapes(existing_100)
        harness.record_created_shapes([{"id": 101, "type": "rectangle"}])

        corrupted_final = copy.deepcopy(existing_100)
        corrupted_final[0]["points"] = [999.0, 999.0, 1000.0, 1000.0]  # Corrupted!

        with pytest.raises(DataSafetyViolationError) as exc_info:
            harness.verify_zero_collateral_damage(corrupted_final)
        assert "DATA CORRUPTION" in str(exc_info.value)
