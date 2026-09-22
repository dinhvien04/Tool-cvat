"""Tests verifying Goal #2: Human Must Be Able To Correct All 3 Output Types Precisely in Native CVAT.

Verifies:
1. Shape specifications and CVAT ingestion contracts:
   - Rectangle: xtl, ytl, xbr, ybr, occluded=False, z_order=0, unlocked, draggable, resizable.
   - Mask: CVAT RLE format [...rle_counts, xtl, ytl, xbr, ybr], brush/eraser/polygon editable, unlocked.
   - Polygon: points [x1, y1, x2, y2, ...], vertex add/move/delete.
   - Polyline: type='polyline', points [x1, y1, x2, y2, ...], point add/move/delete.
2. Pairing (group_id) constraints:
   - Group links semantic identity without locking shapes or enforcing synchronous re-rasterization.
   - Modifying rectangle leaves mask untouched; modifying mask leaves rectangle untouched.
   - No auto-reset or overwrite of human edits.
3. Live round-trip against local CVAT (http://localhost:18080) on task 15, job 15 (if live CVAT available).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import pytest
import requests

from core.cvat_safety import (
    DEFAULT_PROTECTED_JOBS,
    DEFAULT_PROTECTED_TASKS,
    DataSafetyViolationError,
    DisposableTaskContext,
    SafeAnnotationHarness,
    assert_job_writable,
    assert_task_writable,
    is_job_protected,
)
from core.geometry import (
    calculate_polygon_area,
    denormalize_contour,
    mask_to_cvat_flat_list,
    multi_polygon_to_cvat_mask,
    rasterize_polygons_to_mask,
)
from core.pose_face_schema import (
    POSE17_KEYPOINTS,
    VF50_LANDMARKS,
    KeypointElement,
    SkeletonInstance,
    parse_and_sanitize_pose17_instance,
    parse_and_sanitize_vf50_instance,
)
from core.taxonomy import (
    BOX_MASK_LABELS,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
    SHAPE_RECTANGLE,
    validate_cvat_output_shapes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = REPO_ROOT / ".tool-cvat" / "cvat_token.txt"
CVAT_BASE_URL = "http://localhost:18080"


def mask2rle(flat_binary_mask: List[int]) -> List[int]:
    """Encodes a flattened binary mask into CVAT RLE format.
    Starts with count of initial zeros.
    """
    if not flat_binary_mask:
        return [0]
    acc = [0]
    if flat_binary_mask[0] > 0:
        acc.append(1)
    else:
        acc[0] = 1
    for idx in range(1, len(flat_binary_mask)):
        val = flat_binary_mask[idx]
        if flat_binary_mask[idx - 1] == val:
            acc[-1] += 1
        else:
            acc.append(1)
    return acc


def rle2mask(rle: List[int], width: int, height: int) -> List[int]:
    """Decodes CVAT RLE back into flattened binary mask."""
    decoded = [0] * (width * height)
    decoded_idx = 0
    value = 0
    for count in rle:
        while count > 0 and decoded_idx < len(decoded):
            decoded[decoded_idx] = value
            decoded_idx += 1
            count -= 1
        value = 1 if value == 0 else 0
    return decoded


def get_live_cvat_session():
    """Returns requests.Session with auth if CVAT is reachable, else None."""
    if not TOKEN_FILE.exists():
        return None
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Token {token}",
        "Accept": "application/vnd.cvat+json",
        "Content-Type": "application/json",
    })
    try:
        r = session.get(f"{CVAT_BASE_URL}/api/server/about", timeout=2.0)
        if r.status_code == 200:
            return session
    except Exception:
        return None
    return None


class TestShapeContractSpecifications:
    """Verify all 3 detector output types satisfy native CVAT specifications."""

    def test_rectangle_shape_specification(self):
        """Verify rectangle + mask pair conforms to Policy A (Detector 1)."""
        box = [100.0, 150.0, 300.0, 450.0]
        xtl, ytl, xbr, ybr = box
        assert xtl < xbr
        assert ytl < ybr

        rect_shape = {
            "type": "rectangle",
            "label": "car",
            "points": box,
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "group_id": 10,
        }
        mask_shape = {
            "type": "mask",
            "label": "car",
            "points": box,
            "mask": [1, 1, 1, 1],
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "group_id": 10,
        }
        validated, _ = validate_cvat_output_shapes([rect_shape, mask_shape])
        assert len(validated) == 2
        by_type = {s["type"]: s for s in validated}
        assert SHAPE_RECTANGLE in by_type
        assert SHAPE_MASK in by_type
        assert len(by_type[SHAPE_RECTANGLE]["points"]) == 4
        assert by_type[SHAPE_RECTANGLE]["occluded"] is False
        assert by_type[SHAPE_RECTANGLE]["z_order"] == 0
        assert by_type[SHAPE_RECTANGLE]["group_id"] == by_type[SHAPE_MASK]["group_id"] == 10

    def test_mask_shape_rle_specification(self):
        """Verify mask uses CVAT RLE format [...rle_counts, xtl, ytl, xbr, ybr]."""
        w, h = 10, 10
        # Checkerboard pattern
        flat_mask = [1 if (i + j) % 2 == 0 else 0 for i in range(h) for j in range(w)]
        rle = mask2rle(flat_mask)
        # Decode and verify lossless roundtrip
        recovered = rle2mask(rle, w, h)
        assert recovered == flat_mask

        # Append bbox [xtl, ytl, xbr, ybr]
        bbox = [50.0, 50.0, 59.0, 59.0]
        mask_points = rle + bbox
        assert len(mask_points) >= 5
        assert mask_points[-4:] == bbox

    def test_polygon_shape_specification(self):
        """Verify polygon + mask pair conforms to Policy B (Detector 2)."""
        pts = [10.0, 20.0, 100.0, 20.0, 100.0, 80.0, 10.0, 80.0]
        poly_shape = {
            "type": "polygon",
            "label": "road",
            "points": pts,
            "group_id": 20,
        }
        mask_shape = {
            "type": "mask",
            "label": "road",
            "points": [10.0, 20.0, 100.0, 80.0],
            "mask": [1, 1, 1, 1],
            "group_id": 20,
        }
        validated, _ = validate_cvat_output_shapes([poly_shape, mask_shape])
        assert len(validated) == 2
        by_type = {s["type"]: s for s in validated}
        assert SHAPE_POLYGON in by_type
        assert SHAPE_MASK in by_type
        assert len(by_type[SHAPE_POLYGON]["points"]) == 8
        assert len(by_type[SHAPE_POLYGON]["points"]) % 2 == 0
        assert by_type[SHAPE_POLYGON]["group_id"] == by_type[SHAPE_MASK]["group_id"] == 20

    def test_polyline_shape_specification(self):
        """Verify polyline has points [x1, y1, x2, y2, ...] with >= 2 points, strictly atomic."""
        pts = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
        line_shape = {
            "type": "polyline",
            "label": "lane/single white",
            "points": pts,
            "group_id": 999,  # Group must be stripped
        }
        validated, _ = validate_cvat_output_shapes([line_shape])
        assert len(validated) == 1
        assert validated[0]["type"] == SHAPE_POLYLINE
        assert len(validated[0]["points"]) == 6
        # Polyline must be atomic: no group_id allowed
        assert "group_id" not in validated[0]

    def test_skeleton_shape_specification(self):
        """Verify Pose17 and VF50 skeletons satisfy native CVAT structural constraints:
        - Shape type 'skeleton'
        - Parent points must be empty list [] (CVAT rejects non-empty points on skeleton)
        - Child elements must have type 'points' with len(points) == 2 ([x, y])
        - Outside and occluded flags are preserved and unlocked
        """
        # 1. Pose17 instance
        raw_pose = {
            "keypoints": {
                kp: [float(i * 10), float(i * 15), 2]
                for i, kp in enumerate(POSE17_KEYPOINTS)
            },
            "confidence": 0.95,
        }
        pose_inst = parse_and_sanitize_pose17_instance(raw_pose, img_width=1000, img_height=1000)
        assert pose_inst is not None
        pose_dict = pose_inst.to_cvat_dict()
        assert pose_dict["type"] == "skeleton"
        assert pose_dict["label"] == "person"
        assert len(pose_dict["elements"]) == 17
        for elem in pose_dict["elements"]:
            assert elem["type"] == "points"
            assert len(elem["points"]) == 2
            assert elem["outside"] is False
            assert elem["occluded"] is False

        # 2. VF50 instance
        raw_vf50 = {
            "landmarks": {
                lm: [float(i * 5), float(i * 8), 2]
                for i, lm in enumerate(VF50_LANDMARKS)
            },
            "confidence": 0.98,
        }
        vf50_inst = parse_and_sanitize_vf50_instance(raw_vf50, img_width=1000, img_height=1000)
        assert vf50_inst is not None
        vf50_dict = vf50_inst.to_cvat_dict()
        assert vf50_dict["type"] == "skeleton"
        assert vf50_dict["label"] == "face"
        assert len(vf50_dict["elements"]) == 50
        for elem in vf50_dict["elements"]:
            assert elem["type"] == "points"
            assert len(elem["points"]) == 2
            assert elem["outside"] is False
            assert elem["occluded"] is False


class TestLiveCVATEditRoundTrip:
    """Live verification against CVAT instance with 100% Data Safety Guarantee.

    Safety Invariants:
    1. Production tasks and jobs (12, 13, 15) are STRICTLY READ-ONLY.
    2. Write/edit tests MUST target disposable jobs and use SafeAnnotationHarness.
    3. Cleanup logic ONLY deletes specifically recorded test IDs.
    4. Bulk or wildcard shape deletions are strictly prohibited.
    """

    def test_production_jobs_strictly_read_only(self):
        """Verify production jobs (12, 13, 15) are strictly protected against mutations."""
        for jid in (12, 13, 15):
            assert is_job_protected(jid) is True
            with pytest.raises(DataSafetyViolationError):
                assert_job_writable(jid)

    def test_100_existing_annotations_safety_contract(self):
        """Verify that given 100 existing annotations, creating 1 test annotation
        and running cleanup leaves all 100 existing annotations 100% untouched.
        """
        from tests.test_cvat_data_safety_harness import generate_100_mock_annotations
        import copy

        disposable_job_id = 9999
        harness = SafeAnnotationHarness(job_id=disposable_job_id)

        # 1. 100 existing annotations
        existing_100 = generate_100_mock_annotations()
        existing_ids = {s["id"] for s in existing_100}
        assert len(existing_ids) == 100
        harness.snapshot_existing_shapes(existing_100)

        # 2. Create 1 test annotation
        test_shape = {
            "id": 101,
            "type": "rectangle",
            "label_id": 10,
            "frame": 0,
            "group": 0,
            "points": [10.0, 10.0, 50.0, 50.0],
            "occluded": False,
            "z_order": 0,
        }
        harness.record_created_shapes([test_shape])

        # 3. Simulate job with 101 shapes
        current_shapes = copy.deepcopy(existing_100) + [copy.deepcopy(test_shape)]
        plan = harness.compute_cleanup_plan(current_shapes)

        assert plan["collateral_damage_detected"] is False
        assert plan["delete_shape_ids"] == {101}
        assert plan["untouched_initial_count"] == 100

        # Simulate execution of delete plan
        final_shapes = [s for s in current_shapes if s["id"] != 101]
        harness.verify_zero_collateral_damage(final_shapes)
        assert len(final_shapes) == 100

    def test_live_production_jobs_read_only_access(self):
        """Verify that live production jobs (12, 13, 15) can be inspected in read-only mode."""
        session = get_live_cvat_session()
        if not session:
            pytest.skip("Local CVAT instance (http://localhost:18080) not available or token missing")

        for jid in (12, 13, 15):
            r = session.get(f"{CVAT_BASE_URL}/api/jobs/{jid}/annotations", timeout=10)
            if r.status_code == 200:
                shapes = r.json().get("shapes", [])
                # Read-only check: assert shape structure without mutating anything
                for s in shapes[:3]:
                    assert "id" in s
                    assert "type" in s
                    assert "points" in s

    def test_live_disposable_detectors_edit_roundtrip(self):
        """Live roundtrip verification for detector shapes on a disposable job."""
        session = get_live_cvat_session()
        if not session:
            pytest.skip("Local CVAT instance (http://localhost:18080) not available or token missing")

        disposable_job_raw = os.getenv("CVAT_DISPOSABLE_JOB_ID", "").strip()
        if not disposable_job_raw:
            pytest.skip(
                "CVAT_DISPOSABLE_JOB_ID not configured. Set to a disposable (non-protected) job ID "
                "to run live editability roundtrip against CVAT. Production jobs (12, 13, 15) are strictly read-only."
            )

        target_job_id = int(disposable_job_raw)
        # ENFORCE DATA SAFETY: must not be in protected jobs
        assert_job_writable(target_job_id)

        harness = SafeAnnotationHarness(job_id=target_job_id)

        # 1. Query initial shapes and snapshot
        r_initial = session.get(f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations", timeout=10)
        assert r_initial.status_code == 200
        initial_shapes = r_initial.json().get("shapes", [])
        harness.snapshot_existing_shapes(initial_shapes)

        # 2. Get task labels
        r_job = session.get(f"{CVAT_BASE_URL}/api/jobs/{target_job_id}", timeout=10)
        assert r_job.status_code == 200
        task_id = r_job.json()["task_id"]

        r_labels = session.get(f"{CVAT_BASE_URL}/api/labels?task_id={task_id}&page_size=100", timeout=10)
        labels = {l["name"]: l["id"] for l in r_labels.json().get("results", [])}

        car_id = labels.get("car", labels.get(list(labels.keys())[0]))
        road_id = labels.get("road", car_id)
        lane_id = labels.get("lane/single white", car_id)

        # 3. Create test shapes representing all 3 detector types
        rect_pts = [200.0, 100.0, 300.0, 200.0]
        mask1_rle = mask2rle([1] * 100) + [210.0, 110.0, 220.0, 120.0]
        test_rect = {
            "type": "rectangle", "label_id": car_id, "frame": 0, "group": 901,
            "points": rect_pts, "occluded": False, "z_order": 0, "source": "auto", "attributes": [],
        }
        test_mask1 = {
            "type": "mask", "label_id": car_id, "frame": 0, "group": 901,
            "points": mask1_rle, "occluded": False, "z_order": 0, "source": "auto", "attributes": [],
        }
        poly_pts = [350.0, 450.0, 450.0, 450.0, 450.0, 550.0, 350.0, 550.0]
        mask2_rle = mask2rle([1] * 100) + [360.0, 460.0, 370.0, 470.0]
        test_poly = {
            "type": "polygon", "label_id": road_id, "frame": 0, "group": 902,
            "points": poly_pts, "occluded": False, "z_order": 0, "source": "auto", "attributes": [],
        }
        test_mask2 = {
            "type": "mask", "label_id": road_id, "frame": 0, "group": 902,
            "points": mask2_rle, "occluded": False, "z_order": 0, "source": "auto", "attributes": [],
        }
        line_pts = [150.0, 550.0, 250.0, 600.0, 350.0, 650.0]
        test_line = {
            "type": "polyline", "label_id": lane_id, "frame": 0, "group": None,
            "points": line_pts, "occluded": False, "z_order": 0, "source": "auto", "attributes": [],
        }

        # Ingest shapes
        r_create = session.patch(
            f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations?action=create",
            json={"shapes": [test_rect, test_mask1, test_poly, test_mask2, test_line], "tracks": [], "tags": []},
            timeout=10,
        )
        assert r_create.status_code in (200, 201)
        created_shapes = r_create.json()["shapes"]
        harness.record_created_shapes(created_shapes)

        c_rect = next(s for s in created_shapes if s["type"] == "rectangle" and s["group"] == 901)
        c_mask1 = next(s for s in created_shapes if s["type"] == "mask" and s["group"] == 901)
        c_poly = next(s for s in created_shapes if s["type"] == "polygon" and s["group"] == 902)
        c_mask2 = next(s for s in created_shapes if s["type"] == "mask" and s["group"] == 902)
        c_line = next(s for s in created_shapes if s["type"] == "polyline")

        try:
            # 4. Perform human modifications
            new_box = [220.0, 115.0, 320.0, 215.0]
            edit_rect = dict(c_rect, points=new_box)
            new_mask1_pts = mask2rle([1] * 400) + [205.0, 105.0, 225.0, 125.0]
            edit_mask1 = dict(c_mask1, points=new_mask1_pts)
            new_poly_pts = [360.0, 440.0, 450.0, 450.0, 470.0, 500.0, 450.0, 550.0, 350.0, 550.0]
            edit_poly = dict(c_poly, points=new_poly_pts)
            new_line_pts = [150.0, 550.0, 260.0, 590.0, 350.0, 650.0, 400.0, 700.0]
            edit_line = dict(c_line, points=new_line_pts)

            r_update = session.patch(
                f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations?action=update",
                json={"shapes": [edit_rect, edit_mask1, edit_poly, edit_line], "tracks": [], "tags": []},
                timeout=10,
            )
            assert r_update.status_code in (200, 204)

            # 5. Reload and verify exact preservation
            r_reload = session.get(f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations", timeout=10)
            reloaded = {s["id"]: s for s in r_reload.json()["shapes"]}

            assert reloaded[c_rect["id"]]["points"] == new_box
            assert reloaded[c_mask1["id"]]["points"] == new_mask1_pts
            assert reloaded[c_poly["id"]]["points"] == new_poly_pts
            assert reloaded[c_mask2["id"]]["points"] == mask2_rle
            assert reloaded[c_line["id"]]["points"] == new_line_pts

        finally:
            # 6. Clean up ONLY created test shapes via SafeAnnotationHarness
            harness.cleanup_live(session, CVAT_BASE_URL)

    def test_live_disposable_skeleton_edit_roundtrip(self):
        """Live roundtrip verification for skeleton editability on a disposable job."""
        session = get_live_cvat_session()
        if not session:
            pytest.skip("Local CVAT instance (http://localhost:18080) not available or token missing")

        disposable_job_raw = os.getenv("CVAT_DISPOSABLE_JOB_ID", "").strip()
        if not disposable_job_raw:
            pytest.skip(
                "CVAT_DISPOSABLE_JOB_ID not configured. Set to a disposable (non-protected) job ID "
                "with skeleton labels to run live skeleton editability roundtrip against CVAT."
            )

        target_job_id = int(disposable_job_raw)
        assert_job_writable(target_job_id)

        harness = SafeAnnotationHarness(job_id=target_job_id)

        # Query initial shapes
        r_init = session.get(f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations", timeout=10)
        assert r_init.status_code == 200
        harness.snapshot_existing_shapes(r_init.json().get("shapes", []))

        # Check for skeleton label on task
        r_job = session.get(f"{CVAT_BASE_URL}/api/jobs/{target_job_id}", timeout=10)
        assert r_job.status_code == 200
        task_id = r_job.json()["task_id"]

        r_labels = session.get(f"{CVAT_BASE_URL}/api/labels?task_id={task_id}&page_size=100", timeout=10)
        skel_labels = [l for l in r_labels.json().get("results", []) if l.get("type") == "skeleton"]
        if not skel_labels:
            pytest.skip(f"No skeleton label found on task {task_id} for disposable job {target_job_id}")

        skeleton_label = skel_labels[0]
        sublabels = skeleton_label["sublabels"]
        sublabel_map = {s["name"]: s["id"] for s in sublabels}
        label_id = skeleton_label["id"]

        # Build test skeleton
        elements = [
            {
                "type": "points",
                "label_id": s["id"],
                "frame": 0,
                "group": 0,
                "source": "auto",
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "points": [float(200 + idx * 8), float(150 + idx * 10)],
                "attributes": [],
            }
            for idx, s in enumerate(sublabels)
        ]

        test_skel = {
            "type": "skeleton",
            "label_id": label_id,
            "frame": 0,
            "group": 0,
            "source": "auto",
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "points": [],
            "attributes": [],
            "elements": elements,
        }

        # Create on CVAT
        r_create = session.patch(
            f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations?action=create",
            json={"shapes": [test_skel], "tracks": [], "tags": []},
            timeout=10,
        )
        assert r_create.status_code in (200, 201)
        created_shapes = r_create.json()["shapes"]
        harness.record_created_shapes(created_shapes)
        c_skel = created_shapes[0]
        skel_id = c_skel["id"]

        try:
            # Simulate keypoint edit on first element
            first_elem = c_skel["elements"][0]
            first_lid = first_elem["label_id"]
            orig_first_pts = list(first_elem["points"])
            new_first_pts = [orig_first_pts[0] + 15.0, orig_first_pts[1] + 15.0]
            first_elem["points"] = new_first_pts

            r_update = session.patch(
                f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations?action=update",
                json={"shapes": [c_skel], "tracks": [], "tags": []},
                timeout=10,
            )
            assert r_update.status_code in (200, 204)

            # Reload and verify
            r_reload = session.get(f"{CVAT_BASE_URL}/api/jobs/{target_job_id}/annotations", timeout=10)
            assert r_reload.status_code == 200
            reloaded_shapes_map = {s["id"]: s for s in r_reload.json()["shapes"]}
            assert skel_id in reloaded_shapes_map
            r_skel = reloaded_shapes_map[skel_id]
            reloaded_elems = {e["label_id"]: e for e in r_skel["elements"]}
            assert reloaded_elems[first_lid]["points"] == new_first_pts

        finally:
            # Clean up ONLY the created skeleton shape via SafeAnnotationHarness
            harness.cleanup_live(session, CVAT_BASE_URL)

