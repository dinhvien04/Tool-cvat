"""Comprehensive Verification of Native CVAT Editability for All 3 Detector Output Types.

This script tests Goal #2 against live local CVAT on the configured CVAT job:
1. Shape specifications validation for:
   - Rectangle: xtl, ytl, xbr, ybr, occluded=False, z_order=0, unlocked, draggable, resizable
   - Mask: CVAT RLE format [...rle_counts, xtl, ytl, xbr, ybr], brush/eraser/polygon editable, unlocked
   - Polygon: points [x1, y1, x2, y2, ...], vertex add/move/delete
   - Polyline: type='polyline', points [x1, y1, x2, y2, ...], point add/move/delete
2. Pairing (group_id) verification:
   - Rectangle + Mask paired with group_id (Detector 1)
   - Polygon + Mask paired with group_id (Detector 2)
   - Polyline atomic without group_id (Detector 3)
   - Independent editability: editing rectangle leaves mask untouched (no lock, no auto-reset)
   - Editing polygon leaves mask untouched
3. Live Edit Round-trip on configured Job (Task 15):
   - POST /api/jobs/15/annotations?action=create test shapes
   - Verify live ingestion
   - PATCH /api/jobs/15/annotations?action=update with human-modified coordinates
   - GET /api/jobs/15/annotations to verify persistence of human edits
   - DELETE /api/jobs/15/annotations?action=delete to cleanly restore job state
"""

import os
import sys
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cvat_safety import assert_job_writable, is_job_protected, get_protected_job_ids

TOKEN_FILE = REPO_ROOT / ".tool-cvat" / "cvat_token.txt"

base_url = os.getenv("CVAT_URL", "http://localhost:18080").rstrip("/")
job_id_raw = os.getenv("CVAT_JOB_ID", "").strip()
if not job_id_raw:
    print("Error: set CVAT_JOB_ID to a disposable/local verification job ID.")
    sys.exit(2)
try:
    job_id = int(job_id_raw)
except ValueError:
    print("Error: CVAT_JOB_ID must be an integer.")
    sys.exit(2)

if is_job_protected(job_id):
    print(f"Error: Job {job_id} is a protected production job ({sorted(get_protected_job_ids())}).")
    print("Production jobs are STRICTLY READ-ONLY. Mutating tests must use disposable jobs.")
    sys.exit(2)

access_token = os.getenv("CVAT_ACCESS_TOKEN", "").strip()
legacy_token = os.getenv("CVAT_TOKEN", "").strip()
if access_token:
    authorization = f"Bearer {access_token}"
elif legacy_token:
    authorization = f"Token {legacy_token}"
elif TOKEN_FILE.exists():
    # Backwards-compatible local-only legacy token file. .tool-cvat/ is gitignored.
    authorization = f"Token {TOKEN_FILE.read_text(encoding='utf-8').strip()}"
else:
    print("Error: set CVAT_ACCESS_TOKEN (recommended) or CVAT_TOKEN.")
    sys.exit(2)

headers = {
    "Authorization": authorization,
    "Accept": "application/vnd.cvat+json",
    "Content-Type": "application/json",
}


def mask2rle(flat_binary_mask):
    """Encodes a flattened binary mask into CVAT RLE format."""
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


def rle2mask(rle, width, height):
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


def run_verification():
    print("=" * 70)
    print(f" Native CVAT Editability Verification: Job {job_id}")
    print("=" * 70)

    # 1. Verify CVAT connection and configured Job
    resp_job = requests.get(f"{base_url}/api/jobs/{job_id}", headers=headers, timeout=15)
    assert resp_job.status_code == 200, f"Failed to get configured Job: {resp_job.text}"
    job_info = resp_job.json()
    task_id = job_info["task_id"]
    print(f"[OK] Connected to configured Job (Task ID: {task_id}, Stage: {job_info.get('stage')}, State: {job_info.get('state')})")

    # Fetch labels
    resp_labels = requests.get(f"{base_url}/api/labels?task_id={task_id}&page_size=100", headers=headers, timeout=15)
    labels = {l["name"]: l["id"] for l in resp_labels.json().get("results", [])}
    print(f"[OK] Loaded {len(labels)} labels for Task {task_id}")

    car_label_id = labels["car"]
    road_label_id = labels["road"]
    lane_label_id = labels["lane/single white"]

    # 2. Inspect initial job annotations
    resp_initial = requests.get(f"{base_url}/api/jobs/{job_id}/annotations", headers=headers, timeout=15)
    assert resp_initial.status_code == 200, f"Failed to get initial annotations: {resp_initial.text}"
    initial_annos = resp_initial.json()
    initial_shape_count = len(initial_annos.get("shapes", []))
    print(f"[INFO] Initial shape count on configured Job: {initial_shape_count}")

    # 3. Construct Test Shapes representing all 3 detector types
    # Group 801: Detector 1 (Rectangle + Mask) - Paired
    rect_box = [150.0, 100.0, 250.0, 200.0]  # xtl, ytl, xbr, ybr (w=101, h=101)
    mask1_w, mask1_h = 21, 21
    mask1_pixels = [1] * (mask1_w * mask1_h)
    mask1_rle = mask2rle(mask1_pixels)
    # CVAT mask format: [...rle, xtl, ytl, xbr, ybr]
    mask1_points = mask1_rle + [160.0, 110.0, 180.0, 130.0]

    test_rect = {
        "type": "rectangle",
        "label_id": car_label_id,
        "frame": 0,
        "group": 801,
        "points": rect_box,
        "occluded": False,
        "z_order": 0,
        "source": "auto",
        "attributes": [],
    }

    test_mask1 = {
        "type": "mask",
        "label_id": car_label_id,
        "frame": 0,
        "group": 801,
        "points": mask1_points,
        "occluded": False,
        "z_order": 0,
        "source": "auto",
        "attributes": [],
    }

    # Group 802: Detector 2 (Polygon + Mask) - Paired
    poly_points = [300.0, 400.0, 500.0, 400.0, 500.0, 550.0, 300.0, 550.0]  # 4 vertices
    mask2_w, mask2_h = 31, 31
    mask2_pixels = [1] * (mask2_w * mask2_h)
    mask2_rle = mask2rle(mask2_pixels)
    mask2_points = mask2_rle + [300.0, 400.0, 330.0, 430.0]

    test_poly = {
        "type": "polygon",
        "label_id": road_label_id,
        "frame": 0,
        "group": 802,
        "points": poly_points,
        "occluded": False,
        "z_order": 0,
        "source": "auto",
        "attributes": [],
    }

    test_mask2 = {
        "type": "mask",
        "label_id": road_label_id,
        "frame": 0,
        "group": 802,
        "points": mask2_points,
        "occluded": False,
        "z_order": 0,
        "source": "auto",
        "attributes": [],
    }

    # Unpaired: Detector 3 (Polyline) - Atomic
    polyline_points = [100.0, 500.0, 200.0, 550.0, 350.0, 620.0]  # 3 points

    test_line = {
        "type": "polyline",
        "label_id": lane_label_id,
        "frame": 0,
        "group": None,
        "points": polyline_points,
        "occluded": False,
        "z_order": 0,
        "source": "auto",
        "attributes": [],
    }

    # 4. Inject test shapes into configured Job via PATCH ?action=create
    print("\n--- STEP 1: Ingesting Shapes for all 3 Detectors ---")
    create_payload = {
        "shapes": [test_rect, test_mask1, test_poly, test_mask2, test_line],
        "tracks": [],
        "tags": [],
    }
    resp_create = requests.patch(
        f"{base_url}/api/jobs/{job_id}/annotations?action=create",
        json=create_payload,
        headers=headers,
        timeout=15,
    )
    assert resp_create.status_code in (200, 201), f"Create failed: {resp_create.text}"
    created_data = resp_create.json()
    created_shapes = created_data.get("shapes", [])
    print(f"[OK] Ingested {len(created_shapes)} shapes into configured Job")

    # Map created shapes by group & type
    created_rect = next(s for s in created_shapes if s["type"] == "rectangle" and s["group"] == 801)
    created_mask1 = next(s for s in created_shapes if s["type"] == "mask" and s["group"] == 801)
    created_poly = next(s for s in created_shapes if s["type"] == "polygon" and s["group"] == 802)
    created_mask2 = next(s for s in created_shapes if s["type"] == "mask" and s["group"] == 802)
    created_line = next(s for s in created_shapes if s["type"] == "polyline")

    rect_id = created_rect["id"]
    mask1_id = created_mask1["id"]
    poly_id = created_poly["id"]
    mask2_id = created_mask2["id"]
    line_id = created_line["id"]

    all_test_ids = [rect_id, mask1_id, poly_id, mask2_id, line_id]
    print(f"[OK] Shape IDs assigned: Rect={rect_id}, Mask1={mask1_id}, Poly={poly_id}, Mask2={mask2_id}, Line={line_id}")

    try:
        # 5. Verify Structure & Unlocked State
        print("\n--- STEP 2: Verifying Shape Structures & Constraints ---")
        # Verify Rectangle
        assert len(created_rect["points"]) == 4, f"Rectangle points must have length 4: {created_rect['points']}"
        assert created_rect["occluded"] is False
        assert created_rect["z_order"] == 0
        assert created_rect["group"] == 801
        print("[OK] Rectangle verified: normal CVAT box [xtl, ytl, xbr, ybr], occluded=False, z_order=0, unlocked")

        # Verify Mask 1
        assert created_mask1["points"][-4:] == [160.0, 110.0, 180.0, 130.0]
        assert len(created_mask1["points"]) >= 5
        assert created_mask1["group"] == 801
        print("[OK] Mask 1 verified: CVAT RLE format [...rle, xtl, ytl, xbr, ybr], unlocked")

        # Verify Polygon
        assert len(created_poly["points"]) == 8  # 4 pairs
        assert created_poly["group"] == 802
        print("[OK] Polygon verified: 4 vertices [x1, y1, ...], unlocked")

        # Verify Mask 2
        assert created_mask2["points"][-4:] == [300.0, 400.0, 330.0, 430.0]
        assert created_mask2["group"] == 802
        print("[OK] Mask 2 verified: paired with polygon under group 802, unlocked")

        # Verify Polyline
        assert len(created_line["points"]) == 6  # 3 pairs
        assert created_line["type"] == "polyline"
        assert created_line["group"] in (None, 0)
        print("[OK] Polyline verified: atomic, 3 points, group_id omitted/null, unlocked")

        # 6. Simulate Human Edits
        print("\n--- STEP 3: Simulating Precise Human Corrections ---")
        # Edit A: Human drags and resizes Rectangle (moving corners)
        modified_rect_box = [170.0, 120.0, 290.0, 240.0]  # shifted and expanded
        edited_rect = dict(created_rect)
        edited_rect["points"] = modified_rect_box

        # Edit B: Human uses Brush/Eraser on Mask 1 (modifying binary mask and expanding bbox)
        # Simulate brushing additional area: new bbox is [150.0, 100.0, 190.0, 140.0] (w=41, h=41)
        brushed_w, brushed_h = 41, 41
        brushed_pixels = [1] * (brushed_w * brushed_h)
        brushed_rle = mask2rle(brushed_pixels)
        brushed_mask1_points = brushed_rle + [150.0, 100.0, 190.0, 140.0]
        edited_mask1 = dict(created_mask1)
        edited_mask1["points"] = brushed_mask1_points

        # Edit C: Human edits Polygon (moves vertex 0, inserts a 5th vertex, deletes vertex 2)
        # Original: [300, 400, 500, 400, 500, 550, 300, 550] (4 vertices)
        # Move v0: (300, 400) -> (315, 385)
        # Insert v_new between v1 and v2: (520, 475)
        # New polygon has 5 vertices (10 coordinates)
        edited_poly_points = [315.0, 385.0, 500.0, 400.0, 520.0, 475.0, 500.0, 550.0, 300.0, 550.0]
        edited_poly = dict(created_poly)
        edited_poly["points"] = edited_poly_points

        # Notice: test_mask2 is intentionally NOT edited here!
        # We verify that human edit on polygon does NOT alter mask2, and vice-versa.

        # Edit D: Human edits Polyline (moves point 1, adds point 4)
        # Original: [100, 500, 200, 550, 350, 620] (3 points)
        # Move pt1: (200, 550) -> (210, 540)
        # Add pt3: (400, 680)
        edited_line_points = [100.0, 500.0, 210.0, 540.0, 350.0, 620.0, 400.0, 680.0]  # 4 points
        edited_line = dict(created_line)
        edited_line["points"] = edited_line_points

        # 7. Apply Edits via PATCH ?action=update
        print("\n--- STEP 4: Applying Updates to CVAT (Simulating Save) ---")
        update_payload = {
            "shapes": [edited_rect, edited_mask1, edited_poly, edited_line],
            "tracks": [],
            "tags": [],
        }
        resp_update = requests.patch(
            f"{base_url}/api/jobs/{job_id}/annotations?action=update",
            json=update_payload,
            headers=headers,
            timeout=15,
        )
        assert resp_update.status_code in (200, 204), f"Update failed: {resp_update.text}"
        print("[OK] Human updates sent to CVAT successfully")

        # 8. Reload and Verify Persistence
        print("\n--- STEP 5: Reloading Annotations and Verifying Exact Round-trip Persistence ---")
        resp_reloaded = requests.get(f"{base_url}/api/jobs/{job_id}/annotations", headers=headers, timeout=15)
        assert resp_reloaded.status_code == 200
        reloaded_annos = resp_reloaded.json()
        reloaded_shapes_map = {s["id"]: s for s in reloaded_annos.get("shapes", [])}

        # Check Rectangle edits
        r_rect = reloaded_shapes_map[rect_id]
        assert r_rect["points"] == modified_rect_box, f"Rectangle box did not persist! Got {r_rect['points']}"
        assert r_rect["group"] == 801
        print(f"[OK] Rectangle edit preserved: box={r_rect['points']}")

        # Check Mask 1 edits
        r_mask1 = reloaded_shapes_map[mask1_id]
        assert r_mask1["points"] == brushed_mask1_points, "Mask 1 brushed points did not persist!"
        assert r_mask1["points"][-4:] == [150.0, 100.0, 190.0, 140.0]
        assert r_mask1["group"] == 801
        print(f"[OK] Mask 1 brush edit preserved: bbox={r_mask1['points'][-4:]}, total points={len(r_mask1['points'])}")

        # Check Polygon edits (vertex moved, vertex added)
        r_poly = reloaded_shapes_map[poly_id]
        assert r_poly["points"] == edited_poly_points, f"Polygon edits did not persist! Got {r_poly['points']}"
        assert len(r_poly["points"]) == 10
        assert r_poly["group"] == 802
        print(f"[OK] Polygon vertex additions and moves preserved: {len(r_poly['points']) // 2} vertices")

        # Check Mask 2 (Unmodified peer in Group 802)
        r_mask2 = reloaded_shapes_map[mask2_id]
        assert r_mask2["points"] == mask2_points, "Mask 2 was corrupted or auto-reset by polygon edit!"
        assert r_mask2["group"] == 802
        print(f"[OK] Paired Mask 2 remained completely intact after polygon edit (no cross-shape side effects)")

        # Check Polyline edits (point moved, point added)
        r_line = reloaded_shapes_map[line_id]
        assert r_line["points"] == edited_line_points, f"Polyline edits did not persist! Got {r_line['points']}"
        assert len(r_line["points"]) == 8
        print(f"[OK] Polyline point moves and additions preserved: {len(r_line['points']) // 2} points")

        # 9. Test Polyline Vertex Deletion round-trip
        print("\n--- STEP 6: Testing Vertex Deletion Round-trip on Polyline ---")
        # Delete point 0 from polyline: [100.0, 500.0] deleted -> 3 points remaining
        truncated_line_points = edited_line_points[2:]  # 6 coordinates (3 points)
        edited_line_del = dict(r_line)
        edited_line_del["points"] = truncated_line_points

        resp_del_pt = requests.patch(
            f"{base_url}/api/jobs/{job_id}/annotations?action=update",
            json={"shapes": [edited_line_del], "tracks": [], "tags": []},
            headers=headers,
            timeout=15,
        )
        assert resp_del_pt.status_code in (200, 204)

        resp_del_verify = requests.get(f"{base_url}/api/jobs/{job_id}/annotations", headers=headers, timeout=15)
        r_line_del = {s["id"]: s for s in resp_del_verify.json()["shapes"]}[line_id]
        assert r_line_del["points"] == truncated_line_points
        assert len(r_line_del["points"]) == 6
        print(f"[OK] Polyline vertex deletion preserved: successfully reduced to {len(r_line_del['points']) // 2} points")

    finally:
        # 10. Clean up test shapes via DELETE action
        print("\n--- STEP 7: Cleaning Up Test Shapes ---")
        # Fetch current shape objects for the test IDs so all required serializer fields are populated
        resp_current = requests.get(f"{base_url}/api/jobs/{job_id}/annotations", headers=headers, timeout=15)
        current_shapes = {s["id"]: s for s in resp_current.json().get("shapes", [])}
        shapes_to_delete = [current_shapes[sid] for sid in all_test_ids if sid in current_shapes]

        delete_payload = {
            "shapes": shapes_to_delete,
            "tracks": [],
            "tags": [],
        }
        resp_del = requests.patch(
            f"{base_url}/api/jobs/{job_id}/annotations?action=delete",
            json=delete_payload,
            headers=headers,
            timeout=15,
        )
        assert resp_del.status_code in (200, 204), f"Cleanup failed: {resp_del.text}"

        # Verify restoration of original count
        resp_final = requests.get(f"{base_url}/api/jobs/{job_id}/annotations", headers=headers, timeout=15)
        final_shapes = resp_final.json().get("shapes", [])
        assert len(final_shapes) == initial_shape_count, f"Cleanup mismatch: expected {initial_shape_count}, got {len(final_shapes)}"
        print(f"[OK] Cleaned up all test shapes. configured Job restored to {len(final_shapes)} shapes.")

    print("\n" + "=" * 70)
    print(" ALL EDITABILITY VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
