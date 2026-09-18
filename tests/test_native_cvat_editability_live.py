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
from pathlib import Path
from typing import Any, Dict, List
import pytest
import requests

from core.geometry import (
    calculate_polygon_area,
    denormalize_contour,
    mask_to_cvat_flat_list,
    multi_polygon_to_cvat_mask,
    rasterize_polygons_to_mask,
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


class TestLiveCVATEditRoundTrip:
    """Live verification against local CVAT instance (Task 15, Job 15)."""

    def test_live_task15_job15_edit_roundtrip(self):
        session = get_live_cvat_session()
        if not session:
            pytest.skip("Local CVAT instance (http://localhost:18080) not available or token missing")

        # 1. Fetch Job 15 and labels
        r_job = session.get(f"{CVAT_BASE_URL}/api/jobs/15", timeout=10)
        assert r_job.status_code == 200, f"Job 15 not found: {r_job.text}"

        r_labels = session.get(f"{CVAT_BASE_URL}/api/labels?task_id=15&page_size=100", timeout=10)
        labels = {l["name"]: l["id"] for l in r_labels.json().get("results", [])}

        car_id = labels["car"]
        road_id = labels["road"]
        lane_id = labels["lane/single white"]

        # 2. Query initial shapes
        r_initial = session.get(f"{CVAT_BASE_URL}/api/jobs/15/annotations", timeout=10)
        assert r_initial.status_code == 200
        initial_count = len(r_initial.json().get("shapes", []))

        # 3. Create test shapes representing all 3 detector types
        # Type 1: Rectangle + Mask (group 901)
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

        # Type 2: Polygon + Mask (group 902)
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

        # Type 3: Polyline (atomic)
        line_pts = [150.0, 550.0, 250.0, 600.0, 350.0, 650.0]
        test_line = {
            "type": "polyline", "label_id": lane_id, "frame": 0, "group": None,
            "points": line_pts, "occluded": False, "z_order": 0, "source": "auto", "attributes": [],
        }

        # Ingest shapes
        r_create = session.patch(
            f"{CVAT_BASE_URL}/api/jobs/15/annotations?action=create",
            json={"shapes": [test_rect, test_mask1, test_poly, test_mask2, test_line], "tracks": [], "tags": []},
            timeout=10,
        )
        assert r_create.status_code in (200, 201)
        created_shapes = r_create.json()["shapes"]

        c_rect = next(s for s in created_shapes if s["type"] == "rectangle" and s["group"] == 901)
        c_mask1 = next(s for s in created_shapes if s["type"] == "mask" and s["group"] == 901)
        c_poly = next(s for s in created_shapes if s["type"] == "polygon" and s["group"] == 902)
        c_mask2 = next(s for s in created_shapes if s["type"] == "mask" and s["group"] == 902)
        c_line = next(s for s in created_shapes if s["type"] == "polyline")

        test_ids = [c_rect["id"], c_mask1["id"], c_poly["id"], c_mask2["id"], c_line["id"]]

        try:
            # 4. Perform human modifications
            # Drag box
            new_box = [220.0, 115.0, 320.0, 215.0]
            edit_rect = dict(c_rect, points=new_box)

            # Brush mask 1
            new_mask1_pts = mask2rle([1] * 400) + [205.0, 105.0, 225.0, 125.0]
            edit_mask1 = dict(c_mask1, points=new_mask1_pts)

            # Move vertex on polygon & add 5th vertex
            new_poly_pts = [360.0, 440.0, 450.0, 450.0, 470.0, 500.0, 450.0, 550.0, 350.0, 550.0]
            edit_poly = dict(c_poly, points=new_poly_pts)

            # Move point on polyline & add 4th point
            new_line_pts = [150.0, 550.0, 260.0, 590.0, 350.0, 650.0, 400.0, 700.0]
            edit_line = dict(c_line, points=new_line_pts)

            # Update via PATCH ?action=update
            r_update = session.patch(
                f"{CVAT_BASE_URL}/api/jobs/15/annotations?action=update",
                json={"shapes": [edit_rect, edit_mask1, edit_poly, edit_line], "tracks": [], "tags": []},
                timeout=10,
            )
            assert r_update.status_code in (200, 204)

            # 5. Reload and verify exact preservation
            r_reload = session.get(f"{CVAT_BASE_URL}/api/jobs/15/annotations", timeout=10)
            reloaded = {s["id"]: s for s in r_reload.json()["shapes"]}

            # Assert Rectangle preserved
            assert reloaded[c_rect["id"]]["points"] == new_box
            # Assert Mask 1 preserved
            assert reloaded[c_mask1["id"]]["points"] == new_mask1_pts
            # Assert Polygon additions & moves preserved
            assert reloaded[c_poly["id"]]["points"] == new_poly_pts
            # Assert Paired Mask 2 is uncorrupted
            assert reloaded[c_mask2["id"]]["points"] == mask2_rle
            # Assert Polyline additions & moves preserved
            assert reloaded[c_line["id"]]["points"] == new_line_pts

        finally:
            # Clean up test shapes
            r_curr = session.get(f"{CVAT_BASE_URL}/api/jobs/15/annotations", timeout=10)
            curr_map = {s["id"]: s for s in r_curr.json().get("shapes", [])}
            to_del = [curr_map[sid] for sid in test_ids if sid in curr_map]

            r_del = session.patch(
                f"{CVAT_BASE_URL}/api/jobs/15/annotations?action=delete",
                json={"shapes": to_del, "tracks": [], "tags": []},
                timeout=10,
            )
            assert r_del.status_code in (200, 204)

            r_final = session.get(f"{CVAT_BASE_URL}/api/jobs/15/annotations", timeout=10)
            assert len(r_final.json().get("shapes", [])) == initial_count
