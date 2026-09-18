"""Live smoke testing of deployed 3 9Router detectors."""
import base64
import io
import json
from pathlib import Path
import sys
import time
import requests
from PIL import Image

def generate_test_image_b64() -> str:
    test_jpg = Path("test.jpg")
    if test_jpg.exists():
        return base64.b64encode(test_jpg.read_bytes()).decode("utf-8")
    im = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def test_detector(name: str, port: int, expected_policy: str):
    url = f"http://127.0.0.1:{port}"
    print(f"\nTesting {name} on {url} (Policy: {expected_policy})...")
    payload = {
        "image": generate_test_image_b64(),
        "threshold": 0.2
    }
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=90)
        elapsed = time.time() - t0
        print(f"Status: {resp.status_code} in {elapsed:.2f}s")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            return False

        shapes = resp.json()
        print(f"Emitted shapes count: {len(shapes)}")

        # Policy verification
        if expected_policy == "rectangle_mask":
            rects = [s for s in shapes if s.get("type") == "rectangle"]
            masks = [s for s in shapes if s.get("type") == "mask"]
            print(f"  - Rectangles: {len(rects)}, Masks: {len(masks)}")
            for s in shapes:
                assert s.get("type") in ("rectangle", "mask"), f"Invalid shape type {s.get('type')}"
            # Check group_id pairing if instances detected
            rect_groups = {s.get("group_id") for s in rects if "group_id" in s}
            mask_groups = {s.get("group_id") for s in masks if "group_id" in s}
            if rects and masks:
                assert rect_groups == mask_groups, f"Mismatched group IDs: {rect_groups} vs {mask_groups}"
                print("  [PASS] Paired rectangle + mask group_id verified.")

        elif expected_policy == "polygon_mask":
            polys = [s for s in shapes if s.get("type") == "polygon"]
            masks = [s for s in shapes if s.get("type") == "mask"]
            boxes = [s for s in shapes if s.get("type") == "rectangle"]
            print(f"  - Polygons: {len(polys)}, Masks: {len(masks)}, Boxes: {len(boxes)}")
            assert len(boxes) == 0, f"Found {len(boxes)} bounding boxes in polygon_mask!"
            for s in shapes:
                assert s.get("type") in ("polygon", "mask"), f"Invalid shape type {s.get('type')}"
            poly_groups = {s.get("group_id") for s in polys if "group_id" in s}
            mask_groups = {s.get("group_id") for s in masks if "group_id" in s}
            if polys and masks:
                assert poly_groups == mask_groups, f"Mismatched group IDs: {poly_groups} vs {mask_groups}"
                print("  [PASS] Paired polygon + mask group_id verified, boxes suppressed.")

        elif expected_policy == "polyline":
            polylines = [s for s in shapes if s.get("type") == "polyline"]
            others = [s for s in shapes if s.get("type") != "polyline"]
            print(f"  - Polylines: {len(polylines)}, Others: {len(others)}")
            assert len(others) == 0, f"Found non-polyline shapes: {others}"
            for s in shapes:
                assert "group_id" not in s, f"Polyline must not have group_id: {s}"
            print("  [PASS] Polyline-only policy verified.")

        return True
    except Exception as e:
        print(f"Invocation failed: {e}")
        return False

if __name__ == "__main__":
    # Test ports
    ok1 = test_detector("9Router Rectangle + Mask", 4659, "rectangle_mask")
    ok2 = test_detector("9Router Polygon + Mask", 4183, "polygon_mask")
    ok3 = test_detector("9Router Polyline", 14313, "polyline")

    if ok1 and ok2 and ok3:
        print("\nALL THREE DETECTORS PASSED LIVE VERIFICATION!")
        sys.exit(0)
    else:
        print("\nONE OR MORE DETECTORS FAILED.")
        sys.exit(1)
