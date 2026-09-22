"""Test live inference of deployed Week-2 Nuclio functions on test_driver.jpg."""
import base64
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import requests

POSE17_URL = os.getenv("NUCLIO_POSE17_URL", "http://localhost:5997").rstrip("/")
VF50_URL = os.getenv("NUCLIO_VF50_URL", "http://localhost:5886").rstrip("/")

req_headers = {"Content-Type": "application/json"}
nuclio_token = os.getenv("NUCLIO_AUTH_TOKEN", "").strip()
if nuclio_token:
    req_headers["Authorization"] = f"Bearer {nuclio_token}"

repo_root = Path(__file__).resolve().parent.parent
if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
    test_img_path = Path(sys.argv[1])
else:
    test_img_path = repo_root / "test_driver.jpg"

if not test_img_path.exists():
    print(f"Warning: {test_img_path} not found. Creating a synthetic 640x480 test image for ping test.")
    dummy = Image.new("RGB", (640, 480), color=(128, 128, 128))
    dummy.save(test_img_path)

with Image.open(test_img_path) as img:
    w, h = img.size
print(f"Loaded test_driver.jpg: {w}x{h}")

img_bytes = test_img_path.read_bytes()
b64_image = base64.b64encode(img_bytes).decode("utf-8")

payload = {
    "image": b64_image,
    "threshold": 0.5,
}

# 1. Test Human Pose 17
print(f"\n--- Testing 9Router Human Pose 17 ({POSE17_URL}) ---")
t0 = time.perf_counter()
try:
    resp17 = requests.post(
        POSE17_URL,
        json=payload,
        headers=req_headers,
        timeout=60,
    )
    dur17 = time.perf_counter() - t0
    print(f"Pose 17 Response Status: {resp17.status_code} in {dur17:.2f}s")
    if resp17.status_code == 200:
        shapes17 = resp17.json()
        print(f"Returned {len(shapes17)} shapes:")
        for s in shapes17:
            print(f"  Shape type: {s.get('type')}, label: {s.get('label')}, elements: {len(s.get('elements', []))}")
            for el in s.get("elements", [])[:3]:
                print(f"    Sublabel: {el.get('label')}, points: {el.get('points')}, outside: {el.get('outside')}")
    else:
        print(f"Error: {resp17.text[:400]}")
except Exception as e:
    print(f"Pose 17 Request Failed: {e}")

# 2. Test Face VF-50
print(f"\n--- Testing 9Router Face Landmark VF-50 ({VF50_URL}) ---")
t0 = time.perf_counter()
try:
    resp50 = requests.post(
        VF50_URL,
        json=payload,
        headers=req_headers,
        timeout=60,
    )
    dur50 = time.perf_counter() - t0
    print(f"Face VF-50 Response Status: {resp50.status_code} in {dur50:.2f}s")
    if resp50.status_code == 200:
        shapes50 = resp50.json()
        print(f"Returned {len(shapes50)} shapes:")
        for s in shapes50:
            print(f"  Shape type: {s.get('type')}, label: {s.get('label')}, elements: {len(s.get('elements', []))}")
            for el in s.get("elements", [])[:3]:
                print(f"    Sublabel: {el.get('label')}, points: {el.get('points')}, outside: {el.get('outside')}")
    else:
        print(f"Error: {resp50.text[:400]}")
except Exception as e:
    print(f"Face VF-50 Request Failed: {e}")
