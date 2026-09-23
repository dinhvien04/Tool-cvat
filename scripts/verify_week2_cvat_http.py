import os
import sys
import time
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CVAT_URL = os.getenv("CVAT_URL", "http://localhost:18080").rstrip("/")

# Multi-tiered secure token resolution
access_token = os.getenv("CVAT_ACCESS_TOKEN", "").strip()
legacy_token = os.getenv("CVAT_TOKEN", "").strip()
token_file = REPO_ROOT / ".tool-cvat" / "cvat_token.txt"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/vnd.cvat+json",
}

if access_token:
    HEADERS["Authorization"] = f"Bearer {access_token}"
elif legacy_token:
    HEADERS["Authorization"] = f"Token {legacy_token}"
elif token_file.exists():
    token_val = token_file.read_text(encoding="utf-8").strip()
    HEADERS["Authorization"] = f"Token {token_val}"

target_job_id = int(os.getenv("CVAT_JOB_ID", "17"))
target_frame = int(os.getenv("CVAT_FRAME", "0"))

functions_to_test = [
    ("ninerouter-human-pose-17", "9Router Human Pose 17"),
    ("ninerouter-face-vf50", "9Router Face Landmark VF-50"),
]

print("=" * 80)
print(f"E2E CVAT HTTP Verification: Task 20, Job {target_job_id}, Frame {target_frame}")
print("Route: Localhost:18080 -> Traefik -> CVAT Django -> Nuclio -> 9Router")
print("=" * 80)

all_passed = True
results = {}

for func_id, label in functions_to_test:
    url = f"{CVAT_URL}/api/lambda/functions/{func_id}"
    payload = {"job": target_job_id, "frame": target_frame}
    print(f"\n[HTTP POST] Invoking {label} ({func_id}) at {url}...")
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        dur = time.perf_counter() - t0
        print(f"  HTTP Status Code: {resp.status_code} ({dur:.2f}s)")
        if resp.status_code == 200:
            data = resp.json()
            shapes = data.get("shapes", [])
            print(f"  [PASS] Successfully returned {len(shapes)} shapes!")
            for idx, s in enumerate(shapes):
                st = s.get("type", "unknown")
                lid = s.get("label_id")
                elements = s.get("elements", [])
                print(f"    Shape #{idx+1}: type={st}, label_id={lid}, elements={len(elements)}")
                for el in elements[:3]:
                    print(f"      Sub-element label_id={el.get('label_id')}, points={el.get('points')}, outside={el.get('outside')}")
            results[func_id] = {"status": "PASS", "shapes": len(shapes), "duration": dur}
        else:
            all_passed = False
            print(f"  [FAIL] HTTP {resp.status_code}: {resp.text[:500]}")
            results[func_id] = {"status": "FAIL", "error": resp.text[:500]}
    except Exception as e:
        dur = time.perf_counter() - t0
        all_passed = False
        print(f"  [FAIL] Request failed ({dur:.2f}s): {e}")
        results[func_id] = {"status": "FAIL", "error": str(e)}

print("\n" + "=" * 80)
print("E2E CVAT HTTP Verification Summary")
print("=" * 80)
for fid, r in results.items():
    print(f"  {fid:30s} : {r['status']} (details: {r})")

if all_passed:
    print("\nALL WEEK-2 DETECTORS PASSED E2E CVAT HTTP VERIFICATION ON TASK 20!")
    sys.exit(0)
else:
    print("\nONE OR MORE DETECTORS FAILED E2E CVAT HTTP VERIFICATION.")
    sys.exit(1)
