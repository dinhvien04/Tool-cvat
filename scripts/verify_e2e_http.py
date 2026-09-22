import os
import sys
import time
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cvat_safety import is_job_protected

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

job_id_str = os.getenv("CVAT_JOB_ID") or os.getenv("CVAT_TEST_JOB_ID") or "15"
try:
    target_job_id = int(job_id_str)
except ValueError:
    print(f"Error: Invalid CVAT_JOB_ID '{job_id_str}' - must be an integer.")
    sys.exit(2)

if is_job_protected(target_job_id):
    print(f"[DATA SAFETY NOTICE] Target Job {target_job_id} is in protected production jobs list.")
    print("Lambda inference invocation is executed in READ-ONLY mode without writing back annotations.")

functions_to_test = [
    ("ninerouter-polyline", "Polyline"),
    ("ninerouter-rectangle-mask", "Rectangle + Mask"),
    ("ninerouter-polygon-mask", "Polygon + Mask"),
]

print("=" * 70)
print("E2E Traefik -> CVAT Nginx -> Uvicorn -> Nuclio -> 9Router Verification")
print("=" * 70)

for func_id, label in functions_to_test:
    url = f"{CVAT_URL}/api/lambda/functions/{func_id}"
    payload = {"job": target_job_id, "frame": 0}
    print(f"\nCalling {label} ({func_id}) at {url}...")
    start = time.perf_counter()
    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=310)
        elapsed = time.perf_counter() - start
        print(f"Status Code: {resp.status_code} in {elapsed:.2f}s")
        if resp.status_code == 200:
            data = resp.json()
            shapes = data.get("shapes", [])
            print(f"[PASS] Successfully received {len(shapes)} shapes!")
            counts = {}
            for s in shapes:
                st = s.get("type", "unknown")
                counts[st] = counts.get(st, 0) + 1
            print(f"Shape types: {counts}")
        else:
            print(f"[FAIL] Unexpected status {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[ERROR] Request failed in {elapsed:.2f}s: {e}")

print("\n" + "=" * 70)
print("E2E Verification Finished.")
print("=" * 70)
