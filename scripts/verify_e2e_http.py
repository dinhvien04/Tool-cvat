import os
import time
import requests

CVAT_URL = os.getenv("CVAT_URL", "http://localhost:18080")
TOKEN = os.getenv("CVAT_TOKEN", "")
HEADERS = {
    "Content-Type": "application/json",
}
if TOKEN:
    HEADERS["Authorization"] = f"Token {TOKEN}"

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
    payload = {"job": 15, "frame": 0}
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
