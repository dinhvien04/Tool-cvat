"""Check CVAT lambda functions."""
import os
import sys
import requests

cvat_url = os.getenv("CVAT_URL", "http://127.0.0.1:18080")
token = os.getenv("CVAT_TOKEN", "")

headers = {}
if token:
    headers["Authorization"] = f"Token {token}"

try:
    resp = requests.get(f"{cvat_url}/api/lambda/functions", headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        functions = resp.json()
        print(f"Total Lambda Functions: {len(functions)}")
        for fn in functions:
            fn_id = fn.get("id")
            fn_name = fn.get("name")
            fn_type = fn.get("type")
            fn_prov = fn.get("provider")
            print(f"  * [{fn_id}] {fn_name} (type: {fn_type}, provider: {fn_prov})")
    else:
        print(f"Response: {resp.text[:300]}")
except Exception as e:
    print(f"Error querying CVAT: {e}")
