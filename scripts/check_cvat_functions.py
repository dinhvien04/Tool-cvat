"""Check functions visible to CVAT Lambda Manager."""
import json
import subprocess
import sys

py_code = """
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
django.setup()
from cvat.apps.lambda_manager.views import LambdaGateway

gateway = LambdaGateway()
funcs = [f.to_dict() for f in gateway.list()]
items = []
for fn in funcs:
    items.append({
        'id': fn.get('id'),
        'name': fn.get('name'),
        'kind': fn.get('kind'),
        'type': fn.get('type'),
    })
print('JSON_START' + json.dumps(items) + 'JSON_END')
"""

res = subprocess.run(
    ["docker", "exec", "-i", "cvat_server", "python3", "-c", py_code],
    capture_output=True,
    text=True,
)
if res.returncode != 0:
    print("Error querying cvat_server:", res.stderr)
    sys.exit(1)

out = res.stdout
if "JSON_START" in out and "JSON_END" in out:
    raw = out.split("JSON_START")[1].split("JSON_END")[0]
    data = json.loads(raw)
    print(f"Total functions in CVAT Lambda Manager: {len(data)}")
    for item in data:
        print(f"  - ID: {item['id']:<40} | Name: {item['name']:<35} | Type: {item['type']}")
else:
    print("Could not parse output:", out)
