"""End-to-end verification of CVAT LambdaGateway invocation and DetectionResultConverter.

Verifies:
1. nuclio function invocation returns raw detector shapes.
2. conv_mask_to_poly = False -> native mask shapes with RLE + bbox and paired group_id.
3. conv_mask_to_poly = True -> polygon shapes with vector points and paired group_id.
"""

import base64
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cvat.settings.development")
django.setup()

from cvat.apps.engine.models import Task
from cvat.apps.lambda_manager.views import DetectionResultConverter, LambdaGateway

with open("/tmp/test.jpg", "rb") as f:
    b64_img = base64.b64encode(f.read()).decode()

gw = LambdaGateway()
func = gw.get("ninerouter-vision-box-mask")
print("=== Invoking function ninerouter-vision-box-mask from cvat_server ===")
detections = gw.invoke(func, {"image": b64_img, "threshold": 0.5})
print(f"Raw detector returned {len(detections)} shapes:")
for d in detections:
    pts_len = len(d.get("points", [])) if "points" in d else 0
    mask_len = len(d.get("mask", [])) if "mask" in d else 0
    gid = d.get("group_id")
    print(
        f"  label={d['label']} type={d['type']} group_id={gid} points_len={pts_len} mask_len={mask_len}"
    )

task = Task.objects.get(id=4)
labels_dict = {l.name: l.id for l in task.get_labels()}
car_detections = [d for d in detections if d["label"] in labels_dict]

converter = DetectionResultConverter(db_task=task)

print("\n=== CVAT Conversion: conv_mask_to_poly = False (Native Masks) ===")
res_native = converter.convert(
    conv_mask_to_poly=False, frame=0, annotations=car_detections
)
for s in res_native["shapes"]:
    print(
        f"  Shape type={s['type']} group={s.get('group')} points_count={len(s['points'])}"
    )

print("\n=== CVAT Conversion: conv_mask_to_poly = True (Polygons) ===")
res_poly = converter.convert(
    conv_mask_to_poly=True, frame=0, annotations=car_detections
)
for s in res_poly["shapes"]:
    print(
        f"  Shape type={s['type']} group={s.get('group')} points_count={len(s['points'])} points_sample={s['points'][:4]}"
    )
