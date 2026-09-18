import os
import sys
import time
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.development')
django.setup()

from cvat.apps.engine.models import Task, Job
from cvat.apps.lambda_manager.views import LambdaGateway, DetectionResultConverter

task = Task.objects.get(id=15)
job = Job.objects.get(id=15)
gateway = LambdaGateway()

funcs = ['ninerouter-rectangle-mask', 'ninerouter-polygon-mask', 'ninerouter-polyline']
for func_id in funcs:
    print(f"\n{'='*70}")
    print(f"Testing {func_id} via CVAT LambdaGateway on Task 15, Job 15, Frame 0...")
    print(f"{'='*70}")
    start = time.perf_counter()
    try:
        func = gateway.get(func_id)
        converter = DetectionResultConverter(task)
        res = func.invoke(task, {'frame': 0}, db_job=job, converter=converter, is_interactive=True)
        elapsed = time.perf_counter() - start
        shapes_count = len(res.get('shapes', [])) if isinstance(res, dict) else 0
        tags_count = len(res.get('tags', [])) if isinstance(res, dict) else 0
        print(f"[PASS] {func_id} completed successfully in {elapsed:.2f}s!")
        print(f"  Emitted shapes: {shapes_count}, tags: {tags_count}")
        if shapes_count > 0:
            shape_types = {}
            for s in res['shapes']:
                t = s.get('type', 'unknown')
                shape_types[t] = shape_types.get(t, 0) + 1
            print(f"  Shape breakdown: {shape_types}")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[FAIL] {func_id} failed after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()

print("\nLive CVAT LambdaGateway invocation tests completed.")
