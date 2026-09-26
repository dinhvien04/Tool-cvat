"""Generate function.yaml for ninerouter-buddha-multilimbs Nuclio detector."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.buddha_contract import build_cvat_buddha_multilimbs_full_spec

FEEDBACK_HOST_PATH = (ROOT / ".tool-cvat").as_posix()

specs = build_cvat_buddha_multilimbs_full_spec()
spec_json = json.dumps(specs, separators=(",", ":"))

buddha_yaml = f"""metadata:
  name: ninerouter-buddha-multilimbs
  namespace: cvat
  annotations:
    name: 9Router Buddha Multi-Limb Pose
    type: detector
    spec: '{spec_json}'

spec:
  description: 9Router Buddha Multi-Limb Pose & Hand Landmark Detector (Multi-armed deity body, face, arms, and 21-point hands)
  runtime: 'python:3.11'
  handler: main:handler
  eventTimeout: 180s

  env:
    - name: NINEROUTER_URL
      value: 'http://host.docker.internal:20128'
    - name: VISION_MODEL
      value: 'ag/claude-opus-4-6-thinking'
    - name: BUDDHA_MODEL
      value: ''
    - name: BUDDHA_REFINE_MODEL
      value: ''
    - name: BUDDHA_ALLOW_MODEL_FALLBACK
      value: '1'
    - name: DETECTION_MODE
      value: 'buddha_multilimbs'
    - name: NINEROUTER_TIMEOUT
      value: '120.0'
    - name: MAX_IMAGE_SIZE
      value: '1280'
    - name: MAX_TOKENS
      value: '4000'
    - name: BUDDHA_MAX_ARMS
      value: '100'
    - name: BUDDHA_MAX_HAND_REFINEMENTS
      value: '60'
    - name: BUDDHA_TWO_PASS
      value: '1'
    - name: BUDDHA_HAND_REFINE
      value: '1'
    - name: BUDDHA_REFINE_WORKERS
      value: '2'
    - name: BUDDHA_MIN_CONFIDENCE
      value: '0.30'
    - name: FEEDBACK_DATA_DIR
      value: /opt/nuclio/feedback
    - name: FEEDBACK_DB_PATH
      value: /opt/nuclio/feedback/feedback.sqlite3

  volumes:
    - volume:
        name: feedback-data
        hostPath:
          path: {FEEDBACK_HOST_PATH}
      volumeMount:
        name: feedback-data
        mountPath: /opt/nuclio/feedback

  triggers:
    myHttpTrigger:
      maxWorkers: 1
      numWorkers: 1
      kind: http
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        port: 5776
        maxRequestBodySize: 33554432

  build:
    image: cvat.custom.ninerouter.buddha.multilimbs
    baseImage: python:3.11-slim
    directives:
      preCopy:
        - kind: RUN
          value: pip install --no-cache-dir requests pillow pyyaml

  platform:
    attributes:
      restartPolicy:
        name: always
        maximumRetryCount: 3
      mountMode: volume
"""

out_path = ROOT / "serverless" / "ninerouter-buddha-multilimbs" / "nuclio" / "function.yaml"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(buddha_yaml, encoding="utf-8")
print(f"Generated Buddha function.yaml at {out_path} ({len(buddha_yaml)} bytes, {len(specs)} labels)")
