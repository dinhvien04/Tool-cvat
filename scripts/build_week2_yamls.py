"""Generate function.yaml files for Week-2 Pose17 and VF50 Nuclio functions."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.skeleton_contract import (
    build_cvat_pose17_spec,
    build_cvat_vf50_spec,
)

FEEDBACK_HOST_PATH = (ROOT / ".tool-cvat").as_posix()

# 1. Pose 17
pose_spec = [build_cvat_pose17_spec(parent_label="person")]
pose_json = json.dumps(pose_spec, indent=2)
indented_pose_json = "\n".join(("      " + line) if line else "" for line in pose_json.splitlines())

pose_yaml = f"""metadata:
  name: ninerouter-human-pose-17
  namespace: cvat
  annotations:
    name: 9Router Human Pose 17
    type: detector
    spec: |
{indented_pose_json}

spec:
  description: 9Router Human Pose 17 Detector (COCO 17-Keypoint Human Skeleton via local 9Router)
  runtime: 'python:3.11'
  handler: main:handler
  eventTimeout: 60s

  env:
    - name: NINEROUTER_URL
      value: 'http://host.docker.internal:20128'
    - name: VISION_MODEL
      value: 'ag/gemini-3.8-flash-low'
    - name: POSE17_MODEL
      value: 'ag/gemini-3.8-flash-low'
    - name: DETECTION_MODE
      value: 'human_pose_17'
    - name: NINEROUTER_TIMEOUT
      value: '45.0'
    - name: MAX_IMAGE_SIZE
      value: '1280'
    - name: MAX_TOKENS
      value: '2000'
    - name: FEEDBACK_DATA_DIR
      value: '/opt/nuclio/feedback'
    - name: FEEDBACK_DB_PATH
      value: '/opt/nuclio/feedback/feedback.sqlite3'

  volumes:
    - volume:
        name: feedback-data
        hostPath:
          path: '{FEEDBACK_HOST_PATH}'
      volumeMount:
        name: feedback-data
        mountPath: '/opt/nuclio/feedback'

  triggers:
    myHttpTrigger:
      numWorkers: 2
      kind: 'http'
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        maxRequestBodySize: 33554432 # 32MB

  build:
    image: cvat.custom.ninerouter.human.pose.17
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

pose_path = ROOT / "serverless" / "ninerouter-human-pose-17" / "nuclio" / "function.yaml"
pose_path.parent.mkdir(parents=True, exist_ok=True)
pose_path.write_text(pose_yaml, encoding="utf-8")
print(f"Generated {pose_path} ({len(pose_yaml)} bytes)")

# 2. VF50 (Authoritative 7 Component Skeletons, NO parent 'face' skeleton)
vf_spec = build_cvat_vf50_spec()
vf_json = json.dumps(vf_spec, separators=(",", ":"))

vf_yaml = f"""metadata:
  name: ninerouter-face-vf50
  namespace: cvat
  annotations:
    name: 9Router Face Landmark VF-50
    type: detector
    spec: '{vf_json}'

spec:
  description: 9Router Face Landmark VF-50 Detector (VinAI 50 Facial Landmarks across 7 Component Skeletons via local 9Router)
  runtime: 'python:3.11'
  handler: main:handler
  eventTimeout: 60s

  env:
    - name: NINEROUTER_URL
      value: 'http://host.docker.internal:20128'
    - name: VISION_MODEL
      value: 'ag/gemini-3.8-flash-low'
    - name: VF50_MODEL
      value: 'ag/gemini-3.8-flash-low'
    - name: DETECTION_MODE
      value: 'face_vf50'
    - name: NINEROUTER_TIMEOUT
      value: '45.0'
    - name: MAX_IMAGE_SIZE
      value: '1280'
    - name: MAX_TOKENS
      value: '2500'
    - name: FEEDBACK_DATA_DIR
      value: '/opt/nuclio/feedback'
    - name: FEEDBACK_DB_PATH
      value: '/opt/nuclio/feedback/feedback.sqlite3'

  volumes:
    - volume:
        name: feedback-data
        hostPath:
          path: '{FEEDBACK_HOST_PATH}'
      volumeMount:
        name: feedback-data
        mountPath: '/opt/nuclio/feedback'

  triggers:
    myHttpTrigger:
      numWorkers: 2
      kind: 'http'
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        maxRequestBodySize: 33554432 # 32MB

  build:
    image: cvat.custom.ninerouter.face.vf50
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

vf_path = ROOT / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml"
vf_path.parent.mkdir(parents=True, exist_ok=True)
vf_path.write_text(vf_yaml, encoding="utf-8")
print(f"Generated {vf_path} ({len(vf_yaml)} bytes)")
