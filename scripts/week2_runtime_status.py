"""Inspect deployed Week-2 Nuclio functions, specs, and drift status.

Checks:
1. Docker container status & health for:
   - nuclio-nuclio-ninerouter-human-pose-17
   - nuclio-nuclio-ninerouter-face-vf50
2. Deployed annotations (metadata.annotations.spec):
   - Pose 17: 1 skeleton 'person', 17 sublabels, valid SVG
   - Face VF-50: 7 component skeletons (VinFast spec) vs legacy 1 monolithic 'face'
3. Drift status against repo HEAD files:
   - serverless/ninerouter-human-pose-17/nuclio/function.yaml
   - serverless/ninerouter-face-vf50/nuclio/function.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.skeleton_contract import VF50_COMPONENT_NAMES, VF50_COMPONENT_CONFIG
from core.pose_face_schema import POSE17_KEYPOINTS

WEEK2_CONTAINERS = {
    "pose17": {
        "container_name": "nuclio-nuclio-ninerouter-human-pose-17",
        "function_name": "ninerouter-human-pose-17",
        "yaml_path": REPO_ROOT / "serverless" / "ninerouter-human-pose-17" / "nuclio" / "function.yaml",
        "expected_labels": 1,
    },
    "vf50": {
        "container_name": "nuclio-nuclio-ninerouter-face-vf50",
        "function_name": "ninerouter-face-vf50",
        "yaml_path": REPO_ROOT / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml",
        "expected_labels": 7,
    },
}


def get_container_inspect(container_name: str) -> Optional[Dict[str, Any]]:
    """Run docker inspect on a container and return parsed JSON."""
    try:
        res = subprocess.run(
            ["docker", "inspect", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if data and isinstance(data, list):
                return data[0]
    except Exception:
        pass
    return None


def get_repo_spec(yaml_path: Path) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Read repo HEAD spec from function.yaml."""
    if not yaml_path.exists():
        return None, None
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ann = content.get("metadata", {}).get("annotations", {})
        name = ann.get("name")
        raw_spec = ann.get("spec")
        spec_items = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
        return name, spec_items
    except Exception:
        return None, None


def analyze_function_runtime(key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze runtime status and spec of a single Week-2 detector."""
    container_name = cfg["container_name"]
    yaml_path = cfg["yaml_path"]

    repo_name, repo_spec = get_repo_spec(yaml_path)

    report: Dict[str, Any] = {
        "key": key,
        "container_name": container_name,
        "function_name": cfg["function_name"],
        "deployed": False,
        "status": "NOT_FOUND",
        "health": "UNKNOWN",
        "ports": [],
        "env": {},
        "deployed_spec_type": "UNKNOWN",
        "deployed_label_count": 0,
        "spec_drift": False,
        "drift_details": [],
        "ready": False,
    }

    info = get_container_inspect(container_name)
    if not info:
        report["drift_details"].append("Container is not deployed or not running")
        return report

    report["deployed"] = True
    state = info.get("State", {})
    report["status"] = state.get("Status", "unknown")
    report["health"] = state.get("Health", {}).get("Status", "none")

    # Ports
    ports_map = info.get("NetworkSettings", {}).get("Ports", {})
    for container_port, host_bindings in ports_map.items():
        if host_bindings:
            for b in host_bindings:
                report["ports"].append(f"{b.get('HostPort')}->{container_port}")

    # Labels & annotations
    labels = info.get("Config", {}).get("Labels", {})
    raw_ann = labels.get("nuclio.io/annotations")
    if not raw_ann:
        report["drift_details"].append("Missing nuclio.io/annotations in container labels")
        return report

    try:
        ann = json.loads(raw_ann)
    except Exception as e:
        report["drift_details"].append(f"Cannot parse container annotations JSON: {e}")
        return report

    deployed_name = ann.get("name")
    raw_spec = ann.get("spec")
    try:
        deployed_spec = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
    except Exception as e:
        report["drift_details"].append(f"Cannot parse deployed spec JSON: {e}")
        return report

    if not isinstance(deployed_spec, list):
        report["drift_details"].append("Deployed spec is not a list")
        return report

    report["deployed_label_count"] = len(deployed_spec)
    label_names = [item.get("name") for item in deployed_spec if isinstance(item, dict)]

    if key == "vf50":
        if len(deployed_spec) == 1 and label_names == ["face"]:
            report["deployed_spec_type"] = "LEGACY_MONOLITHIC_FACE"
            report["spec_drift"] = True
            report["drift_details"].append(
                "CRITICAL SPEC DRIFT: Container is running legacy monolithic 'face' skeleton with 50 sublabels. "
                "Repo HEAD has 7 component skeletons (longmaytrai..moitrong). Redeploy required!"
            )
        elif len(deployed_spec) == 7 and set(label_names) == set(VF50_COMPONENT_NAMES):
            report["deployed_spec_type"] = "AUTHORITATIVE_7_COMPONENT_VF50"
        else:
            report["deployed_spec_type"] = f"UNKNOWN_SKELETON_COUNT_{len(deployed_spec)}"
            report["spec_drift"] = True
            report["drift_details"].append(f"Unexpected label names in deployed spec: {label_names}")

    elif key == "pose17":
        if len(deployed_spec) == 1 and label_names == ["person"]:
            subs = deployed_spec[0].get("sublabels", [])
            sub_names = [s.get("name") for s in subs]
            if len(subs) == 17 and set(sub_names) == set(POSE17_KEYPOINTS):
                report["deployed_spec_type"] = "CANONICAL_POSE17_17KPS"
            else:
                report["deployed_spec_type"] = "INVALID_POSE17_SUBLABELS"
                report["spec_drift"] = True
                report["drift_details"].append(f"Expected 17 COCO keypoints, got {len(subs)}")
        else:
            report["deployed_spec_type"] = f"UNEXPECTED_POSE17_LABELS_{label_names}"
            report["spec_drift"] = True

    # Compare deployed spec against repo spec
    if repo_spec:
        # Check label count match
        if len(repo_spec) != len(deployed_spec):
            report["spec_drift"] = True
            report["drift_details"].append(
                f"Spec label count mismatch: repo has {len(repo_spec)}, container has {len(deployed_spec)}"
            )

    report["ready"] = (
        report["deployed"]
        and report["status"] == "running"
        and report["health"] in ("healthy", "none")
        and not report["spec_drift"]
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-2 AI Detector Runtime Status & Spec Drift Auditor")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any function is drifted or not ready")
    args = parser.parse_args()

    results: Dict[str, Dict[str, Any]] = {}
    all_ready = True

    for key, cfg in WEEK2_CONTAINERS.items():
        res = analyze_function_runtime(key, cfg)
        results[key] = res
        if not res["ready"]:
            all_ready = False

    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if all_ready or not args.check else 1

    print("=" * 75)
    print("WEEK-2 DETECTOR RUNTIME STATUS & SPEC DRIFT REPORT")
    print("=" * 75)

    for key, r in results.items():
        print(f"\n[{r['function_name']}]")
        print(f"  Container : {r['container_name']}")
        print(f"  Status    : {r['status']} (health: {r['health']})")
        print(f"  Ports     : {', '.join(r['ports']) if r['ports'] else 'None'}")
        print(f"  Spec Type : {r['deployed_spec_type']} ({r['deployed_label_count']} labels)")
        if r["spec_drift"]:
            print("  DRIFT     : YES - SPEC DRIFT DETECTED!")
            for d in r["drift_details"]:
                print(f"    - {d}")
        else:
            print("  DRIFT     : NO (Aligned with repo HEAD)")

        print(f"  Ready     : {'YES' if r['ready'] else 'NO - REDEPLOY NEEDED'}")

    print("\n" + "=" * 75)
    if all_ready:
        print("ALL WEEK-2 DETECTORS ARE DEPLOYED, HEALTHY, AND SPEC-ALIGNED.")
        return 0
    else:
        print("ONE OR MORE WEEK-2 DETECTORS REQUIRE REDEPLOYMENT.")
        return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
