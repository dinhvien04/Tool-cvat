"""Inspect, validate, fingerprint, and check task compatibility for Buddha Multi-Limb CVAT detector.

Usage:
    python scripts/buddha_schema.py [--json] [--yaml] [--summary]
    python scripts/buddha_schema.py --write-json config/buddha_cvat_labels.raw.json
    python scripts/buddha_schema.py --fingerprint
    python scripts/buddha_schema.py --task-id 22 [--strict] [--mapping-plan]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.buddha_contract import (
    build_cvat_buddha_multilimbs_full_spec,
    compute_buddha_spec_fingerprint,
    load_buddha_schema,
)
from core.pose_face_schema import POSE17_ID_TO_KEYPOINT, POSE17_KEYPOINT_TO_ID, POSE17_KEYPOINTS


def print_summary(specs: List[Dict[str, Any]]) -> None:
    """Print formatted summary of the 10-skeleton Buddha detector schema."""
    print("=" * 76)
    print(" 9Router Buddha Multi-Limb Pose Detector - Canonical Label Specification")
    print("=" * 76)
    print(f"Total Parent Labels : {len(specs)}")
    print("Supported Topology  : 10 Skeletons (1 Person + 7 VF50 Facial + 1 Arm + 1 Hand)")
    fingerprint = compute_buddha_spec_fingerprint(specs)
    print(f"Schema Fingerprint  : {fingerprint}")
    print("-" * 76)
    print(f"{'ID':<4} {'Label Name':<20} {'Type':<12} {'Sublabels':<10} {'Color':<10}")
    print("-" * 76)
    for s in specs:
        lbl_id = s.get("id")
        name = s.get("name")
        stype = s.get("type")
        sublabels = s.get("sublabels", [])
        color = s.get("color", "N/A")
        print(f"{lbl_id:<4} {name:<20} {stype:<12} {len(sublabels):<10} {color:<10}")

    print("\n" + "=" * 76)
    print(" Detailed Sublabel Topology")
    print("=" * 76)
    for s in specs:
        name = s.get("name")
        sublabels = s.get("sublabels", [])
        sub_names = [sub.get("name") for sub in sublabels if isinstance(sub, dict)]
        if len(sub_names) <= 6:
            print(f" • {name:<14} ({len(sub_names):>2} pts): {', '.join(sub_names)}")
        else:
            first_few = ", ".join(sub_names[:5])
            print(f" • {name:<14} ({len(sub_names):>2} pts): {first_few}, ... (+{len(sub_names) - 5} more)")
    print("=" * 76)


def get_task_info(task_id: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """Retrieve CVAT task labels with child sublabels.

    Attempts:
    1. Direct Django ORM query via 'docker exec -i cvat_server' (best for local dev).
    2. CVAT REST API if CVAT_TOKEN is provided.
    """
    # 1. Try Docker exec into cvat_server container
    docker_py = f"""
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
try:
    django.setup()
    from cvat.apps.engine.models import Task
    t = Task.objects.get(id={task_id})
    labels = []
    for l in t.get_labels():
        subs = []
        for s in l.sublabels.all():
            subs.append({{'id': s.id, 'name': s.name, 'type': s.type}})
        labels.append({{'id': l.id, 'name': l.name, 'type': l.type, 'sublabels': subs}})
    print("TASK_JSON:" + json.dumps({{'id': t.id, 'name': t.name, 'labels': labels}}))
except Exception as e:
    print("TASK_ERR:" + str(e))
"""
    try:
        proc = subprocess.run(
            ["docker", "exec", "-i", "cvat_server", "python3", "-c", docker_py],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("TASK_JSON:"):
                return json.loads(line[len("TASK_JSON:"): ]), "docker"
            elif line.startswith("TASK_ERR:"):
                err_msg = line[len("TASK_ERR:"): ]
                if "DoesNotExist" in err_msg:
                    return None, f"Task #{task_id} not found in CVAT database"
    except Exception:
        pass

    # 2. Fallback to CVAT REST API
    cvat_url = os.getenv("CVAT_URL", "http://127.0.0.1:8080")
    token = os.getenv("CVAT_TOKEN")
    if token:
        try:
            import requests
            headers = {"Authorization": f"Token {token}"}
            resp = requests.get(f"{cvat_url}/api/tasks/{task_id}", headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.json(), "rest"
            return None, f"CVAT REST API returned HTTP {resp.status_code}"
        except Exception as e:
            return None, f"CVAT REST API connection error: {e}"

    return None, "Unable to query cvat_server container or CVAT REST API"


def inspect_task_compatibility(
    task_id: int,
    specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recursively inspect task compatibility against Buddha Multi-Limb spec."""
    task_data, source = get_task_info(task_id)
    if not task_data:
        return {
            "task_id": task_id,
            "compatible": False,
            "error": source,
            "parent_issues": [],
            "sublabel_issues": [],
            "labels": {},
        }

    task_name = task_data.get("name", str(task_id))
    task_labels_raw = task_data.get("labels", [])

    task_labels_map: Dict[str, Dict[str, Any]] = {}
    for tl in task_labels_raw:
        if isinstance(tl, dict):
            task_labels_map[tl.get("name", "")] = tl

    parent_issues: List[str] = []
    sublabel_issues: List[Dict[str, Any]] = []

    model_parents = [s.get("name") for s in specs]

    for model_spec in specs:
        p_name = model_spec.get("name")
        p_type = model_spec.get("type")
        model_sublabels = [sub.get("name") for sub in model_spec.get("sublabels", [])]

        if p_name not in task_labels_map:
            parent_issues.append(f"Missing parent label in task: '{p_name}'")
            continue

        tl = task_labels_map[p_name]
        t_type = tl.get("type")
        if t_type != p_type:
            parent_issues.append(
                f"Parent label '{p_name}' type mismatch: model expects '{p_type}', task has '{t_type}'"
            )

        task_sublabels = [sub.get("name") for sub in tl.get("sublabels", [])]

        # Compare child sublabels
        model_subs_set = set(model_sublabels)
        task_subs_set = set(task_sublabels)

        if model_subs_set != task_subs_set:
            sublabel_issues.append({
                "parent": p_name,
                "model_count": len(model_sublabels),
                "task_count": len(task_sublabels),
                "model_sublabels": model_sublabels,
                "task_sublabels": task_sublabels,
                "missing_in_task": sorted(list(model_subs_set - task_subs_set)),
                "extra_in_task": sorted(list(task_subs_set - model_subs_set)),
            })

    is_compatible = len(parent_issues) == 0 and len(sublabel_issues) == 0

    return {
        "task_id": task_id,
        "task_name": task_name,
        "source": source,
        "compatible": is_compatible,
        "parent_issues": parent_issues,
        "sublabel_issues": sublabel_issues,
        "total_task_labels": len(task_labels_raw),
    }


def generate_mapping_plan(task_id: int, specs: List[Dict[str, Any]]) -> None:
    """Generate and display exact model-to-task mapping plan."""
    task_data, _ = get_task_info(task_id)
    print("\n" + "=" * 76)
    print(f" MODEL-TO-TASK MAPPING PLAN FOR CVAT TASK #{task_id}")
    print("=" * 76)

    # 1. Person mapping (COCO semantic vs numeric strings)
    print("\n[Parent Skeleton: 'person' (17 Keypoints)]")
    print("  Root Cause Explanation:")
    print("  - Model Spec defines sublabels as numeric strings: \"1\" .. \"17\" (CVAT standard).")
    print("  - Task defines sublabels as semantic names: \"nose\" .. \"left_ankle\".")
    print("  - CVAT UI throws 'Mapping for elements was not specified in skeleton person'")
    print("    if elements are not explicitly mapped in the AI Tools detector modal.")

    print("\n  Required Element Mapping Dictionary (Model Element -> Task Element):")
    pose17_mapping: Dict[str, str] = {}
    for num_id in range(1, 18):
        sem_name = POSE17_ID_TO_KEYPOINT.get(num_id, "")
        pose17_mapping[str(num_id)] = sem_name
        print(f"    • Model \"{num_id:>2}\" ({sem_name:<14})  ==>  Task Element: \"{sem_name}\"")

    # 2. Other 9 skeletons
    print("\n[Remaining 9 Skeleton Labels:]")
    for s in specs:
        name = s.get("name")
        if name == "person":
            continue
        sublabels = [sub.get("name") for sub in s.get("sublabels", [])]
        print(f"  • Parent '{name}' ({len(sublabels)} pts): 1:1 identical sublabel names (Auto-matched)")

    # 3. Actionable remediation
    print("\n" + "-" * 76)
    print(" ACTIONABLE REMEDIATION OPTIONS:")
    print("-" * 76)
    print("  Option A (Recommended for new tasks):")
    print("    Create a fresh CVAT task using the canonical raw labels JSON:")
    print("    python scripts/buddha_schema.py --write-json config/buddha_cvat_labels.raw.json")
    print("    In CVAT Web UI: Create Task -> Raw -> Paste JSON from config/buddha_cvat_labels.raw.json")
    print("    -> 100% zero-configuration automatic element matching.")
    print("\n  Option B (For existing legacy Task 22):")
    print("    In CVAT Web UI on Task 22 -> AI Tools -> Detectors -> Select '9Router Buddha Multi-Limb Pose':")
    print("    In the mapping dialog, expand skeleton 'person' and map numeric IDs 1..17 to semantic keypoints:")
    print(f"    {json.dumps(pose17_mapping, indent=6)}")
    print("=" * 76)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate CVAT Buddha Multi-Limb label specification")
    parser.add_argument("--json", action="store_true", help="Print raw JSON spec array for CVAT task creation")
    parser.add_argument("--yaml", action="store_true", help="Print YAML representation of spec")
    parser.add_argument("--summary", action="store_true", default=False, help="Print formatted human-readable summary")
    parser.add_argument("--fingerprint", action="store_true", help="Print SHA-256 fingerprint of canonical spec")
    parser.add_argument("--write-json", type=str, default="", help="Write canonical raw JSON spec to specified file path")
    parser.add_argument("--task-id", type=int, default=0, help="Optional CVAT task ID to verify compatibility against")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if any compatibility issue is detected")
    parser.add_argument("--mapping-plan", action="store_true", help="Output model-to-task mapping plan for the specified task")
    args = parser.parse_args()

    specs = build_cvat_buddha_multilimbs_full_spec()

    # 1. Output file if requested
    if args.write_json:
        target_path = Path(args.write_json)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
        print(f"[OK] Wrote canonical CVAT labels JSON ({len(specs)} labels) to {target_path}")
        return 0

    # 2. Output fingerprint
    if args.fingerprint:
        fp = compute_buddha_spec_fingerprint(specs)
        print(f"BUDDHA_SPEC_FINGERPRINT={fp}")
        return 0

    # 3. Output JSON or YAML
    if args.json:
        print(json.dumps(specs, indent=2))
        return 0

    if args.yaml:
        import yaml
        print(yaml.dump(specs, sort_keys=False, allow_unicode=True))
        return 0

    # 4. Print Summary
    print_summary(specs)

    # 5. Task compatibility verification
    if args.task_id > 0:
        print(f"\n[Task Compatibility Check: Task #{args.task_id}]")
        comp = inspect_task_compatibility(args.task_id, specs)

        if "error" in comp and comp["error"]:
            print(f"  [ERROR] {comp['error']}")
            if args.strict:
                return 1
            return 0

        print(f"  Task Name          : {comp.get('task_name')}")
        print(f"  Data Source        : {comp.get('source')}")
        print(f"  Total Task Labels  : {comp.get('total_task_labels')}")

        if comp.get("parent_issues"):
            print(f"  [FAIL] Parent Label Issues ({len(comp['parent_issues'])}):")
            for issue in comp["parent_issues"]:
                print(f"    • {issue}")

        if comp.get("sublabel_issues"):
            print(f"  [WARN] Child Sublabel Mismatches ({len(comp['sublabel_issues'])}):")
            for sub_issue in comp["sublabel_issues"]:
                p = sub_issue["parent"]
                print(f"    • Skeleton '{p}':")
                print(f"        Model sublabels ({sub_issue['model_count']}): {sub_issue['model_sublabels']}")
                print(f"        Task sublabels  ({sub_issue['task_count']}): {sub_issue['task_sublabels']}")
                if sub_issue["missing_in_task"]:
                    print(f"        Missing in task : {sub_issue['missing_in_task']}")
                if sub_issue["extra_in_task"]:
                    print(f"        Extra in task   : {sub_issue['extra_in_task']}")

        if comp.get("compatible"):
            print(f"  [PASS] Task #{args.task_id} is 100% natively compatible with Buddha detector!")
        else:
            print(f"  [NOTICE] Task #{args.task_id} has sublabel mapping divergence.")

        if args.mapping_plan or not comp.get("compatible"):
            generate_mapping_plan(args.task_id, specs)

        if args.strict and not comp.get("compatible"):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
