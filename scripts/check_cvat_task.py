"""CVAT Task Label Inspector and Compatibility Checker for Phase 3B.

Inspects a CVAT task's defined labels, compares them against the Phase 3B 31-label
master schema, and produces an actionable compatibility report and mapping table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from core.taxonomy import (
    AMBIGUITY_PAIRS as AMBIGUOUS_PAIRS,
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    MASTER_31_LABELS as ALL_31_LABELS,
    Taxonomy,
)


def get_task_info_from_docker(task_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve task details and labels from cvat_server container via Django ORM."""
    py_code = f"""
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
django.setup()
from cvat.apps.engine.models import Task

try:
    task = Task.objects.get(id={task_id})
    labels = [l.name for l in task.get_labels()]
    print("TASK_DATA_JSON:" + json.dumps({{"id": task.id, "name": task.name, "labels": labels}}))
except Task.DoesNotExist:
    print("TASK_NOT_FOUND")
except Exception as e:
    print("TASK_ERROR:" + str(e))
"""
    try:
        res = subprocess.run(
            ["docker", "exec", "-i", "cvat_server", "python3", "-c", py_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout = res.stdout
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("TASK_DATA_JSON:"):
                return json.loads(line[len("TASK_DATA_JSON:"):])
            elif line == "TASK_NOT_FOUND":
                return None
            elif line.startswith("TASK_ERROR:"):
                raise RuntimeError(line[len("TASK_ERROR:"):])
    except Exception as e:
        print(f"Warning: Could not query CVAT container directly: {e}", file=sys.stderr)
        return None

    return None


def inspect_task(task_id: int, task_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Inspect and analyze task labels against the Phase 3B 31-label master taxonomy."""
    if task_data is None:
        task_data = get_task_info_from_docker(task_id)

    if not task_data:
        return {
            "task_id": task_id,
            "error": f"Task ID {task_id} not found in CVAT or container unreachable.",
        }

    task_name = task_data.get("name", f"Task {task_id}")
    task_labels = [l.strip() for l in task_data.get("labels", []) if l.strip()]
    task_labels_set = set(task_labels)

    master_31_set = set(ALL_31_LABELS)
    taxonomy = Taxonomy()

    matched_31 = sorted(list(task_labels_set & master_31_set))
    foreign_labels = sorted(list(task_labels_set - master_31_set))
    unmapped_31 = sorted(list(master_31_set - task_labels_set))

    # Group counts in matched labels
    instances_matched = [l for l in matched_31 if taxonomy.get_group(l) == GROUP_INSTANCE]
    regions_matched = [l for l in matched_31 if taxonomy.get_group(l) == GROUP_REGION]
    lanes_matched = [l for l in matched_31 if taxonomy.get_group(l) == GROUP_LANE]

    # Check ambiguous label presence
    ambiguities = []
    for pair in AMBIGUOUS_PAIRS:
        in_task = [l for l in pair if l in task_labels_set]
        if len(in_task) > 1:
            ambiguities.append({
                "pair": pair,
                "status": "BOTH_PRESENT",
                "message": f"Both '{pair[0]}' and '{pair[1]}' are present in task labels. Strict segregation policy applies.",
            })
        elif len(in_task) == 1:
            counterpart = pair[1] if in_task[0] == pair[0] else pair[0]
            ambiguities.append({
                "pair": pair,
                "present": in_task[0],
                "counterpart": counterpart,
                "status": "SINGLE_PRESENT",
                "message": f"Only '{in_task[0]}' is defined in task. Counterpart '{counterpart}' will not conflict.",
            })

    # Determine overall compatibility status
    if len(foreign_labels) == 0 and len(unmapped_31) == 0:
        overall_status = "PERFECT_MATCH_31"
        status_desc = "Task contains all 31 labels exactly matching master schema."
    elif len(foreign_labels) == 0 and len(matched_31) > 0:
        overall_status = "STRICT_SUBSET"
        status_desc = f"Task is a strict subset of 31-label master schema ({len(matched_31)}/31 active labels)."
    elif len(foreign_labels) > 0 and len(matched_31) > 0:
        overall_status = "PARTIAL_MATCH_WITH_FOREIGN"
        status_desc = f"Task contains {len(matched_31)} supported labels and {len(foreign_labels)} foreign/unsupported labels."
    else:
        overall_status = "NO_MATCH"
        status_desc = "None of the task labels match the Phase 3B 31-label master schema."

    # Build mapping table rows
    mapping_rows = []
    for lbl in task_labels:
        if lbl in master_31_set:
            grp = taxonomy.get_group(lbl)
            meta = taxonomy.get_metadata(lbl)
            shapes = ", ".join(meta.allowed_shapes) if meta else "any"
            mapping_rows.append({
                "task_label": lbl,
                "model_label": lbl,
                "group": grp,
                "cvat_shapes": shapes,
                "status": "SUPPORTED",
            })
        else:
            # Check if case-insensitive or fuzzy match exists
            lower_to_orig = {l.lower(): l for l in ALL_31_LABELS}
            suggestion = lower_to_orig.get(lbl.lower())
            mapping_rows.append({
                "task_label": lbl,
                "model_label": suggestion or "UNMAPPED",
                "group": "FOREIGN",
                "cvat_shapes": "N/A",
                "status": "FOREIGN_LABEL" if not suggestion else f"CASE_MISMATCH (suggest: '{suggestion}')",
            })

    return {
        "task_id": task_id,
        "task_name": task_name,
        "total_task_labels": len(task_labels),
        "overall_status": overall_status,
        "status_description": status_desc,
        "matched_count": len(matched_31),
        "foreign_count": len(foreign_labels),
        "active_groups": {
            "instances": len(instances_matched),
            "regions": len(regions_matched),
            "lanes": len(lanes_matched),
        },
        "matched_labels": matched_31,
        "foreign_labels": foreign_labels,
        "ambiguities": ambiguities,
        "mapping_table": mapping_rows,
    }


def print_report(report: Dict[str, Any]):
    """Print a terminal report with mapping table."""
    if "error" in report:
        print(f"\n[ERROR] Task {report.get('task_id')}: {report['error']}")
        return

    print("\n" + "=" * 76)
    print(f" CVAT Task Compatibility Report - Task #{report['task_id']}: '{report['task_name']}'")
    print("=" * 76)
    print(f"Overall Status:       {report['overall_status']}")
    print(f"Details:              {report['status_description']}")
    print(f"Matched 31-Labels:    {report['matched_count']} / 31")
    print(f"Foreign Labels:       {report['foreign_count']}")
    print(f"Active Groups:        Instances: {report['active_groups']['instances']} | Regions: {report['active_groups']['regions']} | Lanes: {report['active_groups']['lanes']}")

    if report["ambiguities"]:
        print("\nAmbiguity Analysis:")
        for amb in report["ambiguities"]:
            print(f" - [{amb['status']}] {amb['message']}")

    print("\nLabel Mapping & Routing Table:")
    print("-" * 76)
    header = f"{'Task Label':<22} | {'Model Label':<22} | {'Group':<10} | {'Shapes':<14}"
    print(header)
    print("-" * 76)
    for row in report["mapping_table"]:
        print(f"{row['task_label']:<22} | {row['model_label']:<22} | {row['group']:<10} | {row['cvat_shapes']:<14}")
    print("-" * 76)

    if report["foreign_labels"]:
        print(f"\n[!] Notice: Foreign labels {report['foreign_labels']} will be ignored by 9Router detector.")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect CVAT Task labels against Phase 3B master schema.")
    parser.add_argument("--task-id", "-t", type=int, required=True, help="CVAT Task ID to inspect")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    args = parser.parse_args()

    report = inspect_task(task_id=args.task_id)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
