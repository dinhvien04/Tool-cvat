"""Inspect and print the canonical 10-label CVAT task specification for Buddha Multi-Limb detector.

Usage:
    python scripts/buddha_schema.py [--json] [--yaml] [--summary] [--task-id <id>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.buddha_contract import (
    build_cvat_buddha_multilimbs_full_spec,
    load_buddha_schema,
)


def print_summary(specs: List[Dict[str, Any]]) -> None:
    schema = load_buddha_schema()
    print("=" * 72)
    print(" 9Router Buddha Multi-Limb Pose Detector - CVAT Label Schema")
    print("=" * 72)
    print(f"Total Labels: {len(specs)}")
    print(f"Supported Skeletons: 10 (1 Person + 7 VF50 Facial + 1 Buddha Arm + 1 Buddha Hand)")
    print("-" * 72)
    print(f"{'ID':<4} {'Label Name':<20} {'Shape':<12} {'Sublabels':<10} {'Color':<10}")
    print("-" * 72)
    for s in specs:
        lbl_id = s.get("id")
        name = s.get("name")
        stype = s.get("type")
        sublabels = s.get("sublabels", [])
        color = s.get("color", "N/A")
        print(f"{lbl_id:<4} {name:<20} {stype:<12} {len(sublabels):<10} {color:<10}")

    print("\n" + "=" * 72)
    print(" Detailed Sublabel Topology")
    print("=" * 72)
    for s in specs:
        name = s.get("name")
        sublabels = s.get("sublabels", [])
        sub_names = [sub.get("name") for sub in sublabels if isinstance(sub, dict)]
        if len(sub_names) <= 6:
            print(f" • {name} ({len(sub_names)} pts): {', '.join(sub_names)}")
        else:
            first_few = ", ".join(sub_names[:5])
            print(f" • {name} ({len(sub_names)} pts): {first_few}, ... (+{len(sub_names) - 5} more)")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect CVAT Buddha Multi-Limb label specification")
    parser.add_argument("--json", action="store_true", help="Print raw JSON spec for CVAT task creation")
    parser.add_argument("--yaml", action="store_true", help="Print YAML representation of spec")
    parser.add_argument("--summary", action="store_true", default=True, help="Print formatted human-readable summary")
    parser.add_argument("--task-id", type=int, default=0, help="Optional CVAT task ID to verify compatibility against")
    args = parser.parse_args()

    specs = build_cvat_buddha_multilimbs_full_spec()

    if args.json:
        print(json.dumps(specs, indent=2))
        return 0

    if args.yaml:
        import yaml
        print(yaml.dump(specs, sort_keys=False, allow_unicode=True))
        return 0

    print_summary(specs)

    if args.task_id > 0:
        cvat_url = os.getenv("CVAT_URL", "http://127.0.0.1:8080")
        token = os.getenv("CVAT_TOKEN")
        print(f"\nChecking compatibility with CVAT Task #{args.task_id} at {cvat_url}...")
        if not token:
            print("  [WARN] CVAT_TOKEN not set in environment. Skipping live task verification.")
        else:
            try:
                import requests
                headers = {"Authorization": f"Token {token}"}
                resp = requests.get(f"{cvat_url}/api/tasks/{args.task_id}", headers=headers, timeout=5)
                if resp.status_code == 200:
                    task_data = resp.json()
                    task_labels = [l.get("name") for l in task_data.get("labels", [])]
                    spec_labels = [s.get("name") for s in specs]
                    matching = set(spec_labels).intersection(set(task_labels))
                    missing = set(spec_labels) - set(task_labels)
                    print(f"  [INFO] Task #{args.task_id} contains {len(task_labels)} labels.")
                    print(f"  [INFO] Matching Buddha labels: {len(matching)}/{len(spec_labels)}")
                    if missing:
                        print(f"  [WARN] Missing labels in task: {sorted(missing)}")
                    else:
                        print(f"  [PASS] Task #{args.task_id} is 100% compatible with Buddha Multi-Limb detector!")
                else:
                    print(f"  [FAIL] Failed to retrieve task #{args.task_id}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [ERROR] Could not connect to CVAT: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
