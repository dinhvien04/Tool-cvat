"""CVAT 2.75.1 Skeleton Mapping Preflight Validator for Week-2 Detectors.

Validates CVAT task labels against Week-2 detector models:
1. 9Router Human Pose 17 (ninerouter-human-pose-17)
2. 9Router Face Landmark VF-50 (ninerouter-face-vf50)

Enforces CVAT 2.75.1 Lambda Manager Skeleton Mapping Contract:
- In CVAT 2.75.1 (cvat/apps/lambda_manager/views.py:446-450):
  When model label is skeleton and task label is skeleton:
  mapping_item MUST contain a nested "sublabels" mapping.
  If omitted or empty, CVAT throws:
  'Detection error occurred: Mapping for elements was not specified in skeleton "{model_label}"'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pose_face_schema import POSE17_KEYPOINTS, VF50_LANDMARKS
from core.skeleton_contract import (
    VF50_COMPONENT_NAMES,
    VF50_COMPONENT_CONFIG,
    build_cvat_pose17_spec,
    build_cvat_vf50_spec,
)

FIVE_POINT_FACE_CORRESPONDENCE = {
    "left_eye": {"component": "mattrai", "landmark_id": 18, "desc": "Left eye inner corner (display left)"},
    "right_eye": {"component": "matphai", "landmark_id": 26, "desc": "Right eye inner corner (display right)"},
    "nose": {"component": "songmui", "landmark_id": 13, "desc": "Nose bridge base point 13"},
    "mouth_left": {"component": "moingoai", "landmark_id": 30, "desc": "Outer lip left corner"},
    "mouth_right": {"component": "moingoai", "landmark_id": 36, "desc": "Outer lip right corner"},
}


def get_task_labels_from_cvat(task_id: int) -> Optional[List[Dict[str, Any]]]:
    """Retrieve task labels and their sublabels from CVAT via Docker Django ORM or HTTP API."""
    py_code = f"""
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
django.setup()
from cvat.apps.engine.models import Task

try:
    task = Task.objects.get(id={task_id})
    labels_data = []
    for l in task.get_labels():
        sublabels = [
            {{"id": s.id, "name": s.name, "type": getattr(s, "type", "points")}}
            for s in l.sublabels.all()
        ]
        labels_data.append({{
            "id": l.id,
            "name": l.name,
            "type": getattr(l, "type", "any"),
            "sublabels": sublabels,
        }})
    print("TASK_LABELS_JSON:" + json.dumps(labels_data))
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
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("TASK_LABELS_JSON:"):
                return json.loads(line[len("TASK_LABELS_JSON:"):])
            elif line == "TASK_NOT_FOUND":
                return None
            elif line.startswith("TASK_ERROR:"):
                print(f"Warning: CVAT query error: {line}", file=sys.stderr)
                return None
    except Exception as e:
        print(f"Warning: Could not query CVAT container directly: {e}", file=sys.stderr)

    return None


def get_model_spec(fn_name: str) -> List[Dict[str, Any]]:
    """Return canonical CVAT spec for the given Week-2 detector."""
    if fn_name in ("ninerouter-human-pose-17", "pose17", "pose"):
        return [build_cvat_pose17_spec(parent_label="person")]
    elif fn_name in ("ninerouter-face-vf50", "vf50", "face"):
        return build_cvat_vf50_spec()
    else:
        raise ValueError(f"Unknown Week-2 function: {fn_name!r}")


def validate_cvat_mapping(
    mapping: Dict[str, Any],
    model_spec: List[Dict[str, Any]],
    task_labels: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Simulate CVAT 2.75.1 backend validation logic (lambda_manager/views.py:446-450).

    In CVAT 2.75.1:
    if md_label["type"] == "skeleton" and db_label.type == "skeleton":
        if "sublabels" not in mapping_item:
            raise ValidationError(
                f'Mapping for elements was not specified in skeleton "{model_label_name}" '
            )
    """
    errors: List[str] = []
    task_label_map = {l["name"]: l for l in task_labels}
    model_label_map = {m["name"]: m for m in model_spec}

    for model_name, mapping_item in mapping.items():
        if model_name not in model_label_map:
            errors.append(f"Model label {model_name!r} not found in model specification")
            continue

        md_label = model_label_map[model_name]
        target_task_name = mapping_item.get("name", model_name)

        if target_task_name not in task_label_map:
            errors.append(
                f"Mapping target label {target_task_name!r} for model label {model_name!r} "
                f"does not exist in task labels"
            )
            continue

        db_label = task_label_map[target_task_name]

        if md_label.get("type") == "skeleton" and db_label.get("type") == "skeleton":
            if "sublabels" not in mapping_item:
                errors.append(
                    f'[CRITICAL CVAT 2.75.1 ERROR] Mapping for elements was not specified in skeleton '
                    f'"{model_name}" (missing "sublabels" key in mapping payload)'
                )
                continue

            sub_mapping = mapping_item.get("sublabels", {})
            if not isinstance(sub_mapping, dict) or not sub_mapping:
                errors.append(
                    f'[CRITICAL CVAT 2.75.1 ERROR] Skeleton "{model_name}" has empty sublabels mapping'
                )
                continue

            db_sub_names = {s["name"] for s in db_label.get("sublabels", [])}
            md_sub_names = {s["name"] for s in md_label.get("sublabels", [])}

            for md_sub_name, sub_map_item in sub_mapping.items():
                if md_sub_name not in md_sub_names:
                    errors.append(
                        f'Model sublabel "{md_sub_name}" does not exist in model skeleton "{model_name}"'
                    )
                target_sub = sub_map_item.get("name", md_sub_name)
                if target_sub not in db_sub_names:
                    errors.append(
                        f'Target sublabel "{target_sub}" mapped from "{md_sub_name}" '
                        f'does not exist in task skeleton "{target_task_name}"'
                    )

    return len(errors) == 0, errors


def build_compatible_mapping_payload(
    fn_name: str,
    task_labels: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Build a compliant mapping payload for CVAT 2.75.1 with sublabels guaranteed."""
    model_spec = get_model_spec(fn_name)
    mapping: Dict[str, Any] = {}
    diagnostics: List[str] = []

    task_label_map = {l["name"]: l for l in task_labels}

    if fn_name in ("ninerouter-human-pose-17", "pose17", "pose"):
        # Model has 1 skeleton 'person' with 17 sublabels
        target_name = None
        if "person" in task_label_map and task_label_map["person"].get("type") == "skeleton":
            target_name = "person"
        elif "body" in task_label_map and task_label_map["body"].get("type") == "skeleton":
            target_name = "body"

        if not target_name:
            diagnostics.append(
                "Task does not contain a compatible 'person' or 'body' skeleton label."
            )
            return None, diagnostics

        db_label = task_label_map[target_name]
        db_sub_names = {s["name"] for s in db_label.get("sublabels", [])}

        sublabels_mapping: Dict[str, Any] = {}
        for kp in POSE17_KEYPOINTS:
            if kp in db_sub_names:
                sublabels_mapping[kp] = {"name": kp, "attributes": {}}

        if not sublabels_mapping:
            diagnostics.append(
                f"Task skeleton '{target_name}' has 0 matching sublabels with Pose 17."
            )
            return None, diagnostics

        mapping["person"] = {
            "name": target_name,
            "attributes": {},
            "sublabels": sublabels_mapping,
        }
        diagnostics.append(
            f"Successfully mapped 'person' -> '{target_name}' with {len(sublabels_mapping)}/17 sublabels."
        )

    elif fn_name in ("ninerouter-face-vf50", "vf50", "face"):
        # Model has 7 component skeletons: longmaytrai..moitrong
        matched_comps = [c for c in VF50_COMPONENT_NAMES if c in task_label_map and task_label_map[c].get("type") == "skeleton"]

        if matched_comps:
            for cname in matched_comps:
                db_label = task_label_map[cname]
                db_sub_names = {s["name"] for s in db_label.get("sublabels", [])}
                cfg = VF50_COMPONENT_CONFIG[cname]
                expected_subs = [str(i) for i in range(cfg["start"], cfg["end"] + 1)]

                sub_map: Dict[str, Any] = {}
                for s in expected_subs:
                    if s in db_sub_names:
                        sub_map[s] = {"name": s, "attributes": {}}

                if sub_map:
                    mapping[cname] = {
                        "name": cname,
                        "attributes": {},
                        "sublabels": sub_map,
                    }
            diagnostics.append(
                f"Mapped {len(mapping)}/7 VF-50 component skeletons to task."
            )
        else:
            # Check if task has a 5-point or monolithic 'face' skeleton
            if "face" in task_label_map:
                face_label = task_label_map["face"]
                face_subs = [s["name"] for s in face_label.get("sublabels", [])]
                diagnostics.append(
                    f"[INCOMPATIBLE TOPOLOGY] Task defines a monolithic skeleton 'face' with "
                    f"{len(face_subs)} sublabels {face_subs}. The model defines 7 independent "
                    f"component skeletons ({', '.join(VF50_COMPONENT_NAMES)}). "
                    f"In CVAT 2.75.1, skeleton mapping cannot merge multiple model skeletons "
                    f"into a single task skeleton. Attempting to map 'face' without sublabels "
                    f"WILL trigger: Detection error occurred: Mapping for elements was not specified in skeleton 'face'."
                )
                diagnostics.append(
                    "Recommended action: Add the 7 VinFast component skeletons to the task, or "
                    "use the 5-point face landmark correspondence table below:"
                )
                for k, v in FIVE_POINT_FACE_CORRESPONDENCE.items():
                    diagnostics.append(
                        f"  - {k:<12}: Component '{v['component']}', Landmark {v['landmark_id']} ({v['desc']})"
                    )
            else:
                diagnostics.append(
                    "Task does not contain VF-50 component skeletons."
                )
            return None, diagnostics

    return {"mapping": mapping}, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CVAT 2.75.1 Skeleton Mapping Preflight Validator for Week-2 Detectors"
    )
    parser.add_argument("--task-id", type=int, required=True, help="CVAT Task ID to inspect")
    parser.add_argument(
        "--function",
        choices=["ninerouter-human-pose-17", "ninerouter-face-vf50", "pose17", "vf50"],
        required=True,
        help="Target detector function",
    )
    parser.add_argument("--output-mapping", type=Path, help="Path to write validated mapping JSON")
    parser.add_argument("--validate-mapping", type=Path, help="Validate an existing mapping JSON file")

    args = parser.parse_args()

    fn_canonical = (
        "ninerouter-human-pose-17" if "pose" in args.function else "ninerouter-face-vf50"
    )

    print("=" * 70)
    print(f"CVAT 2.75.1 Skeleton Mapping Preflight: Task {args.task_id} vs {fn_canonical}")
    print("=" * 70)

    task_labels = get_task_labels_from_cvat(args.task_id)
    if not task_labels:
        print(f"[FAIL] Could not retrieve labels for Task {args.task_id} from CVAT.", file=sys.stderr)
        return 1

    print(f"Task {args.task_id} labels found:")
    for l in task_labels:
        sub_count = len(l.get("sublabels", []))
        subs_preview = [s["name"] for s in l.get("sublabels", [])[:5]]
        if sub_count > 5:
            subs_preview.append(f"... ({sub_count} total)")
        print(f"  - {l['name']} (type={l['type']}): sublabels={subs_preview}")

    print("\n--- Compatibility & Preflight Analysis ---")
    mapping_payload, diagnostics = build_compatible_mapping_payload(fn_canonical, task_labels)

    for diag in diagnostics:
        print(f"  {diag}")

    if args.validate_mapping:
        if not args.validate_mapping.exists():
            print(f"[FAIL] File not found: {args.validate_mapping}")
            return 1
        with open(args.validate_mapping, "r", encoding="utf-8") as f:
            custom_payload = json.load(f)
        custom_mapping = custom_payload.get("mapping", custom_payload)
        model_spec = get_model_spec(fn_canonical)
        is_valid, val_errs = validate_cvat_mapping(custom_mapping, model_spec, task_labels)
        if is_valid:
            print(f"\n[PASS] Mapping {args.validate_mapping} is 100% COMPATIBLE with CVAT 2.75.1 backend!")
            return 0
        else:
            print(f"\n[FAIL] Mapping {args.validate_mapping} has CVAT 2.75.1 validation errors:")
            for err in val_errs:
                print(f"  - {err}")
            return 1

    if mapping_payload is not None:
        model_spec = get_model_spec(fn_canonical)
        is_valid, val_errs = validate_cvat_mapping(mapping_payload["mapping"], model_spec, task_labels)
        if is_valid:
            print(f"\n[SUCCESS] Generated compliant CVAT 2.75.1 mapping payload.")
            if args.output_mapping:
                args.output_mapping.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")
                print(f"Saved payload to {args.output_mapping}")
            else:
                print("Payload preview:")
                print(json.dumps(mapping_payload, indent=2)[:400] + "\n  ...")
            return 0
        else:
            print(f"\n[FAIL] Generated mapping failed internal validation:")
            for err in val_errs:
                print(f"  - {err}")
            return 1
    else:
        print(f"\n[NOTICE] Automatic mapping could not be generated due to topology differences.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
