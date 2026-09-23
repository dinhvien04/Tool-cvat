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
import re
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

POSE17_KEYPOINT_SPECS: List[Tuple[int, str, List[str]]] = [
    (1, "nose", ["nose", "nose_point", "head"]),
    (2, "right_eye", ["right_eye", "r_eye", "righteye", "reye", "eye_r", "eye_right"]),
    (3, "left_eye", ["left_eye", "l_eye", "lefteye", "leye", "eye_l", "eye_left"]),
    (4, "right_ear", ["right_ear", "r_ear", "rightear", "rear", "ear_r", "ear_right"]),
    (5, "left_ear", ["left_ear", "l_ear", "leftear", "lear", "ear_l", "ear_left"]),
    (6, "right_shoulder", ["right_shoulder", "r_shoulder", "rightshoulder", "rshoulder", "shoulder_r", "shoulder_right"]),
    (7, "left_shoulder", ["left_shoulder", "l_shoulder", "leftshoulder", "lshoulder", "shoulder_l", "shoulder_left"]),
    (8, "right_elbow", ["right_elbow", "r_elbow", "rightelbow", "relbow", "elbow_r", "elbow_right"]),
    (9, "left_elbow", ["left_elbow", "l_elbow", "leftelbow", "lelbow", "elbow_l", "elbow_left"]),
    (10, "right_wrist", ["right_wrist", "r_wrist", "rightwrist", "rwrist", "wrist_r", "wrist_right"]),
    (11, "left_wrist", ["left_wrist", "l_wrist", "leftwrist", "lwrist", "wrist_l", "wrist_left"]),
    (12, "right_hip", ["right_hip", "r_hip", "righthip", "rhip", "hip_r", "hip_right"]),
    (13, "left_hip", ["left_hip", "l_hip", "lefthip", "lhip", "hip_l", "hip_left"]),
    (14, "right_knee", ["right_knee", "r_knee", "rightknee", "rknee", "knee_r", "knee_right"]),
    (15, "left_knee", ["left_knee", "l_knee", "leftknee", "lknee", "knee_l", "knee_left"]),
    (16, "right_ankle", ["right_ankle", "r_ankle", "rightankle", "rankle", "ankle_r", "ankle_right"]),
    (17, "left_ankle", ["left_ankle", "l_ankle", "leftankle", "lankle", "ankle_l", "ankle_left"]),
]


def normalize_sublabel_name(val: str) -> str:
    """Normalize sublabel string: lowercase, hyphens/spaces to underscores, normalize prefixes."""
    s = str(val).strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace(".", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s.startswith("r_"):
        s = "right_" + s[2:]
    elif s.startswith("l_"):
        s = "left_" + s[2:]
    return s


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
    """Simulate CVAT 2.75.1 backend validation logic (lambda_manager/views.py:446-450)
    and enforce Week-2 skeleton completeness contracts.

    In CVAT 2.75.1:
    if md_label["type"] == "skeleton" and db_label.type == "skeleton":
        if "sublabels" not in mapping_item:
            raise ValidationError(
                f'Mapping for elements was not specified in skeleton "{model_label_name}" '
            )

    Week-2 Completeness Contracts:
    - Pose17: strictly 17/17 sublabels required.
    - VF50: strictly 7/7 component skeletons and 50/50 sublabels required.
    - Monolithic 'face' skeleton rejected as INCOMPATIBLE_FULL_VF50.
    """
    errors: List[str] = []
    task_label_map = {l["name"]: l for l in task_labels}
    model_label_map = {m["name"]: m for m in model_spec}

    # Rejection of legacy monolithic face for VF50
    if any(m.lower() == "face" for m in mapping) and not any(m["name"] == "face" for m in model_spec):
        errors.append(
            "[INCOMPATIBLE_FULL_VF50] Monolithic face skeleton mapping is incompatible with 7-component VF-50 model specification."
        )

    # Completeness check: all model skeletons must be present in mapping
    missing_model_labels = [m["name"] for m in model_spec if m["name"] not in mapping]
    if missing_model_labels:
        if any(m["name"] == "person" for m in model_spec):
            errors.append(
                f"[INCOMPLETE_POSE_MAPPING] Required model skeleton 'person' missing from mapping"
            )
        else:
            errors.append(
                f"[INCOMPLETE_VF50_MAPPING] Missing {len(missing_model_labels)}/7 VF-50 component skeleton(s): {missing_model_labels}"
            )

    for model_name, mapping_item in mapping.items():
        if model_name not in model_label_map:
            if model_name.lower() != "face":
                errors.append(f"Model label {model_name!r} not found in model specification")
            continue

        if not isinstance(mapping_item, dict):
            errors.append(
                f'Mapping item for model label "{model_name}" must be a dictionary, '
                f'got {type(mapping_item).__name__}'
            )
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

        # Verify label type compatibility matching CVAT lambda_manager labels_compatible
        md_type = md_label.get("type", "any")
        db_type = db_label.get("type", "any")
        if md_type != "any" and db_type != "any" and md_type != db_type:
            errors.append(
                f'[INCOMPATIBLE_TYPE] Model label "{model_name}" (type: {md_type}) '
                f'and task label "{target_task_name}" (type: {db_type}) are not compatible'
            )
            continue

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

            # Enforce 17/17 for Pose17 and exact component sublabels for VF50
            if len(sub_mapping) < len(md_sub_names):
                missing_subs = sorted(list(md_sub_names - set(sub_mapping.keys())))
                if model_name == "person":
                    errors.append(
                        f"[INCOMPLETE_POSE_MAPPING] Skeleton '{model_name}' has only "
                        f"{len(sub_mapping)}/{len(md_sub_names)} sublabels mapped. Missing: {missing_subs}"
                    )
                else:
                    errors.append(
                        f"[INCOMPLETE_VF50_MAPPING] Skeleton '{model_name}' has only "
                        f"{len(sub_mapping)}/{len(md_sub_names)} sublabels mapped. Missing: {missing_subs}"
                    )

            target_subs_seen: Dict[str, str] = {}
            for md_sub_name, sub_map_item in sub_mapping.items():
                if not isinstance(sub_map_item, dict):
                    errors.append(
                        f'Sublabel mapping for "{md_sub_name}" in skeleton "{model_name}" '
                        f'must be a dictionary, got {type(sub_map_item).__name__}'
                    )
                    continue

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
                if target_sub in target_subs_seen:
                    errors.append(
                        f'Duplicate mapping to target sublabel "{target_sub}" in skeleton "{model_name}": '
                        f'both "{target_subs_seen[target_sub]}" and "{md_sub_name}" map to it'
                    )
                target_subs_seen[target_sub] = md_sub_name

    return len(errors) == 0, errors


def build_compatible_mapping_payload(
    fn_name: str,
    task_labels: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Build a compliant mapping payload for CVAT 2.75.1 with sublabels guaranteed.

    Enforces:
    1. Pose17:
       - Supports Case A: semantic sublabels ("nose", "right_eye", ..., "right_ankle").
       - Supports Case B: numeric sublabels ("1", "2", ..., "17").
       - Supports Case C: uppercase/normalized aliases ("NOSE", "RIGHT_EYE", etc.).
       - Strictly requires 17/17 matched sublabels; 16/17 or less rejected as INCOMPLETE_POSE_MAPPING.
    2. VF50:
       - Strictly requires 7/7 component skeletons AND 50/50 matched sublabels; 6/7 or 49/50 rejected as INCOMPLETE_VF50_MAPPING.
       - Legacy monolithic "face" skeleton cleanly diagnosed and rejected as INCOMPATIBLE_FULL_VF50.
    """
    model_spec = get_model_spec(fn_name)
    mapping: Dict[str, Any] = {}
    diagnostics: List[str] = []

    task_label_map = {l["name"]: l for l in task_labels}

    if fn_name in ("ninerouter-human-pose-17", "pose17", "pose"):
        # Model has 1 skeleton 'person' with 17 sublabels
        target_name = None
        for l in task_labels:
            if l.get("type") == "skeleton":
                lname = l["name"].strip()
                if lname.lower() in ("person", "body"):
                    target_name = lname
                    break

        if not target_name:
            diagnostics.append(
                "[INCOMPATIBLE_TOPOLOGY] Task does not contain a compatible 'person' or 'body' skeleton label."
            )
            return None, diagnostics

        db_label = task_label_map[target_name]
        task_subs = db_label.get("sublabels", [])
        task_sub_names = [str(s["name"]).strip() for s in task_subs]
        task_sub_set = set(task_sub_names)

        sublabels_mapping: Dict[str, Any] = {}
        used_task_subs: Set[str] = set()

        # Check if task is Case B: strictly numeric sublabels ("1".."17")
        numeric_17 = {str(i) for i in range(1, 18)}
        if numeric_17.issubset(task_sub_set):
            for idx, coco_name, _ in POSE17_KEYPOINT_SPECS:
                sublabels_mapping[coco_name] = {"name": str(idx), "attributes": {}}
            detected_case = "Case B: numeric sublabels ('1'..'17')"
        else:
            # Build normalized lookup map: normalized_name -> actual_task_name
            norm_to_actual: Dict[str, str] = {}
            for name in task_sub_names:
                norm_key = normalize_sublabel_name(name)
                if norm_key not in norm_to_actual:
                    norm_to_actual[norm_key] = name

            is_exact_coco = True
            is_uppercase_or_alias = False

            for idx, coco_name, aliases in POSE17_KEYPOINT_SPECS:
                matched_target: Optional[str] = None

                # Priority 1: Exact lowercase match with canonical coco_name
                if coco_name in task_sub_set and coco_name not in used_task_subs:
                    matched_target = coco_name

                # Priority 2: Normalized match against aliases (handles NOSE, RIGHT_EYE, R Eye, etc.)
                if matched_target is None:
                    for a in aliases:
                        norm_a = normalize_sublabel_name(a)
                        if norm_a in norm_to_actual:
                            candidate = norm_to_actual[norm_a]
                            if candidate not in used_task_subs:
                                matched_target = candidate
                                is_exact_coco = False
                                is_uppercase_or_alias = True
                                break

                # Priority 3: Numeric fallback if numeric string present
                if matched_target is None and str(idx) in task_sub_set and str(idx) not in used_task_subs:
                    matched_target = str(idx)
                    is_exact_coco = False

                if matched_target is not None:
                    sublabels_mapping[coco_name] = {"name": matched_target, "attributes": {}}
                    used_task_subs.add(matched_target)

            if len(sublabels_mapping) == 17:
                if all(item["name"].isdigit() for item in sublabels_mapping.values()):
                    detected_case = "Case B: numeric sublabels ('1'..'17')"
                elif is_exact_coco and not is_uppercase_or_alias:
                    detected_case = "Case A: semantic sublabels ('nose', 'right_eye', ..., 'right_ankle')"
                else:
                    detected_case = "Case C: uppercase/normalized aliases"
            else:
                detected_case = "unresolved"

        matched_count = len(sublabels_mapping)
        if matched_count < 17:
            missing_kps = [
                coco_name for _, coco_name, _ in POSE17_KEYPOINT_SPECS
                if coco_name not in sublabels_mapping
            ]
            diagnostics.append(
                f"[INCOMPLETE_POSE_MAPPING] Pose17 requires strictly 17/17 matched sublabels. "
                f"Only {matched_count}/17 matched in task skeleton '{target_name}'. "
                f"Missing or unmatched keypoints ({len(missing_kps)}): {missing_kps}. "
                f"Task sublabels found ({len(task_sub_names)}): {task_sub_names[:10]}"
                + ("..." if len(task_sub_names) > 10 else "")
            )
            return None, diagnostics

        mapping["person"] = {
            "name": target_name,
            "attributes": {},
            "sublabels": sublabels_mapping,
        }
        diagnostics.append(
            f"Successfully mapped 'person' -> '{target_name}' with 17/17 sublabels ({detected_case})."
        )

    elif fn_name in ("ninerouter-face-vf50", "vf50", "face"):
        # 1. Cleanly diagnose and reject legacy monolithic 'face' skeleton (5-point or 50-point)
        face_label = None
        for l in task_labels:
            if l.get("type") == "skeleton" and l["name"].strip().lower() == "face":
                face_label = l
                break

        if face_label is not None:
            face_subs = [s["name"] for s in face_label.get("sublabels", [])]
            face_subs_preview = face_subs[:10]
            if len(face_subs) > 10:
                face_subs_preview.append(f"... ({len(face_subs)} total)")
            diagnostics.append(
                f"[INCOMPATIBLE_FULL_VF50] Task defines a monolithic skeleton '{face_label['name']}' with "
                f"{len(face_subs)} sublabels {face_subs_preview}. The full VF-50 model defines 7 independent "
                f"component skeletons ({', '.join(VF50_COMPONENT_NAMES)}) with 50 total points. "
                f"A monolithic face task ({len(face_subs)}-point) is NOT equivalent to full VF-50 component topology. "
                f"In CVAT 2.75.1, skeleton mapping cannot merge multiple model skeletons "
                f"into a single task skeleton. Attempting to map 'face' without sublabels "
                f"WILL trigger: Detection error occurred: Mapping for elements was not specified in skeleton 'face'."
            )
            if len(face_subs) <= 5:
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
                    "Recommended action: Reconfigure the task with the 7 canonical VinFast component skeletons: "
                    f"{', '.join(VF50_COMPONENT_NAMES)}."
                )
            return None, diagnostics

        # 2. Match the 7 component skeletons
        task_skeletons_by_lower = {
            l["name"].strip().lower(): l
            for l in task_labels
            if l.get("type") == "skeleton"
        }

        missing_components: List[str] = []
        component_errors: List[str] = []

        for cname in VF50_COMPONENT_NAMES:
            if cname.lower() not in task_skeletons_by_lower:
                missing_components.append(cname)
                continue

            db_label = task_skeletons_by_lower[cname.lower()]
            target_comp_name = db_label["name"]
            task_subs = db_label.get("sublabels", [])
            task_sub_names = [str(s["name"]).strip() for s in task_subs]
            task_sub_set = set(task_sub_names)

            cfg = VF50_COMPONENT_CONFIG[cname]
            expected_ids = [str(i) for i in range(cfg["start"], cfg["end"] + 1)]

            sub_map: Dict[str, Any] = {}
            for s_id in expected_ids:
                matched_target = None
                if s_id in task_sub_set:
                    matched_target = s_id
                else:
                    prefix_2d = f"{cname}_{int(s_id):02d}"
                    prefix_1d = f"{cname}_{s_id}"
                    if prefix_2d in task_sub_set:
                        matched_target = prefix_2d
                    elif prefix_1d in task_sub_set:
                        matched_target = prefix_1d
                    else:
                        for ts in task_sub_names:
                            if ts.endswith(f"_{s_id}") or ts.endswith(f"_{int(s_id):02d}"):
                                matched_target = ts
                                break

                if matched_target is not None:
                    sub_map[s_id] = {"name": matched_target, "attributes": {}}

            if len(sub_map) < len(expected_ids):
                missing_ids = [i for i in expected_ids if i not in sub_map]
                component_errors.append(
                    f"Component '{cname}' incomplete: {len(sub_map)}/{len(expected_ids)} matched sublabels. Missing: {missing_ids}"
                )

            mapping[cname] = {
                "name": target_comp_name,
                "attributes": {},
                "sublabels": sub_map,
            }

        total_matched_subs = sum(len(m["sublabels"]) for m in mapping.values())

        if missing_components or component_errors or len(mapping) < 7 or total_matched_subs < 50:
            diagnostics.append(
                f"[INCOMPLETE_VF50_MAPPING] VF-50 requires strictly 7/7 component skeletons and 50/50 matched sublabels. "
                f"Found {len(mapping)}/7 component skeletons and {total_matched_subs}/50 matched sublabels."
            )
            if missing_components:
                diagnostics.append(
                    f"Missing component skeletons ({len(missing_components)}/7): {missing_components}"
                )
            for err in component_errors:
                diagnostics.append(f"  - {err}")
            return None, diagnostics

        diagnostics.append(
            "Successfully mapped all 7/7 VF-50 component skeletons (50/50 sublabels) to task."
        )

    return {"mapping": mapping}, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CVAT 2.75.1 Skeleton Mapping Preflight Validator for Week-2 Detectors"
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--task-id", type=int, help="CVAT Task ID to inspect")
    target_group.add_argument(
        "--labels-file",
        type=Path,
        help="Path to JSON file containing task labels (offline / testing mode)",
    )
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

    target_desc = f"Task {args.task_id}" if args.task_id is not None else f"Labels file {args.labels_file}"

    print("=" * 70)
    print(f"CVAT 2.75.1 Skeleton Mapping Preflight: {target_desc} vs {fn_canonical}")
    print("=" * 70)

    if args.task_id is not None:
        task_labels = get_task_labels_from_cvat(args.task_id)
        if not task_labels:
            print(f"[FAIL] Could not retrieve labels for Task {args.task_id} from CVAT.", file=sys.stderr)
            return 1
    else:
        if not args.labels_file.exists():
            print(f"[FAIL] Labels file not found: {args.labels_file}", file=sys.stderr)
            return 1
        with open(args.labels_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict) and "labels" in content:
                task_labels = content["labels"]
            elif isinstance(content, list):
                task_labels = content
            else:
                print(f"[FAIL] Unrecognized format in {args.labels_file}. Expected list or dict with 'labels' key.", file=sys.stderr)
                return 1

    print(f"{target_desc} labels found:")
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
