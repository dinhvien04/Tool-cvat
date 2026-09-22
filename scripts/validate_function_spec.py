"""Pre-deployment validator for Tool-cvat Nuclio function.yaml metadata.

Prevents malformed functions from poisoning CVAT's GET /api/lambda/functions endpoint.
Validates:
1. metadata.name, metadata.annotations exist
2. metadata.annotations.name, metadata.annotations.type (FunctionKind: detector, etc.)
3. metadata.annotations.spec is valid JSON
4. Label IDs and names are unique and non-empty
5. Label shape types are supported by CVAT
6. Full 31-label detector preserves exact 31 labels without missing or merged classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.feedback import ALL_31_LABELS
from core.taxonomy import BOX_MASK_LABELS, POLYGON_MASK_LABELS, POLYLINE_LABELS
from core.pose_face_schema import POSE17_KEYPOINTS, VF50_LANDMARKS, validate_svg_node_ids
from core.skeleton_contract import VF50_COMPONENT_NAMES, VF50_COMPONENT_CONFIG

VALID_FUNCTION_KINDS = {"detector", "interactor", "tracker", "reid"}
VALID_CVAT_LABEL_TYPES = {"rectangle", "polygon", "polyline", "points", "mask", "skeleton", "any"}


def validate_function_yaml(yaml_path: Path) -> Tuple[bool, List[str]]:
    """Validate a Nuclio function.yaml for CVAT Lambda Manager compatibility.

    Args:
        yaml_path: Path to function.yaml.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []
    if not yaml_path.exists():
        return False, [f"File does not exist: {yaml_path}"]

    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"YAML parse error in {yaml_path}: {e}"]

    if not isinstance(content, dict):
        return False, [f"YAML root must be a dictionary in {yaml_path}"]

    # 1. Check metadata
    metadata = content.get("metadata")
    if not isinstance(metadata, dict):
        return False, [f"Missing 'metadata' dictionary in {yaml_path}"]

    fn_name = metadata.get("name")
    if not fn_name or not isinstance(fn_name, str):
        errors.append(f"Missing or invalid 'metadata.name' in {yaml_path}")

    # 2. Check metadata.annotations
    annotations = metadata.get("annotations")
    if not isinstance(annotations, dict):
        return False, [f"Missing 'metadata.annotations' dictionary in {yaml_path} - this poisons CVAT Lambda Manager!"]

    display_name = annotations.get("name")
    if not display_name or not isinstance(display_name, str):
        errors.append(f"Missing or invalid 'metadata.annotations.name' in {yaml_path}")

    fn_kind = annotations.get("type")
    if fn_kind not in VALID_FUNCTION_KINDS:
        errors.append(
            f"Invalid function kind 'metadata.annotations.type': {fn_kind!r} in {yaml_path}. "
            f"Must be one of {sorted(VALID_FUNCTION_KINDS)}"
        )

    # 3. Check metadata.annotations.spec
    spec_raw = annotations.get("spec")
    if spec_raw is None:
        return False, [f"Missing 'metadata.annotations.spec' in {yaml_path}"]

    try:
        spec_items = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in 'metadata.annotations.spec' in {yaml_path}: {e}"]

    if not isinstance(spec_items, list):
        return False, [f"'metadata.annotations.spec' must be a JSON array in {yaml_path}"]

    # 4. Check label entries
    seen_ids: Set[Any] = set()
    seen_names: Set[str] = set()
    label_names: List[str] = []

    for idx, item in enumerate(spec_items):
        if not isinstance(item, dict):
            errors.append(f"Spec item #{idx} is not an object in {yaml_path}")
            continue

        label_id = item.get("id")
        if label_id is None:
            errors.append(f"Spec item #{idx} missing 'id' in {yaml_path}")
        elif label_id in seen_ids:
            errors.append(f"Duplicate label id {label_id!r} at item #{idx} in {yaml_path}")
        else:
            seen_ids.add(label_id)

        label_name = item.get("name")
        if not label_name or not isinstance(label_name, str):
            errors.append(f"Spec item #{idx} missing or invalid 'name' in {yaml_path}")
        elif label_name in seen_names:
            errors.append(f"Duplicate label name {label_name!r} at item #{idx} in {yaml_path}")
        else:
            seen_names.add(label_name)
            label_names.append(label_name)

        label_type = item.get("type", "any")
        if label_type not in VALID_CVAT_LABEL_TYPES:
            errors.append(
                f"Unsupported label type {label_type!r} for label {label_name!r} in {yaml_path}. "
                f"Must be one of {sorted(VALID_CVAT_LABEL_TYPES)}"
            )

    # 5. Policy-specific detector validations
    if "ninerouter-rectangle-mask" in str(yaml_path) or fn_name == "ninerouter-rectangle-mask":
        if len(label_names) != 14:
            errors.append(
                f"ninerouter-rectangle-mask must have exactly 14 labels, but found {len(label_names)} in {yaml_path}"
            )
        missing_labels = set(BOX_MASK_LABELS) - set(label_names)
        if missing_labels:
            errors.append(f"ninerouter-rectangle-mask is missing expected labels: {sorted(missing_labels)}")
        extra_labels = set(label_names) - set(BOX_MASK_LABELS)
        if extra_labels:
            errors.append(f"ninerouter-rectangle-mask has unexpected extra labels: {sorted(extra_labels)}")
        for item in spec_items:
            if isinstance(item, dict) and item.get("type") != "any":
                errors.append(
                    f"ninerouter-rectangle-mask label '{item.get('name')}' must have type 'any' for dual-shape compatibility, got '{item.get('type')}'"
                )

    elif "ninerouter-polygon-mask" in str(yaml_path) or fn_name == "ninerouter-polygon-mask":
        if len(label_names) != 10:
            errors.append(
                f"ninerouter-polygon-mask must have exactly 10 labels, but found {len(label_names)} in {yaml_path}"
            )
        missing_labels = set(POLYGON_MASK_LABELS) - set(label_names)
        if missing_labels:
            errors.append(f"ninerouter-polygon-mask is missing expected labels: {sorted(missing_labels)}")
        extra_labels = set(label_names) - set(POLYGON_MASK_LABELS)
        if extra_labels:
            errors.append(f"ninerouter-polygon-mask has unexpected extra labels: {sorted(extra_labels)}")
        for item in spec_items:
            if isinstance(item, dict) and item.get("type") != "any":
                errors.append(
                    f"ninerouter-polygon-mask label '{item.get('name')}' must have type 'any' for dual-shape compatibility, got '{item.get('type')}'"
                )

    elif "ninerouter-polyline" in str(yaml_path) or fn_name == "ninerouter-polyline":
        if len(label_names) != 7:
            errors.append(
                f"ninerouter-polyline must have exactly 7 labels, but found {len(label_names)} in {yaml_path}"
            )
        missing_labels = set(POLYLINE_LABELS) - set(label_names)
        if missing_labels:
            errors.append(f"ninerouter-polyline is missing expected labels: {sorted(missing_labels)}")
        extra_labels = set(label_names) - set(POLYLINE_LABELS)
        if extra_labels:
            errors.append(f"ninerouter-polyline has unexpected extra labels: {sorted(extra_labels)}")
        for item in spec_items:
            if isinstance(item, dict) and item.get("type") != "polyline":
                errors.append(
                    f"ninerouter-polyline label '{item.get('name')}' must have type 'polyline', got '{item.get('type')}'"
                )

    elif "ninerouter-human-pose-17" in str(yaml_path) or fn_name == "ninerouter-human-pose-17":
        if len(label_names) != 1:
            errors.append(
                f"ninerouter-human-pose-17 must have exactly 1 parent label, but found {len(label_names)} in {yaml_path}"
            )
        elif label_names[0] not in ("person", "body"):
            errors.append(
                f"ninerouter-human-pose-17 parent label must be 'person', got {label_names[0]!r} in {yaml_path}"
            )
        for item in spec_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "skeleton":
                errors.append(
                    f"ninerouter-human-pose-17 parent label must have type 'skeleton', got {item.get('type')!r} in {yaml_path}"
                )
            sublabels = item.get("sublabels", [])
            if not isinstance(sublabels, list) or len(sublabels) != 17:
                errors.append(
                    f"ninerouter-human-pose-17 must have exactly 17 sublabels, got {len(sublabels) if isinstance(sublabels, list) else type(sublabels)} in {yaml_path}"
                )
            else:
                sub_names = [s.get("name") for s in sublabels if isinstance(s, dict)]
                missing_kps = set(POSE17_KEYPOINTS) - set(sub_names)
                if missing_kps:
                    errors.append(f"ninerouter-human-pose-17 missing keypoints: {sorted(missing_kps)}")
                for sub in sublabels:
                    if isinstance(sub, dict) and sub.get("type") != "points":
                        errors.append(
                            f"ninerouter-human-pose-17 sublabel '{sub.get('name')}' must have type 'points', got '{sub.get('type')}'"
                        )
            svg_content = item.get("svg", "")
            if not svg_content:
                errors.append(f"ninerouter-human-pose-17 missing 'svg' in skeleton spec in {yaml_path}")
            else:
                is_valid_svg, svg_errs = validate_svg_node_ids(svg_content, expected_node_ids=set(range(1, 18)))
                if not is_valid_svg:
                    errors.extend([f"Pose17 SVG error: {e}" for e in svg_errs])

    elif "ninerouter-face-vf50" in str(yaml_path) or fn_name == "ninerouter-face-vf50":
        if len(label_names) == 7:
            # Authoritative 7-component VinFast VF-50 architecture
            missing_comps = set(VF50_COMPONENT_NAMES) - set(label_names)
            if missing_comps:
                errors.append(f"ninerouter-face-vf50 missing component skeleton labels: {sorted(missing_comps)}")
            extra_comps = set(label_names) - set(VF50_COMPONENT_NAMES)
            if extra_comps:
                errors.append(f"ninerouter-face-vf50 unexpected extra labels: {sorted(extra_comps)}")

            for item in spec_items:
                if not isinstance(item, dict):
                    continue
                cname = item.get("name")
                if item.get("type") != "skeleton":
                    errors.append(
                        f"ninerouter-face-vf50 label '{cname}' must have type 'skeleton', got {item.get('type')!r} in {yaml_path}"
                    )
                cfg = VF50_COMPONENT_CONFIG.get(cname)
                if not cfg:
                    continue
                expected_count = cfg["end"] - cfg["start"] + 1
                sublabels = item.get("sublabels", [])
                if not isinstance(sublabels, list) or len(sublabels) != expected_count:
                    errors.append(
                        f"ninerouter-face-vf50 component '{cname}' must have {expected_count} sublabels, got {len(sublabels) if isinstance(sublabels, list) else type(sublabels)} in {yaml_path}"
                    )
                else:
                    for sub in sublabels:
                        if isinstance(sub, dict) and sub.get("type") != "points":
                            errors.append(
                                f"ninerouter-face-vf50 component '{cname}' sublabel '{sub.get('name')}' must have type 'points', got '{sub.get('type')}'"
                            )
                svg_content = item.get("svg", "")
                if not svg_content:
                    errors.append(f"ninerouter-face-vf50 missing 'svg' in component '{cname}' skeleton spec in {yaml_path}")
                else:
                    # Accept 1-indexed local node IDs (1..count) or 0-indexed (0..count-1)
                    is_valid_svg_1, svg_errs_1 = validate_svg_node_ids(svg_content, expected_node_ids=set(range(1, expected_count + 1)))
                    is_valid_svg_0, svg_errs_0 = validate_svg_node_ids(svg_content, expected_node_ids=set(range(0, expected_count)))
                    if not (is_valid_svg_1 or is_valid_svg_0):
                        errors.extend([f"VF50 SVG error in '{cname}': {e}" for e in svg_errs_1])

        elif len(label_names) == 1:
            # Legacy unified parent skeleton
            if label_names[0] not in ("face", "head"):
                errors.append(
                    f"ninerouter-face-vf50 parent label must be 'face', got {label_names[0]!r} in {yaml_path}"
                )
            for item in spec_items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "skeleton":
                    errors.append(
                        f"ninerouter-face-vf50 parent label must have type 'skeleton', got {item.get('type')!r} in {yaml_path}"
                    )
                sublabels = item.get("sublabels", [])
                if not isinstance(sublabels, list) or len(sublabels) != 50:
                    errors.append(
                        f"ninerouter-face-vf50 must have exactly 50 sublabels, got {len(sublabels) if isinstance(sublabels, list) else type(sublabels)} in {yaml_path}"
                    )
                else:
                    sub_names = [s.get("name") for s in sublabels if isinstance(s, dict)]
                    missing_lms = set(VF50_LANDMARKS) - set(sub_names)
                    if missing_lms:
                        errors.append(f"ninerouter-face-vf50 missing landmarks: {sorted(missing_lms)}")
                    for sub in sublabels:
                        if isinstance(sub, dict) and sub.get("type") != "points":
                            errors.append(
                                f"ninerouter-face-vf50 sublabel '{sub.get('name')}' must have type 'points', got '{sub.get('type')}'"
                            )
                svg_content = item.get("svg", "")
                if not svg_content:
                    errors.append(f"ninerouter-face-vf50 missing 'svg' in skeleton spec in {yaml_path}")
                else:
                    is_valid_svg_0, svg_errs_0 = validate_svg_node_ids(svg_content, expected_node_ids=set(range(0, 50)))
                    is_valid_svg_1, svg_errs_1 = validate_svg_node_ids(svg_content, expected_node_ids=set(range(1, 51)))
                    if not (is_valid_svg_0 or is_valid_svg_1):
                        errors.extend([f"VF50 SVG error: {e}" for e in svg_errs_0])
        else:
            errors.append(
                f"ninerouter-face-vf50 must have either 7 component skeletons (authoritative) or 1 parent skeleton, but found {len(label_names)} in {yaml_path}"
            )

    elif "ninerouter-vision-31" in str(yaml_path) or fn_name == "ninerouter-vision-31":
        if len(label_names) != 31:
            errors.append(
                f"ninerouter-vision-31 must have exactly 31 labels, but found {len(label_names)} in {yaml_path}"
            )
        missing_labels = set(ALL_31_LABELS) - set(label_names)
        if missing_labels:
            errors.append(f"ninerouter-vision-31 is missing expected labels: {sorted(missing_labels)}")
        extra_labels = set(label_names) - set(ALL_31_LABELS)
        if extra_labels:
            errors.append(f"ninerouter-vision-31 has unexpected extra labels: {sorted(extra_labels)}")

    return len(errors) == 0, errors


def validate_all_detectors() -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all detector function.yaml files in serverless/."""
    targets = [
        REPO_ROOT / "serverless" / "ninerouter-rectangle-mask" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-polygon-mask" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-polyline" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-human-pose-17" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-vision" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-vision-mask" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-vision-box-mask" / "nuclio" / "function.yaml",
        REPO_ROOT / "serverless" / "ninerouter-vision-31" / "nuclio" / "function.yaml",
    ]
    results: Dict[str, Tuple[bool, List[str]]] = {}
    for target in targets:
        rel = str(target.relative_to(REPO_ROOT)).replace("\\", "/")
        results[rel] = validate_function_yaml(target)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Nuclio function.yaml metadata for CVAT compatibility")
    parser.add_argument("--yaml-path", type=Path, help="Path to specific function.yaml to validate")
    parser.add_argument("--all", action="store_true", help="Validate all repository function.yaml files")

    args = parser.parse_args()

    if args.yaml_path:
        is_valid, errors = validate_function_yaml(args.yaml_path)
        if is_valid:
            print(f"[PASS] {args.yaml_path} is valid for CVAT Lambda Manager.")
            return 0
        else:
            print(f"[FAIL] {args.yaml_path} has validation errors:")
            for err in errors:
                print(f"  - {err}")
            return 1

    # Default to validating all
    results = validate_all_detectors()
    all_ok = True
    print("=" * 60)
    print("Nuclio Function Specification Validator")
    print("=" * 60)
    for path, (is_valid, errors) in results.items():
        if is_valid:
            print(f"[PASS] {path}")
        else:
            all_ok = False
            print(f"[FAIL] {path}:")
            for err in errors:
                print(f"  - {err}")

    if all_ok:
        print("\nAll detector function.yaml specifications are VALID.")
        return 0
    else:
        print("\nOne or more detector function.yaml specifications FAILED validation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
