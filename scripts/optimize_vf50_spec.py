"""Optimize and compact function.yaml for ninerouter-face-vf50 to fit within OS limits."""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import yaml
from core.pose_face_schema import VF50_LANDMARKS, VF50_EDGES, validate_svg_node_ids
from scripts.validate_function_spec import validate_function_yaml
fn_path = repo_root / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml"

# Build concise SVG
svg_parts = ['<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">']
for idx, name in enumerate(VF50_LANDMARKS):
    svg_parts.append(
        f'<circle id="node_{idx}" data-type="element node" data-element-id="{idx}" '
        f'data-node-id="{idx}" data-label-name="{name}" cx="50" cy="50" r="3"/>'
    )
for u, v in VF50_EDGES:
    svg_parts.append(
        f'<line id="edge_{u}_{v}" data-type="edge" data-node-from="{u}" data-node-to="{v}" stroke="red" stroke-width="1"/>'
    )
svg_parts.append('</svg>')
svg_str = "".join(svg_parts)

is_valid, errs = validate_svg_node_ids(svg_str, expected_node_ids=set(range(50)))
if not is_valid:
    raise ValueError(f"SVG invalid: {errs}")

sublabels = [
    {"id": idx, "name": name, "type": "points", "attributes": []}
    for idx, name in enumerate(VF50_LANDMARKS)
]

spec_obj = [
    {
        "id": 1,
        "name": "face",
        "type": "skeleton",
        "attributes": [],
        "sublabels": sublabels,
        "svg": svg_str,
    }
]

spec_json = json.dumps(spec_obj, separators=(",", ":"))
print(f"Compact spec JSON length: {len(spec_json)} characters")

content = yaml.safe_load(fn_path.read_text(encoding="utf-8"))
content["metadata"]["annotations"]["spec"] = spec_json

# Format as YAML
output_yaml = yaml.dump(content, sort_keys=False, width=10000)
print(f"Total function.yaml length: {len(output_yaml)} characters (down from 25,337)")

fn_path.write_text(output_yaml, encoding="utf-8")

# Validate
is_valid_yaml, yaml_errs = validate_function_yaml(fn_path)
print(f"validate_function_yaml: valid={is_valid_yaml}, errors={yaml_errs}")
assert is_valid_yaml, f"Validation failed: {yaml_errs}"
