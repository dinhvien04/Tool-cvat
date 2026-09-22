"""Optimize and compact function.yaml for ninerouter-face-vf50 to fit within OS limits."""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import yaml
from core.skeleton_contract import build_cvat_vf50_spec
from scripts.validate_function_spec import validate_function_yaml

fn_path = repo_root / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml"

spec_obj = build_cvat_vf50_spec()
spec_json = json.dumps(spec_obj, separators=(",", ":"))
print(f"Compact 7-component spec JSON length: {len(spec_json)} characters")

content = yaml.safe_load(fn_path.read_text(encoding="utf-8"))
content["metadata"]["annotations"]["spec"] = spec_json

# Format as YAML
output_yaml = yaml.dump(content, sort_keys=False, width=100000)
print(f"Total function.yaml length: {len(output_yaml)} characters")

fn_path.write_text(output_yaml, encoding="utf-8")

# Validate
is_valid_yaml, yaml_errs = validate_function_yaml(fn_path)
print(f"validate_function_yaml: valid={is_valid_yaml}, errors={yaml_errs}")
assert is_valid_yaml, f"Validation failed: {yaml_errs}"
