"""Contract tests for the rectangle-only 9Router Box detector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BOX_DIR = ROOT / "serverless" / "ninerouter-box" / "nuclio"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.taxonomy import BOX_MASK_LABELS  # noqa: E402
from scripts.validate_function_spec import validate_function_yaml  # noqa: E402

_spec = importlib.util.spec_from_file_location("box2d_model_handler", BOX_DIR / "model_handler.py")
_handler = importlib.util.module_from_spec(_spec)
sys.modules["box2d_model_handler"] = _handler
_spec.loader.exec_module(_handler)
rectangle_shapes_only = _handler.rectangle_shapes_only
while str(BOX_DIR) in sys.path:
    sys.path.remove(str(BOX_DIR))


def test_box_function_yaml_is_rectangle_only():
    yaml_path = BOX_DIR / "function.yaml"
    is_valid, errors = validate_function_yaml(yaml_path)
    assert is_valid, errors

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["metadata"]["name"] == "ninerouter-box"
    assert data["metadata"]["annotations"]["name"] == "9Router Box"
    env = {item["name"]: item["value"] for item in data["spec"]["env"]}
    assert env["DETECTION_MODE"] == "box"
    spec = yaml.safe_load(data["metadata"]["annotations"]["spec"])
    if isinstance(data["metadata"]["annotations"]["spec"], str):
        import json
        spec = json.loads(data["metadata"]["annotations"]["spec"])
    assert [item["name"] for item in spec] == list(BOX_MASK_LABELS)
    assert {item["type"] for item in spec} == {"rectangle"}


def test_rectangle_shapes_only_drops_masks_and_unknown_labels():
    shapes = [
        {"type": "rectangle", "label": "car", "points": [1, 2, 3, 4]},
        {"type": "mask", "label": "car", "points": [1, 2, 3, 4, 5, 6]},
        {"type": "polygon", "label": "road", "points": [0, 0, 1, 1, 2, 2]},
        {"type": "rectangle", "label": "lane/single white", "points": [1, 2, 3, 4]},
        {"type": "rectangle", "label": "person", "points": [1]},
    ]
    kept = rectangle_shapes_only(shapes, BOX_MASK_LABELS)
    assert kept == [{"type": "rectangle", "label": "car", "points": [1, 2, 3, 4]}]
