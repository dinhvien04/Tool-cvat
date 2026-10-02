"""Polygon-only and mask-only detectors keep a single shape type."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_function_spec import validate_function_yaml  # noqa: E402


def _load(unique_name: str, directory: Path):
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(unique_name, directory / "model_handler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    while str(directory) in sys.path:
        sys.path.remove(str(directory))
    return module


_polygon = _load("polygon_only_handler", ROOT / "serverless" / "ninerouter-polygon" / "nuclio")
_mask = _load("mask_only_handler", ROOT / "serverless" / "ninerouter-mask" / "nuclio")


def test_split_function_specs():
    polygon_yaml = ROOT / "serverless" / "ninerouter-polygon" / "nuclio" / "function.yaml"
    mask_yaml = ROOT / "serverless" / "ninerouter-mask" / "nuclio" / "function.yaml"
    ok_poly, poly_errors = validate_function_yaml(polygon_yaml)
    ok_mask, mask_errors = validate_function_yaml(mask_yaml)
    assert ok_poly, poly_errors
    assert ok_mask, mask_errors
    poly_spec = json.loads(yaml.safe_load(polygon_yaml.read_text(encoding="utf-8"))["metadata"]["annotations"]["spec"])
    mask_spec = json.loads(yaml.safe_load(mask_yaml.read_text(encoding="utf-8"))["metadata"]["annotations"]["spec"])
    assert {item["type"] for item in poly_spec} == {"polygon"}
    assert {item["type"] for item in mask_spec} == {"mask"}


def test_filters_drop_the_paired_shape():
    polygons = _polygon.polygon_shapes_only(
        [
            {"type": "polygon", "label": "road", "points": [0, 0, 10, 0, 10, 10], "group_id": 1},
            {"type": "mask", "label": "road", "mask": [1, 0, 0, 10, 10]},
            {"type": "polygon", "label": "car", "points": [0, 0, 1, 0, 1, 1]},
        ],
        ["road"],
    )
    assert polygons == [{"type": "polygon", "label": "road", "points": [0, 0, 10, 0, 10, 10]}]
    masks = _mask.mask_shapes_only(
        [
            {"type": "mask", "label": "car", "mask": [1, 0, 0, 4, 4]},
            {"type": "rectangle", "label": "car", "points": [0, 0, 4, 4]},
        ],
        ["car"],
    )
    assert masks == [{"label": "car", "type": "mask", "mask": [1, 0, 0, 4, 4]}]
