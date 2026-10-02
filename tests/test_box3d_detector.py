"""Contract tests for the 9Router Box 3D cuboid detector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BOX3D_DIR = ROOT / "serverless" / "ninerouter-box-3d" / "nuclio"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_function_spec import validate_function_yaml  # noqa: E402


def _load(unique_name: str, filename: str):
    sys.path.insert(0, str(BOX3D_DIR))
    spec = importlib.util.spec_from_file_location(unique_name, BOX3D_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    while str(BOX3D_DIR) in sys.path:
        sys.path.remove(str(BOX3D_DIR))
    return module


_geometry = _load("box3d_cuboid_geometry", "cuboid_geometry.py")
sys.modules["cuboid_geometry"] = _geometry
_handler = _load("box3d_model_handler", "model_handler.py")
cuboid_from_front_face = _geometry.cuboid_from_front_face
BOX3D_LABELS = _handler.BOX3D_LABELS
cuboid_shapes_only = _handler.cuboid_shapes_only
objects_to_cuboids = _handler.objects_to_cuboids


CVAT_DRAG_CUBOID = [
    100, 80.1, 100, 200, 260, 80, 260, 200,
    288.8, 58.5, 288.8, 178.3, 128.8, 58.5, 128.8, 178.3,
]


def test_box3d_function_yaml_uses_screenshot_labels():
    yaml_path = BOX3D_DIR / "function.yaml"
    is_valid, errors = validate_function_yaml(yaml_path)
    assert is_valid, errors
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["metadata"]["annotations"]["name"] == "9Router Box 3D"
    spec = json.loads(data["metadata"]["annotations"]["spec"])
    assert [item["name"] for item in spec] == list(BOX3D_LABELS)
    assert {item["type"] for item in spec} == {"cuboid"}


def test_front_face_matches_cvat_cuboid_point_order():
    points = cuboid_from_front_face(100, 80, 260, 200, 0.18, "right")
    assert len(points) == 16
    assert [round(value, 1) for value in points] == CVAT_DRAG_CUBOID


def test_objects_become_cuboids_and_drop_unknown_labels():
    shapes = objects_to_cuboids(
        [
            {"label": "car", "box_2d": [100, 200, 400, 700], "depth_ratio": 0.2, "side": "right", "confidence": 0.91},
            {"label": "sky", "box_2d": [0, 0, 100, 100]},
            {"label": "bus", "box_2d": [10, 10, 20]},
        ],
        BOX3D_LABELS,
        width=1000,
        height=1000,
    )
    assert len(shapes) == 1
    assert shapes[0]["type"] == "cuboid"
    assert shapes[0]["label"] == "car"
    assert len(shapes[0]["points"]) == 16
    assert cuboid_shapes_only(shapes + [{"type": "rectangle", "label": "car", "points": [1, 2, 3, 4]}], BOX3D_LABELS) == shapes
