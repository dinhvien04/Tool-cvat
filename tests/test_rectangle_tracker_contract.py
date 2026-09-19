"""Contract tests for the experimental 9Router rectangle tracker."""

import importlib.util
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "serverless" / "ninerouter-rectangle-tracker" / "nuclio" / "model_handler.py"
SPEC = importlib.util.spec_from_file_location("rectangle_tracker_model_handler", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
ModelHandler = MODULE.ModelHandler


def test_tracker_function_metadata_is_rectangle_tracker():
    path = ROOT / "serverless" / "ninerouter-rectangle-tracker" / "nuclio" / "function.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    annotations = data["metadata"]["annotations"]
    assert annotations["type"] == "tracker"
    assert annotations["supported_shape_types"] == "rectangle"
    assert annotations["spec"] == "[]"


def test_init_tracking_is_local_and_preserves_rectangle(monkeypatch):
    handler = ModelHandler()
    monkeypatch.setattr(handler, "_call_router", lambda *_: (_ for _ in ()).throw(AssertionError("no remote call on init")))
    image = Image.new("RGB", (640, 480), "white")
    shape, state = handler.infer(
        image=image,
        shape={"type": "rectangle", "points": [100, 120, 220, 260]},
        state=None,
    )
    assert shape["type"] == "rectangle"
    assert shape["points"] == [100.0, 120.0, 220.0, 260.0]
    assert state["box"] == shape["points"]
    assert state["template"].startswith("data:image/jpeg;base64,")


def test_next_frame_maps_normalized_box_from_search_roi(monkeypatch):
    handler = ModelHandler()
    image = Image.new("RGB", (640, 480), "white")
    _, state = handler.infer(
        image=image,
        shape={"type": "rectangle", "points": [200, 150, 300, 250]},
        state=None,
    )

    monkeypatch.setattr(
        handler,
        "_call_router",
        lambda _template, _search: ([350, 350, 650, 650], 0.95, True),
    )
    shape, next_state = handler.infer(image=image, shape=None, state=state)
    assert shape["type"] == "rectangle"
    x1, y1, x2, y2 = shape["points"]
    assert x2 > x1 and y2 > y1
    assert next_state["failures"] == 0
    assert next_state["confidence"] == 0.95


def test_failed_remote_track_keeps_previous_box_and_expands_next_search(monkeypatch):
    handler = ModelHandler()
    image = Image.new("RGB", (640, 480), "white")
    shape0, state = handler.infer(
        image=image,
        shape={"type": "rectangle", "points": [100, 100, 180, 180]},
        state=None,
    )
    monkeypatch.setattr(handler, "_call_router", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    shape1, state1 = handler.infer(image=image, shape=None, state=state)
    assert shape1["points"] == shape0["points"]
    assert state1["failures"] == 1
    roi0 = handler._search_box(state["box"], image.width, image.height, 0)
    roi1 = handler._search_box(state1["box"], image.width, image.height, state1["failures"])
    assert (roi1[2] - roi1[0]) >= (roi0[2] - roi0[0])
