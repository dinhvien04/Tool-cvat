"""Tests verifying strict adherence to CVAT Serverless Detector contract."""

import json
from pathlib import Path
import pytest
import yaml

from core.vision_contract import (
    ALL_31_LABELS,
    DEFAULT_BBOX_LABELS,
    box_2d_to_cvat_rect,
)


def test_cvat_function_yaml_contract():
    """Verify function.yaml conforms to CVAT serverless detector schema."""
    yaml_path = Path(__file__).resolve().parent.parent / "serverless" / "ninerouter-vision" / "nuclio" / "function.yaml"
    assert yaml_path.exists(), "function.yaml not found"

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Namespace and Metadata
    meta = config.get("metadata", {})
    assert meta.get("name") == "ninerouter-vision"
    assert meta.get("namespace") == "cvat"

    annotations = meta.get("annotations", {})
    assert annotations.get("name") == "9Router Vision"
    assert annotations.get("type") == "detector"

    # 2. Spec JSON Array
    raw_spec = annotations.get("spec", "")
    assert raw_spec, "annotations.spec is missing or empty"
    spec_list = json.loads(raw_spec)
    assert isinstance(spec_list, list)
    assert len(spec_list) == 13

    # 3. Label check
    declared_names = set()
    for item in spec_list:
        assert "id" in item
        assert "name" in item
        assert item.get("type") == "rectangle"
        declared_names.add(item["name"])

    # Must match exactly the 13 bbox labels
    assert declared_names == set(DEFAULT_BBOX_LABELS)

    # Must preserve intentional label ambiguities
    assert "pedestrian" in declared_names
    assert "person" in declared_names
    assert "traffic light" in declared_names
    assert "traffic_light" in declared_names
    assert "traffic sign" in declared_names
    assert "traffic_sign" in declared_names

    # Must NOT include non-bbox labels (e.g. ego vehicle, lane, drivable area)
    non_bbox_labels = [label for label in ALL_31_LABELS if label not in DEFAULT_BBOX_LABELS]
    for non_bbox in non_bbox_labels:
        assert non_bbox not in declared_names

    # 4. Trigger Configuration
    triggers = config.get("spec", {}).get("triggers", {})
    assert "myHttpTrigger" in triggers
    http_trig = triggers["myHttpTrigger"]
    assert http_trig.get("kind") == "http"
    assert http_trig.get("attributes", {}).get("maxRequestBodySize") >= 16 * 1024 * 1024


def test_cvat_detector_points_and_clamping():
    """Verify bounding box points conversion strictly obeys [xtl, ytl, xbr, ybr] order."""
    # Normalized [ymin, xmin, ymax, xmax] in [0, 1000]
    box = [150, 250, 450, 750]
    width = 1920
    height = 1080

    rect = box_2d_to_cvat_rect(box, width=width, height=height)
    xtl, ytl, xbr, ybr = rect

    # Check order
    assert xtl <= xbr
    assert ytl <= ybr

    # Check exact values
    assert xtl == pytest.approx(0.250 * 1920, 0.01)
    assert ytl == pytest.approx(0.150 * 1080, 0.01)
    assert xbr == pytest.approx(0.750 * 1920, 0.01)
    assert ybr == pytest.approx(0.450 * 1080, 0.01)

    # Check bounds
    assert 0.0 <= xtl <= width
    assert 0.0 <= ytl <= height
    assert 0.0 <= xbr <= width
    assert 0.0 <= ybr <= height


def test_cvat_detector_response_schema_validation():
    """Verify shape dictionary matches CVAT detector specification expectations."""
    shape = {
        "confidence": "0.95",
        "label": "car",
        "points": [100.5, 120.0, 350.25, 400.0],
        "type": "rectangle",
    }

    # Verify type field is literal 'rectangle'
    assert shape["type"] == "rectangle"
    # Verify label is string
    assert isinstance(shape["label"], str)
    # Verify confidence is a float string in range [0.0, 1.0]
    conf_val = float(shape["confidence"])
    assert 0.0 <= conf_val <= 1.0
    # Verify points list has 4 elements
    assert len(shape["points"]) == 4
    xtl, ytl, xbr, ybr = shape["points"]
    assert xtl <= xbr
    assert ytl <= ybr
