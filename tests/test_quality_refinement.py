"""Targeted tests for adaptive geometry refinement."""

import json
from unittest.mock import MagicMock

from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.parser import ParsedObject
from app.service import annotate_image
from core.quality_gate import assess_quality, should_accept_refinement
from core.vision_contract import MODE_POLYGON_MASK, MODE_RECTANGLE_MASK


def test_quality_gate_flags_partial_instance_mask():
    item = ParsedObject(
        label="car",
        box_2d=[100, 100, 700, 700],
        instance_mask=[[100, 100], [220, 100], [220, 220], [100, 220]],
        confidence=0.95,
    )
    report = assess_quality([item], MODE_RECTANGLE_MASK)
    assert report.needs_refine
    assert any("mask_box_mismatch" in r or "implausible_mask_fill" in r for r in report.reasons)


def test_quality_gate_flags_large_coarse_region():
    item = ParsedObject(
        label="area/drivable",
        region_polygon=[[0, 450], [1000, 450], [1000, 1000], [0, 1000]],
        confidence=0.95,
    )
    report = assess_quality([item], MODE_POLYGON_MASK)
    assert report.needs_refine
    assert any("large_region_too_coarse" in r for r in report.reasons)


def test_accept_refinement_when_suspects_drop():
    before = assess_quality([
        ParsedObject(
            label="road",
            region_polygon=[[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
            confidence=0.9,
        )
    ], MODE_POLYGON_MASK)
    after = assess_quality([
        ParsedObject(
            label="road",
            region_polygon=[
                [0, 520], [120, 500], [240, 490], [360, 505], [480, 520], [600, 510],
                [720, 500], [840, 510], [1000, 530], [1000, 1000], [850, 1000],
                [700, 1000], [550, 1000], [400, 1000], [250, 1000], [100, 1000], [0, 1000],
            ],
            confidence=0.93,
        )
    ], MODE_POLYGON_MASK)
    assert should_accept_refinement(before, after)


def test_pipeline_uses_only_one_optional_refine_call(monkeypatch):
    monkeypatch.setenv("QUALITY_REFINE_ENABLED", "true")
    monkeypatch.setenv("QUALITY_REFINE_MODEL", "ag/gemini-3.8-flash-medium")
    monkeypatch.setenv("QUALITY_REFINE_TOTAL_BUDGET", "42")

    client = NineRouterClient(base_url="http://127.0.0.1:20128", timeout=45)
    first = {
        "regions": [{
            "label": "road",
            "confidence": 0.95,
            "polygon": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
        }]
    }
    refined = {
        "regions": [{
            "label": "road",
            "confidence": 0.95,
            "polygon": [
                [0, 520], [100, 510], [200, 500], [300, 505], [400, 515], [500, 525],
                [600, 520], [700, 510], [800, 505], [900, 515], [1000, 530], [1000, 1000],
                [900, 1000], [800, 1000], [700, 1000], [600, 1000], [500, 1000],
                [400, 1000], [300, 1000], [200, 1000], [100, 1000], [0, 1000],
            ],
        }]
    }
    client.send_vision_request = MagicMock(side_effect=[
        VisionResponse(content=json.dumps(first), raw_response={}, duration_seconds=0.1,
                       model="ag/gemini-3.8-flash-low", status_code=200),
        VisionResponse(content=json.dumps(refined), raw_response={}, duration_seconds=0.2,
                       model="ag/gemini-3.8-flash-medium", status_code=200),
    ])

    result = annotate_image(
        image_source=Image.new("RGB", (640, 480), "white"),
        client=client,
        model="ag/gemini-3.8-flash-low",
        mode=MODE_POLYGON_MASK,
        threshold=0.0,
        enable_feedback=False,
    )

    assert client.send_vision_request.call_count == 2
    assert result.refinement_attempted is True
    assert result.refinement_accepted is True
    assert result.model_used == "ag/gemini-3.8-flash-medium"
    assert result.quality_score_after >= result.quality_score_before
    assert {s["type"] for s in result.shapes} == {"polygon", "mask"}


def test_good_instance_stays_single_fast_call(monkeypatch):
    monkeypatch.setenv("QUALITY_REFINE_ENABLED", "true")
    client = NineRouterClient(base_url="http://127.0.0.1:20128", timeout=45)
    payload = {
        "objects": [{
            "label": "car",
            "confidence": 0.96,
            "box_2d": [100, 100, 700, 700],
            "mask": [
                [110, 110], [350, 100], [690, 120], [700, 350],
                [680, 690], [350, 700], [120, 680], [100, 350],
            ],
        }]
    }
    client.send_vision_request = MagicMock(return_value=VisionResponse(
        content=json.dumps(payload), raw_response={}, duration_seconds=0.1,
        model="ag/gemini-3.8-flash-low", status_code=200,
    ))

    result = annotate_image(
        image_source=Image.new("RGB", (640, 480), "white"),
        client=client,
        model="ag/gemini-3.8-flash-low",
        mode=MODE_RECTANGLE_MASK,
        threshold=0.0,
        enable_feedback=False,
    )
    assert client.send_vision_request.call_count == 1
    assert result.refinement_attempted is False


def test_empty_result_triggers_refinement():
    report = assess_quality([], MODE_POLYGON_MASK)
    assert report.needs_refine is True
    assert report.score == 0.0
    assert "empty_detection" in report.reasons


def test_polygon_prompt_distinguishes_drivable_from_road():
    from core.vision_contract import build_polygon_mask_prompt
    prompt = build_polygon_mask_prompt()
    assert '"area/drivable"' in prompt
    assert '"road"' in prompt
    assert "not mutually exclusive" in prompt.lower()
    assert "never replace" in prompt.lower()
