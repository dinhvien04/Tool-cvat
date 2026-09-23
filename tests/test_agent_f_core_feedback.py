"""Unit and integration tests for Agent F: Shared Core, 3-Policy Prompting & Policy-Aware Feedback.

Covers Lead Engineer Mandate Sections 2, 13, 14, 15, 16, 17, 18:
1. 3 canonical policies and clean partitioning (14, 10, 7; union 31, pairwise disjoint).
2. Per-detector output validators (Detector A, Detector B, Detector C).
3. Centralized prompt builders and format_shared_constraints.
4. Policy-aware feedback retrieval and zero cross-policy leakage.
5. End-to-end service shape emission and helper methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.client import NineRouterClient, VisionResponse
from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_LANE_EDIT,
    CORRECTION_MASK_EDIT,
    CORRECTION_REGION_EDIT,
    CORRECTION_RELABEL,
    FeedbackDatabase,
    compute_image_hash,
)
from app.retrieval import CorrectionRetrievalEngine, resolve_policy_from_labels
from app.service import AnnotationResult, annotate_image
from core.taxonomy import (
    BOX_MASK_LABELS,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
    SHAPE_RECTANGLE,
    Taxonomy,
    get_taxonomy,
    validate_cvat_output_shapes,
    validate_detector_a_shapes,
    validate_detector_b_shapes,
    validate_detector_c_shapes,
    validate_policy_a_shapes,
    validate_policy_b_shapes,
    validate_policy_c_shapes,
)
from core.vision_contract import (
    MODE_BOX_MASK,
    MODE_POLYGON_MASK,
    MODE_POLYLINE,
    MODE_RECTANGLE_MASK,
    SYSTEM_PROMPT_POLYGON_MASK,
    SYSTEM_PROMPT_POLYLINE,
    SYSTEM_PROMPT_RECTANGLE_MASK,
    build_openai_vision_payload,
    build_polygon_mask_prompt,
    build_polyline_prompt,
    build_rectangle_mask_prompt,
    build_user_prompt,
    format_shared_constraints,
)


class TestTaxonomy3PolicyPartition:
    """Verify strict 3-policy label partitioning and disjointness."""

    def test_partition_counts_and_union(self):
        tax = get_taxonomy()
        a_labels = set(tax.get_box_mask_labels())
        b_labels = set(tax.get_polygon_mask_labels())
        c_labels = set(tax.get_polyline_labels())

        assert len(a_labels) == 14, f"Policy A must have 14 labels, got {len(a_labels)}"
        assert len(b_labels) == 10, f"Policy B must have 10 labels, got {len(b_labels)}"
        assert len(c_labels) == 7, f"Policy C must have 7 labels, got {len(c_labels)}"

        # Disjointness
        assert a_labels.isdisjoint(b_labels), f"Policy A and B overlap: {a_labels & b_labels}"
        assert a_labels.isdisjoint(c_labels), f"Policy A and C overlap: {a_labels & c_labels}"
        assert b_labels.isdisjoint(c_labels), f"Policy B and C overlap: {b_labels & c_labels}"

        # Union is exactly 31 master labels
        union = a_labels | b_labels | c_labels
        assert len(union) == 31
        assert union == set(tax.get_all_labels())


class TestDetectorValidators:
    """Verify per-detector output validation (Section 17)."""

    def test_detector_a_validation_accepts_valid_pair(self):
        shapes = [
            {"type": SHAPE_RECTANGLE, "label": "car", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1},
            {"type": SHAPE_MASK, "label": "car", "points": [10.0, 10.0, 100.0, 10.0, 100.0, 100.0, 10.0, 100.0], "mask": [1, 1, 10, 10, 100, 100], "group_id": 1},
        ]
        validated, warnings = validate_detector_a_shapes(shapes)
        assert len(validated) == 2
        assert len(warnings) == 0
        # Alias test
        assert validate_policy_a_shapes(shapes)[0] == validated

    def test_detector_a_drops_polygon_polyline_and_unpaired(self):
        shapes = [
            # Valid car pair
            {"type": SHAPE_RECTANGLE, "label": "car", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1},
            {"type": SHAPE_MASK, "label": "car", "points": [10.0, 10.0, 100.0, 10.0, 100.0, 100.0, 10.0, 100.0], "mask": [1, 1, 10, 10, 100, 100], "group_id": 1},
            # Forbidden shapes in Detector A
            {"type": SHAPE_POLYGON, "label": "car", "points": [10.0, 10.0, 20.0, 20.0, 30.0, 10.0], "group_id": 2},
            {"type": SHAPE_POLYLINE, "label": "car", "points": [10.0, 10.0, 20.0, 20.0], "group_id": 3},
            # Policy B/C labels
            {"type": SHAPE_RECTANGLE, "label": "road", "points": [10.0, 10.0, 50.0, 50.0], "group_id": 4},
            # Incomplete (box-only)
            {"type": SHAPE_RECTANGLE, "label": "pedestrian", "points": [5.0, 5.0, 20.0, 50.0], "group_id": 5},
        ]
        validated, warnings = validate_detector_a_shapes(shapes)
        assert len(validated) == 2
        assert all(s["label"] == "car" for s in validated)
        assert any("detector_a_forbidden_shape" in w for w in warnings)
        assert any("detector_a_policy_mismatch" in w for w in warnings)
        assert any("detector_a_violation" in w for w in warnings)

    def test_detector_b_validation_accepts_valid_pair(self):
        shapes = [
            {"type": SHAPE_POLYGON, "label": "road", "points": [0.0, 100.0, 100.0, 100.0, 50.0, 200.0], "group_id": 1},
            {"type": SHAPE_MASK, "label": "road", "points": [0.0, 100.0, 100.0, 100.0, 50.0, 200.0], "mask": [1, 1, 0, 100, 100, 200], "group_id": 1},
        ]
        validated, warnings = validate_detector_b_shapes(shapes)
        assert len(validated) == 2
        assert len(warnings) == 0
        assert validate_policy_b_shapes(shapes)[0] == validated

    def test_detector_b_drops_rectangle_polyline_and_unpaired(self):
        shapes = [
            # Valid road pair
            {"type": SHAPE_POLYGON, "label": "road", "points": [0.0, 100.0, 100.0, 100.0, 50.0, 200.0], "group_id": 1},
            {"type": SHAPE_MASK, "label": "road", "points": [0.0, 100.0, 100.0, 100.0, 50.0, 200.0], "mask": [1, 1, 0, 100, 100, 200], "group_id": 1},
            # Forbidden shapes in Detector B
            {"type": SHAPE_RECTANGLE, "label": "road", "points": [0.0, 100.0, 100.0, 200.0], "group_id": 2},
            {"type": SHAPE_POLYLINE, "label": "road", "points": [0.0, 100.0, 100.0, 200.0], "group_id": 3},
            # Policy A label
            {"type": SHAPE_POLYGON, "label": "car", "points": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0], "group_id": 4},
            # Incomplete (polygon-only)
            {"type": SHAPE_POLYGON, "label": "vegetation", "points": [0.0, 0.0, 20.0, 0.0, 20.0, 20.0], "group_id": 5},
        ]
        validated, warnings = validate_detector_b_shapes(shapes)
        assert len(validated) == 2
        assert all(s["label"] == "road" for s in validated)
        assert any("detector_b_forbidden_shape" in w for w in warnings)
        assert any("detector_b_policy_mismatch" in w for w in warnings)

    def test_detector_c_validation_accepts_polyline_drops_others(self):
        shapes = [
            # Valid lane polyline
            {"type": SHAPE_POLYLINE, "label": "lane/single white", "points": [10.0, 20.0, 50.0, 80.0], "group_id": 99},
            # Forbidden shapes in Detector C
            {"type": SHAPE_RECTANGLE, "label": "lane/single white", "points": [10.0, 20.0, 50.0, 80.0]},
            {"type": SHAPE_POLYGON, "label": "lane/crosswalk", "points": [10.0, 20.0, 50.0, 80.0, 20.0, 90.0]},
            {"type": SHAPE_MASK, "label": "lane/single yellow", "mask": [1, 1, 0, 0, 10, 10]},
            # Policy A/B labels
            {"type": SHAPE_POLYLINE, "label": "road", "points": [10.0, 20.0, 50.0, 80.0]},
            {"type": SHAPE_POLYLINE, "label": "car", "points": [10.0, 20.0, 50.0, 80.0]},
        ]
        validated, warnings = validate_detector_c_shapes(shapes)
        assert len(validated) == 1
        assert validated[0]["label"] == "lane/single white"
        assert validated[0]["type"] == SHAPE_POLYLINE
        # group_id must be stripped for lanes
        assert "group_id" not in validated[0]
        assert validate_policy_c_shapes(shapes)[0] == validated

    def test_validate_cvat_output_shapes_policy_dispatch(self):
        car_pair = [
            {"type": SHAPE_RECTANGLE, "label": "car", "points": [10.0, 10.0, 100.0, 100.0], "group_id": 1},
            {"type": SHAPE_MASK, "label": "car", "points": [10.0, 10.0, 100.0, 10.0, 100.0, 100.0, 10.0, 100.0], "mask": [1, 1, 10, 10, 100, 100], "group_id": 1},
        ]
        road_pair = [
            {"type": SHAPE_POLYGON, "label": "road", "points": [0.0, 100.0, 100.0, 100.0, 50.0, 200.0], "group_id": 2},
            {"type": SHAPE_MASK, "label": "road", "points": [0.0, 100.0, 100.0, 100.0, 50.0, 200.0], "mask": [1, 1, 0, 100, 100, 200], "group_id": 2},
        ]
        lane = [
            {"type": SHAPE_POLYLINE, "label": "lane/single white", "points": [10.0, 20.0, 50.0, 80.0]},
        ]
        mixed = car_pair + road_pair + lane

        # Policy A dispatch
        res_a, _ = validate_cvat_output_shapes(mixed, policy=POLICY_BOX_MASK)
        assert len(res_a) == 2
        assert all(s["label"] == "car" for s in res_a)

        # Policy B dispatch
        res_b, _ = validate_cvat_output_shapes(mixed, policy=POLICY_POLYGON_MASK)
        assert len(res_b) == 2
        assert all(s["label"] == "road" for s in res_b)

        # Policy C dispatch
        res_c, _ = validate_cvat_output_shapes(mixed, policy=POLICY_POLYLINE)
        assert len(res_c) == 1
        assert res_c[0]["label"] == "lane/single white"

        # None dispatch (Full 31 mode)
        res_all, _ = validate_cvat_output_shapes(mixed, policy=None)
        assert len(res_all) == 5


class TestPromptBuildersAndConstraints:
    """Verify Section 16 prompt builders and format_shared_constraints."""

    def test_format_shared_constraints_content(self):
        constraints = format_shared_constraints(include_box=True, include_confidence=True, empty_fallback_key="objects")
        assert "Normalized integer coordinates" in constraints or "normalized integers in the range [0, 1000]" in constraints
        assert "box_2d" in constraints
        assert "Confidence Score" in constraints
        assert "no markdown fences" in constraints
        assert '{"objects": []}' in constraints

    def test_build_rectangle_mask_prompt(self):
        prompt = build_rectangle_mask_prompt()
        assert "2D object detection and instance segmentation" in prompt
        assert "box_2d" in prompt and "mask" in prompt
        assert '"objects":' in prompt
        assert "car" in prompt and "truck" in prompt and "pedestrian" in prompt
        # Policy B/C labels should not be in default allowed labels
        assert "road" not in prompt
        assert "lane/single white" not in prompt

    def test_build_polygon_mask_prompt(self):
        prompt = build_polygon_mask_prompt()
        assert "semantic region segmentation" in prompt
        assert '"regions":' in prompt
        assert "polygon" in prompt
        assert "road" in prompt and "sidewalk" in prompt and "building" in prompt
        # Box should be forbidden
        assert "Do NOT provide bounding boxes (box_2d)" in prompt
        assert "car" not in prompt

    def test_build_polyline_prompt(self):
        prompt = build_polyline_prompt()
        assert "lane demarcation and road geometry delineation" in prompt
        assert '"lanes":' in prompt
        assert "polyline" in prompt
        assert "lane/single white" in prompt and "lane/crosswalk" in prompt
        assert "Do NOT provide polygon, mask, or bounding box" in prompt
        assert "car" not in prompt

    def test_build_user_prompt_routing(self):
        p_rect = build_user_prompt(mode=MODE_RECTANGLE_MASK)
        assert '"objects":' in p_rect

        p_box_mask = build_user_prompt(mode=MODE_BOX_MASK)
        assert '"objects":' in p_box_mask

        p_poly = build_user_prompt(mode=MODE_POLYGON_MASK)
        assert '"regions":' in p_poly

        p_line = build_user_prompt(mode=MODE_POLYLINE)
        assert '"lanes":' in p_line

    def test_build_openai_vision_payload_system_prompts(self):
        payload_a = build_openai_vision_payload("fakeb64", mode=MODE_RECTANGLE_MASK)
        assert payload_a["messages"][0]["content"] == SYSTEM_PROMPT_RECTANGLE_MASK

        payload_b = build_openai_vision_payload("fakeb64", mode=MODE_POLYGON_MASK)
        assert payload_b["messages"][0]["content"] == SYSTEM_PROMPT_POLYGON_MASK

        payload_c = build_openai_vision_payload("fakeb64", mode=MODE_POLYLINE)
        assert payload_c["messages"][0]["content"] == SYSTEM_PROMPT_POLYLINE


class TestPolicyAwareRetrieval:
    """Verify Section 13 & 14 policy-aware few-shot retrieval and zero leakage."""

    @pytest.fixture
    def populated_db(self, tmp_path: Path):
        db = FeedbackDatabase(
            db_path=tmp_path / "unified_feedback.sqlite3",
            examples_dir=tmp_path / "examples",
        )
        # Create dummy crops
        crop_a = tmp_path / "examples" / "crop_a.jpg"
        crop_b = tmp_path / "examples" / "crop_b.jpg"
        crop_c = tmp_path / "examples" / "crop_c.jpg"
        crop_neg = tmp_path / "examples" / "crop_neg.jpg"

        Image.new("RGB", (64, 64), (100, 100, 100)).save(crop_a)
        Image.new("RGB", (64, 64), (80, 80, 80)).save(crop_b)
        Image.new("RGB", (64, 64), (50, 50, 50)).save(crop_c)
        Image.new("RGB", (64, 64), (120, 120, 120)).save(crop_neg)

        with db._get_connection() as conn:
            # Policy A correction: car -> truck
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "hash_a",
                    CORRECTION_RELABEL,
                    "car",
                    "truck",
                    str(crop_a),
                    json.dumps({
                        "crop_coords": [0, 0, 64, 64],
                        "human_rect": {"type": "rectangle", "points": [10, 10, 50, 50]},
                        "human_mask": {"type": "mask", "points": [10, 10, 50, 10, 50, 50, 10, 50]},
                    }),
                    json.dumps({"type": "rectangle", "label": "truck", "points": [10, 10, 50, 50]}),
                    "2026-09-18T10:00:00Z",
                ),
            )
            # Policy B correction: road REGION_EDIT
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "hash_b",
                    CORRECTION_REGION_EDIT,
                    "road",
                    "road",
                    str(crop_b),
                    json.dumps({
                        "crop_coords": [0, 0, 64, 64],
                        "human_poly": {"type": "polygon", "points": [0, 30, 64, 30, 64, 64, 0, 64]},
                    }),
                    json.dumps({"type": "polygon", "label": "road", "points": [0, 30, 64, 30, 64, 64, 0, 64]}),
                    "2026-09-18T10:05:00Z",
                ),
            )
            # Policy C correction: lane LANE_EDIT
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "hash_c",
                    CORRECTION_LANE_EDIT,
                    "lane/single white",
                    "lane/single white",
                    str(crop_c),
                    json.dumps({
                        "crop_coords": [0, 0, 64, 64],
                        "human_polyline": {"type": "polyline", "points": [10, 60, 50, 10]},
                    }),
                    json.dumps({"type": "polyline", "label": "lane/single white", "points": [10, 60, 50, 10]}),
                    "2026-09-18T10:10:00Z",
                ),
            )
            # Negative example: pedestrian deleted
            conn.execute(
                """
                INSERT INTO corrections (image_hash, correction_type, ai_label, human_label,
                                         crop_path, details_json, human_shape_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "hash_neg",
                    CORRECTION_DELETE_FALSE_POSITIVE,
                    "pedestrian",
                    None,
                    str(crop_neg),
                    json.dumps({"crop_coords": [10, 10, 50, 50]}),
                    None,
                    "2026-09-18T10:15:00Z",
                ),
            )
            conn.commit()

        return db

    def test_policy_a_retrieval_zero_leakage_and_single_key(self, populated_db):
        engine = CorrectionRetrievalEngine(db=populated_db)
        res = engine.retrieve(policy=POLICY_BOX_MASK)

        assert len(res.visual_examples) > 0
        for ex in res.visual_examples:
            out = ex["expected_output"]
            # Strict single-key schema for Policy A
            assert "objects" in out
            assert "regions" not in out
            assert "lanes" not in out
            if ex["correction_type"] != CORRECTION_DELETE_FALSE_POSITIVE:
                assert len(out["objects"]) == 1
                obj = out["objects"][0]
                assert "box_2d" in obj and "mask" in obj
                assert obj["label"] in BOX_MASK_LABELS

    def test_policy_b_retrieval_zero_leakage_and_single_key(self, populated_db):
        engine = CorrectionRetrievalEngine(db=populated_db)
        res = engine.retrieve(policy=POLICY_POLYGON_MASK)

        assert len(res.visual_examples) == 1
        ex = res.visual_examples[0]
        out = ex["expected_output"]
        # Strict single-key schema for Policy B
        assert "regions" in out
        assert "objects" not in out
        assert "lanes" not in out
        assert len(out["regions"]) == 1
        reg = out["regions"][0]
        assert "polygon" in reg
        assert "box_2d" not in reg
        assert reg["label"] in POLYGON_MASK_LABELS

    def test_policy_c_retrieval_zero_leakage_and_single_key(self, populated_db):
        engine = CorrectionRetrievalEngine(db=populated_db)
        res = engine.retrieve(policy=POLICY_POLYLINE)

        assert len(res.visual_examples) == 1
        ex = res.visual_examples[0]
        out = ex["expected_output"]
        # Strict single-key schema for Policy C
        assert "lanes" in out
        assert "objects" not in out
        assert "regions" not in out
        assert len(out["lanes"]) == 1
        lane = out["lanes"][0]
        assert "polyline" in lane
        assert "box_2d" not in lane
        assert "polygon" not in lane
        assert lane["label"] in POLYLINE_LABELS

    def test_policy_inference_from_candidate_labels(self):
        assert resolve_policy_from_labels(policy="box_mask") == POLICY_BOX_MASK
        assert resolve_policy_from_labels(policy="rectangle_mask") == POLICY_BOX_MASK
        assert resolve_policy_from_labels(policy="polygon_mask") == POLICY_POLYGON_MASK
        assert resolve_policy_from_labels(policy="polyline") == POLICY_POLYLINE
        assert resolve_policy_from_labels(policy="full_31") is None
        assert resolve_policy_from_labels(policy=None) is None


class TestEndToEndServicePolicyModes:
    """Verify annotate_image end-to-end execution for the 3 policies."""

    def test_annotate_image_rectangle_mask_mode(self, monkeypatch):
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        monkeypatch.setattr(
            client,
            "get_vision_models",
            lambda: [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}],
        )

        mock_resp = VisionResponse(
            content=json.dumps({
                "objects": [
                    {
                        "label": "car",
                        "box_2d": [100, 100, 800, 800],
                        "mask": [[100, 100], [800, 100], [800, 800], [100, 800]],
                        "confidence": 0.95,
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )
        monkeypatch.setattr(client, "send_vision_request", lambda *args, **kwargs: mock_resp)

        img = Image.new("RGB", (200, 200), (50, 50, 50))
        result = annotate_image(
            image_source=img,
            client=client,
            mode=MODE_RECTANGLE_MASK,
            enable_feedback=False,
        )

        assert len(result.shapes) == 2
        rects = result.to_cvat_rectangles()
        masks = result.to_cvat_masks()
        assert len(rects) == 1
        assert len(masks) == 1
        assert rects[0]["group_id"] == masks[0]["group_id"]
        assert rects[0]["group_id"] > 0

    def test_annotate_image_polygon_mask_mode(self, monkeypatch):
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        monkeypatch.setattr(
            client,
            "get_vision_models",
            lambda: [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}],
        )

        mock_resp = VisionResponse(
            content=json.dumps({
                "regions": [
                    {
                        "label": "road",
                        "polygon": [[0, 500], [1000, 500], [1000, 1000], [0, 1000]],
                        "confidence": 0.92,
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )
        monkeypatch.setattr(client, "send_vision_request", lambda *args, **kwargs: mock_resp)

        img = Image.new("RGB", (200, 200), (50, 50, 50))
        result = annotate_image(
            image_source=img,
            client=client,
            mode=MODE_POLYGON_MASK,
            enable_feedback=False,
        )

        assert len(result.shapes) == 2
        polys = result.to_cvat_polygons()
        masks = result.to_cvat_masks()
        assert len(polys) == 1
        assert len(masks) == 1
        assert polys[0]["group_id"] == masks[0]["group_id"]
        assert polys[0]["group_id"] > 0
        # Rectangles must be strictly absent
        assert len(result.to_cvat_rectangles()) == 0

    def test_annotate_image_polyline_mode(self, monkeypatch):
        client = NineRouterClient(base_url="http://127.0.0.1:20128")
        monkeypatch.setattr(
            client,
            "get_vision_models",
            lambda: [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}],
        )

        mock_resp = VisionResponse(
            content=json.dumps({
                "lanes": [
                    {
                        "label": "lane/single white",
                        "polyline": [[100, 800], [400, 500], [700, 200]],
                        "confidence": 0.88,
                    }
                ]
            }),
            raw_response={},
            duration_seconds=0.1,
            model="ag/gemini-3.8-flash-high",
            status_code=200,
        )
        monkeypatch.setattr(client, "send_vision_request", lambda *args, **kwargs: mock_resp)

        img = Image.new("RGB", (200, 200), (50, 50, 50))
        result = annotate_image(
            image_source=img,
            client=client,
            mode=MODE_POLYLINE,
            enable_feedback=False,
        )

        assert len(result.shapes) == 1
        lines = result.to_cvat_polylines()
        assert len(lines) == 1
        assert lines[0]["type"] == "polyline"
        assert "group_id" not in lines[0]
        # Rectangles, polygons, masks strictly absent
        assert len(result.to_cvat_rectangles()) == 0
        assert len(result.to_cvat_masks()) == 0
        assert len(result.to_cvat_polygons()) == 0
