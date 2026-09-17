"""Tests for app.pipeline module covering execution, error handling, and edge cases."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image, UnidentifiedImageError
import pytest

from app.client import (
    NineRouterConnectionError,
    NineRouterError,
    NineRouterRequestError,
    VisionResponse,
)
from app.parser import VisionParseError
from app.pipeline import (
    PipelineOptions,
    resolve_image_path,
    run_pipeline,
    select_best_vision_model,
)


def test_select_best_vision_model_priority():
    """Verify auto-selection picks preferred models in priority order."""
    mock_client = MagicMock()
    mock_client.get_vision_models.return_value = [
        {"id": "unknown-model"},
        {"id": "ag/gemini-3.7-flash-high"},
        {"id": "ag/gemini-3.8-flash-high"},
    ]

    # ag/gemini-3.8-flash-high has higher priority than ag/gemini-3.7-flash-high
    selected = select_best_vision_model(mock_client)
    assert selected == "ag/gemini-3.8-flash-high"

    # Explicit model request overrides auto-selection
    explicit = select_best_vision_model(mock_client, requested_model="custom/model")
    assert explicit == "custom/model"


def test_select_best_vision_model_fallback_first():
    """Verify fallback to first available model when no preferred model is present."""
    mock_client = MagicMock()
    mock_client.get_vision_models.return_value = [
        {"id": "other-vision-1"},
        {"id": "other-vision-2"},
    ]
    selected = select_best_vision_model(mock_client)
    assert selected == "other-vision-1"


def test_select_best_vision_model_empty_raises():
    """Verify NineRouterError raised when no vision models are available."""
    mock_client = MagicMock()
    mock_client.get_vision_models.return_value = []
    with pytest.raises(NineRouterError, match="No vision models available"):
        select_best_vision_model(mock_client)


def test_resolve_image_path_explicit_exists(tmp_path):
    """Verify existing explicit image path resolves."""
    img_file = tmp_path / "valid.jpg"
    img_file.write_text("fake")
    resolved = resolve_image_path(img_file)
    assert resolved == img_file


def test_resolve_image_path_explicit_missing():
    """Verify missing explicit image path raises FileNotFoundError."""
    missing = Path("D:/non_existent_folder_xyz/img.jpg")
    with pytest.raises(FileNotFoundError, match="Specified image not found"):
        resolve_image_path(missing)


def test_resolve_image_path_no_fallbacks():
    """Verify FileNotFoundError raised when no image provided and no fallbacks found."""
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="No image specified"):
            resolve_image_path(None)


def test_run_pipeline_end_to_end(tmp_path):
    """Verify complete pipeline execution and artifact generation."""
    # Create temporary test image
    img = Image.new("RGB", (800, 600), color="silver")
    img_path = tmp_path / "test_input.jpg"
    img.save(img_path)

    output_dir = tmp_path / "output"

    options = PipelineOptions(
        image_path=img_path,
        model="ag/gemini-3.8-flash-high",
        output_dir=output_dir,
        max_image_size=1600,
        labels_config=Path(__file__).resolve().parent.parent / "config" / "labels.yaml",
        strict=False,
    )

    fake_vision_response = VisionResponse(
        content=json.dumps({
            "objects": [
                {
                    "label": "car",
                    "box_2d": [150, 200, 450, 650],
                },
                {
                    "label": "traffic light",
                    "box_2d": [50, 700, 150, 750],
                }
            ]
        }),
        raw_response={"id": "chatcmpl-test"},
        duration_seconds=1.23,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )

    with patch("app.pipeline.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.send_vision_request.return_value = fake_vision_response

        result = run_pipeline(options)

        assert result.success is True
        assert result.detections_count == 2
        assert result.model_used == "ag/gemini-3.8-flash-high"
        assert result.original_dimensions == (800, 600)
        assert result.label_distribution == {"car": 1, "traffic light": 1}

        # Verify all 4 required artifacts were created
        result_bbox = output_dir / "result_bbox.jpg"
        predictions = output_dir / "predictions.json"
        raw_response = output_dir / "raw_response.txt"
        run_report = output_dir / "run_report.json"

        assert result_bbox.exists()
        assert predictions.exists()
        assert raw_response.exists()
        assert run_report.exists()

        # Check predictions.json contents
        with open(predictions, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
            assert len(pred_data["objects"]) == 2
            assert pred_data["model"] == "ag/gemini-3.8-flash-high"
            assert pred_data["image"]["width"] == 800
            assert pred_data["image"]["height"] == 600
            assert "cvat_annotations" in pred_data

        # Check raw_response.txt
        with open(raw_response, "r", encoding="utf-8") as f:
            raw_text = f.read()
            assert "objects" in raw_text

        # Check run_report.json
        with open(run_report, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            assert report_data["success"] is True
            assert report_data["detections"]["total_count"] == 2
            assert report_data["timing"]["request_duration_seconds"] == 1.23


# =====================================================================
# Pipeline Failure Modes
# =====================================================================

def test_run_pipeline_missing_image_file(tmp_path):
    """Verify pipeline fails when specified image does not exist."""
    options = PipelineOptions(
        image_path=tmp_path / "does_not_exist.jpg",
        output_dir=tmp_path / "output",
    )
    with pytest.raises(FileNotFoundError, match="Specified image not found"):
        run_pipeline(options)


def test_run_pipeline_corrupt_image_file(tmp_path):
    """Verify pipeline fails when image file is corrupt or not an image."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"This is definitely not a valid image file content.")

    options = PipelineOptions(
        image_path=corrupt_file,
        output_dir=tmp_path / "output",
    )

    with patch("app.pipeline.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.get_vision_models.return_value = [{"id": "ag/gemini-3.8-flash-high"}]

        with pytest.raises(ValueError, match="Cannot identify image file"):
            run_pipeline(options)


def test_run_pipeline_health_check_failure(tmp_path):
    """Verify pipeline halts immediately when health check fails."""
    img = Image.new("RGB", (100, 100))
    img_path = tmp_path / "img.jpg"
    img.save(img_path)

    options = PipelineOptions(
        image_path=img_path,
        output_dir=tmp_path / "output",
    )

    with patch("app.pipeline.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.side_effect = NineRouterConnectionError("Cannot connect to 9Router")

        with pytest.raises(NineRouterConnectionError):
            run_pipeline(options)


def test_run_pipeline_model_unavailable(tmp_path):
    """Verify pipeline fails when requested model is not available or errors on API."""
    img = Image.new("RGB", (100, 100))
    img_path = tmp_path / "img.jpg"
    img.save(img_path)

    options = PipelineOptions(
        image_path=img_path,
        model="non-existent-model",
        output_dir=tmp_path / "output",
    )

    with patch("app.pipeline.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.send_vision_request.side_effect = NineRouterRequestError(
            "Model non-existent-model not found", status_code=404
        )

        with pytest.raises(NineRouterRequestError) as exc_info:
            run_pipeline(options)
        assert exc_info.value.status_code == 404


def test_run_pipeline_strict_mode_fails_on_disallowed_label(tmp_path):
    """Verify pipeline raises VisionParseError in strict mode when model outputs unallowed label."""
    img = Image.new("RGB", (100, 100))
    img_path = tmp_path / "img.jpg"
    img.save(img_path)

    options = PipelineOptions(
        image_path=img_path,
        model="ag/gemini-3.8-flash-high",
        output_dir=tmp_path / "output",
        strict=True,
    )

    fake_resp = VisionResponse(
        content=json.dumps({"objects": [{"label": "alien_spaceship", "box_2d": [10, 10, 50, 50]}]}),
        raw_response={},
        duration_seconds=0.5,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
    )

    with patch("app.pipeline.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.send_vision_request.return_value = fake_resp

        with pytest.raises(VisionParseError, match="not in allowed labels"):
            run_pipeline(options)
