"""Tests for CLI entry points: app.models and main.py."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.client import NineRouterConnectionError, NineRouterError
from app.models import format_models_table, list_vision_models, main as models_main
from app.pipeline import PipelineResult
from main import main as main_cli, parse_arguments, print_summary


# =====================================================================
# app.models CLI tests
# =====================================================================

def test_models_cli_help_subprocess():
    """Verify python -m app.models --help exit code and help output."""
    result = subprocess.run(
        [sys.executable, "-m", "app.models", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Discover and inspect available 9Router vision models" in result.stdout
    assert "--url" in result.stdout
    assert "--key" in result.stdout
    assert "--json" in result.stdout


def test_main_cli_help_subprocess():
    """Verify python main.py --help exit code and help output."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "CVAT x 9Router 2D Bounding Box AI Annotation Pipeline" in result.stdout
    assert "--image" in result.stdout
    assert "--model" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--max-image-size" in result.stdout
    assert "--strict" in result.stdout


def test_format_models_table_empty():
    """Verify table formatting when no models are returned."""
    output = format_models_table([])
    assert output == "No vision models found."


def test_format_models_table_with_models():
    """Verify table formatting with recommended, gemini, and claude models."""
    sample_models = [
        {
            "id": "ag/gemini-3.8-flash-high",
            "owned_by": "google",
            "capabilities": {"contextWindow": 1000000, "vision": True},
        },
        {
            "id": "ag/claude-opus-4-6-thinking",
            "owned_by": "anthropic",
            "capabilities": {"contextWindow": 200000, "vision": True},
        },
        {
            "id": "custom-vision-model",
            "owner": "custom",
            "capabilities": {"vision": True},
        },
    ]
    output = format_models_table(sample_models)
    assert "Model ID" in output
    assert "ag/gemini-3.8-flash-high" in output
    assert "[Recommended]" in output
    assert "ag/claude-opus-4-6-thinking" in output
    assert "[Claude]" in output
    assert "Total vision models available: 3" in output


def test_list_vision_models_success_table_output(capsys):
    """Verify list_vision_models prints human-readable table on success and returns 0."""
    with patch("app.models.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.get_vision_models.return_value = [
            {"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}},
        ]

        exit_code = list_vision_models(url="http://127.0.0.1:20128")
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "9Router Status: Connected to" in captured.out
        assert "ag/gemini-3.8-flash-high" in captured.out
        assert "Default auto-selection for detection pipeline: 'ag/gemini-3.8-flash-high'" in captured.out


def test_list_vision_models_success_json_output(capsys):
    """Verify list_vision_models with output_json=True prints valid JSON array and returns 0."""
    models_data = [
        {"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}},
        {"id": "ag/gemini-3.7-flash-high", "capabilities": {"vision": True}},
    ]
    with patch("app.models.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.get_vision_models.return_value = models_data

        exit_code = list_vision_models(output_json=True)
        assert exit_code == 0

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "ag/gemini-3.8-flash-high"


def test_list_vision_models_health_failure(capsys):
    """Verify list_vision_models returns 1 and prints error to stderr when health check fails."""
    with patch("app.models.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.side_effect = NineRouterConnectionError("Connection refused")

        exit_code = list_vision_models()
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Error connecting to 9Router" in captured.err


def test_list_vision_models_fetch_failure(capsys):
    """Verify list_vision_models returns 1 and prints error to stderr when model query fails."""
    with patch("app.models.NineRouterClient") as MockClientClass:
        mock_instance = MockClientClass.return_value
        mock_instance.get_health.return_value = {"ok": True}
        mock_instance.get_vision_models.side_effect = NineRouterError("API timeout")

        exit_code = list_vision_models()
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Error fetching vision models" in captured.err


def test_models_main_exit_code():
    """Verify app.models.main calls sys.exit with return code."""
    with patch("sys.argv", ["models.py", "--json"]), \
         patch("app.models.list_vision_models", return_value=0):
        with pytest.raises(SystemExit) as exc_info:
            models_main()
        assert exc_info.value.code == 0

    with patch("sys.argv", ["models.py"]), \
         patch("app.models.list_vision_models", return_value=1):
        with pytest.raises(SystemExit) as exc_info:
            models_main()
        assert exc_info.value.code == 1


# =====================================================================
# main.py CLI tests
# =====================================================================

def test_parse_arguments_defaults():
    """Verify default CLI argument values in main.py."""
    with patch("sys.argv", ["main.py"]):
        args = parse_arguments()
        assert args.image is None
        assert args.model is None
        assert args.output_dir.endswith("output")
        assert args.max_image_size == 1600
        assert args.strict is False
        assert args.temperature == 0.0
        assert args.max_tokens == 4096


def test_parse_arguments_custom_values():
    """Verify parsing custom and short flags in main.py."""
    custom_args = [
        "main.py",
        "-i", "custom_image.png",
        "-m", "custom_model",
        "-o", "custom_output",
        "--max-image-size", "1200",
        "--strict",
        "--temperature", "0.2",
        "--max-tokens", "2048",
        "--url", "http://10.0.0.1:20128",
        "--key", "secret123",
    ]
    with patch("sys.argv", custom_args):
        args = parse_arguments()
        assert args.image == "custom_image.png"
        assert args.model == "custom_model"
        assert args.output_dir == "custom_output"
        assert args.max_image_size == 1200
        assert args.strict is True
        assert args.temperature == 0.2
        assert args.max_tokens == 2048
        assert args.url == "http://10.0.0.1:20128"
        assert args.key == "secret123"


def test_print_summary(capsys):
    """Verify print_summary formats pipeline execution result."""
    fake_result = PipelineResult(
        success=True,
        image_path="test.jpg",
        model_used="ag/gemini-3.8-flash-high",
        original_dimensions=(1920, 1080),
        resized_dimensions=(1600, 900),
        was_resized=True,
        request_duration_seconds=1.45,
        total_duration_seconds=2.10,
        detections_count=3,
        label_distribution={"car": 2, "pedestrian": 1},
        output_files={"result_bbox": "output/result_bbox.jpg"},
    )
    print_summary(fake_result)
    captured = capsys.readouterr()
    assert "CVAT x 9Router AI Annotation - Execution Summary" in captured.out
    assert "SUCCESS" in captured.out
    assert "ag/gemini-3.8-flash-high" in captured.out
    assert "1920x1080" in captured.out
    assert "1600x900 (Resized)" in captured.out
    assert "car              : 2" in captured.out
    assert "pedestrian       : 1" in captured.out


def test_main_cli_success(capsys):
    """Verify main() exits with 0 on pipeline success."""
    fake_result = PipelineResult(
        success=True,
        image_path="test.jpg",
        model_used="test-model",
        original_dimensions=(100, 100),
        resized_dimensions=(100, 100),
        was_resized=False,
        request_duration_seconds=0.5,
        total_duration_seconds=0.8,
        detections_count=0,
        label_distribution={},
        output_files={},
    )
    with patch("sys.argv", ["main.py", "--image", "test.jpg"]), \
         patch("main.run_pipeline", return_value=fake_result):
        with pytest.raises(SystemExit) as exc_info:
            main_cli()
        assert exc_info.value.code == 0


def test_main_cli_failure(capsys):
    """Verify main() prints error to stderr and exits with 1 on pipeline exception."""
    with patch("sys.argv", ["main.py", "--image", "test.jpg"]), \
         patch("main.run_pipeline", side_effect=RuntimeError("Pipeline crash")):
        with pytest.raises(SystemExit) as exc_info:
            main_cli()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Pipeline execution failed: Pipeline crash" in captured.err
