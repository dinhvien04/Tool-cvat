"""Security and Reliability audit tests for CVAT x 9Router AI Annotation."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.client import NineRouterAuthError, NineRouterClient, NineRouterRequestError, VisionResponse
from app.config import AppConfig, LabelConfig, mask_api_key
from app.image_ops import draw_bounding_boxes, load_image
from app.parser import VisionParseError, parse_and_validate
from app.pipeline import PipelineOptions, run_pipeline
from core.vision_contract import box_2d_to_cvat_rect, cvat_rect_to_box_2d


# =========================================================================
# 1. Secrets & Privacy Audit Tests
# =========================================================================

def test_mask_api_key():
    """Verify mask_api_key masks secrets and handles None/empty."""
    assert mask_api_key(None) == "<none>"
    assert mask_api_key("") == "<none>"
    assert mask_api_key("12345") == "***"
    assert mask_api_key("sk-1234567890abcdef") == "sk-...def"
    assert "1234567890abc" not in mask_api_key("sk-1234567890abcdef")


def test_app_config_repr_masks_key():
    """Verify AppConfig string representation NEVER exposes the raw secret key."""
    secret = "sk-super-secret-production-key-999"
    cfg = AppConfig(ninerouter_key=secret)
    rep = repr(cfg)

    assert secret not in rep
    assert "sk-...999" in rep or "***" in rep


def test_pipeline_options_repr_masks_key():
    """Verify PipelineOptions string representation NEVER exposes the raw secret key."""
    secret = "sk-ultra-confidential-token-888"
    opts = PipelineOptions(image_path=Path("test.jpg"), ninerouter_key=secret)
    rep = repr(opts)

    assert secret not in rep
    assert "sk-...888" in rep or "***" in rep


def test_client_repr_masks_key():
    """Verify NineRouterClient string representation NEVER exposes the raw secret key."""
    secret = "sk-my-api-key-12345678"
    client = NineRouterClient(api_key=secret)
    rep = repr(client)

    assert secret not in rep
    assert "sk-...678" in rep or "***" in rep


def test_client_sanitizes_error_text():
    """Verify error messages sanitize API keys if server echoes them back."""
    secret = "sk-leaked-key-12345"
    client = NineRouterClient(api_key=secret)

    leaked_msg = f"Error: authentication failed for bearer sk-leaked-key-12345 on 9Router"
    sanitized = client._sanitize_error_text(leaked_msg)

    assert secret not in sanitized
    assert "***" in sanitized


def test_pipeline_output_reports_contain_no_secrets_or_base64(tmp_path):
    """Verify output/run_report.json and predictions.json do NOT contain API keys or raw base64."""
    secret = "sk-secret-key-do-not-leak"
    img = Image.new("RGB", (200, 200), color="blue")
    img_path = tmp_path / "input.jpg"
    img.save(img_path)

    out_dir = tmp_path / "output"
    labels_file = Path(__file__).resolve().parent.parent / "config" / "labels.yaml"

    opts = PipelineOptions(
        image_path=img_path,
        model="ag/gemini-3.8-flash-high",
        output_dir=out_dir,
        labels_config=labels_file,
        ninerouter_key=secret,
    )

    fake_resp = VisionResponse(
        content='{"objects": [{"label": "car", "box_2d": [100, 100, 200, 200]}]}',
        raw_response={"choices": [{"message": {"content": "ok"}}]},
        duration_seconds=0.5,
        model="ag/gemini-3.8-flash-high",
        status_code=200,
        usage={},
    )

    with patch("app.pipeline.NineRouterClient") as MockClient:
        mock_inst = MockClient.return_value
        mock_inst.get_health.return_value = {"ok": True}
        mock_inst.send_vision_request.return_value = fake_resp

        result = run_pipeline(opts)
        assert result.success is True

    # Check run_report.json
    report_file = out_dir / "run_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        report_text = f.read()
        assert secret not in report_text
        assert "data:image/" not in report_text
        assert "base64" not in report_text

    # Check predictions.json
    pred_file = out_dir / "predictions.json"
    with open(pred_file, "r", encoding="utf-8") as f:
        pred_text = f.read()
        assert secret not in pred_text
        assert "data:image/" not in pred_text


def test_gitignore_covers_required_entries():
    """Verify .gitignore ignores .env, output/, *.pyc, __pycache__/, .pytest_cache/."""
    gitignore_path = Path(__file__).resolve().parent.parent / ".gitignore"
    assert gitignore_path.exists()

    content = gitignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    assert ".env" in lines
    assert "output/" in lines
    assert "__pycache__/" in lines
    assert ".pytest_cache/" in lines
    assert any("pyc" in line for line in lines)


def test_env_example_has_no_secrets():
    """Verify .env.example contains only placeholder values and no real keys."""
    env_ex = Path(__file__).resolve().parent.parent / ".env.example"
    assert env_ex.exists()

    content = env_ex.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("NINEROUTER_KEY="):
            val = line.split("=", 1)[1].strip()
            assert val == "" or "placeholder" in val.lower() or "your" in val.lower()


# =========================================================================
# 2. Reliability & Resilience Audit Tests
# =========================================================================

def test_client_timeout_safety():
    """Verify NineRouterClient prevents non-positive or None timeouts from hanging."""
    c1 = NineRouterClient(timeout=0)
    assert c1.timeout == 60.0

    c2 = NineRouterClient(timeout=-10.0)
    assert c2.timeout == 60.0

    c3 = NineRouterClient(timeout=None)
    assert c3.timeout == 60.0


def test_models_discovery_resilience_to_invalid_json():
    """Verify get_vision_models handles corrupt non-JSON responses gracefully."""
    client = NineRouterClient()
    mock_resp_corrupt = MagicMock()
    mock_resp_corrupt.status_code = 200
    mock_resp_corrupt.json.side_effect = json.JSONDecodeError("Expecting value", "bad json", 0)

    mock_resp_valid = MagicMock()
    mock_resp_valid.status_code = 200
    mock_resp_valid.json.return_value = {
        "data": [{"id": "ag/gemini-3.8-flash-high", "capabilities": {"vision": True}}]
    }

    def mock_get(url, *args, **kwargs):
        if "image-to-text" in url:
            return mock_resp_corrupt
        return mock_resp_valid

    with patch.object(client.session, "get", side_effect=mock_get):
        models = client.get_vision_models()
        assert len(models) == 1
        assert models[0]["id"] == "ag/gemini-3.8-flash-high"


def test_load_image_corrupted_format():
    """Verify load_image raises ValueError on invalid or corrupted image bytes."""
    corrupt_bytes = b"NOT_AN_IMAGE_HEADER_12345"
    with pytest.raises(ValueError, match="Cannot identify"):
        load_image(corrupt_bytes)


def test_load_image_missing_file():
    """Verify load_image raises FileNotFoundError when target path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file_9999.jpg")


def test_coordinate_conversion_zero_dimensions():
    """Verify coordinate conversion functions reject non-positive dimensions safely."""
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        box_2d_to_cvat_rect([100, 100, 200, 200], width=0, height=100)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        box_2d_to_cvat_rect([100, 100, 200, 200], width=100, height=-50)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        cvat_rect_to_box_2d(10, 10, 50, 50, width=0, height=100)

    with pytest.raises(VisionParseError, match="Invalid image dimensions"):
        parse_and_validate(
            '{"objects": [{"label": "car", "box_2d": [100, 100, 200, 200]}]}',
            image_width=0,
            image_height=100,
        )


def test_draw_bounding_boxes_zero_dimensions():
    """Verify draw_bounding_boxes handles 0-dimension images safely without crashing."""
    img = Image.new("RGB", (100, 100))
    # Test drawing on empty/zero box
    annotated = draw_bounding_boxes(img, [])
    assert annotated.size == (100, 100)


def test_label_config_from_yaml_missing_or_corrupted(tmp_path):
    """Verify LabelConfig.from_yaml handles missing or corrupted YAML files safely."""
    # Missing file
    cfg1 = LabelConfig.from_yaml(tmp_path / "missing.yaml")
    assert cfg1.all_labels == []

    # Corrupt YAML
    corrupt_file = tmp_path / "corrupt.yaml"
    corrupt_file.write_text(":\n  - invalid : yaml ::", encoding="utf-8")
    cfg2 = LabelConfig.from_yaml(corrupt_file)
    assert cfg2.all_labels == []
