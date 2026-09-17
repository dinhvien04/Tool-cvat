"""Configuration management for CVAT x 9Router AI Annotation.

Loads settings from environment variables, optional .env files,
and configuration YAML files (e.g. config/labels.yaml).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Try loading python-dotenv if installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_LABELS_PATH = DEFAULT_CONFIG_DIR / "labels.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_NINEROUTER_URL = "http://127.0.0.1:20128"
DEFAULT_VISION_MODEL = "ag/gemini-3.8-flash-high"
DEFAULT_MAX_IMAGE_SIZE = 1600


def mask_api_key(key: Optional[str]) -> str:
    """Return masked representation of API key (e.g. 'sk-***...xyz' or '***').

    Ensures secrets are never exposed in log messages, debug representations, or traces.
    """
    if not key:
        return "<none>"
    stripped = key.strip()
    if len(stripped) <= 6:
        return "***"
    return f"{stripped[:3]}...{stripped[-3:]}"


@dataclass
class LabelConfig:
    """Label definitions loaded from labels.yaml."""
    all_labels: List[str] = field(default_factory=list)
    bbox_labels: List[str] = field(default_factory=list)
    non_bbox_labels: List[str] = field(default_factory=list)
    cvat_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path | str] = None) -> LabelConfig:
        """Load label configurations from a YAML file safely."""
        target_path = Path(yaml_path) if yaml_path else DEFAULT_LABELS_PATH
        if not target_path.exists() or not target_path.is_file():
            return cls()

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return cls()

        return cls(
            all_labels=list(data.get("all_labels", [])),
            bbox_labels=list(data.get("bbox_labels", [])),
            non_bbox_labels=list(data.get("non_bbox_labels", [])),
            cvat_schema=dict(data.get("cvat_schema", {})),
        )


@dataclass
class AppConfig:
    """Central application settings."""
    ninerouter_url: str = DEFAULT_NINEROUTER_URL
    ninerouter_key: Optional[str] = None
    vision_model: str = DEFAULT_VISION_MODEL
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    labels_config_path: Path = DEFAULT_LABELS_PATH
    labels: LabelConfig = field(default_factory=LabelConfig)

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        masked_key = mask_api_key(self.ninerouter_key) if self.ninerouter_key else None
        return (
            f"AppConfig(ninerouter_url={self.ninerouter_url!r}, "
            f"ninerouter_key={masked_key!r}, "
            f"vision_model={self.vision_model!r}, "
            f"max_image_size={self.max_image_size}, "
            f"output_dir={self.output_dir!r}, "
            f"labels_config_path={self.labels_config_path!r})"
        )

    @classmethod
    def load(
        cls,
        env_file: Optional[Path | str] = None,
        labels_yaml: Optional[Path | str] = None,
        url_override: Optional[str] = None,
        key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        output_dir_override: Optional[Path | str] = None,
        max_image_size_override: Optional[int] = None,
    ) -> AppConfig:
        """Load complete configuration from environment, .env file, and config YAML."""
        # Check custom env file
        if env_file:
            target_env = Path(env_file)
            if target_env.exists():
                try:
                    from dotenv import load_dotenv
                    load_dotenv(target_env, override=True)
                except ImportError:
                    _simple_load_env(target_env)

        url = url_override or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL).strip()
        # Clean trailing slash from base url
        if url.endswith("/"):
            url = url.rstrip("/")

        key = key_override or os.getenv("NINEROUTER_KEY")
        if key is not None:
            key = key.strip()
            if not key:
                key = None

        model = model_override or os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL).strip()

        # Image size
        max_size_env = os.getenv("MAX_IMAGE_SIZE")
        if max_image_size_override is not None:
            max_size = int(max_image_size_override)
        elif max_size_env:
            try:
                max_size = int(max_size_env)
            except ValueError:
                max_size = DEFAULT_MAX_IMAGE_SIZE
        else:
            max_size = DEFAULT_MAX_IMAGE_SIZE

        # Output directory
        if output_dir_override is not None:
            out_dir = Path(output_dir_override)
        else:
            out_dir_env = os.getenv("OUTPUT_DIR")
            out_dir = Path(out_dir_env) if out_dir_env else DEFAULT_OUTPUT_DIR

        # Labels configuration path
        target_labels_path = Path(labels_yaml) if labels_yaml else Path(
            os.getenv("LABELS_CONFIG", str(DEFAULT_LABELS_PATH))
        )
        label_cfg = LabelConfig.from_yaml(target_labels_path)

        return cls(
            ninerouter_url=url,
            ninerouter_key=key,
            vision_model=model,
            max_image_size=max_size,
            output_dir=out_dir,
            labels_config_path=target_labels_path,
            labels=label_cfg,
        )


def _simple_load_env(path: Path) -> None:
    """Fallback manual .env parser if python-dotenv is not installed."""
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


# Global config instance for quick access
def get_config() -> AppConfig:
    """Retrieve default global application configuration."""
    return AppConfig.load()
