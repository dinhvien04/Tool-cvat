"""CLI entry point for CVAT x 9Router AI Annotation Pipeline.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --image path/to/image.jpg --model ag/gemini-3.8-flash-high --output-dir output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NINEROUTER_URL,
    DEFAULT_OUTPUT_DIR,
    AppConfig,
)
from app.pipeline import PipelineOptions, PipelineResult, run_pipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CVAT x 9Router 2D Bounding Box AI Annotation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to input image (defaults to searching for test.jpg or test.png)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="9Router vision model ID (if omitted, auto-selects best available vision model)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save generated output artifacts",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help="Maximum width or height for image sent to 9Router API (preserves aspect ratio)",
    )
    parser.add_argument(
        "--labels-config",
        type=str,
        default=str(DEFAULT_LABELS_PATH),
        help="Path to labels YAML configuration file",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="9Router base URL (overrides env NINEROUTER_URL)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="9Router API key (overrides env NINEROUTER_KEY)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict validation mode: fail on unknown labels or malformed boxes rather than skipping them",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="box_and_mask",
        choices=["box", "mask", "box_and_mask", "box-mask"],
        help="Detection mode: 'box' (rectangles only), 'mask' (polygons/masks only), or 'box_and_mask' (both)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM vision model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for vision model output",
    )

    return parser.parse_args()


def print_summary(result: PipelineResult) -> None:
    """Print clean formatted summary of pipeline execution."""
    print("\n" + "=" * 70)
    print(" CVAT x 9Router AI Annotation - Execution Summary")
    print("=" * 70)
    print(f"Status:             {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Model:              {result.model_used}")
    print(f"Image Source:       {result.image_path}")
    print(f"Original Image:     {result.original_dimensions[0]}x{result.original_dimensions[1]}")
    if result.was_resized:
        print(f"API Transmitted:    {result.resized_dimensions[0]}x{result.resized_dimensions[1]} (Resized)")
    else:
        print(f"API Transmitted:    {result.resized_dimensions[0]}x{result.resized_dimensions[1]} (Native)")
    print(f"API Duration:       {result.request_duration_seconds:.2f}s")
    print(f"Total Duration:     {result.total_duration_seconds:.2f}s")
    print(f"Total Detections:   {result.detections_count}")
    if result.label_distribution:
        print("Detections Breakdown:")
        for label, count in sorted(result.label_distribution.items()):
            print(f"  - {label:<16} : {count}")
    print("-" * 70)
    print("Output Artifacts:")
    for name, path in result.output_files.items():
        print(f"  * {name:<14}: {path}")
    print("=" * 70 + "\n")


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    # Load application config with CLI overrides
    cfg = AppConfig.load(
        url_override=args.url,
        key_override=args.key,
        model_override=args.model,
        output_dir_override=args.output_dir,
        max_image_size_override=args.max_image_size,
        labels_yaml=args.labels_config,
    )

    options = PipelineOptions(
        image_path=Path(args.image) if args.image else None,  # type: ignore
        model=args.model or cfg.vision_model,
        output_dir=cfg.output_dir,
        max_image_size=cfg.max_image_size,
        labels_config=cfg.labels_config_path,
        ninerouter_url=cfg.ninerouter_url,
        ninerouter_key=cfg.ninerouter_key,
        strict=args.strict,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mode=args.mode.replace("-", "_"),
    )

    try:
        result = run_pipeline(options)
        print_summary(result)
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
