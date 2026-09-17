"""CLI entry point for CVAT x 9Router AI Annotation Pipeline.

Supports:
1. Standard Pipeline Inference:
    python main.py --image path/to/image.jpg
    python main.py --image path/to/image.jpg --mode full_31

2. Human-in-the-Loop Correction Learning Subcommands:
    python main.py feedback-sync --job <id> [--url <cvat_url>] [--token <token>]
    python main.py feedback-stats
    python main.py feedback-list [--limit 20] [--label <name>]
    python main.py feedback-disable
    python main.py feedback-enable
    python main.py feedback-clear [--all | --crops | --rules]
    python main.py feedback-eval [--image path/to/image.jpg]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NINEROUTER_URL,
    DEFAULT_OUTPUT_DIR,
    AppConfig,
)
from app.pipeline import PipelineOptions, PipelineResult, run_pipeline


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse standard pipeline command-line arguments."""
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
        choices=["box", "mask", "box_and_mask", "box-mask", "full_31"],
        help="Detection mode: 'box', 'mask', 'box_and_mask', or 'full_31'",
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

    return parser.parse_args(args)


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


# -----------------------------------------------------------------------------
# Feedback Subcommand Handlers
# -----------------------------------------------------------------------------

def handle_feedback_sync(argv: List[str]) -> int:
    """Sync annotations from a CVAT job and update correction memory."""
    parser = argparse.ArgumentParser(
        prog="python main.py feedback-sync",
        description="Synchronize annotations from a CVAT job into Tool-cvat correction memory",
    )
    parser.add_argument("--job", "-j", type=int, required=True, help="CVAT Job ID to synchronize")
    parser.add_argument("--url", type=str, default="http://localhost:18080", help="CVAT base URL")
    parser.add_argument("--token", type=str, default=None, help="CVAT authentication token")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum sample threshold for rule derivation")
    args = parser.parse_args(argv)

    from app.cvat_sync import sync_job_feedback
    from app.feedback import FeedbackDatabase

    db = FeedbackDatabase()
    print(f"\nSynchronizing CVAT Job #{args.job} from {args.url}...")
    report = sync_job_feedback(
        job_id=args.job,
        cvat_url=args.url,
        token=args.token,
        db=db,
        min_rule_samples=args.min_samples,
    )

    print("\n" + "=" * 70)
    print(f" CVAT Job Feedback Synchronization Report - Job #{args.job}")
    print("=" * 70)
    print(f"Status:             {'SUCCESS' if report.success else 'FAILED'}")
    print(f"Task ID:            #{report.task_id if report.task_id else 'N/A'}")
    print(f"Frames Checked:     {report.frames_checked}")
    print(f"Frames Reconciled:  {report.frames_reconciled}")
    print(f"Total Human Shapes: {report.total_human_shapes}")
    if report.corrections_found:
        print("Corrections Detected:")
        for ctype, cnt in sorted(report.corrections_found.items()):
            print(f"  - {ctype:<24} : {cnt}")
    else:
        print("Corrections Detected: (none)")
    print(f"Derived Rules Added: {report.rules_derived}")
    print(f"Message:            {report.message}")
    print("=" * 70 + "\n")
    return 0 if report.success else 1


def handle_feedback_stats(argv: List[str]) -> int:
    """Display comprehensive correction statistics across all 31 master labels."""
    from app.feedback import ALL_31_LABELS, FeedbackDatabase

    db = FeedbackDatabase()
    stats = db.get_label_statistics()
    rules = db.get_active_rules()

    print("\n" + "=" * 108)
    print(" Tool-cvat Correction Memory - Per-Label Statistics (Full 31 Labels)")
    print("=" * 108)
    header = (
        f"{'Label':<20} | {'Acc':<5} | {'RelFrom':<7} | {'RelTo':<6} | "
        f"{'BoxMov':<6} | {'BoxRsz':<6} | {'MaskEd':<6} | {'FalsePos':<8} | {'MissAdd':<7} | {'Total':<6}"
    )
    print(header)
    print("-" * 108)

    total_corrections_all = 0
    total_accepted_all = 0

    for lbl in ALL_31_LABELS:
        s = stats.get(lbl, {})
        acc = s.get("accepted", 0)
        rf = s.get("relabeled_from", 0)
        rt = s.get("relabeled_to", 0)
        bm = s.get("box_moved", 0)
        br = s.get("box_resized", 0)
        me = s.get("mask_edited", 0) + s.get("region_edited", 0) + s.get("lane_edited", 0)
        fp = s.get("false_positives", 0)
        ma = s.get("missed_adds", 0)
        tot = s.get("total_corrections", 0)

        total_corrections_all += tot
        total_accepted_all += acc

        # Print all labels or highlight active ones
        row_str = (
            f"{lbl:<20} | {acc:<5} | {rf:<7} | {rt:<6} | "
            f"{bm:<6} | {br:<6} | {me:<6} | {fp:<8} | {ma:<7} | {tot:<6}"
        )
        print(row_str)

    print("-" * 108)
    print(f"Totals: Accepted: {total_accepted_all} | Total Corrections: {total_corrections_all}")
    print("=" * 108)

    if rules:
        print("\nActive Correction-Derived Rules:")
        for idx, r in enumerate(rules, 1):
            print(f"  {idx}. {r}")
    else:
        print("\nActive Correction-Derived Rules: (none yet — requires >= 3 samples per pattern)")
    print()
    return 0


def handle_feedback_list(argv: List[str]) -> int:
    """List recent human corrections from the database."""
    parser = argparse.ArgumentParser(
        prog="python main.py feedback-list",
        description="List recent correction records from local database",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of records to return")
    parser.add_argument("--label", type=str, default=None, help="Filter by label")
    parser.add_argument("--type", type=str, default=None, help="Filter by correction type")
    args = parser.parse_args(argv)

    from app.feedback import FeedbackDatabase

    db = FeedbackDatabase()
    records = db.get_corrections(limit=args.limit, label=args.label, correction_type=args.type)

    print("\n" + "=" * 95)
    print(f" Tool-cvat Correction Records (Showing {len(records)} items)")
    print("=" * 95)
    print(f"{'ID':<6} | {'Type':<22} | {'AI Label':<16} | {'Human Label':<16} | {'IoU':<6} | {'Created At':<20}")
    print("-" * 95)

    for r in records:
        cid = str(r["id"])
        ctype = str(r["correction_type"])
        ai_lbl = str(r.get("ai_label") or "-")
        h_lbl = str(r.get("human_label") or "-")
        iou = f"{r.get('iou', 0.0):.2f}" if r.get("iou") is not None else "-"
        created = str(r.get("created_at", ""))[:19]
        print(f"{cid:<6} | {ctype:<22} | {ai_lbl:<16} | {h_lbl:<16} | {iou:<6} | {created:<20}")

    print("=" * 95 + "\n")
    return 0


def handle_feedback_toggle(enable: bool) -> int:
    """Enable or disable correction memory."""
    from app.feedback import FeedbackDatabase

    db = FeedbackDatabase()
    db.set_enabled(enable)
    state_str = "ENABLED" if enable else "DISABLED"
    print(f"\n[Tool-cvat] Correction memory & adaptive prompting is now {state_str}.\n")
    return 0


def handle_feedback_clear(argv: List[str]) -> int:
    """Clear feedback database, crops, or rules."""
    parser = argparse.ArgumentParser(
        prog="python main.py feedback-clear",
        description="Clear correction memory records, crops, or rules",
    )
    parser.add_argument("--all", action="store_true", help="Clear all corrections, predictions, and crops")
    parser.add_argument("--crops", action="store_true", help="Clear only local image crops")
    parser.add_argument("--rules", action="store_true", help="Clear only derived rules")
    args = parser.parse_args(argv)

    from app.feedback import FeedbackDatabase

    db = FeedbackDatabase()
    if args.crops and not args.all:
        db.clear(all_data=False, crops=True)
        print("\n[Tool-cvat] Local example crops cleared.")
    elif args.rules and not args.all:
        db.clear(all_data=False, rules=True)
        print("\n[Tool-cvat] Derived rules cleared.")
    else:
        db.clear(all_data=True, crops=True)
        print("\n[Tool-cvat] All feedback records, predictions, rules, and crops cleared.")
    return 0


def handle_feedback_eval(argv: List[str]) -> int:
    """Run comparative evaluation comparing baseline vs. adaptive inference."""
    parser = argparse.ArgumentParser(
        prog="python main.py feedback-eval",
        description="Compare baseline inference vs adaptive correction-aware inference",
    )
    parser.add_argument("--image", "-i", type=str, default="test.jpg", help="Path to evaluation test image")
    parser.add_argument("--model", "-m", type=str, default=None, help="Vision model ID")
    parser.add_argument("--mode", type=str, default="full_31", help="Detection mode")
    args = parser.parse_args(argv)

    from app.client import NineRouterClient
    from app.feedback import FeedbackDatabase
    from app.service import annotate_image

    db = FeedbackDatabase()
    client = NineRouterClient()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        return 1

    img_bytes = img_path.read_bytes()

    print("\n" + "=" * 70)
    print(" Running Adaptive vs. Baseline Evaluation")
    print("=" * 70)
    print(f"Image: {args.image} ({len(img_bytes)} bytes)")
    print(f"Mode:  {args.mode}")

    # Pass 1: Baseline without correction memory
    print("\n[1/2] Running Pass 1: Baseline Inference (Feedback Disabled)...")
    res_base = annotate_image(
        image_source=img_bytes,
        client=client,
        model=args.model,
        mode=args.mode,
        enable_feedback=False,
    )
    print(f"  -> Baseline Shapes: {len(res_base.shapes)} ({res_base.api_duration_seconds:.2f}s)")

    # Pass 2: Adaptive with correction memory
    print("\n[2/2] Running Pass 2: Adaptive Inference (Correction Memory Enabled)...")
    res_adapt = annotate_image(
        image_source=img_bytes,
        client=client,
        model=args.model,
        mode=args.mode,
        enable_feedback=True,
        feedback_db=db,
    )
    print(f"  -> Adaptive Shapes: {len(res_adapt.shapes)} ({res_adapt.api_duration_seconds:.2f}s)")
    print(f"  -> Rules Injected:  {len(res_adapt.rules_injected)}")
    for r in res_adapt.rules_injected:
        print(f"      * {r}")

    print("\n" + "=" * 70)
    print(" Evaluation Comparison Summary")
    print("=" * 70)
    print(f"Baseline Detections: {len(res_base.shapes)}")
    print(f"Adaptive Detections: {len(res_adapt.shapes)}")
    print(f"Rules Injected:      {len(res_adapt.rules_injected)}")
    print("=" * 70 + "\n")
    return 0


def handle_feedback_webhook_setup(argv: List[str]) -> int:
    """Inspect or configure CVAT feedback webhook."""
    import os
    from app.cvat_sync import CVATSyncClient

    parser = argparse.ArgumentParser(
        prog="python main.py feedback-webhook-setup",
        description="Inspect, register, or update CVAT feedback webhook",
    )
    parser.add_argument("--url", type=str, default=os.getenv("CVAT_URL", "http://localhost:18080"), help="CVAT server base URL")
    parser.add_argument("--token", type=str, default=os.getenv("CVAT_TOKEN"), help="CVAT authentication token")
    parser.add_argument("--target-url", type=str, default="http://nuclio-nuclio-ninerouter-vision-31:8080", help="Webhook destination URL")
    parser.add_argument("--secret", type=str, default=os.getenv("CVAT_WEBHOOK_SECRET"), help="Webhook HMAC shared secret")
    parser.add_argument("--list", action="store_true", help="List existing webhooks only")
    args = parser.parse_args(argv)

    client = CVATSyncClient(base_url=args.url, token=args.token)

    if args.list:
        hooks = client.list_webhooks()
        print(f"\nRegistered CVAT Webhooks ({len(hooks)}):")
        for h in hooks:
            print(f"  ID: {h.get('id')} | URL: {h.get('target_url')} | Desc: {h.get('description')}")
        print()
        return 0

    res = client.setup_webhook(target_url=args.target_url, secret=args.secret)
    print(f"\n[Tool-cvat] Webhook setup result: {res['action']} (Webhook ID: {res.get('id')})\n")
    return 0


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Main execution function with subcommand dispatch."""
    raw_args = argv if argv is not None else sys.argv[1:]

    # Check for feedback subcommands
    if raw_args and raw_args[0] in (
        "feedback-sync",
        "feedback-stats",
        "feedback-list",
        "feedback-disable",
        "feedback-enable",
        "feedback-clear",
        "feedback-eval",
        "feedback-webhook-setup",
    ):
        subcmd = raw_args[0]
        sub_argv = raw_args[1:]

        if subcmd == "feedback-sync":
            code = handle_feedback_sync(sub_argv)
        elif subcmd == "feedback-stats":
            code = handle_feedback_stats(sub_argv)
        elif subcmd == "feedback-list":
            code = handle_feedback_list(sub_argv)
        elif subcmd == "feedback-disable":
            code = handle_feedback_toggle(False)
        elif subcmd == "feedback-enable":
            code = handle_feedback_toggle(True)
        elif subcmd == "feedback-clear":
            code = handle_feedback_clear(sub_argv)
        elif subcmd == "feedback-eval":
            code = handle_feedback_eval(sub_argv)
        elif subcmd == "feedback-webhook-setup":
            code = handle_feedback_webhook_setup(sub_argv)
        else:
            code = 1
        sys.exit(code)

    # Standard pipeline execution
    args = parse_arguments(raw_args)

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
