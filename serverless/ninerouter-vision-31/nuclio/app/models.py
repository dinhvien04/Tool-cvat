"""CLI tool and module for discovering 9Router vision models.

Usage:
    python -m app.models [--url http://127.0.0.1:20128] [--key KEY] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from app.client import NineRouterClient, NineRouterError
from app.config import DEFAULT_NINEROUTER_URL, AppConfig

RECOMMENDED_MODELS = [
    "ag/gemini-3.8-flash-high",
    "ag/gemini-3.8-flash-medium",
    "ag/gemini-3.8-flash",
    "ag/gemini-3.7-flash-high",
    "ag/claude-sonnet-4-6",
]


def format_models_table(models: List[Dict[str, Any]]) -> str:
    """Format models list as a clear terminal table."""
    if not models:
        return "No vision models found."

    lines = []
    header = f"{'#':<3} | {'Model ID':<30} | {'Owner':<10} | {'Context':<10} | {'Vision':<7} | {'Notes'}"
    separator = "-" * len(header)
    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    for i, m in enumerate(models, 1):
        mid = str(m.get("id", "unknown"))
        owner = str(m.get("owned_by", m.get("owner", "-")))
        caps = m.get("capabilities", {})
        ctx = str(caps.get("contextWindow", m.get("context_length", "-")))
        has_vision = "Yes" if caps.get("vision") is True or "vision" in mid.lower() else "Unknown"

        note = ""
        if mid in RECOMMENDED_MODELS:
            note = "[Recommended]"
        elif "gemini-3.8" in mid:
            note = "[Gemini 3.8]"
        elif "claude" in mid:
            note = "[Claude]"

        line = f"{i:<3} | {mid:<30} | {owner:<10} | {ctx:<10} | {has_vision:<7} | {note}"
        lines.append(line)

    lines.append(separator)
    lines.append(f"Total vision models available: {len(models)}")
    return "\n".join(lines)


def list_vision_models(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    output_json: bool = False,
) -> int:
    """Query 9Router and print discovered vision models."""
    cfg = AppConfig.load(url_override=url, key_override=api_key)
    client = NineRouterClient(base_url=cfg.ninerouter_url, api_key=cfg.ninerouter_key)

    try:
        health = client.get_health()
    except NineRouterError as e:
        print(f"Error connecting to 9Router at {cfg.ninerouter_url}: {e}", file=sys.stderr)
        return 1

    try:
        models = client.get_vision_models()
    except NineRouterError as e:
        print(f"Error fetching vision models: {e}", file=sys.stderr)
        return 1

    if output_json:
        print(json.dumps(models, indent=2))
        return 0

    print(f"9Router Status: Connected to {cfg.ninerouter_url} (Health: {health})")
    print(format_models_table(models))

    # Show first recommended pick
    model_ids = [str(m.get("id")) for m in models if "id" in m]
    selected = None
    for rec in RECOMMENDED_MODELS:
        if rec in model_ids:
            selected = rec
            break
    if not selected and model_ids:
        selected = model_ids[0]

    if selected:
        print(f"\nDefault auto-selection for detection pipeline: '{selected}'")

    return 0


def main() -> None:
    """CLI entry point for python -m app.models."""
    parser = argparse.ArgumentParser(
        description="Discover and inspect available 9Router vision models.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="9Router base URL (default: env NINEROUTER_URL or http://127.0.0.1:20128)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="9Router API key (optional, default: env NINEROUTER_KEY)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON array of models",
    )
    args = parser.parse_args()

    sys.exit(list_vision_models(url=args.url, api_key=args.key, output_json=args.json))


if __name__ == "__main__":
    main()
