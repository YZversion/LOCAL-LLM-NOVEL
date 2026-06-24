#!/usr/bin/env python3
"""One-command System B MVP update: reviewed facts -> kg.json -> Markdown cards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kg_render import render_kg
from scripts.kg_update import KGUpdateError, load_facts, load_kg, merge_facts, write_kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed facts and render System B memory cards.")
    parser.add_argument("--facts", required=True, help="Reviewed facts JSON file.")
    parser.add_argument("--kg", default="data/story_bible/kg.json", help="KG JSON path.")
    parser.add_argument(
        "--out-dir",
        default="data/story_bible/generated/system_b",
        help="Rendered Markdown card directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write files.")
    parser.add_argument("--prune", action="store_true", help="Delete old rendered cards before rendering.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        kg_path = Path(args.kg)
        kg = load_kg(kg_path)
        facts = load_facts(Path(args.facts))
        kg, merge_stats = merge_facts(kg, facts)
        if args.dry_run:
            print(
                f"KG dry-run ok: inserted={merge_stats['inserted']} "
                f"updated={merge_stats['updated']} entries={len(kg['entries'])}"
            )
            return 0
        write_kg(kg_path, kg)
        render_stats = render_kg(kg_path, Path(args.out_dir), prune=args.prune)
        print(
            f"KG update ok: inserted={merge_stats['inserted']} updated={merge_stats['updated']} "
            f"entries={len(kg['entries'])} rendered_facts={render_stats['facts']} "
            f"rendered_entities={render_stats['entities']}"
        )
        return 0
    except (KGUpdateError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"update_kg error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
