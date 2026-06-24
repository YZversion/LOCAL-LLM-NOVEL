#!/usr/bin/env python3
"""Create an editable System B facts draft from accepted chapter text.

This MVP does not pretend to solve extraction.  It writes a conservative draft
JSON that the author can edit before running update_kg.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _excerpt(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an editable System B facts draft.")
    parser.add_argument("--chapter", required=True, type=int, help="Accepted chapter number.")
    parser.add_argument("--input", required=True, help="Accepted chapter text path.")
    parser.add_argument("--out", required=True, help="Output draft facts JSON path.")
    parser.add_argument("--title", default="", help="Optional event title.")
    parser.add_argument("--entities", default="", help="Comma-separated entities to seed the draft.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        print(f"kg_extract error: input not found: {src}")
        return 2
    text = src.read_text(encoding="utf-8").strip()
    entities = [e.strip() for e in re.split(r"[,，、]", args.entities) if e.strip()]
    title = args.title.strip() or f"第{args.chapter}章新增事件"
    draft = {
        "schema_version": "system_b.facts_draft.v1",
        "chapter": args.chapter,
        "notes": [
            "Edit this file before update_kg.py.",
            "Split state changes into separate facts; do not overwrite old facts.",
        ],
        "facts": [
            {
                "type": "event",
                "title": title,
                "summary": _excerpt(text),
                "chapter": args.chapter,
                "entities": entities,
                "valid_from": args.chapter,
                "valid_to": None,
                "confidence": 0.3,
                "status": "needs_human_review",
                "evidence": [
                    {
                        "chapter": args.chapter,
                        "quote": _excerpt(text, 80),
                    }
                ],
            }
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote editable facts draft: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
