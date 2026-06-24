#!/usr/bin/env python3
"""Render System B kg.json into retrievable Markdown memory cards."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kg_update import KGUpdateError, SCHEMA_VERSION, load_kg


def _slug(value: str, fallback: str = "item") -> str:
    value = value.strip() or fallback
    value = re.sub(r"[\\/:*?\"<>|\s]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:80] or fallback


def _yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _entry_body(entry: dict[str, Any]) -> str:
    lines = [
        f"# {entry.get('title') or entry.get('id')}",
        "",
        f"- 类型: {entry.get('type')}",
        f"- 章节: {entry.get('chapter')}",
        f"- 有效期: {entry.get('valid_from')} -> {entry.get('valid_to')}",
        f"- 置信度: {entry.get('confidence')}",
    ]
    entities = entry.get("entities") or []
    if entities:
        lines.append(f"- 相关实体: {'、'.join(entities)}")
    lines.extend(["", "## 记忆", "", str(entry.get("summary") or "").strip()])
    evidence = entry.get("evidence") or []
    if evidence:
        lines.extend(["", "## Evidence"])
        for item in evidence:
            if isinstance(item, dict):
                ch = item.get("chapter", entry.get("chapter"))
                quote = item.get("quote") or item.get("source_offset") or item.get("note") or ""
                lines.append(f"- ch{ch}: {quote}")
            else:
                lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def _entry_meta(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": entry.get("title") or entry.get("id"),
        "type": f"system_b_{entry.get('type')}",
        "revealed_in": entry.get("revealed_in", entry.get("valid_from")),
        "valid_from": entry.get("valid_from"),
        "valid_to": entry.get("valid_to"),
        "system_b": True,
        "kg_id": entry.get("id"),
    }


def _entity_body(entity: str, entries: list[dict[str, Any]]) -> str:
    lines = [f"# {entity}", "", "System B entity memory.", ""]
    for entry in sorted(entries, key=lambda e: (e.get("valid_from") or 0, e.get("id") or "")):
        ch = entry.get("chapter")
        fact_type = entry.get("type")
        summary = str(entry.get("summary") or "").strip()
        lines.append(f"- ch{ch} [{fact_type}] {summary}")
        evidence = entry.get("evidence") or []
        if evidence:
            first = evidence[0]
            if isinstance(first, dict):
                quote = first.get("quote") or first.get("note") or first.get("source_offset")
                if quote:
                    lines.append(f"  - evidence: {quote}")
    return "\n".join(lines).strip() + "\n"


def render_kg(kg_path: Path, out_dir: Path, prune: bool = False) -> dict[str, int]:
    kg = load_kg(kg_path)
    if kg.get("schema_version") != SCHEMA_VERSION:
        raise KGUpdateError(f"Unsupported kg schema: {kg.get('schema_version')!r}")

    facts_dir = out_dir / "facts"
    entities_dir = out_dir / "entities"
    facts_dir.mkdir(parents=True, exist_ok=True)
    entities_dir.mkdir(parents=True, exist_ok=True)

    if prune:
        for old in out_dir.rglob("*.md"):
            old.unlink()

    written_facts = 0
    entity_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in kg.get("entries", []):
        entry_id = str(entry.get("id") or "")
        if not entry_id:
            continue
        card = facts_dir / f"{_slug(entry_id)}.md"
        card.write_text(_frontmatter(_entry_meta(entry)) + _entry_body(entry), encoding="utf-8")
        written_facts += 1
        for entity in entry.get("entities") or []:
            entity = str(entity).strip()
            if entity:
                entity_map[entity].append(entry)

    written_entities = 0
    for entity, entries in entity_map.items():
        valid_from = min(e.get("valid_from") for e in entries if isinstance(e.get("valid_from"), int))
        valid_to_values = [e.get("valid_to") for e in entries]
        meta = {
            "title": entity,
            "type": "system_b_entity",
            "revealed_in": valid_from,
            "valid_from": valid_from,
            "valid_to": None if any(v is None for v in valid_to_values) else max(valid_to_values),
            "system_b": True,
        }
        card = entities_dir / f"{_slug(entity)}.md"
        card.write_text(_frontmatter(meta) + _entity_body(entity, entries), encoding="utf-8")
        written_entities += 1

    return {"facts": written_facts, "entities": written_entities}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render System B kg.json to Markdown cards.")
    parser.add_argument("--kg", default="data/story_bible/kg.json", help="KG JSON path.")
    parser.add_argument(
        "--out-dir",
        default="data/story_bible/generated/system_b",
        help="Directory for rendered Markdown cards.",
    )
    parser.add_argument("--prune", action="store_true", help="Delete old rendered .md cards first.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stats = render_kg(Path(args.kg), Path(args.out_dir), prune=args.prune)
        print(f"KG render ok: facts={stats['facts']} entities={stats['entities']} out={args.out_dir}")
        return 0
    except (KGUpdateError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"kg_render error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
