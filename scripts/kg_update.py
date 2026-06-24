#!/usr/bin/env python3
"""Merge reviewed System B facts into a durable kg.json file.

The extractor is intentionally outside this merge step.  This script accepts
plain JSON facts that a human can inspect and edit before they become memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "system_b.v1"
FACT_TYPES = {
    "event",
    "character_state",
    "relationship_delta",
    "location_state",
    "plot_thread",
}


class KGUpdateError(Exception):
    """Raised for user-facing KG merge errors."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(fact: dict[str, Any]) -> str:
    chapter = fact.get("chapter") or fact.get("valid_from") or "na"
    digest = hashlib.sha1(_compact({
        "type": fact.get("type"),
        "title": fact.get("title"),
        "summary": fact.get("summary"),
        "entities": fact.get("entities", []),
        "chapter": chapter,
    }).encode("utf-8")).hexdigest()[:12]
    return f"{fact.get('type', 'fact')}_ch{chapter}_{digest}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_facts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise KGUpdateError(f"Facts file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        facts = data
    elif isinstance(data, dict):
        facts = data.get("facts") or data.get("entries") or []
    else:
        raise KGUpdateError("Facts JSON must be an object or list.")
    if not isinstance(facts, list):
        raise KGUpdateError("facts must be a list.")
    out: list[dict[str, Any]] = []
    for i, fact in enumerate(facts, 1):
        if not isinstance(fact, dict):
            raise KGUpdateError(f"Fact #{i} is not an object.")
        out.append(fact)
    return out


def load_kg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "entries": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise KGUpdateError("kg.json must be a JSON object.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise KGUpdateError(
            f"Unsupported kg schema: {data.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )
    data.setdefault("entries", [])
    if not isinstance(data["entries"], list):
        raise KGUpdateError("kg.entries must be a list.")
    return data


def normalize_fact(raw: dict[str, Any]) -> dict[str, Any]:
    fact_type = raw.get("type")
    if fact_type not in FACT_TYPES:
        raise KGUpdateError(f"Unsupported fact type: {fact_type!r}")

    summary = str(raw.get("summary") or "").strip()
    if not summary:
        raise KGUpdateError("Fact summary is required.")

    chapter = raw.get("chapter")
    if chapter is not None and not isinstance(chapter, int):
        raise KGUpdateError("chapter must be an integer when provided.")

    valid_from = raw.get("valid_from", chapter)
    if not isinstance(valid_from, int):
        raise KGUpdateError("valid_from is required and must be an integer.")
    revealed_in = raw.get("revealed_in", valid_from)
    if not isinstance(revealed_in, int):
        raise KGUpdateError("revealed_in must be an integer.")
    valid_to = raw.get("valid_to")
    if valid_to is not None and not isinstance(valid_to, int):
        raise KGUpdateError("valid_to must be null or an integer.")

    entities = [str(e).strip() for e in _as_list(raw.get("entities")) if str(e).strip()]
    evidence_items = []
    for item in _as_list(raw.get("evidence")):
        if isinstance(item, dict):
            evidence_items.append(item)
        elif item:
            evidence_items.append({"note": str(item)})

    fact = {
        "id": str(raw.get("id") or "").strip(),
        "type": fact_type,
        "title": str(raw.get("title") or "").strip() or (entities[0] if entities else fact_type),
        "summary": summary,
        "chapter": chapter,
        "entities": entities,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "revealed_in": revealed_in,
        "confidence": float(raw.get("confidence", 0.7)),
        "evidence": evidence_items,
        "created_at": raw.get("created_at") or _now(),
        "updated_at": _now(),
    }
    if not fact["id"]:
        fact["id"] = _stable_id(fact)
    for key in ("status", "supersedes", "notes"):
        if key in raw:
            fact[key] = raw[key]
    return fact


def merge_facts(kg: dict[str, Any], facts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    existing = {entry.get("id"): i for i, entry in enumerate(kg.get("entries", []))}
    stats = {"inserted": 0, "updated": 0}
    for raw in facts:
        fact = normalize_fact(raw)
        idx = existing.get(fact["id"])
        if idx is None:
            kg["entries"].append(fact)
            existing[fact["id"]] = len(kg["entries"]) - 1
            stats["inserted"] += 1
        else:
            # Idempotent reruns update the same exact fact id. State changes
            # should be represented as new facts, not by reusing old ids.
            old_created = kg["entries"][idx].get("created_at")
            fact["created_at"] = old_created or fact["created_at"]
            kg["entries"][idx] = fact
            stats["updated"] += 1
    kg["updated_at"] = _now()
    return kg, stats


def write_kg(path: Path, kg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed System B facts into kg.json.")
    parser.add_argument("--facts", required=True, help="Reviewed facts JSON file.")
    parser.add_argument("--kg", default="data/story_bible/kg.json", help="KG JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print stats without writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        kg_path = Path(args.kg)
        kg = load_kg(kg_path)
        facts = load_facts(Path(args.facts))
        kg, stats = merge_facts(kg, facts)
        if not args.dry_run:
            write_kg(kg_path, kg)
        print(
            f"KG merge ok: inserted={stats['inserted']} updated={stats['updated']} "
            f"entries={len(kg['entries'])} dry_run={args.dry_run}"
        )
        return 0
    except (KGUpdateError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"kg_update error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
