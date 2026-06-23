#!/usr/bin/env python3
"""
阶段4：合并第一本与第二本 QLoRA 训练样本。

只新增 data/processed/merged_train_samples.jsonl，不覆盖来源文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


CONTENT_UNLABELED = "unlabeled"
NOVEL2_SECTIONS = {"main", "extras", "vol4"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_labels(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for label in labels:
        sample_id = label.get("sample_id")
        if not sample_id:
            raise ValueError("novel2 label missing sample_id")
        if sample_id in by_id:
            raise ValueError(f"duplicate novel2 label sample_id: {sample_id}")
        by_id[sample_id] = label
    return by_id


def add_novel1_trace(sample: dict[str, Any], index: int, merged_index: int) -> dict[str, Any]:
    row = deepcopy(sample)
    row.update(
        {
            "merged_sample_id": f"merged_{merged_index:06d}",
            "source_book": "novel1",
            "source_sample_id": f"novel1_{index:06d}",
            "source_section": None,
            "source_section_confidence": None,
            "content_sensitivity": CONTENT_UNLABELED,
            "content_sensitivity_confidence": 0.0,
        }
    )
    return row


def add_novel2_trace(
    sample: dict[str, Any],
    label: dict[str, Any],
    merged_index: int,
) -> dict[str, Any]:
    section = label.get("source_section")
    if section not in NOVEL2_SECTIONS:
        raise ValueError(f"invalid novel2 source_section: {section!r}")
    if sample.get("source_section") != section:
        raise ValueError(f"novel2 source_section mismatch for {sample.get('sample_id')}")

    row = deepcopy(sample)
    row.update(
        {
            "merged_sample_id": f"merged_{merged_index:06d}",
            "source_book": "novel2",
            "source_sample_id": sample["sample_id"],
            "source_section": section,
            "source_section_confidence": label.get("source_section_confidence"),
            "content_sensitivity": label.get("content_sensitivity"),
            "content_sensitivity_confidence": label.get("content_sensitivity_confidence"),
        }
    )
    return row


def merge_samples(
    novel1: list[dict[str, Any]],
    novel2: list[dict[str, Any]],
    novel2_labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    label_by_id = index_labels(novel2_labels)
    novel2_ids = {sample.get("sample_id") for sample in novel2}
    if novel2_ids != set(label_by_id):
        missing = novel2_ids.symmetric_difference(set(label_by_id))
        raise ValueError(f"novel2 sample/label id mismatch: {sorted(missing)[:5]}")

    merged: list[dict[str, Any]] = []
    for i, sample in enumerate(novel1, 1):
        merged.append(add_novel1_trace(sample, i, len(merged) + 1))
    for sample in novel2:
        sample_id = sample.get("sample_id")
        if not sample_id:
            raise ValueError("novel2 sample missing sample_id")
        merged.append(add_novel2_trace(sample, label_by_id[sample_id], len(merged) + 1))
    return merged


def validate_merged(merged: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    source_book_counts = Counter(row.get("source_book") for row in merged)
    source_section_counts = Counter(row.get("source_section") for row in merged)
    content_counts = Counter(row.get("content_sensitivity") for row in merged)

    seen_ids: set[str] = set()
    for row in merged:
        merged_id = row.get("merged_sample_id")
        if not merged_id or merged_id in seen_ids:
            errors.append(f"bad merged_sample_id: {merged_id!r}")
        seen_ids.add(merged_id)

        if row.get("source_book") not in {"novel1", "novel2"}:
            errors.append(f"bad source_book: {row.get('source_book')!r}")
        if row.get("source_book") == "novel2" and row.get("source_section") not in NOVEL2_SECTIONS:
            errors.append(f"bad novel2 source_section: {row.get('source_section')!r}")
        if row.get("source_book") == "novel1" and row.get("source_section") is not None:
            errors.append("novel1 source_section should be null")

        for key in ("messages", "completion", "target_chapter", "max_chapter_used"):
            if key not in row:
                errors.append(f"{row.get('merged_sample_id')}: missing {key}")
        if not str(row.get("completion", "")).strip():
            errors.append(f"{row.get('merged_sample_id')}: empty completion")

        messages = row.get("messages")
        roles = [m.get("role") for m in messages] if isinstance(messages, list) else []
        if not {"system", "user"}.issubset(set(roles)):
            errors.append(f"{row.get('merged_sample_id')}: missing system/user messages")

        for key in (
            "merged_sample_id",
            "source_book",
            "source_sample_id",
            "source_section",
            "source_section_confidence",
            "content_sensitivity",
            "content_sensitivity_confidence",
        ):
            if key not in row:
                errors.append(f"{row.get('merged_sample_id')}: missing trace field {key}")

    return {
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors[:20],
        "total_samples": len(merged),
        "source_book": dict(source_book_counts),
        "source_section": {str(k): v for k, v in source_section_counts.items()},
        "content_sensitivity": dict(content_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge novel1 and novel2 train samples")
    parser.add_argument("--novel1", default="data/processed/train_samples.jsonl")
    parser.add_argument("--novel2", default="data/processed/novel2_samples.jsonl")
    parser.add_argument("--novel2-labels", default="data/processed/novel2_labels.jsonl")
    parser.add_argument("--output", default="data/processed/merged_train_samples.jsonl")
    args = parser.parse_args()

    novel1_path = Path(args.novel1)
    novel2_path = Path(args.novel2)
    labels_path = Path(args.novel2_labels)
    output_path = Path(args.output)

    novel1 = load_jsonl(novel1_path)
    novel2 = load_jsonl(novel2_path)
    labels = load_jsonl(labels_path)
    merged = merge_samples(novel1, novel2, labels)
    validation = validate_merged(merged)
    if not validation["ok"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    write_jsonl(output_path, merged)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Wrote {len(merged)} samples -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
