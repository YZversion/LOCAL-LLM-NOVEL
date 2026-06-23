#!/usr/bin/env python3
"""
阶段4：第二本素材的切分与打标。

本脚本只处理 data/raw/novel2_raw.txt，不修改第一本 train_samples。

Outputs:
    data/processed/novel2_samples.jsonl
    data/processed/novel2_labels.jsonl
    data/processed/novel2_samples.meta.json

Label file deliberately contains no source text snippets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cowriter.prompts import build_prompt


MAIN_END = 910_812
EXTRAS_END = 1_099_487

CONTENT_LABELS = (
    "general",
    "mature_nonexplicit",
    "explicit_sensitive",
    "violence_sensitive",
    "mixed_or_unclear",
)

SOURCE_SECTIONS = ("main", "extras", "vol4")

SECTION_PATTERNS = {
    "main": re.compile(r"(?m)^[ \t\u3000]*(第\s*\d+\s*章[^\n\r]*)[ \t\u3000]*$"),
    "extras": re.compile(r"(?m)^[ \t\u3000]*(番外\s*第\s*\d+\s*章[^\n\r]*)[ \t\u3000]*$"),
    "vol4": re.compile(
        r"(?m)^[ \t\u3000]*(第\s*[零〇一二三四五六七八九十百千万两]+\s*章[^\n\r]*)[ \t\u3000]*$"
    ),
}

SECTION_RANGES = {
    "main": (0, MAIN_END),
    "extras": (MAIN_END, EXTRAS_END),
    "vol4": (EXTRAS_END, None),
}


@dataclass(frozen=True)
class ChapterSlice:
    section: str
    section_index: int
    heading: str
    heading_start: int
    body_start: int
    body_end: int


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_text(path: Path) -> str:
    # Text mode intentionally normalizes CRLF to LF. The known offsets for
    # novel2_raw.txt were measured with LF-equivalent line accounting.
    return path.read_text(encoding="utf-8")


def source_section_for_offset(offset: int) -> str:
    if offset < MAIN_END:
        return "main"
    if offset < EXTRAS_END:
        return "extras"
    return "vol4"


def left_trim_len(text: str) -> int:
    return len(text) - len(text.lstrip())


def parse_section_chapters(text: str, section: str) -> list[ChapterSlice]:
    start, maybe_end = SECTION_RANGES[section]
    end = len(text) if maybe_end is None else maybe_end
    section_text = text[start:end]
    matches = list(SECTION_PATTERNS[section].finditer(section_text))

    chapters: list[ChapterSlice] = []
    for i, match in enumerate(matches):
        body_start = start + match.end()
        body_end = start + (matches[i + 1].start() if i + 1 < len(matches) else len(section_text))
        chapters.append(
            ChapterSlice(
                section=section,
                section_index=i + 1,
                heading=match.group(1).strip(),
                heading_start=start + match.start(),
                body_start=body_start,
                body_end=body_end,
            )
        )
    return chapters


def section_prelude(text: str, section: str, first_heading_start: int) -> str:
    section_start, _ = SECTION_RANGES[section]
    return text[section_start:first_heading_start].strip()


def build_samples(
    text: str,
    *,
    context_chars: int,
    completion_chars: int,
    min_completion_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    parse_counts: dict[str, int] = {}
    skipped = Counter()

    for section in SOURCE_SECTIONS:
        chapters = parse_section_chapters(text, section)
        parse_counts[section] = len(chapters)
        if not chapters:
            continue

        for i, chapter in enumerate(chapters):
            if section == "extras" and i == 0:
                skipped["extras_first_without_same_section_context"] += 1
                continue

            raw_body = text[chapter.body_start:chapter.body_end]
            body_left = left_trim_len(raw_body)
            body = raw_body.strip()
            completion_start = chapter.body_start + body_left
            completion_end = min(completion_start + completion_chars, chapter.body_end)
            completion = body[:completion_chars]

            if len(completion) < min_completion_chars:
                skipped[f"{section}_completion_lt_min"] += 1
                continue
            if source_section_for_offset(completion_start) != section:
                skipped[f"{section}_completion_start_mismatch"] += 1
                continue
            if source_section_for_offset(max(completion_end - 1, completion_start)) != section:
                skipped[f"{section}_completion_end_mismatch"] += 1
                continue

            if i == 0:
                context_source = section_prelude(text, section, chapter.heading_start)
            else:
                prev = chapters[i - 1]
                context_source = text[prev.body_start:prev.body_end].strip()
            context = context_source[-context_chars:] if len(context_source) > context_chars else context_source
            if not context.strip():
                skipped[f"{section}_empty_context"] += 1
                continue

            sample_id = f"novel2_{section}_{len(samples) + 1:06d}"
            messages = build_prompt(
                recent_text=context,
                summary="",
                retrieval={"bible": [], "grep": []},
                instruction=chapter.heading,
                prior_summary="",
            )
            samples.append(
                {
                    "sample_id": sample_id,
                    "source_file": "novel2_raw.txt",
                    "source_section": section,
                    "target_chapter": len(samples) + 1,
                    "max_chapter_used": len(samples),
                    "section_index": chapter.section_index,
                    "heading": chapter.heading,
                    "messages": messages,
                    "completion": completion,
                    "_meta": {
                        "context_chars": context_chars,
                        "actual_context_chars": len(context),
                        "completion_chars": completion_chars,
                        "actual_completion_chars": len(completion),
                        "min_completion_chars": min_completion_chars,
                        "heading_start_char": chapter.heading_start,
                        "completion_start_char": completion_start,
                        "completion_end_char": completion_start + len(completion),
                        "source_section": section,
                    },
                }
            )

    meta = {
        "parsed_chapter_markers": parse_counts,
        "skipped": dict(skipped),
    }
    return samples, meta


def ollama_generate_json(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "prompt": prompt,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 48,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_classifier_response(content: str) -> tuple[str, float]:
    stripped = content.strip()
    try:
        data = json.loads(stripped)
        label = str(data.get("label", "")).strip()
        confidence = float(data.get("confidence", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        label = ""
        confidence = 0.0
        for candidate in CONTENT_LABELS:
            if candidate in stripped:
                label = candidate
                confidence = 0.5
                break

    if label not in CONTENT_LABELS:
        raise ValueError(f"invalid content_sensitivity label: {label!r}")
    return label, max(0.0, min(1.0, confidence))


def classify_content_sensitivity(
    sample: dict[str, Any],
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text_parts = []
    for message in sample["messages"]:
        if message["role"] == "user":
            text_parts.append(message["content"])
    text_parts.append(sample["completion"])
    sample_text = "\n\n".join(text_parts)

    prompt = (
        "You are a dataset safety labeler. Classify Chinese fiction text into exactly one enum.\n"
        "Return only compact JSON with keys label and confidence. Do not think, explain, quote, or summarize.\n\n"
        "Allowed labels:\n"
        "- general: no notable mature, explicit, or violent material\n"
        "- mature_nonexplicit: adult themes or sensual/romantic material without explicit sexual detail\n"
        "- explicit_sensitive: explicit sexual content or pornographic detail\n"
        "- violence_sensitive: graphic violence, coercion, or serious physical harm is dominant\n"
        "- mixed_or_unclear: multiple categories or insufficient confidence\n\n"
        'Return exactly this shape: {"label":"one_allowed_label","confidence":0.0}\n\n'
        f"Text:\n{sample_text}"
    )

    attempts = 0
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            response = ollama_generate_json(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout_s=timeout_s,
            )
            content = response.get("response", "")
            label, confidence = parse_classifier_response(content)
            return (
                {
                    "sample_id": sample["sample_id"],
                    "source_section": sample["source_section"],
                    "source_section_confidence": 1.0,
                    "content_sensitivity": label,
                    "content_sensitivity_confidence": confidence,
                    "ollama_attempts": attempts,
                    "ollama_failed": False,
                },
                {"attempts": attempts, "failed": False, "errors": errors},
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            errors.append(type(exc).__name__)
            if attempt < max_retries:
                time.sleep(min(2.0 * (attempt + 1), 5.0))

    return (
        {
            "sample_id": sample["sample_id"],
            "source_section": sample["source_section"],
            "source_section_confidence": 1.0,
            "content_sensitivity": "mixed_or_unclear",
            "content_sensitivity_confidence": 0.0,
            "ollama_attempts": attempts,
            "ollama_failed": True,
        },
        {"attempts": attempts, "failed": True, "errors": errors},
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_outputs(samples: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    errors: list[str] = []

    if len(sample_by_id) != len(samples):
        errors.append("duplicate sample_id in samples")
    if len(labels) != len(samples):
        errors.append("label/sample count mismatch")

    forbidden_label_fields = {"context", "completion", "messages", "heading", "text", "raw_text"}
    for label in labels:
        sid = label.get("sample_id")
        sample = sample_by_id.get(sid)
        if sample is None:
            errors.append(f"label sample_id not found: {sid}")
            continue
        if forbidden_label_fields.intersection(label):
            errors.append(f"label contains forbidden text-bearing field: {sid}")
        if label.get("source_section") not in SOURCE_SECTIONS:
            errors.append(f"invalid source_section: {sid}")
        if label.get("source_section") != sample.get("source_section"):
            errors.append(f"source_section mismatch: {sid}")
        if label.get("content_sensitivity") not in CONTENT_LABELS:
            errors.append(f"invalid content_sensitivity: {sid}")
        for key in ("source_section_confidence", "content_sensitivity_confidence"):
            value = label.get(key)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                errors.append(f"invalid confidence {key}: {sid}")
        offset = sample["_meta"]["completion_start_char"]
        if source_section_for_offset(offset) != label.get("source_section"):
            errors.append(f"offset/source_section mismatch: {sid}")

    return {
        "ok": not errors,
        "errors": errors[:20],
        "error_count": len(errors),
        "checked_samples": len(samples),
        "checked_labels": len(labels),
    }


def distribution_tables(labels: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(label["source_section"] for label in labels)
    content_counts = Counter(label["content_sensitivity"] for label in labels)
    cross: dict[str, dict[str, int]] = {
        section: {content: 0 for content in CONTENT_LABELS}
        for section in SOURCE_SECTIONS
    }
    for label in labels:
        cross[label["source_section"]][label["content_sensitivity"]] += 1
    return {
        "source_section": {section: source_counts.get(section, 0) for section in SOURCE_SECTIONS},
        "content_sensitivity": {content: content_counts.get(content, 0) for content in CONTENT_LABELS},
        "cross": cross,
    }


def print_report(meta: dict[str, Any]) -> None:
    total = meta["total_samples"]
    distributions = meta["distributions"]
    print("\n=== novel2 split + labels ===")
    print(f"total_samples: {total}")
    print("\nsource_section:")
    for section, count in distributions["source_section"].items():
        print(f"  {section}: {count}")
    print("\ncontent_sensitivity:")
    for label, count in distributions["content_sensitivity"].items():
        pct = (count * 100.0 / total) if total else 0.0
        print(f"  {label}: {count} ({pct:.2f}%)")
    print("\ncross:")
    header = ["source_section", *CONTENT_LABELS]
    print("\t".join(header))
    for section in SOURCE_SECTIONS:
        row = [section] + [str(distributions["cross"][section][label]) for label in CONTENT_LABELS]
        print("\t".join(row))

    ollama = meta["ollama"]
    print("\nollama:")
    print(f"  failed: {ollama['failed_calls']} / {ollama['total_calls']}")
    print(f"  failure_rate: {ollama['failure_rate_pct']:.2f}%")
    print(f"  retried_samples: {ollama['retried_samples']}")
    print(f"  total_attempts: {ollama['total_attempts']}")

    validation = meta["validation"]
    print("\nvalidation:")
    print(f"  ok: {validation['ok']}")
    print(f"  checked_samples: {validation['checked_samples']}")
    print(f"  checked_labels: {validation['checked_labels']}")
    print(f"  error_count: {validation['error_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and label novel2 training samples")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw", default="data/raw/novel2_raw.txt")
    parser.add_argument("--samples-output", default="data/processed/novel2_samples.jsonl")
    parser.add_argument("--labels-output", default="data/processed/novel2_labels.jsonl")
    parser.add_argument("--meta-output", default="data/processed/novel2_samples.meta.json")
    parser.add_argument("--context-chars", type=int, default=1000)
    parser.add_argument("--completion-chars", type=int, default=250)
    parser.add_argument("--min-completion-chars", type=int, default=150)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    parser.add_argument("--ollama-retries", type=int, default=2)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    model = args.ollama_model or cfg["model"]["ollama_model"]
    base_url = args.ollama_base_url or cfg["model"]["ollama_base_url"]

    raw_path = Path(args.raw)
    text = normalize_text(raw_path)
    if len(text) < EXTRAS_END:
        print(f"ERROR: raw text shorter than expected section boundary: {raw_path}", file=sys.stderr)
        return 1

    samples, split_meta = build_samples(
        text,
        context_chars=args.context_chars,
        completion_chars=args.completion_chars,
        min_completion_chars=args.min_completion_chars,
    )
    write_jsonl(Path(args.samples_output), samples)

    labels: list[dict[str, Any]] = []
    call_stats = []
    for sample in tqdm(samples, desc="content_sensitivity", unit="sample"):
        label, stats = classify_content_sensitivity(
            sample,
            base_url=base_url,
            model=model,
            timeout_s=args.ollama_timeout,
            max_retries=args.ollama_retries,
        )
        labels.append(label)
        call_stats.append(stats)
    write_jsonl(Path(args.labels_output), labels)

    validation = validate_outputs(samples, labels)
    distributions = distribution_tables(labels)

    total_calls = len(call_stats)
    failed_calls = sum(1 for stat in call_stats if stat["failed"])
    retried_samples = sum(1 for stat in call_stats if stat["attempts"] > 1)
    total_attempts = sum(int(stat["attempts"]) for stat in call_stats)
    errors = Counter(error for stat in call_stats for error in stat["errors"])

    meta = {
        "raw_file": str(raw_path),
        "samples_output": args.samples_output,
        "labels_output": args.labels_output,
        "total_samples": len(samples),
        "params": {
            "context_chars": args.context_chars,
            "completion_chars": args.completion_chars,
            "min_completion_chars": args.min_completion_chars,
            "section_boundaries": {
                "main": [0, MAIN_END],
                "extras": [MAIN_END, EXTRAS_END],
                "vol4": [EXTRAS_END, len(text)],
            },
        },
        "split": split_meta,
        "ollama": {
            "model": model,
            "base_url": base_url,
            "max_retries": args.ollama_retries,
            "timeout_s": args.ollama_timeout,
            "total_calls": total_calls,
            "failed_calls": failed_calls,
            "failure_rate_pct": (failed_calls * 100.0 / total_calls) if total_calls else 0.0,
            "retried_samples": retried_samples,
            "total_attempts": total_attempts,
            "error_types": dict(errors),
        },
        "distributions": distributions,
        "validation": validation,
    }
    Path(args.meta_output).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(meta)

    if not validation["ok"]:
        print("ERROR: validation failed; see meta validation errors", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
