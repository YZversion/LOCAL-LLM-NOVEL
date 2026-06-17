#!/usr/bin/env python3
"""Produce a text-free sanitized metrics file from a full eval_style JSON report.
Keeps all numeric metrics and risk levels; drops every field that embeds text excerpts.
Usage: python _sanitize_baseline.py <full_report.json> <sanitized_output.json>"""
import json
import sys
from pathlib import Path


def sanitize_char_ngrams(ngrams: dict) -> dict:
    """Keep only counts per n-gram rank, not the gram text itself."""
    result = {}
    for n, items in ngrams.items():
        result[n] = [item["count"] for item in items]
    return result


def sanitize_repetition(rep: dict) -> dict:
    near_dup = rep.get("near_duplicate_adjacent_sentences", {})
    loops = rep.get("short_sentence_loops", {})
    return {
        "duplicate_line_count": rep["duplicate_line_count"],
        "duplicate_line_ratio": rep["duplicate_line_ratio"],
        "duplicate_paragraph_count": rep["duplicate_paragraph_count"],
        "duplicate_paragraph_ratio": rep["duplicate_paragraph_ratio"],
        "consecutive_repeated_sentence_count": rep["consecutive_repeated_sentence_count"],
        "longest_repeated_sentence_run": rep["longest_repeated_sentence_run"],
        "near_duplicate_adjacent_sentences": {
            "count": near_dup.get("count", 0),
        },
        "char_ngrams": sanitize_char_ngrams(rep.get("char_ngrams", {})),
        "short_sentence_loops": {
            "detected": loops.get("detected", False),
            "count": loops.get("count", 0),
        },
        "repetition_risk": rep["repetition_risk"],
    }


def sanitize_contamination(con: dict) -> dict:
    exact = con.get("exact_sentence_overlap", {})
    normalized = con.get("normalized_sentence_overlap", {})
    near = con.get("near_sentence_overlap", {})
    shingles = con.get("char_shingle_overlap", {})
    longest = con.get("longest_common_substring", {})
    paragraphs = con.get("paragraph_overlap", {})
    return {
        "overlapping_sentence_count": con["overlapping_sentence_count"],
        "overlapping_sentence_ratio": con["overlapping_sentence_ratio"],
        "longest_common_substring_length": con["longest_common_substring_length"],
        "contamination_risk": con["contamination_risk"],
        "exact_sentence_overlap": {
            "count": exact.get("count", 0),
            "ratio": exact.get("ratio", 0.0),
        },
        "normalized_sentence_overlap": {
            "count": normalized.get("count", 0),
            "ratio": normalized.get("ratio", 0.0),
        },
        "near_sentence_overlap": {
            "count": near.get("count", 0),
            "ratio": near.get("ratio", 0.0),
        },
        "char_shingle_overlap": shingles,
        "longest_common_substring": {
            "length": longest.get("length", 0),
        },
        "paragraph_overlap": {
            "exact_count": paragraphs.get("exact_count", 0),
            "normalized_count": paragraphs.get("normalized_count", 0),
        },
    }


def sanitize_style_score(score: dict) -> dict:
    components = {}
    for name, item in score.get("components", {}).items():
        components[name] = {"score": item["score"], "weight": item["weight"]}
    return {
        "overall": score["overall"],
        "level": score["level"],
        "components": components,
    }


def sanitize(report: dict, model_name: str, config_note: str) -> dict:
    meta = dict(report["meta"])
    meta["baseline_model"] = model_name
    meta["config_note"] = config_note
    meta["sanitization"] = "text-bearing example fields removed; numeric metrics retained"

    inputs = report["inputs"]
    sanitized_inputs = {
        "reference": inputs["reference"],
        "candidate": inputs["candidate"],
        "reference_exists": inputs["reference_exists"],
        "candidate_exists": inputs["candidate_exists"],
        "reference_size_bytes": inputs["reference_size_bytes"],
        "candidate_size_bytes": inputs["candidate_size_bytes"],
    }

    return {
        "meta": meta,
        "inputs": sanitized_inputs,
        "reference_stats": report["reference_stats"],
        "candidate_stats": report["candidate_stats"],
        "segmentation": report["segmentation"],
        "repetition": sanitize_repetition(report["repetition"]),
        "contamination": sanitize_contamination(report["contamination"]),
        "diff": report["diff"],
        "style_score": sanitize_style_score(report["style_score"]),
        "summary": {
            "overall_warning": report["summary"]["overall_warning"],
            "notes": report["summary"]["notes"],
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <full_report.json> <sanitized_output.json>", file=sys.stderr)
        return 1
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    report = json.loads(src.read_text(encoding="utf-8"))

    model_name = "huihui_ai/qwen3-abliterated:8b-v2"
    config_note = "config.yaml generation params as of 2026-06-17: temperature=0.8 top_p=0.8 top_k=20 repeat_penalty=1.15 repeat_last_n=256 presence_penalty=0.3 frequency_penalty=0.3 dry_multiplier=0.8 dry_base=1.75 dry_allowed_length=2 dry_penalty_last_n=512 output_tokens=600"

    sanitized = sanitize(report, model_name, config_note)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sanitized metrics written to: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
