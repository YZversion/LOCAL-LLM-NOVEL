#!/usr/bin/env python3
"""Deterministic style evaluation for reference/candidate novel text."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PARAGRAPH_BOUNDARY_RE = re.compile(r"\n[ \t\f\v]*\n(?:[ \t\f\v]*\n)*")
SENTENCE_END_CHARS = set("。！？!?；;.")
ELLIPSIS = "…"
CLOSING_QUOTES = set("”’」』\"')]）】》")
OPENING_QUOTES = ("“", "‘", "「", "『", '"', "'")
DIALOGUE_PAIR_RE = re.compile(r"“[^”]+”|‘[^’]+’|「[^」]+」|『[^』]+』|\"[^\"]+\"|'[^']+'")
DIALOGUE_COLON_QUOTE_RE = re.compile(r"[：:]\s*[“‘「『\"']")
SPEECH_VERB_QUOTE_RE = re.compile(
    r"(?:说|道|问|喊|叫|答|笑道|冷声道|低声道|轻声道|低声问|喃喃道|开口)"
    r"\s*[：:]\s*[“‘「『\"']"
)
PUNCT_CHARS = set(" \t\r\n。！？!?；;，,、：:“”‘’「」『』\"'（）()【】《》….-—")
SHORT_SENTENCE_MAX_CHARS = 12
NEAR_DUPLICATE_THRESHOLD = 0.82
STYLE_WEIGHTS = {
    "length_profile": 20,
    "sentence_profile": 25,
    "paragraph_profile": 15,
    "dialogue_profile": 15,
    "repetition_penalty": 15,
    "contamination_penalty": 10,
}


class EvalStyleError(Exception):
    """Raised for user-facing input/output errors."""


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvalStyleError(f"Input file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise EvalStyleError(f"Input file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise EvalStyleError(f"Could not read input file {path}: {exc}") from exc


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def norm_unit(text: str) -> str:
    return compact(text).strip()


def ratio(part: int | float, total: int | float) -> float:
    return round(float(part) / float(total), 4) if total else 0.0


def metric_ratio(candidate_value: int | float, reference_value: int | float) -> float:
    if not reference_value:
        return 1.0 if not candidate_value else 0.0
    return rounded(float(candidate_value) / float(reference_value))


def rounded(value: float) -> float:
    return round(float(value), 4)


def truncate_text(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def effective_text(text: str) -> str:
    return "".join(ch for ch in compact(text) if ch not in PUNCT_CHARS)


def normalized_text(text: str) -> str:
    return effective_text(text).lower()


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [p.strip() for p in PARAGRAPH_BOUNDARY_RE.split(normalized) if p.strip()]


def _is_decimal_dot(text: str, index: int) -> bool:
    return (
        text[index] == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _ellipsis_run_end(text: str, index: int) -> int:
    end = index
    while end < len(text) and text[end] == ELLIPSIS:
        end += 1
    return end


def _is_ellipsis_sentence_end(text: str, index: int) -> bool:
    end = _ellipsis_run_end(text, index)
    if end >= len(text):
        return True
    return text[end].isspace() or text[end] in CLOSING_QUOTES or text[end] in SENTENCE_END_CHARS


def _is_sentence_end(text: str, index: int) -> bool:
    ch = text[index]
    if ch == ELLIPSIS:
        return _is_ellipsis_sentence_end(text, index)
    if ch == "." and _is_decimal_dot(text, index):
        return False
    return ch in SENTENCE_END_CHARS


def _consume_sentence_tail(text: str, index: int) -> int:
    end = index + 1
    while end < len(text):
        ch = text[end]
        if ch == ELLIPSIS:
            end = _ellipsis_run_end(text, end)
        elif ch in SENTENCE_END_CHARS and not _is_decimal_dot(text, end):
            end += 1
        else:
            break
    while end < len(text) and text[end] in CLOSING_QUOTES:
        end += 1
    return end


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(normalized):
        if _is_sentence_end(normalized, index):
            end = _consume_sentence_tail(normalized, index)
            sentence = normalized[start:end].strip()
            if sentence:
                sentences.append(sentence)
            while end < len(normalized) and normalized[end].isspace():
                end += 1
            start = end
            index = end
            continue
        index += 1
    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def is_dialogue_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return (
        line.startswith(OPENING_QUOTES)
        or bool(DIALOGUE_PAIR_RE.search(line))
        or bool(DIALOGUE_COLON_QUOTE_RE.search(line))
        or bool(SPEECH_VERB_QUOTE_RE.search(line))
    )


def sentence_lengths(sentences: list[str]) -> list[int]:
    return [len(norm_unit(s)) for s in sentences if norm_unit(s)]


def basic_stats(text: str) -> dict[str, Any]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    lengths = sentence_lengths(sentences)
    dialogue_lines = sum(1 for line in non_empty_lines if is_dialogue_line(line))
    return {
        "char_count": len(text),
        "non_whitespace_char_count": len(compact(text)),
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
        "sentence_count": len(lengths),
        "average_sentence_length": rounded(statistics.mean(lengths)) if lengths else 0.0,
        "median_sentence_length": rounded(statistics.median(lengths)) if lengths else 0.0,
        "longest_sentence_length": max(lengths) if lengths else 0,
        "dialogue_line_count": dialogue_lines,
        "dialogue_line_ratio": ratio(dialogue_lines, len(non_empty_lines)),
    }


def segmentation_stats(text: str) -> dict[str, Any]:
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    dialogue_lines = sum(1 for line in lines if line.strip() and is_dialogue_line(line))
    return {
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "dialogue_line_count": dialogue_lines,
        "avg_sentences_per_paragraph": ratio(len(sentences), len(paragraphs)),
    }


def top_char_ngrams(text: str, n: int, limit: int = 10) -> list[dict[str, Any]]:
    body = compact(text)
    if len(body) < n:
        return []
    counts = Counter(body[i : i + n] for i in range(len(body) - n + 1))
    return [
        {"gram": gram, "count": count}
        for gram, count in counts.most_common()
        if count > 1 and any(ch not in PUNCT_CHARS for ch in gram)
    ][:limit]


def duplicate_examples(items: list[str], limit: int = 5, text_limit: int = 80) -> list[dict[str, Any]]:
    counts = Counter(items)
    return [
        {"text": truncate_text(text, text_limit), "count": count}
        for text, count in counts.most_common()
        if count > 1
    ][:limit]


def consecutive_sentence_repeats(sentences: list[str]) -> tuple[int, list[dict[str, Any]], int]:
    count = 0
    examples: list[dict[str, Any]] = []
    longest_run = 1 if sentences else 0
    i = 0
    while i < len(sentences):
        j = i + 1
        while j < len(sentences) and sentences[j] == sentences[i]:
            j += 1
        run_len = j - i
        if run_len > 1:
            count += run_len - 1
            longest_run = max(longest_run, run_len)
            if len(examples) < 5:
                examples.append({"sentence": truncate_text(sentences[i]), "repeat_count": run_len})
        i = j
    return count, examples, longest_run


def sentence_similarity(a: str, b: str) -> float:
    a_eff = effective_text(a)
    b_eff = effective_text(b)
    if len(a_eff) < 6 or len(b_eff) < 6:
        return 0.0
    seq_ratio = SequenceMatcher(None, a_eff, b_eff).ratio()
    overlap = sum((Counter(a_eff) & Counter(b_eff)).values())
    shorter = min(len(a_eff), len(b_eff))
    tolerated_gap = min(2, shorter // 4)
    overlap_ratio = overlap / max(1, shorter - tolerated_gap)
    return rounded(min(1.0, max(seq_ratio, overlap_ratio)))


def near_duplicate_adjacent(sentences: list[str]) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    count = 0
    for left, right in zip(sentences, sentences[1:]):
        similarity = sentence_similarity(left, right)
        if similarity >= NEAR_DUPLICATE_THRESHOLD and left != right:
            count += 1
            if len(examples) < 5:
                examples.append({
                    "sentence_a": truncate_text(left),
                    "sentence_b": truncate_text(right),
                    "similarity": similarity,
                })
    return {"count": count, "examples": examples}


def char_ngram_stats(text: str) -> dict[str, list[dict[str, Any]]]:
    return {str(n): top_char_ngrams(text, n) for n in (2, 3, 4)}


def short_sentence_loops(sentences: list[str]) -> dict[str, Any]:
    short = [s for s in sentences if 0 < len(effective_text(s)) <= SHORT_SENTENCE_MAX_CHARS]
    examples: list[dict[str, Any]] = []
    count = 0

    i = 0
    while i < len(short):
        j = i + 1
        while j < len(short) and short[j] == short[i]:
            j += 1
        run_len = j - i
        if run_len >= 3:
            count += 1
            if len(examples) < 5:
                examples.append({
                    "type": "single_sentence_run",
                    "pattern": [truncate_text(short[i])],
                    "repeat_count": run_len,
                })
        i = max(j, i + 1)

    for i in range(len(short) - 3):
        a, b, c, d = short[i : i + 4]
        if a == c and b == d and a != b:
            count += 1
            if len(examples) < 5:
                examples.append({
                    "type": "abab",
                    "pattern": [truncate_text(a), truncate_text(b)],
                    "repeat_count": 2,
                })

    return {"detected": count > 0, "count": count, "examples": examples}


def repetition_risk_level(duplicate_line_ratio: float, duplicate_paragraph_ratio: float,
                          consecutive_count: int, near_duplicate_count: int,
                          loops: dict[str, Any], char_ngrams: dict[str, list[dict[str, Any]]]) -> str:
    max_2 = char_ngrams["2"][0]["count"] if char_ngrams["2"] else 0
    max_3 = char_ngrams["3"][0]["count"] if char_ngrams["3"] else 0
    max_4 = char_ngrams["4"][0]["count"] if char_ngrams["4"] else 0

    if loops["detected"] or consecutive_count >= 2 or duplicate_paragraph_ratio >= 0.25:
        return "high"
    if near_duplicate_count >= 3 or duplicate_line_ratio >= 0.3:
        return "high"
    if near_duplicate_count > 0 or duplicate_line_ratio >= 0.1 or duplicate_paragraph_ratio > 0:
        return "medium"
    if max_2 >= 12 or max_3 >= 8 or max_4 >= 6:
        return "medium"
    return "low"


def repetition_stats(candidate: str) -> dict[str, Any]:
    lines = [norm_unit(line) for line in candidate.splitlines() if norm_unit(line)]
    line_counts = Counter(lines)
    duplicate_line_count = sum(count - 1 for count in line_counts.values() if count > 1)
    duplicate_line_ratio = ratio(duplicate_line_count, len(lines))

    sentences = [norm_unit(s) for s in split_sentences(candidate) if norm_unit(s)]
    consecutive_count, consecutive_examples, longest_run = consecutive_sentence_repeats(sentences)

    paragraphs = [norm_unit(p) for p in split_paragraphs(candidate) if norm_unit(p)]
    paragraph_counts = Counter(paragraphs)
    duplicate_paragraph_count = sum(count - 1 for count in paragraph_counts.values() if count > 1)
    # Character-weighted ratio: chars in ALL occurrences of duplicate paragraphs (originals + copies)
    # divided by total paragraph chars. More accurately reflects content fraction lost to repetition
    # than the old paragraph-count ratio. Threshold (0.25) retained but needs empirical re-calibration.
    _total_para_chars = sum(len(p) for p in paragraphs)
    _dup_para_chars = sum(len(p) * count for p, count in paragraph_counts.items() if count > 1)
    duplicate_paragraph_ratio = ratio(_dup_para_chars, _total_para_chars)

    ngrams = char_ngram_stats(candidate)
    loops = short_sentence_loops(sentences)
    near_duplicates = near_duplicate_adjacent(sentences)
    risk = repetition_risk_level(
        duplicate_line_ratio,
        duplicate_paragraph_ratio,
        consecutive_count,
        near_duplicates["count"],
        loops,
        ngrams,
    )

    return {
        "duplicate_line_count": duplicate_line_count,
        "duplicate_line_ratio": duplicate_line_ratio,
        "repeated_line_examples": duplicate_examples(lines),
        "duplicate_paragraph_count": duplicate_paragraph_count,
        "duplicate_paragraph_ratio": duplicate_paragraph_ratio,
        "repeated_paragraph_examples": duplicate_examples(paragraphs, text_limit=120),
        "consecutive_repeated_sentence_count": consecutive_count,
        "consecutive_repeated_sentence_examples": consecutive_examples,
        "longest_repeated_sentence_run": longest_run,
        "near_duplicate_adjacent_sentences": near_duplicates,
        "char_ngrams": ngrams,
        "short_sentence_loops": loops,
        "repetition_risk": risk,
        # Backward-compatible aliases from stage 3.1/3.2.
        "repeated_line_count": duplicate_line_count,
        "repeated_line_ratio": duplicate_line_ratio,
        "top_2grams": ngrams["2"],
        "top_3grams": ngrams["3"],
        "obvious_short_sentence_loop": loops["detected"],
        "short_loop_sentence": loops["examples"][0]["pattern"][0] if loops["examples"] else "",
        "short_loop_count": loops["count"],
    }


def candidate_sentence_items(text: str, min_effective_chars: int = 8) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sentence in split_sentences(text):
        stripped = sentence.strip()
        effective = effective_text(stripped)
        if len(effective) < min_effective_chars:
            continue
        items.append({
            "sentence": stripped,
            "exact": norm_unit(stripped),
            "normalized": effective.lower(),
            "length": len(effective),
            "visible_length": len(norm_unit(stripped)),
        })
    return items


def exact_sentence_overlap(reference: str, candidate: str) -> dict[str, Any]:
    reference_items = candidate_sentence_items(reference, 8)
    candidate_items = candidate_sentence_items(candidate, 8)
    reference_exact = {item["exact"] for item in reference_items}
    matches = [item for item in candidate_items if item["exact"] in reference_exact]
    return {
        "count": len(matches),
        "ratio": ratio(len(matches), len(candidate_items)),
        "examples": [
            {"sentence": truncate_text(item["sentence"], 120), "length": item["length"]}
            for item in matches[:5]
        ],
    }


def normalized_sentence_overlap(reference: str, candidate: str) -> dict[str, Any]:
    reference_items = candidate_sentence_items(reference, 8)
    candidate_items = candidate_sentence_items(candidate, 8)
    reference_norms = {item["normalized"] for item in reference_items if item["normalized"]}
    matches = [item for item in candidate_items if item["normalized"] in reference_norms]
    return {
        "count": len(matches),
        "ratio": ratio(len(matches), len(candidate_items)),
        "examples": [
            {"sentence": truncate_text(item["sentence"], 120), "length": item["length"]}
            for item in matches[:5]
        ],
    }


def char_shingles(text: str, size: int) -> list[str]:
    if len(text) < size:
        return []
    return [text[i : i + size] for i in range(len(text) - size + 1)]


def near_sentence_overlap(reference: str, candidate: str) -> dict[str, Any]:
    reference_items = candidate_sentence_items(reference, 8)
    candidate_items = candidate_sentence_items(candidate, 8)
    reference_exact = {item["exact"] for item in reference_items}
    reference_norms = {item["normalized"] for item in reference_items if item["normalized"]}
    reference_candidates: list[dict[str, Any]] = []
    for item in reference_items:
        if item["visible_length"] < 12 or not item["normalized"]:
            continue
        item = dict(item)
        item["tri"] = set(char_shingles(item["normalized"], 3))
        reference_candidates.append(item)

    examples: list[dict[str, Any]] = []
    count = 0
    for candidate_item in candidate_items:
        c_norm = candidate_item["normalized"]
        if candidate_item["visible_length"] < 12 or not c_norm:
            continue
        if candidate_item["exact"] in reference_exact or c_norm in reference_norms:
            continue

        candidate_tri = set(char_shingles(c_norm, 3))
        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        for reference_item in reference_candidates:
            r_norm = reference_item["normalized"]
            max_len = max(len(c_norm), len(r_norm))
            if abs(len(c_norm) - len(r_norm)) > max(8, int(max_len * 0.4)):
                continue
            if candidate_tri and reference_item["tri"] and not (candidate_tri & reference_item["tri"]):
                continue
            similarity = SequenceMatcher(None, c_norm, r_norm, autojunk=False).ratio()
            if similarity > best[0]:
                best = (similarity, reference_item)

        if best[1] is not None and best[0] >= 0.88:
            count += 1
            if len(examples) < 5:
                examples.append({
                    "candidate_sentence": truncate_text(candidate_item["sentence"], 120),
                    "reference_sentence": truncate_text(best[1]["sentence"], 120),
                    "similarity": rounded(best[0]),
                })

    return {"count": count, "ratio": ratio(count, len(candidate_items)), "examples": examples}


def char_shingle_overlap(reference: str, candidate: str) -> dict[str, dict[str, Any]]:
    reference_norm = normalized_text(reference)
    candidate_norm = normalized_text(candidate)
    result: dict[str, dict[str, Any]] = {}
    for size in (8, 12, 20):
        candidate_shingles = char_shingles(candidate_norm, size)
        reference_shingles = set(char_shingles(reference_norm, size))
        overlap_count = sum(1 for shingle in candidate_shingles if shingle in reference_shingles)
        result[str(size)] = {
            "candidate_total": len(candidate_shingles),
            "overlap_count": overlap_count,
            "overlap_ratio": ratio(overlap_count, len(candidate_shingles)),
        }
    return result


def find_common_substring(candidate: str, reference: str, length: int) -> tuple[int, int] | None:
    if length <= 0:
        return (0, 0)
    for c_start in range(len(candidate) - length + 1):
        piece = candidate[c_start : c_start + length]
        r_start = reference.find(piece)
        if r_start >= 0:
            return c_start, r_start
    return None


def longest_common_substring(reference: str, candidate: str) -> dict[str, Any]:
    reference_compact = compact(reference)
    candidate_compact = compact(candidate)
    if not reference_compact or not candidate_compact:
        return {"length": 0, "candidate_excerpt": "", "reference_excerpt": ""}
    lo, hi = 0, len(candidate_compact)
    best: tuple[int, int] | None = (0, 0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        found = find_common_substring(candidate_compact, reference_compact, mid)
        if found is not None:
            lo = mid
            best = found
        else:
            hi = mid - 1
    if lo:
        best = find_common_substring(candidate_compact, reference_compact, lo)
    c_start, r_start = best or (0, 0)
    return {
        "length": lo,
        "candidate_excerpt": truncate_text(candidate_compact[c_start : c_start + lo], 120),
        "reference_excerpt": truncate_text(reference_compact[r_start : r_start + lo], 120),
    }


def longest_common_substring_length(a: str, b: str) -> int:
    return int(longest_common_substring(a, b)["length"])


def paragraph_overlap(reference: str, candidate: str) -> dict[str, Any]:
    def paragraph_items(text: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for paragraph in split_paragraphs(text):
            stripped = paragraph.strip()
            effective = effective_text(stripped)
            if len(effective) < 30:
                continue
            items.append({
                "paragraph": stripped,
                "exact": norm_unit(stripped),
                "normalized": effective.lower(),
                "length": len(effective),
            })
        return items

    reference_items = paragraph_items(reference)
    candidate_items = paragraph_items(candidate)
    reference_exact = {item["exact"] for item in reference_items}
    reference_norms = {item["normalized"] for item in reference_items if item["normalized"]}
    exact_matches = [item for item in candidate_items if item["exact"] in reference_exact]
    normalized_matches = [item for item in candidate_items if item["normalized"] in reference_norms]

    examples: list[dict[str, Any]] = []
    for item in candidate_items:
        match_type = ""
        if item["exact"] in reference_exact:
            match_type = "exact"
        elif item["normalized"] in reference_norms:
            match_type = "normalized"
        if match_type and len(examples) < 3:
            examples.append({
                "match_type": match_type,
                "paragraph": truncate_text(item["paragraph"], 120),
                "length": item["length"],
            })

    return {
        "exact_count": len(exact_matches),
        "normalized_count": len(normalized_matches),
        "examples": examples,
    }


def contamination_risk_level(exact: dict[str, Any], normalized: dict[str, Any],
                             near: dict[str, Any], shingles: dict[str, dict[str, Any]],
                             longest: dict[str, Any], paragraphs: dict[str, Any]) -> str:
    if (
        (exact["ratio"] >= 0.30 and exact["count"] >= 2)
        or (normalized["ratio"] >= 0.30 and normalized["count"] >= 2)
        or longest["length"] >= 200
        or shingles["20"]["overlap_ratio"] >= 0.35
        or paragraphs["exact_count"] > 0
        or paragraphs["normalized_count"] > 0
    ):
        return "high"
    if (
        exact["ratio"] >= 0.10
        or normalized["ratio"] >= 0.10
        or near["count"] >= 2
        or near["ratio"] >= 0.20
        or longest["length"] >= 80
        or shingles["12"]["overlap_ratio"] >= 0.25
    ):
        return "medium"
    return "low"


def contamination_stats(reference: str, candidate: str) -> dict[str, Any]:
    exact = exact_sentence_overlap(reference, candidate)
    normalized = normalized_sentence_overlap(reference, candidate)
    near = near_sentence_overlap(reference, candidate)
    shingles = char_shingle_overlap(reference, candidate)
    longest = longest_common_substring(reference, candidate)
    paragraphs = paragraph_overlap(reference, candidate)
    risk = contamination_risk_level(exact, normalized, near, shingles, longest, paragraphs)

    return {
        "overlapping_sentence_count": exact["count"],
        "overlapping_sentence_ratio": exact["ratio"],
        "longest_continuous_overlap_chars": longest["length"],
        "longest_common_substring_length": longest["length"],
        "contamination_risk": risk,
        "exact_sentence_overlap": exact,
        "normalized_sentence_overlap": normalized,
        "near_sentence_overlap": near,
        "char_shingle_overlap": shingles,
        "longest_common_substring": longest,
        "paragraph_overlap": paragraphs,
    }


def diff_summary(reference_stats: dict[str, Any], candidate_stats: dict[str, Any],
                 repetition: dict[str, Any], contamination: dict[str, Any]) -> dict[str, Any]:
    ref_chars = reference_stats["non_whitespace_char_count"]
    cand_chars = candidate_stats["non_whitespace_char_count"]
    char_ratio = rounded((cand_chars - ref_chars) / ref_chars) if ref_chars else (0.0 if cand_chars == 0 else 1.0)
    avg_sentence_length_diff = rounded(
        candidate_stats["average_sentence_length"] - reference_stats["average_sentence_length"]
    )
    ref_spp = ratio(reference_stats["sentence_count"], reference_stats["paragraph_count"])
    cand_spp = ratio(candidate_stats["sentence_count"], candidate_stats["paragraph_count"])
    return {
        "char_count_difference_ratio": char_ratio,
        "char_count_ratio": metric_ratio(candidate_stats["char_count"], reference_stats["char_count"]),
        "non_whitespace_char_count_ratio": metric_ratio(
            candidate_stats["non_whitespace_char_count"], reference_stats["non_whitespace_char_count"]
        ),
        "sentence_count_ratio": metric_ratio(candidate_stats["sentence_count"], reference_stats["sentence_count"]),
        "paragraph_count_ratio": metric_ratio(candidate_stats["paragraph_count"], reference_stats["paragraph_count"]),
        "avg_sentence_length_diff": avg_sentence_length_diff,
        "average_sentence_length_diff": avg_sentence_length_diff,
        "avg_sentence_length_ratio": metric_ratio(
            candidate_stats["average_sentence_length"], reference_stats["average_sentence_length"]
        ),
        "median_sentence_length_diff": rounded(
            candidate_stats["median_sentence_length"] - reference_stats["median_sentence_length"]
        ),
        "dialogue_line_ratio_diff": rounded(
            candidate_stats["dialogue_line_ratio"] - reference_stats["dialogue_line_ratio"]
        ),
        "avg_sentences_per_paragraph_diff": rounded(cand_spp - ref_spp),
        "repetition_risk": repetition["repetition_risk"],
        "contamination_risk": contamination["contamination_risk"],
    }


def bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def ratio_similarity(metric_value: float, tolerance: float, zero_at: float) -> float:
    distance = abs(metric_value - 1.0)
    if distance <= tolerance:
        return 100.0
    if distance >= zero_at:
        return 0.0
    return rounded((zero_at - distance) / (zero_at - tolerance) * 100.0)


def diff_similarity(diff_value: float, tolerance: float, zero_at: float) -> float:
    distance = abs(diff_value)
    if distance <= tolerance:
        return 100.0
    if distance >= zero_at:
        return 0.0
    return rounded((zero_at - distance) / (zero_at - tolerance) * 100.0)


def component(name: str, raw_score: float, notes: list[str]) -> dict[str, Any]:
    weight = STYLE_WEIGHTS[name]
    return {
        "score": rounded(bounded(raw_score) * weight / 100.0),
        "weight": weight,
        "notes": notes,
    }


def risk_score(risk: str, medium_score: float, high_score: float) -> float:
    return {"low": 100.0, "medium": medium_score, "high": high_score}.get(risk, 0.0)


def style_score(reference_stats: dict[str, Any], candidate_stats: dict[str, Any],
                repetition: dict[str, Any], contamination: dict[str, Any],
                diff: dict[str, Any]) -> dict[str, Any]:
    if candidate_stats["non_whitespace_char_count"] == 0:
        notes = ["Candidate is empty; style score is invalid."]
        return {
            "overall": 0.0,
            "level": "invalid",
            "components": {name: component(name, 0.0, notes if name == "length_profile" else []) for name in STYLE_WEIGHTS},
            "notes": notes,
        }
    if reference_stats["non_whitespace_char_count"] == 0:
        notes = ["Reference is empty; style score is invalid."]
        return {
            "overall": 0.0,
            "level": "invalid",
            "components": {name: component(name, 0.0, notes if name == "length_profile" else []) for name in STYLE_WEIGHTS},
            "notes": notes,
        }

    notes: list[str] = []
    length_raw = statistics.mean([
        ratio_similarity(diff["char_count_ratio"], 0.20, 1.00),
        ratio_similarity(diff["non_whitespace_char_count_ratio"], 0.20, 1.00),
    ])
    length_notes: list[str] = []
    if diff["non_whitespace_char_count_ratio"] < 0.70:
        length_notes.append("Candidate is much shorter than reference.")
    elif diff["non_whitespace_char_count_ratio"] > 1.40:
        length_notes.append("Candidate is much longer than reference.")
    else:
        length_notes.append("Candidate length is close to reference.")

    sentence_raw = statistics.mean([
        ratio_similarity(diff["avg_sentence_length_ratio"], 0.20, 1.00),
        ratio_similarity(metric_ratio(candidate_stats["median_sentence_length"], reference_stats["median_sentence_length"]), 0.20, 1.00),
        ratio_similarity(metric_ratio(candidate_stats["longest_sentence_length"], reference_stats["longest_sentence_length"]), 0.35, 1.50),
    ])
    sentence_notes = ["Candidate sentence length is close to reference."] if abs(diff["avg_sentence_length_ratio"] - 1.0) <= 0.20 else [
        "Candidate sentence length differs from reference."
    ]

    paragraph_raw = statistics.mean([
        ratio_similarity(diff["paragraph_count_ratio"], 0.35, 1.20),
        diff_similarity(diff["avg_sentences_per_paragraph_diff"], 0.50, 3.00),
    ])
    paragraph_notes = ["Paragraph rhythm is close to reference."] if paragraph_raw >= 80 else [
        "Paragraph rhythm differs from reference."
    ]

    dialogue_raw = diff_similarity(diff["dialogue_line_ratio_diff"], 0.05, 0.50)
    dialogue_notes = ["Dialogue ratio is close to reference."] if dialogue_raw >= 80 else [
        "Dialogue ratio differs significantly."
    ]

    repetition_raw = risk_score(repetition["repetition_risk"], 60.0, 20.0)
    repetition_notes = []
    if repetition["repetition_risk"] != "low":
        repetition_notes.append("Repetition risk reduces style score.")
        notes.append("Repetition risk reduces style score.")

    contamination_raw = risk_score(contamination["contamination_risk"], 55.0, 10.0)
    contamination_notes = []
    if contamination["contamination_risk"] != "low":
        contamination_notes.append("Contamination risk reduces style score.")
        notes.append("Contamination risk reduces style score.")

    components = {
        "length_profile": component("length_profile", length_raw, length_notes),
        "sentence_profile": component("sentence_profile", sentence_raw, sentence_notes),
        "paragraph_profile": component("paragraph_profile", paragraph_raw, paragraph_notes),
        "dialogue_profile": component("dialogue_profile", dialogue_raw, dialogue_notes),
        "repetition_penalty": component("repetition_penalty", repetition_raw, repetition_notes),
        "contamination_penalty": component("contamination_penalty", contamination_raw, contamination_notes),
    }
    overall = rounded(sum(item["score"] for item in components.values()))
    level = "close" if overall >= 80 else "moderate" if overall >= 60 else "far"
    if not notes:
        notes.append("Style score is based on deterministic form metrics, not literary quality.")
    return {"overall": overall, "level": level, "components": components, "notes": notes}


def build_summary(candidate_stats: dict[str, Any], diff: dict[str, Any],
                  score: dict[str, Any]) -> dict[str, list[str]]:
    warnings: list[str] = []
    notes: list[str] = []
    if candidate_stats["non_whitespace_char_count"] == 0:
        warnings.append("Candidate text is empty.")
        warnings.append("Candidate is empty; style score is invalid.")
    if diff["repetition_risk"] != "low":
        warnings.append(f"Repetition risk is {diff['repetition_risk']}.")
    if diff["contamination_risk"] == "high":
        warnings.append("High contamination risk: candidate appears to copy reference text.")
        warnings.append("Candidate may reuse substantial text from reference.")
    elif diff["contamination_risk"] == "medium":
        warnings.append("Candidate may reuse substantial text from reference.")
    if abs(diff["char_count_difference_ratio"]) >= 0.8:
        notes.append("Reference and candidate lengths differ substantially.")
    notes.append(f"Style score level is {score['level']} ({score['overall']}/100).")
    notes.extend(score["notes"][:3])
    if not notes:
        notes.append("Basic deterministic checks completed.")
    return {"overall_warning": warnings, "notes": notes}


def report_meta() -> dict[str, Any]:
    return {
        "tool": "eval_style.py",
        "version": "3.6",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_version": "1.0",
        "notes": [],
    }


def input_info(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_exists = reference_path.exists()
    candidate_exists = candidate_path.exists()
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_exists": reference_exists,
        "candidate_exists": candidate_exists,
        "reference_size_bytes": reference_path.stat().st_size if reference_exists else 0,
        "candidate_size_bytes": candidate_path.stat().st_size if candidate_exists else 0,
    }


def evaluate(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = read_text(reference_path)
    candidate = read_text(candidate_path)
    reference_stats = basic_stats(reference)
    candidate_stats = basic_stats(candidate)
    segmentation = {
        "reference": segmentation_stats(reference),
        "candidate": segmentation_stats(candidate),
    }
    repetition = repetition_stats(candidate)
    contamination = contamination_stats(reference, candidate)
    diff = diff_summary(reference_stats, candidate_stats, repetition, contamination)
    score = style_score(reference_stats, candidate_stats, repetition, contamination, diff)
    return {
        "meta": report_meta(),
        "inputs": input_info(reference_path, candidate_path),
        "reference_stats": reference_stats,
        "candidate_stats": candidate_stats,
        "segmentation": segmentation,
        "repetition": repetition,
        "contamination": contamination,
        "diff": diff,
        "style_score": score,
        "summary": build_summary(candidate_stats, diff, score),
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def stats_table(stats: dict[str, Any]) -> str:
    labels = [
        ("Characters", "char_count"),
        ("Non-whitespace characters", "non_whitespace_char_count"),
        ("Lines", "line_count"),
        ("Paragraphs", "paragraph_count"),
        ("Sentences", "sentence_count"),
        ("Average sentence length", "average_sentence_length"),
        ("Median sentence length", "median_sentence_length"),
        ("Longest sentence length", "longest_sentence_length"),
        ("Dialogue lines", "dialogue_line_count"),
        ("Dialogue line ratio", "dialogue_line_ratio"),
    ]
    rows = ["| Metric | Value |", "|---|---:|"]
    for label, key in labels:
        value = stats[key]
        rows.append(f"| {label} | {percent(value) if key.endswith('_ratio') else value} |")
    return "\n".join(rows)


def segmentation_table(segmentation: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| Text | Paragraphs | Sentences | Dialogue lines | Avg sentences / paragraph |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("reference", "candidate"):
        stats = segmentation[label]
        rows.append(
            f"| {label.title()} | {stats['paragraph_count']} | {stats['sentence_count']} | "
            f"{stats['dialogue_line_count']} | {stats['avg_sentences_per_paragraph']} |"
        )
    return "\n".join(rows)


def format_text_examples(examples: list[dict[str, Any]], key: str = "text") -> str:
    if not examples:
        return "None"
    return "; ".join(f"`{item[key]}` x{item.get('count', item.get('repeat_count', 1))}" for item in examples[:5])


def format_near_duplicate_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "None"
    return "; ".join(
        f"`{item['sentence_a']}` ~ `{item['sentence_b']}` ({item['similarity']})"
        for item in examples[:5]
    )


def format_loop_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "None"
    return "; ".join(
        f"{item['type']} " + " / ".join(f"`{part}`" for part in item["pattern"])
        + f" x{item['repeat_count']}"
        for item in examples[:5]
    )


def format_ngrams(items: list[dict[str, Any]]) -> str:
    return ", ".join(f"`{x['gram']}`:{x['count']}" for x in items[:10]) or "None"


def format_sentence_overlap_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "None"
    return "; ".join(f"`{item['sentence']}` ({item['length']})" for item in examples[:5])


def format_near_overlap_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "None"
    return "; ".join(
        f"`{item['candidate_sentence']}` ~ `{item['reference_sentence']}` ({item['similarity']})"
        for item in examples[:5]
    )


def format_paragraph_overlap_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "None"
    return "; ".join(
        f"{item['match_type']} `{item['paragraph']}` ({item['length']})"
        for item in examples[:3]
    )


def format_shingle_overlap(shingles: dict[str, dict[str, Any]], size: str) -> str:
    item = shingles[size]
    return f"{item['overlap_count']}/{item['candidate_total']} ({percent(item['overlap_ratio'])})"


def format_style_components(score: dict[str, Any]) -> str:
    rows = ["| Component | Score | Notes |", "|---|---:|---|"]
    for name, item in score["components"].items():
        notes = "; ".join(item["notes"]) or "None"
        rows.append(f"| {name} | {item['score']}/{item['weight']} | {notes} |")
    return "\n".join(rows)


def render_markdown(report: dict[str, Any]) -> str:
    repetition = report["repetition"]
    contamination = report["contamination"]
    diff = report["diff"]
    score = report["style_score"]
    warnings = report["summary"]["overall_warning"]
    notes = report["summary"]["notes"]
    near_dups = repetition["near_duplicate_adjacent_sentences"]
    loops = repetition["short_sentence_loops"]
    char_ngrams = repetition["char_ngrams"]
    exact_overlap = contamination["exact_sentence_overlap"]
    normalized_overlap = contamination["normalized_sentence_overlap"]
    near_overlap = contamination["near_sentence_overlap"]
    paragraph_overlap_data = contamination["paragraph_overlap"]
    longest = contamination["longest_common_substring"]
    shingles = contamination["char_shingle_overlap"]
    return "\n\n".join([
        "# Style Evaluation Report",
        "## Summary\n\n"
        f"- Overall style score: **{score['overall']}/100**\n"
        f"- Level: **{score['level']}**\n"
        f"- Repetition risk: **{repetition['repetition_risk']}**\n"
        f"- Contamination risk: **{contamination['contamination_risk']}**\n"
        f"- Generated at: `{report['meta']['created_at']}`",
        "## Style Score\n\n"
        f"- Overall: **{score['overall']}/100**\n"
        f"- Level: **{score['level']}**\n"
        f"- Notes: {'; '.join(score['notes']) or 'None'}\n\n"
        + format_style_components(score),
        "## Key Warnings\n\n"
        + ("\n".join(f"- {w}" for w in warnings) if warnings else "- None"),
        "## Basic Statistics\n\n"
        "### Reference\n\n" + stats_table(report["reference_stats"])
        + "\n\n### Candidate\n\n" + stats_table(report["candidate_stats"]),
        "## Segmentation Diagnostics\n\n" + segmentation_table(report["segmentation"]),
        "## Repetition Risk\n\n"
        f"- Risk: **{repetition['repetition_risk']}**\n"
        f"- Duplicate lines: {repetition['duplicate_line_count']} ({percent(repetition['duplicate_line_ratio'])})\n"
        f"- Duplicate line examples: {format_text_examples(repetition['repeated_line_examples'])}\n"
        f"- Duplicate paragraphs: {repetition['duplicate_paragraph_count']} "
        f"({percent(repetition['duplicate_paragraph_ratio'])})\n"
        f"- Duplicate paragraph examples: {format_text_examples(repetition['repeated_paragraph_examples'])}\n"
        f"- Consecutive repeated sentences: {repetition['consecutive_repeated_sentence_count']}\n"
        f"- Consecutive examples: {format_text_examples(repetition['consecutive_repeated_sentence_examples'], 'sentence')}\n"
        f"- Near duplicate adjacent sentences: {near_dups['count']}\n"
        f"- Near duplicate examples: {format_near_duplicate_examples(near_dups['examples'])}\n"
        f"- Short sentence loops: {loops['detected']} ({loops['count']})\n"
        f"- Short loop examples: {format_loop_examples(loops['examples'])}\n"
        f"- Top char 4-grams: {format_ngrams(char_ngrams['4'])}",
        "## Contamination Risk\n\n"
        f"- Risk: **{contamination['contamination_risk']}**\n"
        f"- Overlapping candidate sentences: {contamination['overlapping_sentence_count']} "
        f"({percent(contamination['overlapping_sentence_ratio'])})\n"
        f"- Exact sentence overlap: {exact_overlap['count']} ({percent(exact_overlap['ratio'])})\n"
        f"- Exact examples: {format_sentence_overlap_examples(exact_overlap['examples'])}\n"
        f"- Normalized sentence overlap: {normalized_overlap['count']} "
        f"({percent(normalized_overlap['ratio'])})\n"
        f"- Normalized examples: {format_sentence_overlap_examples(normalized_overlap['examples'])}\n"
        f"- Near sentence overlap: {near_overlap['count']} ({percent(near_overlap['ratio'])})\n"
        f"- Near examples: {format_near_overlap_examples(near_overlap['examples'])}\n"
        f"- Paragraph overlap: exact {paragraph_overlap_data['exact_count']}, "
        f"normalized {paragraph_overlap_data['normalized_count']}\n"
        f"- Paragraph examples: {format_paragraph_overlap_examples(paragraph_overlap_data['examples'])}\n"
        f"- Longest common substring: {longest['length']} chars, "
        f"candidate `{longest['candidate_excerpt']}`\n"
        f"- Char shingle 8: {format_shingle_overlap(shingles, '8')}\n"
        f"- Char shingle 12: {format_shingle_overlap(shingles, '12')}\n"
        f"- Char shingle 20: {format_shingle_overlap(shingles, '20')}",
        "## Difference Metrics\n\n"
        f"- Character count difference ratio: {percent(diff['char_count_difference_ratio'])}\n"
        f"- Character count ratio: {diff['char_count_ratio']}\n"
        f"- Non-whitespace character ratio: {diff['non_whitespace_char_count_ratio']}\n"
        f"- Sentence count ratio: {diff['sentence_count_ratio']}\n"
        f"- Paragraph count ratio: {diff['paragraph_count_ratio']}\n"
        f"- Average sentence length diff: {diff['average_sentence_length_diff']}\n"
        f"- Average sentence length ratio: {diff['avg_sentence_length_ratio']}\n"
        f"- Median sentence length diff: {diff['median_sentence_length_diff']}\n"
        f"- Avg sentences / paragraph diff: {diff['avg_sentences_per_paragraph_diff']}\n"
        f"- Dialogue ratio diff: {percent(diff['dialogue_line_ratio_diff'])}\n"
        f"- Repetition risk: {diff['repetition_risk']}\n"
        f"- Contamination risk: {diff['contamination_risk']}",
        "## Inputs\n\n"
        f"- Reference: `{report['inputs']['reference']}`\n"
        f"- Candidate: `{report['inputs']['candidate']}`\n"
        f"- Reference exists: {report['inputs']['reference_exists']}\n"
        f"- Candidate exists: {report['inputs']['candidate_exists']}\n"
        f"- Reference size bytes: {report['inputs']['reference_size_bytes']}\n"
        f"- Candidate size bytes: {report['inputs']['candidate_size_bytes']}\n"
        f"- Schema version: `{report['meta']['schema_version']}`\n"
        f"- Notes: {'; '.join(notes) or 'None'}",
    ]) + "\n"


def render_terminal_summary(report: dict[str, Any]) -> str:
    warnings = report["summary"]["overall_warning"]
    score = report["style_score"]
    lines = [
        "# Style Evaluation Summary",
        f"- Overall style score: {score['overall']}/100",
        f"- Level: {score['level']}",
        f"- Repetition risk: {report['repetition']['repetition_risk']}",
        f"- Contamination risk: {report['contamination']['contamination_risk']}",
        f"- Reference: `{report['inputs']['reference']}`",
        f"- Candidate: `{report['inputs']['candidate']}`",
        "- Key warnings:",
    ]
    lines.extend(f"  - {warning}" for warning in warnings[:5])
    if not warnings:
        lines.append("  - None")
    return "\n".join(lines) + "\n"


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic reference/candidate style evaluation.")
    parser.add_argument("--reference", required=True, help="UTF-8 reference text path.")
    parser.add_argument("--candidate", required=True, help="UTF-8 candidate text path.")
    parser.add_argument("--out-json", help="Optional JSON report output path.")
    parser.add_argument("--out-md", help="Optional Markdown report output path.")
    parser.add_argument("--verbose", action="store_true", help="Print the full Markdown report when writing to terminal.")
    parser.add_argument("--quiet", action="store_true", help="Suppress success messages when writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    try:
        report = evaluate(Path(args.reference), Path(args.candidate))
        if args.out_json:
            write_json(Path(args.out_json), report)
            if not args.quiet:
                print(f"Wrote JSON report: {args.out_json}")
        if args.out_md:
            write_markdown(Path(args.out_md), report)
            if not args.quiet:
                print(f"Wrote Markdown report: {args.out_md}")
        if not args.out_json and not args.out_md:
            if not args.quiet:
                print(render_markdown(report) if args.verbose else render_terminal_summary(report))
        return 0
    except EvalStyleError as exc:
        print(f"eval_style error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
