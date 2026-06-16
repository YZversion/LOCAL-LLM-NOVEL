#!/usr/bin/env python3
"""Regression tests for pipeline/eval_style.py using stable fixtures."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json

from pipeline import eval_style


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures" / "eval_style"
OUT = ROOT / "outputs"

REFERENCE = FIXTURES / "reference_basic.txt"
CANDIDATE_CLOSE = FIXTURES / "candidate_close.txt"
CANDIDATE_REPETITION = FIXTURES / "candidate_repetition.txt"
CANDIDATE_CONTAMINATION = FIXTURES / "candidate_contamination.txt"
CANDIDATE_FAR = FIXTURES / "candidate_far.txt"
CANDIDATE_EMPTY = FIXTURES / "candidate_empty.txt"

REPORT_JSON = OUT / "eval_style_fixture_report.json"
REPORT_MD = OUT / "eval_style_fixture_report.md"
NESTED_REPORT_JSON = OUT / "reports" / "eval_style_fixture_nested.json"
NESTED_REPORT_MD = OUT / "reports" / "eval_style_fixture_nested.md"
EMPTY_REPORT_MD = OUT / "eval_style_fixture_empty.md"


def assert_schema(data: dict) -> None:
    expected_keys = {
        "meta",
        "inputs",
        "reference_stats",
        "candidate_stats",
        "segmentation",
        "repetition",
        "contamination",
        "diff",
        "summary",
        "style_score",
    }
    assert expected_keys.issubset(data.keys())
    assert {"tool", "version", "created_at", "schema_version"}.issubset(data["meta"].keys())
    assert data["meta"]["tool"] == "eval_style.py"
    assert data["meta"]["version"] == "3.6"
    assert data["meta"]["schema_version"] == "1.0"
    assert {
        "reference",
        "candidate",
        "reference_exists",
        "candidate_exists",
        "reference_size_bytes",
        "candidate_size_bytes",
    }.issubset(data["inputs"].keys())
    assert data["inputs"]["reference_exists"] is True
    assert data["inputs"]["candidate_exists"] is True
    assert {
        "length_profile",
        "sentence_profile",
        "paragraph_profile",
        "dialogue_profile",
        "repetition_penalty",
        "contamination_penalty",
    } == set(data["style_score"]["components"].keys())


def assert_markdown_sections(markdown: str) -> None:
    for heading in (
        "# Style Evaluation Report",
        "## Summary",
        "## Style Score",
        "## Key Warnings",
        "## Basic Statistics",
        "## Segmentation Diagnostics",
        "## Repetition Risk",
        "## Contamination Risk",
        "## Difference Metrics",
        "## Inputs",
    ):
        assert heading in markdown


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (
        REFERENCE,
        CANDIDATE_CLOSE,
        CANDIDATE_REPETITION,
        CANDIDATE_CONTAMINATION,
        CANDIDATE_FAR,
        CANDIDATE_EMPTY,
    ):
        assert path.exists(), f"missing fixture: {path}"

    code = eval_style.main([
        "--reference", str(REFERENCE),
        "--candidate", str(CANDIDATE_CLOSE),
        "--out-json", str(REPORT_JSON),
        "--out-md", str(REPORT_MD),
    ])
    assert code == 0

    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert_schema(data)
    assert data["segmentation"]["reference"]["paragraph_count"] == 3
    assert data["segmentation"]["candidate"]["dialogue_line_count"] > 0
    assert data["repetition"]["repetition_risk"] == "low"
    assert data["contamination"]["contamination_risk"] == "low"
    assert data["style_score"]["level"] in {"close", "moderate"}

    markdown = REPORT_MD.read_text(encoding="utf-8")
    assert_markdown_sections(markdown)
    assert "invalid" not in markdown

    close_report = eval_style.evaluate(REFERENCE, CANDIDATE_CLOSE)
    far_report = eval_style.evaluate(REFERENCE, CANDIDATE_FAR)
    assert close_report["style_score"]["overall"] > far_report["style_score"]["overall"]

    repetition_report = eval_style.evaluate(REFERENCE, CANDIDATE_REPETITION)
    assert repetition_report["repetition"]["repetition_risk"] == "high"
    assert repetition_report["repetition"]["short_sentence_loops"]["detected"] is True
    assert any("Repetition risk reduces style score." in note for note in repetition_report["style_score"]["notes"])

    contamination_report = eval_style.evaluate(REFERENCE, CANDIDATE_CONTAMINATION)
    assert contamination_report["contamination"]["contamination_risk"] in {"medium", "high"}
    assert (
        contamination_report["contamination"]["exact_sentence_overlap"]["count"] > 0
        or contamination_report["contamination"]["normalized_sentence_overlap"]["count"] > 0
    )
    assert contamination_report["contamination"]["longest_common_substring_length"] > 0

    empty_report = eval_style.evaluate(REFERENCE, CANDIDATE_EMPTY)
    assert empty_report["candidate_stats"]["non_whitespace_char_count"] == 0
    assert empty_report["style_score"]["level"] == "invalid"
    assert "Candidate is empty; style score is invalid." in empty_report["style_score"]["notes"]
    empty_code = eval_style.main([
        "--reference", str(REFERENCE),
        "--candidate", str(CANDIDATE_EMPTY),
        "--out-md", str(EMPTY_REPORT_MD),
        "--quiet",
    ])
    assert empty_code == 0
    assert "invalid" in EMPTY_REPORT_MD.read_text(encoding="utf-8")

    quiet_stdout = io.StringIO()
    with redirect_stdout(quiet_stdout):
        quiet_code = eval_style.main([
            "--reference", str(REFERENCE),
            "--candidate", str(CANDIDATE_REPETITION),
            "--out-json", str(NESTED_REPORT_JSON),
            "--out-md", str(NESTED_REPORT_MD),
            "--quiet",
        ])
    assert quiet_code == 0
    assert quiet_stdout.getvalue() == ""
    assert NESTED_REPORT_JSON.exists()
    assert NESTED_REPORT_MD.exists()

    verbose_stdout = io.StringIO()
    with redirect_stdout(verbose_stdout):
        verbose_code = eval_style.main([
            "--reference", str(REFERENCE),
            "--candidate", str(CANDIDATE_CLOSE),
            "--verbose",
        ])
    assert verbose_code == 0
    assert "## Difference Metrics" in verbose_stdout.getvalue()

    default_stdout = io.StringIO()
    with redirect_stdout(default_stdout):
        default_code = eval_style.main(["--reference", str(REFERENCE), "--candidate", str(CANDIDATE_CLOSE)])
    assert default_code == 0
    assert "# Style Evaluation Summary" in default_stdout.getvalue()

    print("eval_style fixture regression tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
