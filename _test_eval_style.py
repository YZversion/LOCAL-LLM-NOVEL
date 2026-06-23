#!/usr/bin/env python3
"""Regression tests for pipeline/eval_style.py using stable fixtures."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import io
import json

from pipeline import eval_style
from scripts import eval_draft


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
DRAFT_REPORT_JSON = OUT / "reports" / "eval_draft_repetition_test.json"
DRAFT_REPORT_MD = OUT / "reports" / "eval_draft_repetition_test.md"
DRAFT_SINGLE_RAW = OUT / "eval_draft_single_raw"
DRAFT_SINGLE_CONFIG = OUT / "eval_draft_single_config.yaml"
DRAFT_MULTI_RAW = OUT / "eval_draft_multi_raw"
DRAFT_MULTI_CONFIG = OUT / "eval_draft_multi_config.yaml"


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

    draft_stdout = io.StringIO()
    with redirect_stdout(draft_stdout):
        draft_code = eval_draft.main(["--reference", str(REFERENCE), "--candidate", str(CANDIDATE_CLOSE)])
    assert draft_code == 0
    assert "# Style Evaluation Summary" in draft_stdout.getvalue()

    draft_verbose_stdout = io.StringIO()
    with redirect_stdout(draft_verbose_stdout):
        draft_verbose_code = eval_draft.main([
            "--reference", str(REFERENCE),
            "--candidate", str(CANDIDATE_CLOSE),
            "--verbose",
        ])
    assert draft_verbose_code == 0
    assert "## Difference Metrics" in draft_verbose_stdout.getvalue()

    draft_quiet_stdout = io.StringIO()
    with redirect_stdout(draft_quiet_stdout):
        draft_quiet_code = eval_draft.main([
            "--reference", str(REFERENCE),
            "--candidate", str(CANDIDATE_REPETITION),
            "--out-json", str(DRAFT_REPORT_JSON),
            "--out-md", str(DRAFT_REPORT_MD),
            "--quiet",
        ])
    assert draft_quiet_code == 0
    assert draft_quiet_stdout.getvalue() == ""
    assert DRAFT_REPORT_JSON.exists()
    assert DRAFT_REPORT_MD.exists()
    assert_schema(json.loads(DRAFT_REPORT_JSON.read_text(encoding="utf-8")))

    DRAFT_SINGLE_RAW.mkdir(parents=True, exist_ok=True)
    single_ref = DRAFT_SINGLE_RAW / "reference.txt"
    single_ref.write_text(REFERENCE.read_text(encoding="utf-8"), encoding="utf-8")
    DRAFT_SINGLE_CONFIG.write_text(f"paths:\n  raw_data: \"{DRAFT_SINGLE_RAW}\"\n", encoding="utf-8")
    config_stdout = io.StringIO()
    with redirect_stdout(config_stdout):
        config_code = eval_draft.main(["--config", str(DRAFT_SINGLE_CONFIG), "--candidate", str(CANDIDATE_CLOSE)])
    assert config_code == 0
    assert "# Style Evaluation Summary" in config_stdout.getvalue()

    missing_stderr = io.StringIO()
    with redirect_stderr(missing_stderr):
        missing_code = eval_draft.main(["--reference", str(REFERENCE), "--candidate", str(OUT / "missing_draft.txt")])
    assert missing_code == 2
    assert "Candidate file not found" in missing_stderr.getvalue()

    DRAFT_MULTI_RAW.mkdir(parents=True, exist_ok=True)
    (DRAFT_MULTI_RAW / "a.txt").write_text("第一份参考。", encoding="utf-8")
    (DRAFT_MULTI_RAW / "b.txt").write_text("第二份参考。", encoding="utf-8")
    DRAFT_MULTI_CONFIG.write_text(f"paths:\n  raw_data: \"{DRAFT_MULTI_RAW}\"\n", encoding="utf-8")
    multi_stderr = io.StringIO()
    with redirect_stderr(multi_stderr):
        multi_code = eval_draft.main(["--config", str(DRAFT_MULTI_CONFIG), "--candidate", str(CANDIDATE_CLOSE)])
    assert multi_code == 2
    assert "Multiple .txt files found" in multi_stderr.getvalue()

    # Verify character-based duplicate_paragraph_ratio (post-fix behaviour).
    # One large paragraph (40 chars) duplicated among four short unique ones (2 chars each).
    # Old count-ratio: 1 extra copy / 6 total paragraphs = 0.1667 (< 0.25 → medium).
    # New char-ratio:  (40*2) / (40+2+2+2+2+40) = 80/88 ≈ 0.9091 (>= 0.25 → high).
    _big = "甲" * 40
    _char_ratio_text = "\n\n".join([_big, "乙。", "丙。", "丁。", "戊。", _big])
    _cr = eval_style.repetition_stats(_char_ratio_text)
    assert _cr["duplicate_paragraph_count"] == 1, "extra-copy count should remain 1"
    assert _cr["duplicate_paragraph_ratio"] > 0.85, (
        f"char-based ratio should be ~0.9091, got {_cr['duplicate_paragraph_ratio']}"
    )
    assert _cr["repetition_risk"] == "high", (
        "large duplicated paragraph should now trigger high risk via char-based threshold"
    )

    print("eval_style fixture regression tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
