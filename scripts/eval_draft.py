#!/usr/bin/env python3
"""Optional wrapper for evaluating an existing draft with pipeline.eval_style."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import eval_style  # noqa: E402


class EvalDraftError(Exception):
    """Raised for user-facing wrapper errors."""


def _strip_yaml_value(value: str) -> str:
    value = value.split("#", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def raw_data_from_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise EvalDraftError(f"Config file not found: {config_path}")
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvalDraftError(f"Config file is not valid UTF-8: {config_path}") from exc
    in_paths = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^\S[^:]*:", line):
            in_paths = line.split(":", 1)[0].strip() == "paths"
            continue
        if in_paths:
            match = re.match(r"^\s+raw_data\s*:\s*(.+?)\s*$", line)
            if match:
                raw_data = Path(_strip_yaml_value(match.group(1)))
                return raw_data if raw_data.is_absolute() else config_path.parent / raw_data
    raise EvalDraftError(f"Could not find paths.raw_data in config: {config_path}")


def reference_from_raw_data(raw_data: Path) -> Path:
    if not raw_data.exists() or not raw_data.is_dir():
        raise EvalDraftError(f"raw_data directory not found: {raw_data}")
    txt_files = sorted(path for path in raw_data.glob("*.txt") if path.is_file())
    if not txt_files:
        raise EvalDraftError(f"No .txt reference files found in raw_data: {raw_data}")
    if len(txt_files) > 1:
        names = ", ".join(path.name for path in txt_files[:5])
        raise EvalDraftError(
            f"Multiple .txt files found in raw_data ({names}); pass --reference explicitly."
        )
    return txt_files[0]


def resolve_reference(reference: str | None, config: str | None) -> Path:
    if reference:
        return Path(reference)
    if config:
        return reference_from_raw_data(raw_data_from_config(Path(config)))
    raise EvalDraftError("Reference required: pass --reference or --config.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing draft with pipeline/eval_style.py.")
    parser.add_argument("--reference", help="UTF-8 reference text path. Takes precedence over --config.")
    parser.add_argument("--config", help="config.yaml path used to locate paths.raw_data when --reference is omitted.")
    parser.add_argument("--candidate", required=True, help="UTF-8 draft/candidate text path.")
    parser.add_argument("--out-json", help="Optional JSON report output path.")
    parser.add_argument("--out-md", help="Optional Markdown report output path.")
    parser.add_argument("--verbose", action="store_true", help="Print the full Markdown report when writing to terminal.")
    parser.add_argument("--quiet", action="store_true", help="Suppress success messages when writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    eval_style.configure_stdio()
    args = parse_args(argv)
    try:
        reference_path = resolve_reference(args.reference, args.config)
        candidate_path = Path(args.candidate)
        if not reference_path.exists():
            raise EvalDraftError(f"Reference file not found: {reference_path}")
        if not candidate_path.exists():
            raise EvalDraftError(f"Candidate file not found: {candidate_path}")
        report = eval_style.evaluate(reference_path, candidate_path)
        if args.out_json:
            eval_style.write_json(Path(args.out_json), report)
            if not args.quiet:
                print(f"Wrote JSON report: {args.out_json}")
        if args.out_md:
            eval_style.write_markdown(Path(args.out_md), report)
            if not args.quiet:
                print(f"Wrote Markdown report: {args.out_md}")
        if not args.out_json and not args.out_md and not args.quiet:
            print(eval_style.render_markdown(report) if args.verbose else eval_style.render_terminal_summary(report))
        return 0
    except (EvalDraftError, eval_style.EvalStyleError) as exc:
        print(f"eval_draft error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
