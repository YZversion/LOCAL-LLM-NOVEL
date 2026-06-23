#!/usr/bin/env python3
"""
阶段4：多轮 LoRA 生成，累计至 ~2300c 供 eval_draft.py 评测。

原则：每轮使用 cowriter.Retriever + build_prompt 真实推理链路构建 prompt，
不使用训练样本的压缩结构。生成参数与 config.yaml 一致。

Usage:
    .venv-train\\Scripts\\Activate.ps1
    python pipeline/generate_lora_multi.py
    python scripts/eval_draft.py --candidate outputs/lora_candidate_v2.txt --config config.yaml
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gc
import yaml

from pipeline.eval_style import norm_unit
from pipeline.eval_style import split_paragraphs as _split_paragraphs

ADAPTER_PATH        = "outputs/qlora_run_v4"
PROMPT_PATH         = "outputs/debug/last_prompt.txt"
OUTPUT_PATH         = "outputs/lora_candidate_v4.txt"
CONFIG_PATH         = "config.yaml"
MAX_SEQ_LENGTH_INFER = 8192
MAX_NEW_TOKENS       = 800   # per round; ~600-700c Chinese per round
TARGET_CHARS         = 2300
MAX_ROUNDS           = 8
MAX_RECENT_CHARS     = 2000

TEMPERATURE         = 0.8
TOP_P               = 0.8
TOP_K               = 20
REPETITION_PENALTY  = 1.15


def parse_context_from_prompt(path: Path) -> str:
    """Extract 【当前上文】 content from last_prompt.txt."""
    text = path.read_text(encoding="utf-8")
    marker = "【当前上文】\n"
    end_marker = "\n[ASSISTANT]"
    s = text.find(marker)
    if s == -1:
        return ""
    s += len(marker)
    e = text.find(end_marker, s)
    return text[s:e].strip() if e != -1 else text[s:].strip()


def strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


_MIN_DEDUP_CHARS = 10  # paragraphs shorter than this are not checked for dedup


def _dedup_truncate(new_text: str, seen_keys: set) -> tuple:
    """Split new_text into paragraphs; return content up to (not including)
    the first paragraph whose norm_unit key is already in seen_keys.
    Returns (kept_text, was_truncated, was_fully_skipped)."""
    paras = _split_paragraphs(new_text)
    kept: list = []
    truncated = False
    for p in paras:
        key = norm_unit(p)
        if len(key) < _MIN_DEDUP_CHARS or key not in seen_keys:
            kept.append(p)
        else:
            truncated = True
            break
    kept_text = "\n\n".join(kept).strip()
    return kept_text, truncated, (truncated and not kept_text)


def generate_one_round(model, tokenizer, messages: list[dict]) -> str:
    import torch
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
        )

    new_ids = output_ids[0][input_len:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return strip_think(generated)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not found — activate .venv-train/ first", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    from unsloth import FastLanguageModel
    from cowriter.retriever import Retriever
    from cowriter.prompts import build_prompt

    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}", file=sys.stderr)
        return 1

    prompt_path = Path(PROMPT_PATH)
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))

    print("--- Parsing context from last_prompt.txt ---")
    context_text = parse_context_from_prompt(prompt_path)
    print(f"  Context: {len(context_text)}c")
    if not context_text:
        print("ERROR: could not parse 【当前上文】 from prompt file", file=sys.stderr)
        return 1

    print("\n--- Loading Retriever (BM25 + story_bible) ---")
    retriever = Retriever(cfg)
    print(f"  {len(retriever._docs)} bible docs indexed")

    print(f"\n--- Loading model + adapter from {adapter_path}/ ---")
    print(f"  max_seq_length={MAX_SEQ_LENGTH_INFER}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH_INFER,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("  Model ready.")

    accumulated_text = context_text
    all_new_texts: list[str] = []
    total_new_chars = 0
    seen_para_keys: set = set()
    truncation_rounds = 0
    skip_rounds = 0
    rounds_attempted = 0

    for rnd in range(1, MAX_ROUNDS + 1):
        rounds_attempted += 1
        print(f"\n=== Round {rnd} (new so far: {total_new_chars}c / target {TARGET_CHARS}c) ===")

        recent_text = accumulated_text[-MAX_RECENT_CHARS:]
        retrieval = retriever.retrieve(recent_text, max_chapter=None)
        messages = build_prompt(
            recent_text=recent_text,
            summary="",
            retrieval=retrieval,
            instruction="",
            prior_summary="",
        )
        # If build_prompt added an assistant prefill, remove it for HF generation
        # (we use add_generation_prompt=True instead)
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        # Token count estimate
        try:
            input_text_check = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            input_text_check = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        n_input_tokens = tokenizer(input_text_check, return_tensors="pt")["input_ids"].shape[1]
        print(f"  Input tokens: {n_input_tokens}")

        new_text = generate_one_round(model, tokenizer, messages)
        print(f"  Generated: {len(new_text)}c")
        if new_text:
            print(f"  Preview: {new_text[:120]}...")

        if not new_text or len(new_text) < 5:
            print("  [WARN] Empty generation, stopping early.")
            break

        kept_text, was_truncated, was_skipped = _dedup_truncate(new_text, seen_para_keys)
        if was_truncated:
            truncation_rounds += 1
            print(f"  [DEDUP] Truncated: {len(new_text)}c → {len(kept_text)}c (duplicate paragraph detected)")
        if was_skipped:
            skip_rounds += 1
            print(f"  [SKIP] Entire round is duplicate — skipping.")
            continue

        text_to_add = kept_text if was_truncated else new_text
        for p in _split_paragraphs(text_to_add):
            key = norm_unit(p)
            if len(key) >= _MIN_DEDUP_CHARS:
                seen_para_keys.add(key)
        all_new_texts.append(text_to_add)
        accumulated_text += "\n\n" + text_to_add
        total_new_chars += len(text_to_add)

        if total_new_chars >= TARGET_CHARS:
            print(f"\n  Target reached ({total_new_chars}c >= {TARGET_CHARS}c). Stopping.")
            break

    candidate = "\n\n".join(all_new_texts)
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(candidate, encoding="utf-8")

    print(f"\n=== Generation complete ===")
    print(f"  Rounds completed: {len(all_new_texts)}")
    print(f"  Total new text: {len(candidate)}c")
    print(f"  Saved to: {out_path}")
    print(f"\n=== Deduplication Summary ===")
    print(f"  Rounds attempted:   {rounds_attempted}")
    print(f"  Truncations:        {truncation_rounds}")
    print(f"  Full skips:         {skip_rounds}")
    if skip_rounds > 0:
        print(f"  [Note] {skip_rounds} skip(s) consumed round budget without output.")
    print()
    print("Next step:")
    print("  python scripts/eval_draft.py --candidate outputs/lora_candidate_v4.txt --config config.yaml")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
