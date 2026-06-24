#!/usr/bin/env python3
"""
v4 正式评测候选生成（无交互，自动接受每轮）。

与 adapter_cli.py 完全一致的推理链路：
  - MAX_RECENT_CHARS = 800（已验证安全的配置）
  - /reject 释放逻辑（gc.collect + empty_cache）— 无交互时不会触发，但保留常量
  - Retriever + build_prompt 真实推理链路
  - 跨轮 dedup（与 generate_lora_multi.py 一致）

每轮：
  - 报告 input tokens、generated chars
  - 报告 VRAM 峰值（reset_peak 在 generate 前）
  - 报告是否发生去重截断

生成完成后：
  - 打印最终【当前上文】（最后 800c），供人工确认干净
  - 保存候选到 outputs/adapter_candidate_v4_eval.txt
  - 报告 Deduplication Summary

运行环境：.venv-train/
"""
import sys, gc
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ADAPTER_PATH      = "outputs/qlora_run_v4"
CONTEXT_FILE      = "outputs/debug/v4_eval_context.txt"
OUTPUT_FILE       = "outputs/adapter_candidate_v4_eval.txt"
CONFIG_PATH       = "config.yaml"
MAX_SEQ_LENGTH    = 4096
MAX_NEW_TOKENS    = 800   # 与 generate_lora_multi.py 保持一致
TARGET_CHARS      = 2300
MAX_ROUNDS        = 8
MAX_RECENT_CHARS  = 800   # 已验证安全；与 adapter_cli.py 修改后一致
TEMPERATURE       = 0.8
TOP_P             = 0.8
TOP_K             = 20
REPETITION_PENALTY = 1.15
_MIN_DEDUP_CHARS  = 10
BUDGET_GB         = 8.59


def strip_think(text):
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


def vram_report(label):
    import torch
    alloc  = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved()  / 1024**3
    peak   = torch.cuda.max_memory_allocated() / 1024**3
    margin = BUDGET_GB - peak
    safety = "[OK]" if peak < 8.3 else ("[WARN]" if peak < 8.5 else "[OOM]")
    print(f"  VRAM {safety} alloc={alloc:.3f}GB  reserved={reserv:.3f}GB"
          f"  peak={peak:.3f}GB  margin={margin:.3f}GB")
    return peak


def dedup_truncate(new_text, seen_keys, split_fn, norm_fn):
    paras = split_fn(new_text)
    kept, truncated = [], False
    for p in paras:
        key = norm_fn(p)
        if len(key) < _MIN_DEDUP_CHARS or key not in seen_keys:
            kept.append(p)
        else:
            truncated = True
            break
    kept_text = "\n\n".join(kept).strip()
    return kept_text, truncated, (truncated and not kept_text)


def main():
    try:
        import torch
    except ImportError:
        print("ERROR: torch not found — activate .venv-train/", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    import yaml

    ctx_path = Path(CONTEXT_FILE)
    if not ctx_path.exists():
        print(f"ERROR: context file not found: {ctx_path}", file=sys.stderr)
        return 1

    initial_context = ctx_path.read_text(encoding="utf-8").strip()
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))

    print("=" * 65)
    print("  v4 正式评测候选生成")
    print(f"  adapter : {ADAPTER_PATH}")
    print(f"  context : {CONTEXT_FILE} ({len(initial_context)}c)")
    print(f"  MAX_RECENT_CHARS={MAX_RECENT_CHARS}  TARGET={TARGET_CHARS}c  MAX_ROUNDS={MAX_ROUNDS}")
    print(f"  budget  : {BUDGET_GB} GB")
    print("=" * 65)

    # 内部重复验证（启动时再次确认）
    from pipeline.eval_style import norm_unit, split_paragraphs as _split
    paras_init = _split(initial_context)
    seen_init, dupes_init = set(), []
    for p in paras_init:
        key = norm_unit(p)
        if len(key) < _MIN_DEDUP_CHARS: continue
        if key in seen_init: dupes_init.append(p[:60])
        else: seen_init.add(key)
    if dupes_init:
        print(f"[FAIL] 初始上文内部重复检测未通过（{len(dupes_init)} 处）：")
        for d in dupes_init: print(f"  - {d}")
        return 1
    print(f"[OK] 初始上文内部重复检测通过（{len([p for p in paras_init if len(norm_unit(p))>=_MIN_DEDUP_CHARS])} 段落）")

    # 加载 Retriever
    print("\n[Step 1] 加载 Retriever...")
    from cowriter.retriever import Retriever
    from cowriter.prompts  import build_prompt
    retriever = Retriever(cfg)
    print(f"  {len(retriever._docs)} bible docs indexed")

    # 加载模型
    print(f"\n[Step 2] 加载 v4 adapter (max_seq_length={MAX_SEQ_LENGTH})...")
    torch.cuda.reset_peak_memory_stats()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    peak_load = vram_report("加载后")

    # 预种 seen_para_keys（防止生成内容重复初始上文）
    accumulated_text = initial_context
    seen_para_keys: set = set()
    for p in _split(initial_context):
        key = norm_unit(p)
        if len(key) >= _MIN_DEDUP_CHARS:
            seen_para_keys.add(key)

    all_new_texts: list = []
    total_new_chars  = 0
    truncation_rounds = 0
    skip_rounds       = 0
    peak_per_round: list = []

    # ── 多轮生成 ─────────────────────────────────────────────────────────────
    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*65}")
        print(f"  Round {rnd}  (累积新文本: {total_new_chars}c / 目标 {TARGET_CHARS}c)")
        print(f"{'='*65}")

        recent_text = accumulated_text[-MAX_RECENT_CHARS:]
        print(f"  recent_text: {len(recent_text)}c  "
              f"(总累积: {len(accumulated_text)}c，取后 {MAX_RECENT_CHARS}c)")

        retrieval = retriever.retrieve(recent_text, max_chapter=None)
        messages  = build_prompt(recent_text=recent_text, summary="",
                                 retrieval=retrieval, instruction="", prior_summary="")
        msgs_gen  = [m for m in messages if m["role"] != "assistant"]

        try:
            input_text = tokenizer.apply_chat_template(
                msgs_gen, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            input_text = tokenizer.apply_chat_template(
                msgs_gen, tokenize=False, add_generation_prompt=True)

        inputs    = tokenizer(input_text, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        print(f"  input tokens: {input_len}  (max_seq 利用率 {input_len/MAX_SEQ_LENGTH*100:.1f}%)")

        torch.cuda.reset_peak_memory_stats()
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
        new_ids   = output_ids[0][input_len:]
        new_text  = strip_think(tokenizer.decode(new_ids, skip_special_tokens=True))
        n_new_tok = len(new_ids)
        peak_rnd  = vram_report(f"Round {rnd} 生成后")
        peak_per_round.append(peak_rnd)

        print(f"  生成: {n_new_tok} tokens → {len(new_text)}c")
        if new_text:
            print(f"  预览: {new_text[:120]}")

        if not new_text or len(new_text) < 5:
            print("  [WARN] 空生成，提前停止。")
            break

        kept_text, was_truncated, was_skipped = dedup_truncate(
            new_text, seen_para_keys, _split, norm_unit)

        if was_truncated:
            truncation_rounds += 1
            print(f"  [DEDUP] 截断: {len(new_text)}c -> {len(kept_text)}c")
        if was_skipped:
            skip_rounds += 1
            print(f"  [SKIP] 整轮均重复，跳过。")
            continue

        text_to_add = kept_text if was_truncated else new_text
        for p in _split(text_to_add):
            key = norm_unit(p)
            if len(key) >= _MIN_DEDUP_CHARS:
                seen_para_keys.add(key)

        all_new_texts.append(text_to_add)
        accumulated_text += "\n\n" + text_to_add
        total_new_chars  += len(text_to_add)
        print(f"  [接受] 本轮 +{len(text_to_add)}c，累积新文本: {total_new_chars}c")

        if total_new_chars >= TARGET_CHARS:
            print(f"\n  目标 {TARGET_CHARS}c 已达到，停止。")
            break

    # ── 生成完成，汇总 ───────────────────────────────────────────────────────
    candidate = "\n\n".join(all_new_texts)
    out = Path(OUTPUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(candidate, encoding="utf-8")

    print(f"\n{'='*65}")
    print("  Deduplication Summary")
    print(f"{'='*65}")
    print(f"  完成轮数   : {len(all_new_texts)}")
    print(f"  总新文本   : {len(candidate)}c")
    print(f"  去重截断   : {truncation_rounds} 轮")
    print(f"  整轮跳过   : {skip_rounds} 轮")
    print(f"  保存路径   : {OUTPUT_FILE}")
    print()
    print("  逐轮显存峰值：")
    for i, pk in enumerate(peak_per_round, 1):
        flag = "[OK]" if pk < 8.3 else ("[WARN]" if pk < 8.5 else "[OOM]")
        print(f"    Round {i}: {pk:.3f} GB {flag}")

    print(f"\n{'='*65}")
    print("  最终【当前上文】（最后 800c，供人工确认干净）")
    print(f"{'='*65}")
    final_context = accumulated_text[-MAX_RECENT_CHARS:]
    print(final_context)

    print(f"\n{'='*65}")
    print("  生成候选全文")
    print(f"{'='*65}")
    print(candidate)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
