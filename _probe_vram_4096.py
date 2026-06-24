#!/usr/bin/env python3
"""
4096 配置显存探针 — 模拟真实单轮续写推理的 VRAM 占用。

与之前 50-token 测试的区别：
  - max_seq_length : 4096（之前 8192）
  - max_new_tokens : 400（之前 50；实际单轮约 600-700c 汉字需要约 400-500t）
  - 推理链路       : Retriever.retrieve + build_prompt（与 adapter_cli.py 完全一致）
  - 上文           : test_context_ch1_clean.txt（720c 干净文本）

VRAM 报告三个阶段：加载后、首轮 400-token 生成后、清理后。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ADAPTER_PATH    = "outputs/qlora_run_v4"
CONTEXT_FILE    = "outputs/debug/test_context_ch1_clean.txt"
CONFIG_PATH     = "config.yaml"
MAX_SEQ_LENGTH  = 4096
MAX_NEW_TOKENS  = 400   # realistic single-round: ~600-700c Chinese ~ 400-500 tokens
TEMPERATURE     = 0.8
TOP_P           = 0.8
TOP_K           = 20
REPETITION_PENALTY = 1.15
MAX_RECENT_CHARS   = 2000
BUDGET_GB          = 8.59  # RTX 4070 Laptop hardware total


def strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


def vram_report(label: str):
    import torch
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    peak  = torch.cuda.max_memory_allocated() / 1024**3
    margin = BUDGET_GB - peak
    print(f"[VRAM] {label}")
    print(f"       alloc={alloc:.3f} GB  reserved={reserv:.3f} GB  peak={peak:.3f} GB")
    print(f"       余量(budget - peak) = {margin:.3f} GB")
    return peak, margin


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not found — activate .venv-train/", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    import yaml
    import gc

    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}", file=sys.stderr)
        return 1

    ctx_path = Path(CONTEXT_FILE)
    if not ctx_path.exists():
        print(f"ERROR: context file not found: {ctx_path}", file=sys.stderr)
        return 1

    context_text = ctx_path.read_text(encoding="utf-8").strip()
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))

    print("=" * 60)
    print(f"  4096 配置显存探针")
    print(f"  max_seq_length : {MAX_SEQ_LENGTH}")
    print(f"  max_new_tokens : {MAX_NEW_TOKENS}")
    print(f"  上文            : {len(context_text)}c")
    print(f"  硬件总量        : {BUDGET_GB} GB")
    print("=" * 60)

    # ── Retriever ────────────────────────────────────────────────────────────
    print("\n[Step 1] 初始化 Retriever...")
    from cowriter.retriever import Retriever
    retriever = Retriever(cfg)
    print(f"  {len(retriever._docs)} bible docs indexed")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"\n[Step 2] 加载 v4 adapter (max_seq_length={MAX_SEQ_LENGTH})...")
    torch.cuda.reset_peak_memory_stats()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    peak_load, margin_load = vram_report("加载后")

    # ── 构建真实 Prompt ───────────────────────────────────────────────────────
    print("\n[Step 3] 构建推理 prompt（Retriever + build_prompt）...")
    from cowriter.prompts import build_prompt
    recent_text = context_text[-MAX_RECENT_CHARS:]
    retrieval = retriever.retrieve(recent_text, max_chapter=None)
    messages = build_prompt(
        recent_text=recent_text,
        summary="",
        retrieval=retrieval,
        instruction="",
        prior_summary="",
    )
    msgs_gen = [m for m in messages if m["role"] != "assistant"]

    try:
        input_text = tokenizer.apply_chat_template(
            msgs_gen, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            msgs_gen, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    n_input_tokens = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {n_input_tokens}")
    print(f"  max_seq_length 利用率: {n_input_tokens}/{MAX_SEQ_LENGTH} = {n_input_tokens/MAX_SEQ_LENGTH*100:.1f}%")

    # ── 生成 400 token ────────────────────────────────────────────────────────
    print(f"\n[Step 4] 生成 {MAX_NEW_TOKENS} tokens...")
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
    new_ids = output_ids[0][n_input_tokens:]
    generated_text = strip_think(tokenizer.decode(new_ids, skip_special_tokens=True))
    n_new_tokens = len(new_ids)

    peak_gen, margin_gen = vram_report(f"生成后（{n_new_tokens} new tokens，约 {len(generated_text)}c）")

    print(f"\n  [续写预览（前 200c）]")
    print(f"  {generated_text[:200]}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VRAM 汇总对比")
    print("=" * 60)
    print(f"  配置              : max_seq_length={MAX_SEQ_LENGTH}, max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  Input tokens      : {n_input_tokens}")
    print(f"  New tokens        : {n_new_tokens}")
    print(f"  峰值（加载后）     : {peak_load:.3f} GB  余量 {margin_load:.3f} GB")
    print(f"  峰值（生成后）     : {peak_gen:.3f} GB   余量 {margin_gen:.3f} GB  ← 关键指标")
    print(f"  上次(8192配置)对比 : 8.47 GB            余量 0.12  GB")
    print(f"  改善              : {8.47 - peak_gen:+.3f} GB")
    print()
    if margin_gen >= 1.0:
        print(f"  ✅ 4096配置安全：余量 {margin_gen:.2f} GB >= 1.0 GB 阈值，可进入多轮生成。")
    elif margin_gen >= 0.5:
        print(f"  ⚠️  4096配置余量 {margin_gen:.2f} GB（0.5-1.0 GB之间），多轮累积可能偏紧。")
    else:
        print(f"  ❌ 4096配置余量 {margin_gen:.2f} GB < 0.5 GB，多轮生成风险较高。")
    print("=" * 60)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
