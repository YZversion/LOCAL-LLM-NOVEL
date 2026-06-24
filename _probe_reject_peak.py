#!/usr/bin/env python3
"""
/reject 双份峰值测量探针

完全模拟 adapter_cli.py 的 /reject 路径：
  - 生成完成后只调用 reset_peak_memory_stats()，不调用 empty_cache()
  - 立即开始下一次 generate()，模拟用户按下 /reject 的时序

测量三个阶段：
  Pass 1 : 正常首次生成（对照）
  Pass 2 : /reject 后第一次重新生成（前次张量可能尚在缓存）
  Pass 3 : 连续第二次 /reject（模拟用户再次不满意）

每次 reset_peak_memory_stats() 在 generate() 开始前调用（和 cli 一致）。
"""
import sys, gc
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ADAPTER_PATH    = "outputs/qlora_run_v4"
CONTEXT_FILE    = "outputs/debug/test_context_ch1_clean.txt"
CONFIG_PATH     = "config.yaml"
MAX_SEQ_LENGTH  = 4096
MAX_NEW_TOKENS  = 400
TEMPERATURE     = 0.8
TOP_P           = 0.8
TOP_K           = 20
REPETITION_PENALTY = 1.15
MAX_RECENT_CHARS   = 2000
BUDGET_GB          = 8.59


def strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


def vram_snapshot(label: str) -> dict:
    import torch
    alloc   = torch.cuda.memory_allocated() / 1024**3
    reserv  = torch.cuda.memory_reserved()  / 1024**3
    peak    = torch.cuda.max_memory_allocated() / 1024**3
    margin  = BUDGET_GB - peak
    print(f"  [VRAM] {label}")
    print(f"         alloc={alloc:.3f} GB  reserved={reserv:.3f} GB  peak={peak:.3f} GB  余量={margin:.3f} GB")
    return {"label": label, "alloc": alloc, "reserved": reserv, "peak": peak, "margin": margin}


def do_one_generate(model, tokenizer, inputs, input_len) -> tuple[str, int]:
    """Returns (text, n_new_tokens). Matches _generate_one_round() exactly."""
    import torch
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
    n_new     = len(new_ids)
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return strip_think(generated), n_new
    # output_ids, new_ids go out of scope here → refcount → freed (but cached by pytorch allocator)


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

    adapter_path = Path(ADAPTER_PATH)
    context_text = Path(CONTEXT_FILE).read_text(encoding="utf-8").strip()
    cfg          = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))

    print("=" * 65)
    print("  /reject 双份峰值测量探针")
    print(f"  max_seq_length={MAX_SEQ_LENGTH}  max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  硬件总量={BUDGET_GB} GB")
    print("=" * 65)

    # ── 1. 加载 ──────────────────────────────────────────────────────────────
    print("\n[Step 1] 加载 Retriever...")
    from cowriter.retriever import Retriever
    from cowriter.prompts  import build_prompt
    retriever = Retriever(cfg)

    print(f"[Step 2] 加载 v4 adapter (max_seq_length={MAX_SEQ_LENGTH})...")
    torch.cuda.reset_peak_memory_stats()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    snap_load = vram_snapshot("加载后")

    # ── 2. 构建 Prompt（每次 /reject 用完全相同的 msgs_gen）────────────────
    print("\n[Step 3] 构建推理 prompt...")
    recent_text = context_text[-MAX_RECENT_CHARS:]
    retrieval   = retriever.retrieve(recent_text, max_chapter=None)
    messages    = build_prompt(
        recent_text=recent_text, summary="",
        retrieval=retrieval, instruction="", prior_summary="",
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
    inputs    = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_len}  (max_seq 利用率 {input_len/MAX_SEQ_LENGTH*100:.1f}%)")

    # ── 3. Pass 1: 正常首次生成（对照）───────────────────────────────────────
    print("\n[Pass 1] 正常首次生成（对照）")
    torch.cuda.reset_peak_memory_stats()
    text1, n1 = do_one_generate(model, tokenizer, inputs, input_len)
    snap1 = vram_snapshot(f"Pass1 生成后 ({n1} new tokens, {len(text1)}c)")
    # === 此时 output_ids 已超出 do_one_generate 的作用域 ===
    # PyTorch allocator 将其内存放入缓存，但不会立即归还给 CUDA driver
    # 这就是"前次张量可能仍在缓存"的状态

    # ── 4. Pass 2: /reject 路径（不调用 empty_cache，直接再次 generate）────
    print("\n[Pass 2] /reject 第1次（前次张量尚在 allocator 缓存，直接重新生成）")
    # ↑ 这里完全复制 adapter_cli.py 的 /reject handler：
    #   torch.cuda.reset_peak_memory_stats()  ← cli 有，此处也有
    #   new_text = _generate_one_round(...)   ← 重新生成
    torch.cuda.reset_peak_memory_stats()
    text2, n2 = do_one_generate(model, tokenizer, inputs, input_len)
    snap2 = vram_snapshot(f"Pass2 生成后 ({n2} new tokens, {len(text2)}c)")

    # ── 5. Pass 3: 连续第二次 /reject ────────────────────────────────────────
    print("\n[Pass 3] /reject 第2次（连续双reject）")
    torch.cuda.reset_peak_memory_stats()
    text3, n3 = do_one_generate(model, tokenizer, inputs, input_len)
    snap3 = vram_snapshot(f"Pass3 生成后 ({n3} new tokens, {len(text3)}c)")

    # ── 6. 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  峰值汇总")
    print("=" * 65)
    print(f"  {'场景':<30} {'peak':>8}  {'余量':>8}")
    print(f"  {'-'*30} {'-------':>8}  {'-------':>8}")
    for snap in [snap_load, snap1, snap2, snap3]:
        print(f"  {snap['label']:<30} {snap['peak']:>7.3f}GB  {snap['margin']:>7.3f}GB")
    print()

    peak_gen   = snap1["peak"]
    peak_rej1  = snap2["peak"]
    peak_rej2  = snap3["peak"]
    worst_peak = max(peak_gen, peak_rej1, peak_rej2)
    worst_margin = BUDGET_GB - worst_peak

    print(f"  单次生成峰值              : {peak_gen:.3f} GB")
    print(f"  /reject 第1次峰值         : {peak_rej1:.3f} GB  (Δ {peak_rej1-peak_gen:+.3f} GB vs 单次)")
    print(f"  /reject 第2次峰值         : {peak_rej2:.3f} GB  (Δ {peak_rej2-peak_gen:+.3f} GB vs 单次)")
    print(f"  最坏场景峰值              : {worst_peak:.3f} GB")
    print(f"  最坏场景余量              : {worst_margin:.3f} GB")
    print()

    if worst_margin >= 0.5:
        print(f"  ✅  /reject 路径峰值 {worst_peak:.3f} GB，余量 {worst_margin:.3f} GB >= 0.5 GB 阈值。")
        print(f"      4096 配置下，/reject 场景可以接受。")
    elif worst_margin >= 0.2:
        print(f"  ⚠️  /reject 路径峰值 {worst_peak:.3f} GB，余量 {worst_margin:.3f} GB（偏紧）。")
        print(f"      建议在 adapter_cli.py 的 /reject handler 加入 empty_cache()。")
    else:
        print(f"  ❌  /reject 路径峰值 {worst_peak:.3f} GB，余量 {worst_margin:.3f} GB < 0.2 GB（危险）。")
        print(f"      必须在 /reject handler 加入显式 gc.collect() + empty_cache()。")
    print("=" * 65)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
