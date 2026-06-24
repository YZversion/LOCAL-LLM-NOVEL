#!/usr/bin/env python3
"""
验证 /reject handler 加入 gc.collect()+empty_cache() 后的峰值变化。
与 _probe_reject_peak.py 的区别：
  Pass2/Pass3 在 reset_peak_memory_stats() 之前调用 gc.collect()+empty_cache()
  完全对应修改后的 adapter_cli.py 路径。
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


def strip_think(text):
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


def vram_snapshot(label):
    import torch
    alloc   = torch.cuda.memory_allocated() / 1024**3
    reserv  = torch.cuda.memory_reserved()  / 1024**3
    peak    = torch.cuda.max_memory_allocated() / 1024**3
    margin  = BUDGET_GB - peak
    print(f"  [VRAM] {label}")
    print(f"         alloc={alloc:.3f} GB  reserved={reserv:.3f} GB  peak={peak:.3f} GB  余量={margin:.3f} GB")
    return {"label": label, "peak": peak, "margin": margin}


def do_one_generate(model, tokenizer, inputs, input_len):
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


def main():
    try:
        import torch
    except ImportError:
        print("ERROR: torch not found"); return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); return 1

    import yaml

    context_text = Path(CONTEXT_FILE).read_text(encoding="utf-8").strip()
    cfg          = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))

    print("=" * 65)
    print("  /reject 修复后验证探针（加入 gc.collect()+empty_cache()）")
    print(f"  max_seq_length={MAX_SEQ_LENGTH}  max_new_tokens={MAX_NEW_TOKENS}")
    print("=" * 65)

    print("\n[Step 1] 加载 Retriever + 模型...")
    from cowriter.retriever import Retriever
    from cowriter.prompts  import build_prompt
    retriever = Retriever(cfg)

    torch.cuda.reset_peak_memory_stats()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    snap_load = vram_snapshot("加载后")

    print("\n[Step 2] 构建 prompt...")
    recent_text = context_text[-MAX_RECENT_CHARS:]
    retrieval   = retriever.retrieve(recent_text, max_chapter=None)
    messages    = build_prompt(recent_text=recent_text, summary="",
                               retrieval=retrieval, instruction="", prior_summary="")
    msgs_gen    = [m for m in messages if m["role"] != "assistant"]
    try:
        input_text = tokenizer.apply_chat_template(
            msgs_gen, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            msgs_gen, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_len}")

    # Pass 1: 正常生成
    print("\n[Pass 1] 正常首次生成（对照）")
    torch.cuda.reset_peak_memory_stats()
    text1, n1 = do_one_generate(model, tokenizer, inputs, input_len)
    snap1 = vram_snapshot(f"Pass1 ({n1} new tokens)")

    # Pass 2: /reject 修复后路径
    print("\n[Pass 2] /reject 第1次（修复后：empty_cache 先行）")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text2, n2 = do_one_generate(model, tokenizer, inputs, input_len)
    snap2 = vram_snapshot(f"Pass2 ({n2} new tokens)")

    # Pass 3: 连续第二次 /reject
    print("\n[Pass 3] /reject 第2次（连续双reject，修复后）")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text3, n3 = do_one_generate(model, tokenizer, inputs, input_len)
    snap3 = vram_snapshot(f"Pass3 ({n3} new tokens)")

    # 汇总对比
    print("\n" + "=" * 65)
    print("  峰值汇总（修复后 vs 修复前）")
    print("=" * 65)
    print(f"  {'场景':<35} {'peak':>8}  {'余量':>8}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}")
    for s in [snap_load, snap1, snap2, snap3]:
        print(f"  {s['label']:<35} {s['peak']:>7.3f}GB  {s['margin']:>7.3f}GB")

    print()
    p1, p2, p3 = snap1["peak"], snap2["peak"], snap3["peak"]
    print(f"  Pass1 正常生成     : {p1:.3f} GB")
    print(f"  Pass2 /reject 第1次: {p2:.3f} GB  (Δ vs Pass1: {p2-p1:+.3f} GB)")
    print(f"  Pass3 /reject 第2次: {p3:.3f} GB  (Δ vs Pass1: {p3-p1:+.3f} GB)")
    print()
    print(f"  修复前 /reject 峰值: 8.045 GB（对比基准）")
    print(f"  修复后 /reject 峰值: {max(p2,p3):.3f} GB")
    print(f"  修复效果（降低）   : {8.045 - max(p2,p3):+.3f} GB")
    print(f"  修复后最差余量     : {BUDGET_GB - max(p2,p3):.3f} GB")
    print("=" * 65)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
