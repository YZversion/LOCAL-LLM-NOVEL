#!/usr/bin/env python3
"""
800c MAX_RECENT_CHARS 配置下的真实显存测试。

使用 ~900c 的原文文本作为输入，让 MAX_RECENT_CHARS=800 的截断
真正生效（取 recent_text = context[-800:]），与 adapter_cli.py 修改后
的行为完全一致。

测量：
  Pass1 : 正常生成（400 max_new_tokens），基础峰值
  Pass2 : /reject 路径（gc.collect+empty_cache 先行，与 adapter_cli.py 一致）

对比基准：720c 实测 7.891 GB，估算 800c ~7.96 GB。
硬件总量：8.59 GB
安全阈值：余量 > 0.3 GB
"""
import sys, gc
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ADAPTER_PATH       = "outputs/qlora_run_v4"
RAW_FILE           = "data/raw/风丝引_原文.txt"
CONFIG_PATH        = "config.yaml"
MAX_SEQ_LENGTH     = 4096
MAX_NEW_TOKENS     = 400
TEMPERATURE        = 0.8
TOP_P              = 0.8
TOP_K              = 20
REPETITION_PENALTY = 1.15
MAX_RECENT_CHARS   = 800   # 与 adapter_cli.py 修改后保持一致
TARGET_EXTRACT_C   = 900   # 截取略多于 800c，确保 MAX_RECENT_CHARS 截断生效
BUDGET_GB          = 8.59
SAFETY_MARGIN_GB   = 0.30
_MIN_DEDUP_CHARS   = 10


def strip_think(text):
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


def vram_snapshot(label):
    import torch
    alloc  = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved()  / 1024**3
    peak   = torch.cuda.max_memory_allocated() / 1024**3
    margin = BUDGET_GB - peak
    print(f"  [VRAM] {label}")
    print(f"         alloc={alloc:.3f} GB  reserved={reserv:.3f} GB"
          f"  peak={peak:.3f} GB  margin={margin:.3f} GB")
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
        print("ERROR: torch not found", file=sys.stderr); return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr); return 1

    import yaml

    print("=" * 65)
    print("  800c MAX_RECENT_CHARS 配置下的真实显存测试")
    print(f"  max_seq_length={MAX_SEQ_LENGTH}  MAX_RECENT_CHARS={MAX_RECENT_CHARS}")
    print(f"  max_new_tokens={MAX_NEW_TOKENS}  budget={BUDGET_GB} GB")
    print("=" * 65)

    # ── 1. 从原文截取 ~900c 文本（确保 800c 截断生效）────────────────────
    print("\n[Step 1] 准备测试文本（目标 ~900c，让 800c 截断实际生效）")
    raw_text = Path(RAW_FILE).read_text(encoding="utf-8")

    # 从第二章正文开始，跳过章节标题行
    ch2_pos = raw_text.find("第二章")
    lines_after = raw_text[ch2_pos:].split("\n")
    offset = 0
    for i, line in enumerate(lines_after):
        if i < 3:
            offset += len(line) + 1
            continue
        if line.strip():
            break
        offset += len(line) + 1
    start_pos = ch2_pos + offset
    raw_900c = raw_text[start_pos:start_pos + TARGET_EXTRACT_C].strip()
    actual_len = len(raw_900c)

    print(f"  截取长度: {actual_len}c（起点 pos {start_pos}）")
    print(f"  前60c: {raw_900c[:60]}...")

    # ── 2. 内部重复检测 ──────────────────────────────────────────────────
    print("\n[Step 2] 内部重复自查")
    from pipeline.eval_style import norm_unit, split_paragraphs as _split_paragraphs
    paras = _split_paragraphs(raw_900c)
    seen: set = set()
    dupes: list = []
    for p in paras:
        key = norm_unit(p)
        if len(key) < _MIN_DEDUP_CHARS: continue
        if key in seen: dupes.append(p[:60])
        else: seen.add(key)
    if dupes:
        print(f"  [WARN] 检测到 {len(dupes)} 处内部重复，停止测试。")
        return 1
    para_count = len([p for p in paras if len(norm_unit(p)) >= _MIN_DEDUP_CHARS])
    print(f"  [OK] 无内部重复（{para_count} 段落）")

    # recent_text = 截取 context 的后 800c，模拟运行时行为
    recent_text_actual = raw_900c[-MAX_RECENT_CHARS:]
    print(f"  recent_text（截断后）: {len(recent_text_actual)}c（原始 {actual_len}c → 截取后 {MAX_RECENT_CHARS}c）")

    # ── 3. 加载模型 ──────────────────────────────────────────────────────
    print("\n[Step 3] 加载 Retriever + v4 adapter...")
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    from cowriter.retriever import Retriever
    from cowriter.prompts  import build_prompt
    retriever = Retriever(cfg)
    print(f"  {len(retriever._docs)} bible docs indexed")

    torch.cuda.reset_peak_memory_stats()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    snap_load = vram_snapshot("加载后")

    # ── 4. 构建 prompt（MAX_RECENT_CHARS=800 后的 recent_text）──────────
    print(f"\n[Step 4] 构建 prompt（recent_text={len(recent_text_actual)}c）")
    retrieval = retriever.retrieve(recent_text_actual, max_chapter=None)
    messages  = build_prompt(recent_text=recent_text_actual, summary="",
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
    print(f"  Input tokens: {input_len}  (max_seq 利用率 {input_len/MAX_SEQ_LENGTH*100:.1f}%)")
    print(f"  对比: 720c 时 2263 tokens，2000c 时 3329 tokens，800c 估算 ~2330 tokens")

    # ── 5. Pass1：正常生成 ────────────────────────────────────────────────
    print("\n[Pass 1] 正常生成（800c recent_text）")
    torch.cuda.reset_peak_memory_stats()
    text1, n1 = do_one_generate(model, tokenizer, inputs, input_len)
    snap1 = vram_snapshot(f"Pass1 ({n1} new tokens, {len(text1)}c)")

    # ── 6. Pass2：/reject 修复后路径 ─────────────────────────────────────
    print("\n[Pass 2] /reject（gc.collect+empty_cache 先行，对应修复后 adapter_cli.py）")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text2, n2 = do_one_generate(model, tokenizer, inputs, input_len)
    snap2 = vram_snapshot(f"Pass2 /reject ({n2} new tokens, {len(text2)}c)")

    # ── 7. 汇总判断 ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  汇总：800c MAX_RECENT_CHARS 配置验证")
    print("=" * 65)
    header = f"  {'场景':<42} {'peak':>8}  {'margin':>8}"
    print(header)
    print(f"  {'-'*42} {'-'*8}  {'-'*8}")
    refs = [
        ("720c 正常生成（实测参考）",        7.891, 0.699),
        ("720c /reject 修复后（实测参考）",  7.891, 0.699),
        ("2000c 正常生成（实测，OOM）",       9.759, -1.169),
        ("800c 线性估算（参考）",             7.960, 0.630),
    ]
    for label, peak, margin in refs:
        print(f"  {label:<42} {peak:>7.3f}GB  {margin:>7.3f}GB")
    print(f"  {'-'*42} {'-'*8}  {'-'*8}")
    for snap in [snap_load, snap1, snap2]:
        tag = " <-- 实测" if snap is not snap_load else ""
        print(f"  {snap['label']:<42} {snap['peak']:>7.3f}GB  {snap['margin']:>7.3f}GB{tag}")

    worst = max(snap1["peak"], snap2["peak"])
    worst_m = BUDGET_GB - worst

    print()
    print(f"  input tokens（800c recent_text） : {input_len}")
    print(f"  Pass1 基础峰值                   : {snap1['peak']:.3f} GB  (估算差异: {snap1['peak']-7.960:+.3f} GB)")
    print(f"  Pass2 /reject 峰值               : {snap2['peak']:.3f} GB")
    print(f"  最坏场景峰值                     : {worst:.3f} GB")
    print(f"  最坏场景余量                     : {worst_m:.3f} GB")
    print()

    if worst_m >= SAFETY_MARGIN_GB:
        print(f"  [PASS] 800c 配置安全：最坏峰值 {worst:.3f} GB，余量 {worst_m:.3f} GB >= {SAFETY_MARGIN_GB} GB。")
        print(f"         adapter_cli.py MAX_RECENT_CHARS=800 经实测验证，可进入第二步。")
    else:
        print(f"  [FAIL] 800c 配置不安全：最坏峰值 {worst:.3f} GB，余量 {worst_m:.3f} GB < {SAFETY_MARGIN_GB} GB。")
        print(f"         建议备选值（等用户决定，不擅自执行）：")
        print(f"           - MAX_RECENT_CHARS=600c（估算 input ~2183 tokens，峰值 ~7.7 GB）")
        print(f"           - 或重新评估 max_seq_length / 生成轮数限制")
    print("=" * 65)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
