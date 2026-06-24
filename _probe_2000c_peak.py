#!/usr/bin/env python3
"""
2000c 输入下的真实最坏情况峰值测试。

步骤：
  1. 从风丝引_原文.txt 截取连续 ~2000c 文本（从第二章起，避免前言/章标题）
  2. 用 _check_internal_duplicates 自查内部重复（与 adapter_cli.py 逻辑完全一致）
  3. Pass1：正常生成一次（400 max_new_tokens），报告基础峰值
  4. Pass2：/reject 路径（gc.collect+empty_cache 先行，与修复后的 adapter_cli.py 一致），报告峰值
  5. 汇总：与 720c 实测值（7.891 GB）直接对比，给出明确安全判断

硬件预算：8.59 GB（RTX 4070 Laptop）
安全阈值判断：最坏峰值 >= 8.3 GB（余量 < 0.3 GB）视为不安全。
"""
import sys, gc
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ADAPTER_PATH    = "outputs/qlora_run_v4"
RAW_FILE        = "data/raw/风丝引_原文.txt"
CONFIG_PATH     = "config.yaml"
MAX_SEQ_LENGTH  = 4096
MAX_NEW_TOKENS  = 400
TEMPERATURE     = 0.8
TOP_P           = 0.8
TOP_K           = 20
REPETITION_PENALTY = 1.15
MAX_RECENT_CHARS   = 2000
BUDGET_GB          = 8.59
TARGET_CONTEXT_C   = 2000   # 目标上文长度
_MIN_DEDUP_CHARS   = 10

# 安全阈值：余量小于此值视为不安全
SAFETY_MARGIN_GB   = 0.30


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
        print("ERROR: torch not found — activate .venv-train/", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    import yaml

    # ── 1. 从原文提取 ~2000c 文本 ────────────────────────────────────────────
    print("=" * 65)
    print("  2000c 输入峰值测试探针")
    print(f"  max_seq_length={MAX_SEQ_LENGTH}  max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  硬件总量={BUDGET_GB} GB  安全阈值：余量>{SAFETY_MARGIN_GB} GB")
    print("=" * 65)

    raw_text = Path(RAW_FILE).read_text(encoding="utf-8")

    # 从第二章正文开始截取，跳过前言/楔子/章节标题行
    # 找"第二章"位置后，找第一个正文行（非空、非章节标题）
    ch2_pos = raw_text.find("第二章")
    if ch2_pos == -1:
        print("ERROR: 找不到第二章标记", file=sys.stderr)
        return 1

    # 跳过章节标题行（找第二章之后的第一个非空非标题行）
    lines_after_ch2 = raw_text[ch2_pos:].split("\n")
    content_start_offset = 0
    char_count = 0
    skip_lines = 3  # 跳过"第二章 XXX"、空行
    for i, line in enumerate(lines_after_ch2):
        if i < skip_lines:
            content_start_offset += len(line) + 1
            continue
        if line.strip():  # 第一个非空行：正文开始
            break
        content_start_offset += len(line) + 1

    start_pos = ch2_pos + content_start_offset
    context_2000c = raw_text[start_pos:start_pos + TARGET_CONTEXT_C].strip()
    actual_len = len(context_2000c)

    print(f"\n[Step 1] 截取测试文本")
    print(f"  起点: 第二章正文开始（pos {start_pos}）")
    print(f"  长度: {actual_len}c（目标 {TARGET_CONTEXT_C}c）")
    print(f"  前50c: {context_2000c[:50]}...")

    # ── 2. 内部重复自查 ────────────────────────────────────────────────────────
    print("\n[Step 2] 内部重复自查（与 adapter_cli.py _check_internal_duplicates 一致）")
    from pipeline.eval_style import norm_unit, split_paragraphs as _split_paragraphs

    paras = _split_paragraphs(context_2000c)
    seen_keys: set = set()
    dupes: list = []
    for p in paras:
        key = norm_unit(p)
        if len(key) < _MIN_DEDUP_CHARS:
            continue
        if key in seen_keys:
            dupes.append(p[:60])
        else:
            seen_keys.add(key)

    if dupes:
        print(f"  [WARN] 检测到 {len(dupes)} 处内部重复：")
        for d in dupes:
            print(f"    - {d}")
        print("  测试文本不干净，请换取一段文本。")
        return 1
    else:
        para_count = len([p for p in paras if len(norm_unit(p)) >= _MIN_DEDUP_CHARS])
        print(f"  [OK] 无内部重复（{para_count} 段落，均唯一）")

    # ── 3. 加载模型 ───────────────────────────────────────────────────────────
    import yaml
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))

    print(f"\n[Step 3] 加载 Retriever + v4 adapter...")
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

    # ── 4. 构建 2000c 上文的 Prompt ──────────────────────────────────────────
    print(f"\n[Step 4] 构建 prompt（上文={actual_len}c，MAX_RECENT_CHARS={MAX_RECENT_CHARS}）")
    # MAX_RECENT_CHARS=2000，上文恰好约为上限
    recent_text = context_2000c[-MAX_RECENT_CHARS:]
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
    print(f"  Input tokens: {input_len}  (max_seq 利用率 {input_len/MAX_SEQ_LENGTH*100:.1f}%)")
    print(f"  对比：720c 上文时 input_len=2263 tokens")
    print(f"  输入 token 增量：{input_len - 2263:+d} tokens")

    # ── 5. Pass1：正常生成（基础峰值）────────────────────────────────────────
    print(f"\n[Pass 1] 正常生成（2000c 输入，基础峰值）")
    torch.cuda.reset_peak_memory_stats()
    text1, n1 = do_one_generate(model, tokenizer, inputs, input_len)
    snap1 = vram_snapshot(f"Pass1 ({n1} new tokens, {len(text1)}c)")

    # ── 6. Pass2：/reject 路径（修复后：先 gc+empty_cache）─────────────────
    print(f"\n[Pass 2] /reject（修复后路径：gc.collect+empty_cache 先行）")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text2, n2 = do_one_generate(model, tokenizer, inputs, input_len)
    snap2 = vram_snapshot(f"Pass2 /reject ({n2} new tokens, {len(text2)}c)")

    # ── 7. 汇总与安全判断 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  最终汇总：2000c vs 720c 对比")
    print("=" * 65)
    print(f"  {'场景':<40} {'peak':>8}  {'余量':>8}")
    print(f"  {'-'*40} {'-'*8}  {'-'*8}")
    rows = [
        ("720c 加载后（参考）",        6.019,            2.571),
        ("720c 正常生成（参考）",       7.891,            0.699),
        ("720c /reject 修复后（参考）", 7.891,            0.699),
        (snap_load["label"] + "（2000c）", snap_load["peak"], snap_load["margin"]),
        (snap1["label"] + "（2000c）",    snap1["peak"],    snap1["margin"]),
        (snap2["label"] + "（2000c）",    snap2["peak"],    snap2["margin"]),
    ]
    for label, peak, margin in rows:
        flag = " ←" if "2000c" in label else ""
        print(f"  {label:<40} {peak:>7.3f}GB  {margin:>7.3f}GB{flag}")

    print()
    p1_ref  = 7.891   # 720c 正常生成参考值
    p1_2000 = snap1["peak"]
    p2_2000 = snap2["peak"]
    worst   = max(p1_2000, p2_2000)
    worst_m = BUDGET_GB - worst

    print(f"  输入从 720c→2000c，基础峰值变化   : {p1_ref:.3f} → {p1_2000:.3f} GB  (Δ {p1_2000-p1_ref:+.3f} GB)")
    print(f"  /reject 修复后峰值（2000c 场景）   : {p2_2000:.3f} GB")
    print(f"  最坏场景峰值                       : {worst:.3f} GB")
    print(f"  最坏场景余量                       : {worst_m:.3f} GB")
    print()

    if worst_m >= SAFETY_MARGIN_GB:
        print(f"  [PASS] 2000c 场景最坏峰值 {worst:.3f} GB，余量 {worst_m:.3f} GB >= {SAFETY_MARGIN_GB} GB 阈值。")
        print(f"         4096 配置 + 选项A修复 经完整测试（720c 和 2000c 两个场景），确认安全。")
        print(f"         可以进入下一步：生成正式 v4 评测候选。")
    else:
        print(f"  [FAIL] 2000c 场景最坏峰值 {worst:.3f} GB，余量 {worst_m:.3f} GB < {SAFETY_MARGIN_GB} GB 阈值。")
        print(f"         当前配置不安全，不能直接进入生成正式评测候选。")
        print(f"         备选方向（等用户决定，不擅自执行）：")
        print(f"           A2. 缩减 MAX_RECENT_CHARS（如从 2000c 降至 1500c），减少输入 token 数")
        print(f"           B2. 进一步降低 max-seq-length（但 2048 已低于当前 input token 数，不可用）")
        print(f"           C2. 重新评估是否接受现有余量（如果真实评测上文不会达到 2000c）")
    print("=" * 65)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
