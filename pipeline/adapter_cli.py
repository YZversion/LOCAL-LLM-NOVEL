#!/usr/bin/env python3
"""
pipeline/adapter_cli.py — LoRA adapter 交互式续写 CLI

运行环境：.venv-train/（同 train_qlora.py / generate_lora_multi.py）

功能：用指定 adapter 做多轮续写，全程 rich 显示，支持查看 prompt 结构、
      跨轮去重、草稿保存。初始上文做内部重复自检，防止输入污染。

Usage:
    .venv-train\\Scripts\\Activate.ps1
    python pipeline/adapter_cli.py --adapter outputs/qlora_run_v4/
    python pipeline/adapter_cli.py --adapter outputs/qlora_run_v4/ \\
        --context-file outputs/debug/test_context_ch1_clean.txt
    python pipeline/adapter_cli.py --adapter outputs/qlora_run_v4/ \\
        --raw-prompt-file outputs/eval_anchors/ch1_clean.txt
    python pipeline/adapter_cli.py \\
        --adapter huihui-ai/Huihui-Qwen3-8B-abliterated-v2 \\
        --raw-prompt-file outputs/eval_anchors/ch1_clean.txt
"""
import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path
import re

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.eval_style import norm_unit
from pipeline.eval_style import split_paragraphs as _split_paragraphs

# ── 推理参数 ──────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH_INFER = 8192
MAX_NEW_TOKENS       = 800
TARGET_CHARS         = 2300
MAX_ROUNDS           = 8
MAX_RECENT_CHARS     = 800
TEMPERATURE          = 0.8
TOP_P                = 0.8
TOP_K                = 20
REPETITION_PENALTY   = 1.15
_MIN_DEDUP_CHARS     = 10

# 显存实测（8B 4-bit + v4 adapter，max_seq=8192，输入~2263t）
# 加载后: alloc=6.30 GB  peak=6.46 GB
# 生成时: peak=8.47 GB（硬件总量 8.59 GB，余量 ~120 MB，不宜再增大 max_seq）
_VRAM_LOAD_GB   = 6.30   # 加载后 alloc
_VRAM_INFER_PEAK_GB = 8.47  # 推理时峰值（实测）
_VRAM_BUDGET_GB = 8.59   # 硬件总量

HELP_TEXT = """
命令列表：
  [Enter]            接受当前生成，加入累积，进入下一轮
  文字 + Enter       用你的版本替换生成内容，加入累积
  /reject /拒绝      丢弃当前，重新生成
  /retry [指令]      丢弃当前，重新生成（可附加续写方向）
  /重试 [指令]       同 /retry
  /context /上下文   显示当前完整 prompt 结构（rich 面板）
  /save /保存        保存草稿到 outputs/adapter_candidate_<时间戳>.txt
  /help /帮助        显示本帮助
  /quit  q           退出（提示是否保存）
"""


# ── 去重工具（与 generate_lora_multi.py 逻辑完全一致）────────────────────────

def _dedup_truncate(new_text: str, seen_keys: set) -> tuple:
    """Return (kept_text, was_truncated, was_fully_skipped).

    Split new_text by paragraphs; stop at first para already in seen_keys.
    """
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


def _check_internal_duplicates(text: str) -> list[str]:
    """Detect paragraphs that appear more than once within text.

    Returns list of duplicate paragraph preview strings (first 60 chars each).
    """
    paras = _split_paragraphs(text)
    seen: set = set()
    dupes: list = []
    for p in paras:
        key = norm_unit(p)
        if len(key) < _MIN_DEDUP_CHARS:
            continue
        if key in seen:
            dupes.append(p[:60])
        else:
            seen.add(key)
    return dupes


# ── 生成 ──────────────────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</think>\s*", "", text)
    text = re.sub(r"\s*/no_think\s*", "", text)
    return text.strip()


def _generate_one_round(model, tokenizer, messages: list) -> str:
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
    return _strip_think(tokenizer.decode(new_ids, skip_special_tokens=True))


# ── Rich 显示 ─────────────────────────────────────────────────────────────────

def _vram_line(console) -> float:
    """Print one-line VRAM status; return current alloc_gb."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    peak_gb  = torch.cuda.max_memory_allocated() / 1024**3
    used_frac = peak_gb / _VRAM_BUDGET_GB
    bar  = "█" * int(used_frac * 20)
    bar  = bar.ljust(20, "░")
    color = "green" if peak_gb < 7.5 else ("yellow" if peak_gb < 8.3 else "red")
    console.print(
        f"  [{color}]VRAM alloc {alloc_gb:.2f} GB  peak {peak_gb:.2f} GB / {_VRAM_BUDGET_GB:.2f} GB  [{bar}][/{color}]"
    )
    return alloc_gb


def _parse_prompt_sections(user_content: str) -> list[tuple]:
    """Split user content string into [(title, body, rich_style), ...].

    Recognises the section markers written by build_prompt().
    """
    markers = [
        ("【相关设定】",     "cyan"),
        ("【前情提要】",     "blue"),
        ("【剧情摘要】",     "blue"),
        ("【原文命中段落】", "dim blue"),
        ("续写方向：",      "yellow"),
        ("【当前上文】",     "bright_green"),
    ]
    found = []
    for marker, style in markers:
        pos = user_content.find(marker)
        if pos != -1:
            found.append((pos, marker, style))
    found.sort(key=lambda x: x[0])

    sections = []
    for i, (pos, marker, style) in enumerate(found):
        start = pos + len(marker)
        end   = found[i + 1][0] if i + 1 < len(found) else len(user_content)
        body  = user_content[start:end].strip()
        sections.append((marker, body, style))
    return sections


def _show_prompt_panels(messages: list, retrieval: dict, console):
    """Render prompt structure as rich Panels (called by /context command)."""
    from rich.panel import Panel

    user_content = messages[1]["content"] if len(messages) > 1 else ""
    sections = _parse_prompt_sections(user_content)

    if not sections:
        console.print(Panel(user_content[:1000], title="[USER]", border_style="dim"))
        return

    for title, body, style in sections:
        # Cap display at 800c to keep terminal manageable; full content is still
        # used by the model — this is display only
        display_body = body if len(body) <= 800 else body[:800] + "\n[dim]…（已截断显示，模型实际接收完整内容）[/dim]"
        console.print(Panel(display_body, title=title, border_style=style, expand=True))

    bible_n = len(retrieval.get("bible", []))
    grep_n  = len(retrieval.get("grep", []))
    ctx_len = next((len(b) for t, b, _ in sections if "上文" in t), 0)
    console.print(
        f"[dim]  Bible docs: {bible_n} | Grep hits: {grep_n} | 上文: {ctx_len}c[/dim]"
    )


def _save_draft(all_new_texts: list) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path("outputs") / f"adapter_candidate_{ts}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(all_new_texts), encoding="utf-8")
    return str(out)


def _print_final_summary(console, rounds_done, total_chars,
                          trunc, skips, saved_path):
    from rich.table import Table
    t = Table(title="生成汇总", show_header=False, box=None, padding=(0, 2))
    t.add_column("指标", style="dim")
    t.add_column("值",   style="bold")
    t.add_row("完成轮数", str(rounds_done))
    t.add_row("累积字数", f"{total_chars}c")
    t.add_row("去重截断", str(trunc))
    t.add_row("跳过轮次", str(skips))
    t.add_row("保存路径", saved_path or "（未保存）")
    console.print(t)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> int:
    global TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY

    parser = argparse.ArgumentParser(description="LoRA adapter 交互式续写 CLI")
    parser.add_argument("--adapter", required=True,
                        help="LoRA adapter 目录路径，或可由 Unsloth 加载的基座模型名")
    parser.add_argument("--context-file", default=None,
                        help="初始上文 .txt 文件；不提供则交互粘贴")
    parser.add_argument("--raw-prompt-file", default=None,
                        help="完整 raw prompt .txt；启用后不走 Retriever/build_prompt")
    parser.add_argument("--max-rounds",    type=int, default=MAX_ROUNDS)
    parser.add_argument("--target-chars",  type=int, default=TARGET_CHARS)
    parser.add_argument("--max-seq-length",type=int, default=MAX_SEQ_LENGTH_INFER)
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--seed",          type=int, default=None)
    args = parser.parse_args()

    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    if args.context_file and args.raw_prompt_file:
        console.print("[red]ERROR: --context-file 与 --raw-prompt-file 只能二选一[/red]")
        return 1
    raw_prompt_mode = bool(args.raw_prompt_file)

    # ── 0. 前置检查 ───────────────────────────────────────────────────────────
    adapter_path = Path(args.adapter)
    looks_local_path = (
        "\\" in args.adapter
        or args.adapter.startswith((".", "/"))
        or re.match(r"^[A-Za-z]:", args.adapter)
        or args.adapter.split("/", 1)[0] in {"outputs", "models"}
    )
    if looks_local_path and not adapter_path.exists():
        console.print(f"[red]ERROR: adapter 目录不存在: {adapter_path}[/red]")
        return 1

    import torch
    if not torch.cuda.is_available():
        console.print("[red]ERROR: CUDA not available[/red]")
        return 1

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen_cfg = cfg.get("generation", {}) or {}
    TEMPERATURE = gen_cfg.get("temperature", TEMPERATURE)
    TOP_P = gen_cfg.get("top_p", TOP_P)
    TOP_K = gen_cfg.get("top_k", TOP_K)
    REPETITION_PENALTY = gen_cfg.get("repeat_penalty", REPETITION_PENALTY)

    retriever = None
    if raw_prompt_mode:
        console.print("[dim]--- Raw prompt mode: 跳过 Retriever / build_prompt ---[/dim]")
    else:
        # ── 1. Retriever ──────────────────────────────────────────────────────
        console.print("[dim]--- 初始化 Retriever (BM25 + story_bible) ---[/dim]")
        from cowriter.retriever import Retriever
        retriever = Retriever(cfg)
        console.print(f"[dim]  {len(retriever._docs)} bible docs indexed[/dim]")

    # ── 2. 加载模型 + adapter ─────────────────────────────────────────────────
    console.print(f"\n[bold]--- 加载模型 + adapter ---[/bold]")
    console.print(f"[dim]  {args.adapter}  max_seq_length={args.max_seq_length}[/dim]")

    from unsloth import FastLanguageModel
    with console.status("[green]Loading (4-bit)…[/green]", spinner="dots"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path if adapter_path.exists() else args.adapter),
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)

    if args.seed is not None:
        from transformers import set_seed
        set_seed(args.seed)
        console.print(
            f"[dim]  generation params: seed={args.seed} temperature={TEMPERATURE} "
            f"top_p={TOP_P} top_k={TOP_K} repetition_penalty={REPETITION_PENALTY}[/dim]"
        )

    console.print("[green]  模型就绪。[/green]")
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    peak_gb  = torch.cuda.max_memory_allocated() / 1024**3
    console.print(
        f"[cyan]  加载后 VRAM: {alloc_gb:.2f} GB  peak {peak_gb:.2f} GB  / 预算 {_VRAM_BUDGET_GB:.1f} GB[/cyan]"
    )
    console.print(
        f"[dim]  实测: 加载后 ~{_VRAM_LOAD_GB:.2f} GB，生成时 peak ~{_VRAM_INFER_PEAK_GB:.2f} GB"
        f"（硬件总量 {_VRAM_BUDGET_GB:.2f} GB，余量约 120 MB）。"
        f"每轮后 VRAM 提示颜色：绿=安全 / 黄=接近 / 红=危险。[/dim]"
    )

    # ── 3. 初始上文 ───────────────────────────────────────────────────────────
    console.print("\n[bold]--- 初始上文 ---[/bold]")
    if args.raw_prompt_file:
        ctx_p = Path(args.raw_prompt_file)
        if not ctx_p.exists():
            console.print(f"[red]ERROR: --raw-prompt-file 不存在: {ctx_p}[/red]")
            return 1
        initial_context = ctx_p.read_text(encoding="utf-8").strip()
        console.print(f"[dim]  从 raw prompt 文件加载: {ctx_p} ({len(initial_context)}c)[/dim]")
    elif args.context_file:
        ctx_p = Path(args.context_file)
        if not ctx_p.exists():
            console.print(f"[red]ERROR: --context-file 不存在: {ctx_p}[/red]")
            return 1
        initial_context = ctx_p.read_text(encoding="utf-8").strip()
        console.print(f"[dim]  从文件加载: {ctx_p} ({len(initial_context)}c)[/dim]")
    else:
        console.print("[bold]请粘贴初始上文（单独一行空行结束）：[/bold]")
        lines: list = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        initial_context = "\n".join(lines).strip()

    if not initial_context:
        console.print("[red]ERROR: 初始上文不能为空[/red]")
        return 1

    # ── 4. 内部重复自检 ───────────────────────────────────────────────────────
    dupes = _check_internal_duplicates(initial_context)
    if dupes:
        console.print()
        console.print(Panel(
            f"⚠️  检测到初始上文内部存在 [bold]{len(dupes)}[/bold] 处重复段落！\n\n"
            + "\n".join(f'  • "{d[:60]}…"' for d in dupes[:5])
            + "\n\n建议检查上文来源，确认不是历史生成内容污染。",
            title="[red bold]初始上文内部重复警告[/red bold]",
            border_style="red",
        ))
        resp = input("继续使用此上文？(y/N)：").strip().lower()
        if resp != "y":
            console.print("[yellow]已取消。请提供干净的初始上文重新运行。[/yellow]")
            return 1
    else:
        console.print("[green]  ✓ 初始上文内部重复检测：无重复段落[/green]")

    # ── 5. 首轮 prompt 预览（自动展示，供用户确认上文干净）─────────────────
    console.print("\n[bold]--- 首轮 Prompt 结构（生成前确认）---[/bold]")
    if raw_prompt_mode:
        first_prompt = initial_context[-MAX_RECENT_CHARS:]
        console.print(Panel(first_prompt, title="[yellow]RAW PROMPT[/yellow]",
                            border_style="yellow", expand=True))
        console.print(f"[dim]  Raw prompt chars: {len(first_prompt)} / {len(initial_context)}[/dim]")
    else:
        retrieval_init = retriever.retrieve(
            initial_context[-MAX_RECENT_CHARS:], max_chapter=None
        )
        from cowriter.prompts import build_prompt
        msgs_init = build_prompt(
            recent_text=initial_context[-MAX_RECENT_CHARS:],
            summary="",
            retrieval=retrieval_init,
            instruction="",
            prior_summary="",
        )
        msgs_display = [m for m in msgs_init if m["role"] != "assistant"]
        _show_prompt_panels(msgs_display, retrieval_init, console)

    input("\n按 Enter 开始生成，Ctrl+C 取消：")

    # ── 6. 交互循环 ───────────────────────────────────────────────────────────
    # 预种：用初始上文段落初始化 seen_keys，防止模型重复初始内容
    accumulated_text = initial_context
    seen_para_keys: set = set()
    for p in _split_paragraphs(initial_context):
        key = norm_unit(p)
        if len(key) >= _MIN_DEDUP_CHARS:
            seen_para_keys.add(key)

    all_new_texts: list = []
    total_new_chars  = 0
    truncation_rounds = 0
    skip_rounds       = 0
    saved_path        = ""

    for rnd in range(1, args.max_rounds + 1):
        tail = accumulated_text[-100:].replace("\n", " ")
        console.print(
            f"\n[bold]══ Round {rnd} ({total_new_chars}c / {args.target_chars}c) ══[/bold]"
        )
        console.print(f'[dim]上文末尾: "…{tail}"  ({len(accumulated_text)}c)[/dim]')

        recent_text = accumulated_text[-MAX_RECENT_CHARS:]
        if raw_prompt_mode:
            retrieval = {}
            msgs_gen = [{"role": "user", "content": recent_text}]
        else:
            retrieval = retriever.retrieve(recent_text, max_chapter=None)
            messages = build_prompt(
                recent_text=recent_text,
                summary="",
                retrieval=retrieval,
                instruction="",
                prior_summary="",
            )
            # build_prompt 可能附加 assistant prefill；生成时去掉
            msgs_gen = [m for m in messages if m["role"] != "assistant"]

        # token 估算
        try:
            _check_text = tokenizer.apply_chat_template(
                msgs_gen, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            _check_text = tokenizer.apply_chat_template(
                msgs_gen, tokenize=False, add_generation_prompt=True,
            )
        n_tok = tokenizer(_check_text, return_tensors="pt")["input_ids"].shape[1]
        console.print(f"[dim]  Input tokens: {n_tok}[/dim]")

        # 生成
        console.print("[dim]  [正在生成…][/dim]")
        if args.seed is not None:
            console.print(
                f"[dim]  generation params: seed={args.seed} temperature={TEMPERATURE} "
                f"top_p={TOP_P} top_k={TOP_K} repetition_penalty={REPETITION_PENALTY}[/dim]"
            )
        torch.cuda.reset_peak_memory_stats()
        new_text = _generate_one_round(model, tokenizer, msgs_gen)
        _vram_line(console)

        if not new_text or len(new_text) < 5:
            console.print("[yellow]  [WARN] 空生成，提前停止。[/yellow]")
            break

        # 去重
        kept_text, was_truncated, was_skipped = _dedup_truncate(new_text, seen_para_keys)
        if was_truncated:
            truncation_rounds += 1
            console.print(
                f"[dim]  [DEDUP] Truncated: {len(new_text)}c → {len(kept_text)}c[/dim]"
            )
        if was_skipped:
            skip_rounds += 1
            console.print("[dim]  [SKIP] 整轮均重复，跳过。[/dim]")
            cmd = input("  [r=重新生成 / Enter=跳过此轮 / q=退出]：").strip().lower()
            if cmd == "q":
                break
            continue

        text_to_show = kept_text if was_truncated else new_text
        console.print(Panel(text_to_show, title="[yellow]模型续写[/yellow]",
                            border_style="yellow"))

        # ── 命令循环 ──────────────────────────────────────────────────────────
        while True:
            cmd = input(
                "  [Enter=接受 / 文字=替换 / /reject /retry /context /save /help /quit]："
            ).strip()

            if cmd == "":
                accumulated_text += "\n\n" + text_to_show
                total_new_chars  += len(text_to_show)
                for p in _split_paragraphs(text_to_show):
                    k = norm_unit(p)
                    if len(k) >= _MIN_DEDUP_CHARS:
                        seen_para_keys.add(k)
                all_new_texts.append(text_to_show)
                console.print("[green]  [已接受][/green]")
                break

            elif not cmd.startswith("/"):
                accumulated_text += "\n\n" + cmd
                total_new_chars  += len(cmd)
                for p in _split_paragraphs(cmd):
                    k = norm_unit(p)
                    if len(k) >= _MIN_DEDUP_CHARS:
                        seen_para_keys.add(k)
                all_new_texts.append(cmd)
                console.print(f"[green]  [已记录用户版本 ({len(cmd)}c)][/green]")
                break

            elif cmd in ("/reject", "/拒绝"):
                console.print("[yellow]  [重新生成…][/yellow]")
                if args.seed is not None:
                    console.print(
                        f"[dim]  generation params: seed={args.seed} temperature={TEMPERATURE} "
                        f"top_p={TOP_P} top_k={TOP_K} repetition_penalty={REPETITION_PENALTY}[/dim]"
                    )
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                new_text = _generate_one_round(model, tokenizer, msgs_gen)
                _vram_line(console)
                kept_text, was_truncated, _ = _dedup_truncate(new_text, seen_para_keys)
                text_to_show = kept_text if was_truncated else new_text
                console.print(Panel(text_to_show, title="[yellow]重新生成[/yellow]",
                                    border_style="yellow"))
                continue

            elif cmd.startswith("/retry") or cmd.startswith("/重试"):
                prefix = "/retry" if cmd.startswith("/retry") else "/重试"
                instr  = cmd[len(prefix):].strip()
                if raw_prompt_mode:
                    retry_text = recent_text if not instr else recent_text + "\n\n" + instr
                    retry_msgs = [{"role": "user", "content": retry_text}]
                else:
                    retry_msgs = build_prompt(
                        recent_text=recent_text, summary="",
                        retrieval=retrieval, instruction=instr, prior_summary="",
                    )
                    retry_msgs = [m for m in retry_msgs if m["role"] != "assistant"]
                console.print(f"[yellow]  [重新生成: {instr or '无附加指令'}][/yellow]")
                if args.seed is not None:
                    console.print(
                        f"[dim]  generation params: seed={args.seed} temperature={TEMPERATURE} "
                        f"top_p={TOP_P} top_k={TOP_K} repetition_penalty={REPETITION_PENALTY}[/dim]"
                    )
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                new_text = _generate_one_round(model, tokenizer, retry_msgs)
                _vram_line(console)
                kept_text, was_truncated, _ = _dedup_truncate(new_text, seen_para_keys)
                text_to_show = kept_text if was_truncated else new_text
                console.print(Panel(text_to_show, title="[yellow]重新生成[/yellow]",
                                    border_style="yellow"))
                continue

            elif cmd in ("/context", "/上下文"):
                if raw_prompt_mode:
                    console.print(Panel(recent_text, title="[yellow]RAW PROMPT[/yellow]",
                                        border_style="yellow", expand=True))
                else:
                    _show_prompt_panels(msgs_gen, retrieval, console)
                continue

            elif cmd in ("/save", "/保存"):
                if all_new_texts:
                    saved_path = _save_draft(all_new_texts)
                    console.print(f"[green]  [已保存] {saved_path}[/green]")
                else:
                    console.print("[dim]  还没有生成内容。[/dim]")
                continue

            elif cmd in ("/help", "/帮助"):
                console.print(HELP_TEXT)
                continue

            elif cmd in ("/quit", "q", "exit"):
                if all_new_texts and input("  保存草稿？(y/N)：").strip().lower() == "y":
                    saved_path = _save_draft(all_new_texts)
                    console.print(f"[green]  [已保存] {saved_path}[/green]")
                total_new_chars = sum(len(t) for t in all_new_texts)
                _print_final_summary(console, rnd, total_new_chars,
                                     truncation_rounds, skip_rounds, saved_path)
                del model, tokenizer
                gc.collect(); torch.cuda.empty_cache()
                return 0

        if total_new_chars >= args.target_chars:
            console.print(
                f"\n[green]  目标字数达到 ({total_new_chars}c ≥ {args.target_chars}c)，停止。[/green]"
            )
            break

    # 退出后自动保存（如果还没保存）
    if all_new_texts and not saved_path:
        saved_path = _save_draft(all_new_texts)
        console.print(f"\n[green]草稿已保存: {saved_path}[/green]")

    total_new_chars = sum(len(t) for t in all_new_texts)
    _print_final_summary(console, len(all_new_texts), total_new_chars,
                         truncation_rounds, skip_rounds, saved_path)
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
