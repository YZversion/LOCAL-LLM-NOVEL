#!/usr/bin/env python3
"""
从原文生成缺失章节摘要，追加到 chapter_summaries.md。

用法：
  python scripts/gen_chapter_summaries.py [--dry-run] [--start N] [--end N]

  --dry-run   只打印待处理列表，不调用 LLM
  --start N   只处理 >= N 章（默认 1）
  --end   N   只处理 <= N 章（默认全部）
  --model M   指定 ollama 模型（默认读 config.yaml）
  --raw   P   原文路径（默认 data/raw/风丝引_原文.txt）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import ollama
import yaml

# ── 中文数字转 int ────────────────────────────────────────────────────────────

_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
       "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_to_int(s: str) -> int | None:
    s = s.strip()
    if s.isdigit():
        return int(s)
    total = section = num = 0
    used = False
    for c in s:
        if c in _CN:
            num = _CN[c]; used = True
        elif c in _UNIT:
            u = _UNIT[c]; used = True
            if u == 10000:
                section = (section + num) * u; total += section; section = 0
            else:
                section += (num or 1) * u
            num = 0
        else:
            return None
    return (total + section + num) if used else None


# ── 原文章节切割 ──────────────────────────────────────────────────────────────

_CHAPTER_HEADER_RE = re.compile(
    r'^第([0-9]{1,4}|[零〇一二三四五六七八九十百千万两]{1,8})章[^\n]*',
    re.MULTILINE,
)


def split_chapters(text: str) -> list[tuple[int, str, str]]:
    """返回 [(chapter_num, title_line, body), ...]，按章节号排序。"""
    matches = list(_CHAPTER_HEADER_RE.finditer(text))
    result = []
    for i, m in enumerate(matches):
        header = m.group(0).strip()
        num = _cn_to_int(m.group(1))
        if num is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        result.append((num, header, body))
    result.sort(key=lambda x: x[0])
    return result


# ── 已有章节号检测 ────────────────────────────────────────────────────────────

_EXISTING_NUM_RE = re.compile(r'\*\*章节编号\*\*\s*[:：]\s*(\d+)')


def existing_chapter_nums(summaries_path: Path) -> set[int]:
    if not summaries_path.exists():
        return set()
    text = summaries_path.read_text(encoding="utf-8")
    return {int(m.group(1)) for m in _EXISTING_NUM_RE.finditer(text)}


# ── 加载 JSON 辅助上下文（ch31-58 综合分析）────────────────────────────────────

def load_json_context(json_path: Path) -> dict:
    """返回 {char_summary, plot_summary}，用于 ch31-58 的 prompt 辅助。"""
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    chars = data.get("characters", [])
    char_lines = []
    for c in chars:
        name = c.get("name", "")
        dev = c.get("dev", "")
        role = c.get("role", "")
        if name and dev:
            char_lines.append(f"- {name}（{role}）：{dev}")
    plots = data.get("plots", [])
    plot_lines = [f"- {p}" for p in plots if p]
    return {
        "char_summary": "\n".join(char_lines),
        "plot_summary": "\n".join(plot_lines),
    }


# ── LLM 调用 ─────────────────────────────────────────────────────────────────

# ── story_bible 已有设定检索 ─────────────────────────────────────────────────


def get_bible_context(retriever, body: str, chapter_num: int,
                      max_chars: int = 1500) -> str:
    """检索本章前已有的相关设定，供模型对比识别新增内容。

    使用 max_chapter = chapter_num - 1，确保只看本章之前已确立的设定。
    """
    max_chap = chapter_num - 1
    if max_chap < 1:
        return ""
    result = retriever.retrieve(body[:600], max_chapter=max_chap)
    hits = result.get("bible", [])
    if not hits:
        return ""
    lines: list[str] = []
    total = 0
    for h in hits:
        line = f"【{h['source']}】{h['text'][:300]}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


# ── Prompt 模板 ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一位专业的小说章节摘要助手，擅长提炼中文网络小说的核心情节。
请阅读提供的章节原文，严格按照以下格式输出摘要，不要添加任何多余内容，不要写解释：

## {header}

- **章节编号**: {num}
- **本章概要**: （60-120字，概括本章核心事件，直接开始写，不要以"本章"开头）
- **出场人物**: （本章出现的重要角色，逗号分隔）
- **新增设定**:
  - （格式："名称：简短说明"，例如"凰羽笺：太阳神鸟产出，可传信"）
  - （已在【已有设定】中出现的内容不算新增；没有新设定则只写一行"无"）
- **情节推进**: （一句话说明本章推动了哪条主线）
- **伏笔**: （本章章节内留下的未解悬念或暗示，用分号分隔；没有则写"无"）
- **情绪氛围**: （2-4个词，逗号分隔，例如"孤寞、凄凉、悲漠"）
- **关键词**: （4-8个关键词，逗号分隔）
- **来源章节**: 第{num}章

---"""

_USER_TEMPLATE = """\
{bible_block}{json_block}【本章原文】
{body}

请为【{header}】生成摘要，严格按系统提示中的格式输出，不要添加其他内容。"""

_BIBLE_BLOCK_TEMPLATE = """\
【已有设定（本章之前已确立，不算新增）】
{bible_ctx}

"""

_JSON_BLOCK_TEMPLATE = """\
【角色发展参考（第31-58章综合分析）】
{char_summary}

【情节线参考】
{plot_summary}

"""


def call_ollama(model: str, num: int, header: str, body: str,
                json_ctx: dict, bible_ctx: str = "") -> str:
    system = _SYSTEM_PROMPT.format(header=header, num=num)

    bible_block = ""
    if bible_ctx:
        bible_block = _BIBLE_BLOCK_TEMPLATE.format(bible_ctx=bible_ctx)

    json_block = ""
    if num >= 31 and json_ctx:
        json_block = _JSON_BLOCK_TEMPLATE.format(
            char_summary=json_ctx.get("char_summary", ""),
            plot_summary=json_ctx.get("plot_summary", ""),
        )

    # 原文截取：最多送 3000 字（为上下文留出空间）
    body_excerpt = body[:3000]
    if len(body) > 3000:
        body_excerpt += "\n\n（……原文较长，以上为节选）"

    user = _USER_TEMPLATE.format(
        bible_block=bible_block,
        json_block=json_block,
        body=body_excerpt,
        header=header,
    )

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3, "num_predict": 800},
        think=False,
    )
    return resp.message.content.strip()


# ── 格式修复：确保结果含必要字段 ──────────────────────────────────────────────

_REQUIRED_FIELDS = ["章节编号", "本章概要"]


def ensure_format(result: str, num: int, header: str) -> str:
    """如果 LLM 输出缺少必要字段，包装成最小合法格式。"""
    if all(f in result for f in _REQUIRED_FIELDS):
        return result
    # 简单兜底：把输出内容塞进本章概要
    summary = result.replace("\n", " ")[:200]
    return (
        f"## {header}\n\n"
        f"- **章节编号**: {num}\n"
        f"- **本章概要**: {summary}\n"
        f"- **出场人物**: （见原文）\n"
        f"- **新增设定**:\n  - 无\n"
        f"- **情节推进**: 未提取\n"
        f"- **伏笔**: 无\n"
        f"- **情绪氛围**: 未提取\n"
        f"- **关键词**: 未提取\n"
        f"- **来源章节**: 第{num}章\n\n"
        f"---"
    )


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从原文生成缺失章节摘要，追加到 chapter_summaries.md"
    )
    parser.add_argument("--raw", default="data/raw/风丝引_原文.txt",
                        help="原文路径")
    parser.add_argument("--summaries", default="data/story_bible/chapter_summaries.md",
                        help="chapter_summaries.md 路径")
    parser.add_argument("--json-ctx", default="data/分析结果_31-58章.json",
                        help="31-58章分析 JSON 路径")
    parser.add_argument("--model", default=None,
                        help="ollama 模型名（默认读 config.yaml）")
    parser.add_argument("--start", type=int, default=1,
                        help="从第N章开始处理")
    parser.add_argument("--end", type=int, default=9999,
                        help="到第N章结束")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印待处理列表，不调用 LLM")
    args = parser.parse_args(argv)

    # 读 config.yaml 获取默认模型
    model = args.model
    if model is None:
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            model = cfg.get("model", {}).get("ollama_model",
                                             "huihui_ai/qwen3-abliterated:8b-v2")
        else:
            model = "huihui_ai/qwen3-abliterated:8b-v2"

    raw_path = Path(args.raw)
    summaries_path = Path(args.summaries)
    json_path = Path(args.json_ctx)

    if not raw_path.exists():
        print(f"[错误] 原文不存在: {raw_path}", file=sys.stderr)
        return 1

    print(f"模型: {model}")
    print(f"原文: {raw_path}")

    text = raw_path.read_text(encoding="utf-8")
    chapters = split_chapters(text)
    print(f"原文共识别到 {len(chapters)} 章")

    existing = existing_chapter_nums(summaries_path)
    print(f"chapter_summaries.md 已有: 第{min(existing)}-{max(existing)}章 共{len(existing)}章" if existing else "chapter_summaries.md 为空")

    json_ctx = load_json_context(json_path)
    print(f"JSON 辅助上下文（ch31+）: {'已加载' if json_ctx else '未找到，跳过'}")

    # 初始化 Retriever，用于检索已有设定
    retriever = None
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from cowriter.retriever import Retriever as _Retriever
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            _cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            retriever = _Retriever(_cfg)
            print("story_bible 检索器: 已初始化")
        else:
            print("story_bible 检索器: config.yaml 未找到，跳过设定联动")
    except Exception as e:
        print(f"story_bible 检索器: 初始化失败（{e}），跳过设定联动")

    to_process = [
        (num, header, body)
        for num, header, body in chapters
        if num not in existing and args.start <= num <= args.end
    ]
    print(f"\n待生成: {len(to_process)} 章")
    for num, header, _ in to_process:
        print(f"  第{num}章 {header}")

    if args.dry_run or not to_process:
        print("\n[dry-run] 退出，未调用 LLM。" if args.dry_run else "\n无需生成，已全部存在。")
        return 0

    print()
    errors: list[tuple[int, str]] = []
    done = 0
    try:
        for i, (num, header, body) in enumerate(to_process):
            ctx = json_ctx if num >= 31 else {}
            bible_ctx = get_bible_context(retriever, body, num) if retriever else ""
            print(f"[{i + 1}/{len(to_process)}] {header} ({len(body)}字) ...", end=" ", flush=True)
            try:
                result = call_ollama(model, num, header, body, ctx, bible_ctx)
                result = ensure_format(result, num, header)
                with open(summaries_path, "a", encoding="utf-8") as f:
                    f.write("\n\n" + result.rstrip() + "\n")
                print("✓")
                done += 1
            except Exception as e:
                print(f"✗  {e}")
                errors.append((num, str(e)))

            if i < len(to_process) - 1:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n[中断] 用户按下 Ctrl+C")

    print()
    print(f"本次完成: {done}/{len(to_process)} 章")
    if errors:
        print(f"[失败] {len(errors)} 章: {[n for n, _ in errors]}")
    if done < len(to_process) and not errors:
        print("提示：重新运行脚本可从断点续跑（已完成章节自动跳过）")
    if done > 0:
        print(f"已追加到 {summaries_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
