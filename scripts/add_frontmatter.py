#!/usr/bin/env python3
"""
一次性脚本：给 data/story_bible/ 下的 .md 文件补 YAML frontmatter。

规则：
  generated/characters/*.md  → type:character，revealed_in/valid_from = 来源章节最小值，
                                aliases 提取自 **别名/称呼**，source_chapters 列出所有章节
  手写根目录卡片              → type 来自 ## 类型 字段，revealed_in: 1
  world.md / style.md / glossary.md → revealed_in: 1（世界观/文风/词汇全时段可见）
  聚合文件（characters/relationships/timeline/
    plot_threads/chapter_summaries）→ 跳过，temporal filter 下永不可见

用法：
  python scripts/add_frontmatter.py [--story-bible <路径>] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── 聚合文件：跳过，不加 frontmatter ─────────────────────────────────────────
SKIP_FILES = {
    "characters",
    "relationships",
    "timeline",
    "plot_threads",
    "chapter_summaries",
}

# ── 世界观/文风/词汇表：全时段可见，不含剧情设定 ────────────────────────────
ALWAYS_VISIBLE_FILES = {
    "world": "worldbuilding",
    "style": "style",
    "glossary": "glossary",
}

# ── 中文数字 → int ────────────────────────────────────────────────────────────
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


_SOURCE_CHAPTER_RE = re.compile(
    r"第\s*([0-9]{1,4}|[零〇一二三四五六七八九十百千万两]{1,8})\s*章"
)
_FM_RE = re.compile(r"^---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def already_has_frontmatter(text: str) -> bool:
    return bool(_FM_RE.match(text))


def extract_chapters(text: str) -> list[int]:
    chapters = []
    for m in _SOURCE_CHAPTER_RE.finditer(text):
        n = _cn_to_int(m.group(1))
        if n is not None and n > 0:
            chapters.append(n)
    return sorted(set(chapters))


def extract_min_chapter(text: str) -> int | None:
    chs = extract_chapters(text)
    return chs[0] if chs else None


def _extract_aliases(text: str) -> list[str]:
    """从 **别名/称呼**: ... 字段提取别名列表。"""
    m = re.search(r'\*\*别名[/／]称呼\*\*\s*[:：]\s*(.+)', text)
    if not m:
        return []
    raw = m.group(1).strip()
    # 拆分：中文顿号、逗号、/、空格
    parts = re.split(r'[、，,/\s]+', raw)
    return [p.strip() for p in parts if p.strip() and p.strip() not in ("未明确", "无")]


def _extract_handwritten_type(text: str) -> str:
    """从手写卡片的 ## 类型 节提取类型值，映射到英文 type 标识。"""
    m = re.search(r'##\s*类型\s*\n+(.+)', text)
    if not m:
        return "misc"
    raw = m.group(1).strip()
    mapping = {
        "人物": "character",
        "地点": "location",
        "场景氛围": "scene",
        "场景": "scene",
        "物件": "prop",
        "物件 / 场景道具": "prop",
        "道具": "prop",
        "势力": "faction",
        "组织": "faction",
        "功法": "ability",
        "法宝": "item",
    }
    for key, val in mapping.items():
        if key in raw:
            return val
    return "misc"


def make_frontmatter(data: dict) -> str:
    """从 dict 生成 YAML frontmatter 字符串（手动拼接，避免 pyyaml 格式问题）。"""
    lines = ["---"]
    if data.get("title"):
        # 含特殊字符时加引号
        t = str(data["title"])
        if any(c in t for c in ':{}[]|>&*!,#?@`'):
            t = f'"{t}"'
        lines.append(f"title: {t}")
    if data.get("type"):
        lines.append(f"type: {data['type']}")
    if data.get("aliases"):
        lines.append("aliases:")
        for a in data["aliases"]:
            lines.append(f"  - {a}")
    lines.append(f"revealed_in: {data['revealed_in']}")
    lines.append(f"valid_from: {data['valid_from']}")
    vt = data.get("valid_to")
    lines.append(f"valid_to: {'null' if vt is None else vt}")
    if data.get("source_chapters"):
        lines.append("source_chapters:")
        for ch in sorted(set(data["source_chapters"])):
            lines.append(f"  - {ch}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def patch_file(path: Path, fm_data: dict, *, dry_run: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if already_has_frontmatter(text):
        return "skip:already_has_fm"
    fm = make_frontmatter(fm_data)
    if not dry_run:
        path.write_text(fm + text, encoding="utf-8")
    return f"add:revealed_in={fm_data['revealed_in']},valid_from={fm_data['valid_from']}"


def process_bible_dir(bible_dir: Path, *, dry_run: bool) -> None:
    results: list[tuple[str, str]] = []

    for md_file in sorted(bible_dir.rglob("*.md")):
        rel = md_file.relative_to(bible_dir)
        stem = md_file.stem

        # 隐藏文件
        if stem.startswith("."):
            continue

        # 聚合文件：跳过
        if stem in SKIP_FILES and len(rel.parts) == 1:
            results.append((str(rel), "skip:aggregate"))
            continue

        # 世界观/文风/词汇表 → revealed_in: 1
        if stem in ALWAYS_VISIBLE_FILES and len(rel.parts) == 1:
            fm_data = {
                "title": stem,
                "type": ALWAYS_VISIBLE_FILES[stem],
                "revealed_in": 1,
                "valid_from": 1,
                "valid_to": None,
            }
            action = patch_file(md_file, fm_data, dry_run=dry_run)
            results.append((str(rel), action))
            continue

        # generated/characters/*.md → 解析来源章节 + 别名
        if len(rel.parts) >= 2 and rel.parts[0] == "generated":
            text = md_file.read_text(encoding="utf-8")
            if already_has_frontmatter(text):
                results.append((str(rel), "skip:already_has_fm"))
                continue
            chs = extract_chapters(text)
            min_ch = chs[0] if chs else 1
            aliases = _extract_aliases(text)
            fm_data = {
                "title": stem,
                "type": "character",
                "aliases": aliases,
                "revealed_in": min_ch,
                "valid_from": min_ch,
                "valid_to": None,
                "source_chapters": chs or [min_ch],
            }
            action = patch_file(md_file, fm_data, dry_run=dry_run)
            results.append((str(rel), action))
            continue

        # 根目录手写卡片
        if len(rel.parts) == 1:
            text = md_file.read_text(encoding="utf-8")
            if already_has_frontmatter(text):
                results.append((str(rel), "skip:already_has_fm"))
                continue
            card_type = _extract_handwritten_type(text)
            fm_data = {
                "title": stem,
                "type": card_type,
                "revealed_in": 1,
                "valid_from": 1,
                "valid_to": None,
                "source_chapters": [1],
            }
            action = patch_file(md_file, fm_data, dry_run=dry_run)
            results.append((str(rel), action))
            continue

    prefix = "[dry-run] " if dry_run else ""
    for rel_path, action in results:
        print(f"{prefix}{action:<45} {rel_path}")

    added = sum(1 for _, a in results if a.startswith("add:"))
    skipped = len(results) - added
    print(f"\n{prefix}完成：{added} 个文件{'将' if dry_run else ''}添加 frontmatter，{skipped} 个跳过")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="给 story_bible .md 文件补 frontmatter")
    parser.add_argument("--story-bible", default="data/story_bible",
                        help="story_bible 目录路径（默认 data/story_bible）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不写文件")
    args = parser.parse_args(argv)

    bible_dir = Path(args.story_bible)
    if not bible_dir.exists():
        print(f"[错误] 目录不存在：{bible_dir}", file=sys.stderr)
        return 1

    process_bible_dir(bible_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
