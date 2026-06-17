#!/usr/bin/env python3
"""
从 data/story_bible/characters.md 按人物拆分。
输出：data/story_bible/generated/characters/<人物名>.md
用法：python scripts/split_characters.py
"""
import re
import sys
from pathlib import Path

import yaml

_SOURCE_CH_RE = re.compile(
    r"第\s*([0-9]{1,4}|[零〇一二三四五六七八九十百千万两]{1,8})\s*章"
)
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


def _min_chapter(text: str) -> int:
    chapters = [n for m in _SOURCE_CH_RE.finditer(text)
                if (n := _cn_to_int(m.group(1))) is not None and n > 0]
    return min(chapters) if chapters else 1


def split(src: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    # 按 ### 标题切分，每个人物一块
    parts = re.split(r"\n(?=### )", text)
    created = []
    for part in parts:
        m = re.match(r"### (.+)", part)
        if not m:
            continue
        raw_name = m.group(1).strip()
        # 去掉括号注释：凤倾汐（凰后） → 凤倾汐
        file_name = re.sub(r"[（(][^）)]*[）)]", "", raw_name).strip()
        if not file_name:
            continue
        revealed_in = _min_chapter(part)
        frontmatter = f"---\nrevealed_in: {revealed_in}\n---\n\n"
        (out_dir / f"{file_name}.md").write_text(frontmatter + part.strip(), encoding="utf-8")
        created.append(file_name)
    return created


def main():
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        print(f"[错误] 找不到 {cfg_path}，请在项目根目录运行")
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bible_dir = Path(cfg["paths"]["story_bible"])
    src = bible_dir / "characters.md"
    out_dir = bible_dir / "generated" / "characters"

    if not src.exists():
        print(f"[错误] 找不到 {src}")
        sys.exit(1)

    names = split(src, out_dir)
    print(f"[完成] {len(names)} 个人物文件 → {out_dir}")
    for n in names:
        print(f"  {n}.md")


if __name__ == "__main__":
    main()
