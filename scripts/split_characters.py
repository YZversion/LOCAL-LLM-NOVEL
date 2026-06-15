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
        (out_dir / f"{file_name}.md").write_text(part.strip(), encoding="utf-8")
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
