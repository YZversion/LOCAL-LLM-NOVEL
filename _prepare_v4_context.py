#!/usr/bin/env python3
"""
为 v4 正式评测候选准备干净的起始上文。

策略：从第五十八章正文末尾截取约 800c（风丝引最后一章的结尾），
作为续写起始点，符合"从故事当前进展处续写"的评测语义。

同时运行内部重复检测，确认文本干净后写入
outputs/debug/v4_eval_context.txt。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RAW_FILE   = "data/raw/风丝引_原文.txt"
OUTPUT     = "outputs/debug/v4_eval_context.txt"
TARGET_C   = 800
_MIN_DEDUP = 10


def main():
    raw = Path(RAW_FILE).read_text(encoding="utf-8")

    # 找到第五十八章正文
    ch58_pos = raw.find("第五十八章")
    assert ch58_pos != -1, "找不到第五十八章"

    ch58_text = raw[ch58_pos:]

    # 跳过章节标题行、空行，定位正文开始
    lines = ch58_text.split("\n")
    content_lines = []
    skip = True
    for line in lines:
        if skip:
            stripped = line.strip()
            # 跳过章节标题和紧随的空行
            if "第五十八章" in stripped or stripped == "":
                continue
            skip = False
        # 过滤掉末尾的广告行
        if "soushu" in line or "Download" in line or "www." in line:
            continue
        content_lines.append(line)

    ch58_body = "\n".join(content_lines).strip()

    # 取末尾 ~800c 作为续写上文
    context = ch58_body[-TARGET_C:].strip()
    actual_len = len(context)

    print(f"Ch58 正文长度: {len(ch58_body)}c")
    print(f"截取末尾 {actual_len}c 作为续写上文")
    print(f"前 80c: {context[:80]}")
    print(f"后 80c: {context[-80:]}")
    print()

    # 内部重复检测
    from pipeline.eval_style import norm_unit, split_paragraphs as _split
    paras = _split(context)
    seen, dupes = set(), []
    for p in paras:
        key = norm_unit(p)
        if len(key) < _MIN_DEDUP:
            continue
        if key in seen:
            dupes.append(p[:60])
        else:
            seen.add(key)

    if dupes:
        print(f"[WARN] 检测到 {len(dupes)} 处内部重复，上文不干净：")
        for d in dupes:
            print(f"  - {d}")
        return 1

    valid_paras = len([p for p in paras if len(norm_unit(p)) >= _MIN_DEDUP])
    print(f"[OK] 内部重复检测通过（{valid_paras} 段落，全部唯一）")

    # 保存
    out = Path(OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(context, encoding="utf-8")
    print(f"[OK] 已保存到 {OUTPUT} ({actual_len}c)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
