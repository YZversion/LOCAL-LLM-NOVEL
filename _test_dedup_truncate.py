#!/usr/bin/env python3
"""Verification A: unit-test _dedup_truncate in generate_lora_multi, no GPU needed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.generate_lora_multi import _dedup_truncate, _MIN_DEDUP_CHARS
from pipeline.eval_style import norm_unit

def main() -> int:
    para_a = "这是测试段落甲，描述了湖边的景色，水波荡漾，林清雪静立其中。"
    para_b = "这是测试段落乙，林清雪抬起头望向远处，目光悠远不知所踪。"
    para_c = "这是测试段落丙，柳家二公子站在岸边，玉笛横陈于掌心。"
    short  = "好。"  # 2 chars, below _MIN_DEDUP_CHARS

    # S1: partial truncation
    seen = {norm_unit(para_a)}
    kept, trunc, skip = _dedup_truncate(para_b + "\n\n" + para_a + "\n\n" + para_c, seen)
    assert trunc is True  and skip  is False, f"S1 flags wrong: trunc={trunc} skip={skip}"
    assert norm_unit(para_b) in norm_unit(kept), f"S1: para_b missing from {kept!r}"
    assert norm_unit(para_a) not in norm_unit(kept), f"S1: para_a(dup) present in {kept!r}"
    assert norm_unit(para_c) not in norm_unit(kept), f"S1: para_c(after dup) present in {kept!r}"
    print("S1 PASS  partial truncation: kept para_b, dropped para_a(dup)+para_c(after)")

    # S2: full skip (first para already seen)
    seen2 = {norm_unit(para_a)}
    kept2, trunc2, skip2 = _dedup_truncate(para_a + "\n\n" + para_b, seen2)
    assert trunc2 is True and skip2 is True, f"S2 flags wrong: trunc={trunc2} skip={skip2}"
    assert kept2 == "", f"S2: kept_text must be empty, got {kept2!r}"
    print("S2 PASS  full skip: dup is first para → empty kept_text")

    # S3: all seen → full skip
    seen3 = {norm_unit(para_a), norm_unit(para_b)}
    kept3, trunc3, skip3 = _dedup_truncate(para_a + "\n\n" + para_b, seen3)
    assert trunc3 is True and skip3 is True, f"S3 flags wrong"
    print("S3 PASS  all-seen round: full skip")

    # S4: clean round (nothing seen)
    seen4 = {norm_unit(para_c)}
    kept4, trunc4, skip4 = _dedup_truncate(para_a + "\n\n" + para_b, seen4)
    assert trunc4 is False and skip4 is False, f"S4 flags wrong"
    assert norm_unit(para_a) in norm_unit(kept4) and norm_unit(para_b) in norm_unit(kept4)
    print("S4 PASS  clean round: all content kept")

    # S5: short paragraph below _MIN_DEDUP_CHARS bypasses dedup
    assert len(norm_unit(short)) < _MIN_DEDUP_CHARS
    seen5 = {norm_unit(short)}
    kept5, trunc5, skip5 = _dedup_truncate(short + "\n\n" + para_a, seen5)
    assert trunc5 is False, f"S5: short para must NOT trigger dedup, trunc={trunc5}"
    print(f"S5 PASS  short para (<{_MIN_DEDUP_CHARS} chars) bypasses dedup check")

    print("\nAll Verification A scenarios PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
