# -*- coding: utf-8 -*-
"""
_dedup_output 两层保险丝验证
单元测试（无需 Ollama）+ 三组实机测试：
  组1: 刀疤/花园老人场景（触发大段复读的上下文类型）× 5
  组2: loop_prone 躲开双手场景 × 3（回归：短句循环防御保持）
  组3: baseline 扁舟场景 × 3（回归：正常文本不被误截）
"""
import re
import yaml
from cowriter.session import Session, _dedup_output

# ═══════════════════════════════════════════════════════════════
# 单元测试
# ═══════════════════════════════════════════════════════════════
print("=== _dedup_output 单元测试 ===")
all_ut = True

def ut(label, text, expect_truncated: bool):
    global all_ut
    out = _dedup_output(text)
    actual = len(out) < len(text.strip())
    ok = actual == expect_truncated
    if not ok:
        all_ut = False
    flag = "OK" if ok else "FAIL"
    print(f"  {flag} [{label}] in={len(text.strip())} out={len(out)} truncated={actual}")

# ── 段落层（换行分割，≥50字出现2次截断）─────────────────────────────────────
LONG_PARA = "老人浑浊的眼睛望向远方，整个花园在晨光中如梦如幻，百花盛开，香气馥郁，令人沉醉其中，仿佛时间凝固在这片宁静之中。"
assert len(LONG_PARA) >= 50, f"para too short: {len(LONG_PARA)}"

ut("段落层-≥50字×2次=截断",     LONG_PARA + "\n" + LONG_PARA, True)
ut("段落层-≥50字×1次=不截断",   LONG_PARA, False)
ut("段落层-<50字×2次=不截断",   "她站在那里，神色平静。\n她站在那里，神色平静。", False)

# 排比句（短，<50字，多次重复，不应误伤）
PAIBI = "夜色如墨。\n寂静无声。\n唯有风声。\n夜色如墨。\n寂静无声。\n唯有风声。\n"
ut("段落层-排比短句不误伤",       PAIBI, False)

# ── 句段层（句末标点分割，≥8字出现3次截断）──────────────────────────────────
SENT = "她躲开双手，咯咯笑道你还当真了。"
assert len(SENT) >= 8
ut("句段层-≥8字×3次=截断",       SENT * 3, True)
ut("句段层-≥8字×2次=不截断",     SENT * 2, False)

# 极短叹词（<8字）不误触发
ut("句段层-短叹词不误触发",       "好。" * 6 + "林清雪缓缓睁开美眸望向远方湖面。", False)

# 两层共同作用：段落层先截，句段层再处理剩余
COMBO = LONG_PARA + "\n" + LONG_PARA + "\n" + SENT * 4
ut("组合-段落先截",               COMBO, True)

# 正常无重复文本不截断
NORMAL = "林清雪缓缓睁开双眸。湖面波光粼粼。小女童轻摇长篙。远处山峦若隐若现。老人独坐于石凳之上，眼神深邃。"
ut("正常文本不截断",              NORMAL, False)

print(f"  单元测试: {'ALL PASS' if all_ut else 'SOME FAILED'}\n")

# ═══════════════════════════════════════════════════════════════
# 实机测试
# ═══════════════════════════════════════════════════════════════
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

g = cfg["generation"]
print(f"[模型] {cfg['model']['ollama_model']}")
print(f"[参数] repeat_penalty={g['repeat_penalty']} repeat_last_n={g.get('repeat_last_n','?')} "
      f"dry={g.get('dry_multiplier','?')} presence={g.get('presence_penalty','?')}")

# 组1：刀疤/花园场景（和 bug 触发时的上下文类型一致：大段氛围描写+老人）
SEED_GARDEN = (
    "凤倾汐随着刀疤老人缓步走进庭院深处，满园花木在暮色中朦胧绰约。"
    "老人驻足，浑浊的眼睛望向花圃中央那株独自盛开的白梅，"
    "布满刀疤的手背缓缓抬起，像是要触碰什么，又像是在回忆什么。"
    "花香随晚风飘来，轻轻拂过二人衣袂，庭中一时寂静无声。"
    "凤倾汐低着头，不敢直视老人神色，只听见自己的心跳声。"
)

# 组2：loop_prone（躲开双手）
SEED_LOOP = (
    "林清雪轻轻笑了笑，旋身避开他伸来的双手，退后半步，衣袂翻飞。"
    "那人犹不死心，又伸手去拦，被她再次轻巧地躲开，落了个空。"
    "\"咯咯，\"她掩口轻笑，\"你还当真了？\""
    "莞尔一笑，旋身而去，留下他一人怔在原地。"
    "谁料他一个大步追上，伸手便要扯她衣袖——她再次躲开双手"
)

# 组3：baseline（扁舟）
SEED_BASE = (
    "林清雪盘腿坐在一叶扁舟之上，月白衣衫垂至湖面，衣袖随风微动。"
    "小女童站在船尾轻摇长篹，湖水荡开一圈圈绿色波纹。"
    "过了好半晌，林清雪才缓缓睁开美眸，荡漾的湖水映在她的眸心深处，"
    "悠悠的，仿佛她心底流转的思绪。"
)

BAD_PATTERNS = [
    (re.compile(r"/no_think"),                              "/no_think 泄漏"),
    (re.compile(r"</?think>"),                              "think 残留"),
    (re.compile(r"^/no\b", re.MULTILINE),                   "/no 残片行首"),
    (re.compile(r"^好的[。，：]?\s*$", re.MULTILINE),        "好的 独占行"),
    (re.compile(r"好的，|接下来我将|以下是"),                 "助手语气"),
    (re.compile(r"\d+[.．、]"),                              "列表"),
    (re.compile(r"[A-Za-z]{4,}"),                           "英文乱码"),
]

def _has_large_repeat(text):
    """检查是否还有≥50字的段落出现2次（即保险丝没拦住）"""
    lines = text.splitlines()
    seen = {}
    for ln in lines:
        k = ln.strip()
        if len(k) >= 50:
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 2:
                return True
    return False

def _has_sent_loop(text):
    """检查是否还有≥8字句段出现3次"""
    parts = re.split(r'(?<=[。！？\n])', text)
    seen = {}
    for p in parts:
        k = p.strip()
        if len(k) >= 8:
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 3:
                return True
    return False

def run_group(label, seed, n, expect_no_truncation=False):
    print(f"\n{'='*62}")
    print(f"=== 组{label} (n={n}) seed='{seed[:35]}...' ===")
    sess = Session(cfg)
    sess.seed(seed)
    results = []
    for i in range(n):
        print(f"  --- 第{i+1}次 ---")
        out = sess.generate()
        sess.accept(out)
        issues = []
        if not out.strip():
            issues.append("空输出")
        for pat, lbl in BAD_PATTERNS:
            m = pat.search(out)
            if m:
                issues.append(f"{lbl}:{repr(m.group())}")
        if _has_large_repeat(out):
            issues.append("大段复读未被拦截（段落层漏检）")
        if _has_sent_loop(out):
            issues.append("短句循环未被拦截（句段层漏检）")
        # 若期望不截断，检查是否被误截（字数极短）
        if expect_no_truncation and len(out) < 40 and out.strip():
            issues.append(f"疑似误截（仅{len(out)}字）")
        print(f"  {out[:200]}")
        status = "CLEAN" if not issues else "ISSUES: " + "; ".join(issues)
        print(f"  [{status}] 字数={len(out)}")
        results.append({"issues": issues, "len": len(out)})
    return results

r1 = run_group("1[garden/刀疤]", SEED_GARDEN, 5)
r2 = run_group("2[loop_prone]",  SEED_LOOP,   3, expect_no_truncation=False)
r3 = run_group("3[baseline扁舟]", SEED_BASE,  3, expect_no_truncation=True)

print(f"\n{'='*62}")
print("=== 最终汇总 ===")
ok1 = all(not r["issues"] for r in r1)
ok2 = all(not r["issues"] for r in r2)
ok3 = all(not r["issues"] for r in r3)
print(f"  组1 garden/刀疤  5次: {'ALL CLEAN' if ok1 else 'SOME ISSUES'}")
print(f"  组2 loop_prone   3次: {'ALL CLEAN' if ok2 else 'SOME ISSUES'}")
print(f"  组3 baseline扁舟 3次: {'ALL CLEAN' if ok3 else 'SOME ISSUES'}")
print(f"  单元测试: {'PASS' if all_ut else 'FAIL'}")
print("RESULT:", "ALL PASS" if (ok1 and ok2 and ok3 and all_ut) else "SOME FAILED")
