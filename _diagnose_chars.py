#!/usr/bin/env python3
"""诊断：训练样本中各角色出现情况"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SAMPLES_FILE = "data/processed/train_samples_full_57.jsonl"
OUT_FILE     = "outputs/debug/char_diagnosis.txt"

samples = [json.loads(l) for l in Path(SAMPLES_FILE).read_text(encoding="utf-8").splitlines() if l.strip()]

lines = []
def p(s=""):
    lines.append(s)

targets = ["老刀疤", "叶欢", "凰后", "凤倾汐"]

for name in targets:
    hits_ctx  = [(i+1, s) for i, s in enumerate(samples) if name in s.get("messages", [{}])[1].get("content", "") if len(s.get("messages",[])) > 1]
    hits_comp = [(i+1, s) for i, s in enumerate(samples) if name in s.get("completion", "")]
    p(f"{'='*60}")
    p(f"  {name}: completion 出现 {len(hits_comp)} 条，context(user) 出现 {len(hits_ctx)} 条")
    p(f"{'='*60}")
    for rank, (idx, s) in enumerate(hits_comp[:3], 1):
        comp = s.get("completion", "")
        # 找到名字首次出现位置，打印前后 150c
        pos = comp.find(name)
        snippet_start = max(0, pos - 80)
        snippet_end   = min(len(comp), pos + 200)
        snippet = comp[snippet_start:snippet_end]
        chapter = s.get("chapter") or "?"
        p(f"\n  [样本 #{idx} / ch{chapter}] (completion 片段，前后共 ~{len(snippet)}c):")
        p(f"  {snippet}")
    p()

# 特别核查：叶欢 vs 叶临 是否混用
p("="*60)
p("  叶临 vs 叶欢 区分检查")
p("="*60)
for i, s in enumerate(samples):
    comp = s.get("completion", "")
    has_linlin = "叶临" in comp
    has_huan   = "叶欢" in comp
    if has_linlin and has_huan:
        p(f"  Sample #{i+1} (ch{s.get('chapter','?')}): 同时包含「叶临」和「叶欢」")

# 老刀疤在各样本的出现章节范围
p()
p("="*60)
p("  老刀疤 出现的完整章节列表")
p("="*60)
for i, s in enumerate(samples):
    comp = s.get("completion", "")
    if "老刀疤" in comp:
        ch = s.get("chapter") or "?"
        cnt = comp.count("老刀疤")
        p(f"  Sample #{i+1} ch{ch}: 出现 {cnt} 次，completion={len(comp)}c")

p()
p("="*60)
p("  叶欢 出现的完整章节列表")
p("="*60)
for i, s in enumerate(samples):
    comp = s.get("completion", "")
    if "叶欢" in comp:
        ch = s.get("chapter") or "?"
        cnt = comp.count("叶欢")
        p(f"  Sample #{i+1} ch{ch}: 出现 {cnt} 次，completion={len(comp)}c")

Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_FILE).write_text("\n".join(lines), encoding="utf-8")
print(f"Written to {OUT_FILE}")
