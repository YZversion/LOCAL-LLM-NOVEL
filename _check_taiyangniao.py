#!/usr/bin/env python3
import json
from pathlib import Path

samples = [json.loads(l) for l in Path("data/processed/train_samples_full_57.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

terms = ["太阳神鸟", "金丹", "凝气", "昆仑", "轩辕神阳"]
out = []
for term in terms:
    hits_comp = [(i+1, s) for i, s in enumerate(samples) if term in s.get("completion", "")]
    hits_ctx  = [(i+1, s) for i, s in enumerate(samples)
                 if len(s.get("messages", [])) > 1 and term in s["messages"][1].get("content", "")]
    out.append(f"{term}: completion={len(hits_comp)}条  context={len(hits_ctx)}条")
    for idx, s in hits_comp[:2]:
        comp = s.get("completion", "")
        pos = comp.find(term)
        snip = comp[max(0,pos-50):pos+100]
        out.append(f"  [completion #{ idx }] ...{snip}...")

Path("outputs/debug/taiyangniao_check.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
