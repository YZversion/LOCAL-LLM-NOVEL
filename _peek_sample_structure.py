#!/usr/bin/env python3
"""查看第一条样本的字段结构"""
import json
from pathlib import Path

line = Path("data/processed/train_samples_full_57.jsonl").read_text(encoding="utf-8").splitlines()[0]
s = json.loads(line)
print("Keys:", list(s.keys()))
for k, v in s.items():
    if isinstance(v, str):
        print(f"  {k}: ({len(v)}c) {v[:80]!r}...")
    elif isinstance(v, list):
        print(f"  {k}: list[{len(v)}]")
        for i, item in enumerate(v):
            if isinstance(item, dict):
                print(f"    [{i}] role={item.get('role','?')} content=({len(item.get('content',''))}c) {item.get('content','')[:60]!r}...")
    else:
        print(f"  {k}: {v!r}")
