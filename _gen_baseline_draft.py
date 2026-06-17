#!/usr/bin/env python3
"""One-off script: generate a multi-continuation baseline draft for phase-4 pre-validation.
Produces outputs/draft_baseline_phase4.txt. Does NOT modify any boundary files.
Delete this script after use if desired."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml
from cowriter.session import Session

config_path = ROOT / "config.yaml"
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

session = Session(config)

# Seed with the existing single-sentence draft
seed_text = (ROOT / "outputs" / "draft_20260615_165719.txt").read_text(encoding="utf-8")
session.seed(seed_text)

print(f"Seeded with {len(seed_text)} chars. Generating 5 continuations...", flush=True)
for i in range(1, 6):
    print(f"  Continuation {i}/5 ...", end="", flush=True)
    draft = session.generate()
    session.accept(draft)
    print(f" {len(draft)} chars", flush=True)

path = session.save_draft()
# Rename to a stable baseline filename
stable = ROOT / "outputs" / "draft_baseline_phase4.txt"
path.rename(stable)
total = len(session.accepted_text)
print(f"\nSaved: {stable}  ({total} chars total)")
