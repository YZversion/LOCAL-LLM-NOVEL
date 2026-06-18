# 本地小说续写助手

A local-first AI co-writer for long-form Chinese fiction. Runs entirely on your own hardware via [Ollama](https://ollama.com/) — no cloud, no data leakage.

**Current model:** `huihui_ai/qwen3-abliterated:8b-v2` · **Hardware:** RTX 4070 Laptop 8GB

---

## What it does

Given a chapter target and a short writing instruction, the system builds a structured prompt from:

1. **Character & world settings** — retrieved from `story_bible/` via BM25, filtered to only show info revealed before the target chapter
2. **Prior chapter summaries** — auto-generated from the raw novel text, temporally gated
3. **Your writing instruction** — the outline for this chapter
4. **Recent context** — the last paragraphs of the manuscript so far

Then calls the local LLM to continue the story.

---

## Current status

| Component | Status |
|-----------|--------|
| System A — prompt pipeline + BM25 retrieval | ✅ Complete |
| System A — temporal filtering (`revealed_in` / `valid_from` / `valid_to`) | ✅ Complete, 45 tests passing |
| Chapter summaries (`chapter_summaries.md`) | ✅ ch1–58 complete |
| Story bible character cards | ✅ 21 characters, ch1–20 coverage |
| Phase 4 — QLoRA fine-tuning | 🔧 Current focus |
| Phase 4 — CUDA + Unsloth forward | ✅ Passed |
| Phase 4 — 8B 4-bit inference VRAM | ✅ Passed, peak 5.80 GB |
| Phase 4 — training samples | ✅ 20 samples, ch2–21, validation passed |
| Phase 4 — training run | ⚠️ OOM at `max_seq_length=2048`; next test uses shorter seq/sample |
| System B — dynamic memory update | ⏸ Deferred until QLoRA path is proven |
| Web UI (Gradio) | ✅ Available via `cowriter/web.py` |

---

## Quick start

```powershell
# Install dependencies
pip install -r requirements.txt

# Launch web UI
python cowriter/web.py

# Or use CLI
python cowriter/app.py
```

Requires Ollama running locally with the model pulled:
```powershell
ollama pull huihui_ai/qwen3-abliterated:8b-v2
```

---

## Project structure

```
cowriter/
  session.py       # LLM call, context compression, output cleaning
  retriever.py     # BM25 search + temporal filter + entity extraction
  prompts.py       # Prompt assembly (5-block structure)
  chapter.py       # max_chapter_for_target(N) = N-1
  web.py           # Gradio UI
  app.py           # CLI entry point

scripts/
  gen_chapter_summaries.py   # Batch-generate chapter summaries from raw text
  build_story_bible.py       # Build world/style/glossary cards
  split_characters.py        # Split character data into individual .md files
  add_frontmatter.py         # One-time: add YAML frontmatter to story_bible files
  eval_draft.py              # Style evaluation wrapper

pipeline/
  eval_style.py     # Deterministic style metrics (repetition, contamination, score)
  build_train_samples.py  # Build temporally safe QLoRA training samples
  train_qlora.py    # QLoRA VRAM probe and small-sample training entry point
  generate_lora.py  # Generate candidate text from a trained LoRA adapter
  export_gguf.py    # GGUF export entry point (placeholder)

data/
  raw/              # Original novel text (gitignored)
  story_bible/      # Retrieved context cards (.md with YAML frontmatter)
    generated/
      characters/   # Per-character .md files
    chapter_summaries.md
    kg.json         # [System B, planned] Knowledge graph (gitignored)

config.yaml         # Single source of truth for all parameters
```

---

## Phase 4 — QLoRA fine-tuning (current focus)

Training dependencies are isolated in `.venv-train/`; do not install them into the main runtime environment.

```powershell
# Create / activate the training venv first, then install torch cu130 before the rest:
python -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision==0.25.0 torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-train.txt
```

Validation commands:

```powershell
# Small model platform check: passed on 2026-06-18
.\.venv-train\Scripts\python.exe _test_unsloth_forward.py

# 8B 4-bit inference VRAM check: passed, peak 5.80 GB
.\.venv-train\Scripts\python.exe _test_unsloth_forward.py --model huihui-ai/Huihui-Qwen3-8B-abliterated-v2

# Build and validate temporally safe training samples
python pipeline/build_train_samples.py
python _test_train_samples.py
```

Current training blocker: `max_seq_length=2048` with 4-bit + LoRA r=16 reaches about 7.44 GB reserved and OOMs in fused cross entropy. Next step is to retry with `max_seq_length=512` or rebuild shorter samples, e.g. `context_chars=300` / `completion_chars=200`.

```powershell
.\.venv-train\Scripts\python.exe pipeline/train_qlora.py --max-seq-length 512
```

QLoRA is evaluated only on writing style, rhythm, and instruction following. Dynamic memory for new characters is a separate System B concern.

---

## System B — Dynamic memory (deferred)

System B will eventually update `story_bible/` after each accepted chapter. It is deliberately deferred until the QLoRA path is proven.

Planned shape:

```powershell
python scripts/update_kg.py --chapter 59 --input outputs/chapter_059.txt
```

Planned behavior:
- Extract new characters, relationship changes, and state updates via LLM
- Merge them into `data/story_bible/kg.json`
- Re-render affected character `.md` cards with updated frontmatter
- Make new info available to the retriever for chapter 60+

---

## Temporal filtering

Every retrievable `.md` file in `story_bible/` must have YAML frontmatter:

```yaml
---
title: 林清雪
type: character
revealed_in: 1
valid_from: 1
valid_to: null
---
```

When writing chapter N, `max_chapter = N - 1`. Files without both `revealed_in` and `valid_from` are invisible to the retriever — no accidental future spoilers.

---

## Style evaluation

```powershell
python scripts/eval_draft.py --reference data/raw/reference.txt --candidate outputs/draft.txt
```

Phase 3 baseline: `style_score 50.92/100`, `repetition_risk: high`, `contamination_risk: low`  
Target after fine-tuning: `style_score > 50.92`

---

## Data privacy

`data/`, `models/`, `outputs/` are all gitignored. Raw novel text never enters version control.
