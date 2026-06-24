# 本地小说续写助手

A local-first AI co-writer for long-form Chinese fiction. Runs entirely on your own hardware — no cloud, no data leakage.

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
| Phase 4 — QLoRA fine-tuning | 🔧 v2 selected for local adapter UI |
| Phase 4 — CUDA + Unsloth forward | ✅ Passed |
| Phase 4 — 8B 4-bit inference VRAM | ✅ Passed, peak 5.80 GB |
| Phase 4 — v3/v4 | ✗ Not used for product |
| System B — dynamic memory update | ✅ MVP scripts + regression test |
| Local UI | ✅ PowerShell terminal UI via v2 adapter |
| Web UI (Gradio) | Available, but not required for local/offline product |

---

## Quick start

```powershell
# Install dependencies
pip install -r requirements.txt

# Launch the local v2 adapter UI (no web, no browser)
.\scripts\run_v2_local_ui.ps1

# Or pass a starting context file
.\scripts\run_v2_local_ui.ps1 -ContextFile outputs\debug\test_context_ch1_clean.txt
```

The older Ollama CLI remains available:
```powershell
python cowriter/app.py
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
  kg_extract.py              # System B: write editable facts draft
  kg_update.py               # System B: merge reviewed facts into kg.json
  kg_render.py               # System B: render Markdown cards for Retriever
  update_kg.py               # System B: facts -> kg.json -> Markdown cards
  run_v2_local_ui.ps1        # Local terminal UI for qlora_run_v2

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

## Local v2 Adapter UI

The current local product path uses `outputs/qlora_run_v2/` and the terminal UI in `pipeline/adapter_cli.py`.
It does not use Gradio or any web server.

```powershell
.\.venv-train\Scripts\Activate.ps1
.\scripts\run_v2_local_ui.ps1 -ContextFile <context.txt>
```

`run_v2_local_ui.ps1` sets a local Hugging Face cache proxy under `outputs/hf_stage0_proxy/`.
This avoids the Windows sandbox read-only cache issue during Unsloth import.

## System B — Dynamic memory MVP

System B now has the first controllable loop:

```text
accepted chapter text -> editable facts JSON -> kg.json -> Markdown cards -> Retriever BM25
```

```powershell
# Create an editable draft from accepted text
python scripts\kg_extract.py --chapter 59 --input outputs\chapter_059.txt --out outputs\system_b\ch59_facts.draft.json --entities 林清雪,颜儿

# After editing/reviewing the JSON, merge and render cards
python scripts\update_kg.py --facts outputs\system_b\ch59_facts.draft.json

# Regression test for kg.json -> cards -> Retriever temporal visibility
python _test_system_b.py
```

`kg.json` is the fact source. Markdown cards under `data/story_bible/generated/system_b/`
are only the projection layer for the existing Retriever.

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
