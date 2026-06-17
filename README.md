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
| System A — temporal filtering (`revealed_in` / `valid_from` / `valid_to`) | ✅ Complete, 43 tests passing |
| Chapter summaries (`chapter_summaries.md`) | ✅ ch1–58 complete |
| Story bible character cards | ✅ 21 characters, ch1–20 coverage |
| System B — knowledge graph + dynamic story_bible update | 🔧 In progress |
| Phase 4 — QLoRA fine-tuning | ⏸ Deferred (after System B) |
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
  kg_extract.py              # [System B] Extract entities from chapter text
  kg_update.py               # [System B] Merge into knowledge graph
  kg_render.py               # [System B] Render .md cards from graph
  update_kg.py               # [System B] One-command pipeline: extract→update→render

pipeline/
  eval_style.py     # Deterministic style metrics (repetition, contamination, score)
  train_qlora.py    # QLoRA fine-tuning entry point (placeholder)
  export_gguf.py    # GGUF export entry point (placeholder)

data/
  raw/              # Original novel text (gitignored)
  story_bible/      # Retrieved context cards (.md with YAML frontmatter)
    generated/
      characters/   # Per-character .md files
    chapter_summaries.md
    kg.json         # [System B] Knowledge graph (gitignored)

config.yaml         # Single source of truth for all parameters
```

---

## System B — Knowledge graph (in progress)

After writing each new chapter, run:

```powershell
python scripts/update_kg.py --chapter 59 --input outputs/chapter_059.txt
```

This will:
- Extract new characters, relationship changes, and state updates via LLM
- Merge them into `data/story_bible/kg.json`
- Re-render affected character `.md` cards with updated frontmatter
- Make new info available to the retriever for chapter 60+

Character states are tracked as timelines, so the system always shows a character's state **as of the chapter being written**, not a future state.

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
