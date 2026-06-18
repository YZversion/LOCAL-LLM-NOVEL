#!/usr/bin/env python3
"""
阶段4：构造 QLoRA 小样本训练数据

在主 Python 环境下运行（不是 .venv-train/）——依赖 cowriter/ 的 jieba、rank_bm25。
不导入 unsloth，不加载 LLM。

每条样本遵守时序口径：续写第 N 章时 max_chapter = N-1，
由 cowriter.chapter.max_chapter_for_target 计算（不重新实现）。
prompt 结构由 cowriter.prompts.build_prompt 构造（与生产链路完全相同）。

Usage:
    python pipeline/build_train_samples.py
    python pipeline/build_train_samples.py --chapters 2-21
    python pipeline/build_train_samples.py --chapters 2-11 --context-chars 500 --completion-chars 400

Output:
    data/processed/train_samples.jsonl    — 每行一条 JSON 训练样本
    data/processed/train_samples.meta.json — 构建元数据（章节范围、参数等）
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cowriter.chapter import max_chapter_for_target
from cowriter.prompts import build_prompt
from cowriter.retriever import Retriever


# ── Chapter parsing ──────────────────────────────────────────────────────────

_CH_HDR_RE = re.compile(r'\n(第[^\n]{1,20}章[^\n]*)\n')


def parse_chapters(raw_text: str) -> list[dict]:
    """Split raw novel into chapters. Chapter number = 1-indexed position order.

    Returns list of {'chapter': N, 'heading': '第X章 ...', 'body': '...'}
    Chapter N = 1-indexed position (first heading → chapter 1).
    """
    matches = list(_CH_HDR_RE.finditer(raw_text))
    chapters: list[dict] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        chapters.append({'chapter': i + 1, 'heading': heading, 'body': body})
    return chapters


# ── Sample construction ──────────────────────────────────────────────────────

def build_sample(
    chapter_idx: int,
    chapters: list[dict],
    retriever: Retriever,
    context_chars: int,
    completion_chars: int,
    bible_top_k: int = 3,
    bible_max_chars: int | None = None,
    prior_max_chars: int = 400,
) -> dict:
    """Build one training sample for target chapter at chapters[chapter_idx].

    Input side (prompt):
      - recent_text: last `context_chars` of the preceding chapter body
      - retrieval: story_bible BM25 + grep, filtered by max_chapter = N-1
        bible_top_k / bible_max_chars let training samples use a tighter budget
        than production without touching prompts.py or retriever.py.
      - prior_summary: chapter summaries up to N-1, capped at prior_max_chars
      - instruction: chapter heading (用章节标题作续写方向)
      All assembled by cowriter.prompts.build_prompt (same as production).

    Target side (completion):
      - First `completion_chars` of the target chapter body
    """
    target = chapters[chapter_idx]
    prev = chapters[chapter_idx - 1]
    N = target['chapter']
    max_chap = max_chapter_for_target(N)  # = N-1, from cowriter/chapter.py

    # recent_text: 上一章末尾，作为"当前上文"输入
    recent_text = prev['body'][-context_chars:] if len(prev['body']) > context_chars else prev['body']

    # 通过 retriever 做带时序过滤的检索（与生产链路相同接口）
    retrieval = retriever.retrieve(recent_text, max_chapter=max_chap)

    # 训练样本专用：限制 bible 条数与每条字符上限
    # 在传入 build_prompt 之前剪裁，不修改 retriever.py / prompts.py
    bible_entries = retrieval["bible"][:bible_top_k]
    if bible_max_chars is not None:
        bible_entries = [
            {**e, "text": e["text"][:bible_max_chars]} for e in bible_entries
        ]
    # 训练样本不使用 grep（grep_raw 无章节边界，已确认4个具体案例跨章节泄漏；
    # 参见第零步调查，ch12 案例最严重：续写ch12时grep返回了ch12自身原文段落）
    retrieval_for_prompt = {**retrieval, "bible": bible_entries, "grep": []}

    # 前情提要（已由 retriever 按 chapter_number <= max_chap 过滤）
    prior_summary = retriever.get_prior_summaries(max_chap, max_chars=prior_max_chars)

    # 用与生产完全相同的 build_prompt 构造消息列表
    messages = build_prompt(
        recent_text=recent_text,
        summary="",                     # 小样本训练不引入会话内滚动摘要
        retrieval=retrieval_for_prompt,
        instruction=target['heading'],  # 章节标题作续写方向
        prior_summary=prior_summary,
    )

    # 目标侧：该章节实际原文开头
    completion = target['body'][:completion_chars]

    return {
        'target_chapter': N,
        'max_chapter_used': max_chap,
        'heading': target['heading'],
        'messages': messages,
        'completion': completion,
        # 验证元数据（_test_train_samples.py 使用）
        '_meta': {
            'context_chars': context_chars,
            'actual_context_chars': len(recent_text),
            'completion_chars': completion_chars,
            'actual_completion_chars': len(completion),
            'bible_top_k': bible_top_k,
            'bible_max_chars': bible_max_chars,
            'prior_max_chars': prior_max_chars,
            'bible_hits': [r['source'] for r in bible_entries],
            'prior_summary_chars': len(prior_summary),
            'grep_stripped': True,  # grep 已主动清空，见 build_sample() 注释
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_range(s: str) -> tuple[int, int]:
    parts = s.split('-')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"章节范围格式应为 'lo-hi'，如 '2-21'，得到: {s!r}")
    return int(parts[0]), int(parts[1])


def main() -> int:
    parser = argparse.ArgumentParser(description='Build QLoRA training samples (Task 1)')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml (default: config.yaml)')
    parser.add_argument('--chapters', default='2-21',
                        help='Target chapter range, e.g. "2-21" (default). '
                             'Chapter 1 cannot be a target (no prior context).')
    parser.add_argument('--context-chars', type=int, default=1000,
                        help='Chars of previous chapter body to use as recent_text '
                             '(default: 1000; production uses 2000 but shorter fits '
                             'training seq_len better)')
    parser.add_argument('--completion-chars', type=int, default=600,
                        help='Chars of target chapter body to use as completion '
                             '(default: 600, matching config output_tokens=600)')
    parser.add_argument('--bible-top-k', type=int, default=3,
                        help='Max bible entries per sample (default: 3, same as config; '
                             'use 2 for 1024-token budget)')
    parser.add_argument('--bible-max-chars', type=int, default=None,
                        help='Truncate each bible entry body to N chars before build_prompt '
                             '(default: None = prompts.py clip of 400c; use 250 for 1024-token budget)')
    parser.add_argument('--prior-max-chars', type=int, default=400,
                        help='Max chars for prior chapter summaries '
                             '(default: 400, same as retriever default; use 50 for 1024-token budget)')
    parser.add_argument('--output', default='data/processed/train_samples.jsonl',
                        help='Output JSONL path')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f'ERROR: config not found: {cfg_path}', file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

    lo, hi = parse_range(args.chapters)
    if lo < 2:
        print('ERROR: minimum target chapter is 2 (chapter 1 has no prior context)',
              file=sys.stderr)
        return 1

    # Load raw novel text
    raw_dir = Path(cfg['paths']['raw_data'])
    txts = sorted(raw_dir.glob('*.txt'))
    if not txts:
        print(f'ERROR: no .txt files found in {raw_dir}', file=sys.stderr)
        return 1
    raw_text = txts[0].read_text(encoding='utf-8')
    chapters = parse_chapters(raw_text)
    total_ch = len(chapters)
    print(f'Parsed {total_ch} chapters from {txts[0].name}')

    if hi > total_ch:
        print(f'WARNING: hi={hi} exceeds total chapters={total_ch}, clamping to {total_ch}')
        hi = total_ch
    if lo > hi:
        print('ERROR: lo > hi after clamping', file=sys.stderr)
        return 1

    # Init retriever (reads story_bible, builds BM25 index)
    retriever = Retriever(cfg)
    print(f'Retriever: {len(retriever._docs)} bible docs indexed')

    # Build samples
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for N in range(lo, hi + 1):
        sample = build_sample(
            chapter_idx=N - 1,
            chapters=chapters,
            retriever=retriever,
            context_chars=args.context_chars,
            completion_chars=args.completion_chars,
            bible_top_k=args.bible_top_k,
            bible_max_chars=args.bible_max_chars,
            prior_max_chars=args.prior_max_chars,
        )
        samples.append(sample)
        m = sample['_meta']
        print(
            f'  ch{N:02d} (max_chap={sample["max_chapter_used"]:2d}) '
            f'context={m["actual_context_chars"]}c  '
            f'completion={m["actual_completion_chars"]}c  '
            f'bible={m["bible_hits"]}(top_k={m["bible_top_k"]},max={m["bible_max_chars"]}c)  '
            f'prior={m["prior_summary_chars"]}c(max={m["prior_max_chars"]}c)'
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    meta = {
        'total_samples': len(samples),
        'chapter_range': f'{lo}-{hi}',
        'source_file': txts[0].name,
        'context_chars': args.context_chars,
        'completion_chars': args.completion_chars,
        'bible_top_k': args.bible_top_k,
        'bible_max_chars': args.bible_max_chars,
        'prior_max_chars': args.prior_max_chars,
        'bible_docs_in_index': len(retriever._docs),
    }
    meta_path = out_path.with_suffix('').with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\nWrote {len(samples)} samples → {out_path}')
    print(f'Meta   → {meta_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
