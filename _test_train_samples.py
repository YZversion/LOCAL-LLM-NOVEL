#!/usr/bin/env python3
"""
阶段4验证：检查 train_samples.jsonl 的时序过滤和章节对齐。

Usage:
    python _test_train_samples.py
    python _test_train_samples.py --samples data/processed/train_samples.jsonl

检查项（每条样本）：
  1. max_chapter_used == target_chapter - 1  （时序口径正确性）
  2. completion 开头 50 字符出现在对应章节原文中  （章节对齐）
  3. _meta.bible_hits 中每个来源的 revealed_in <= max_chapter_used
     （时序过滤确实生效，未泄露未来设定）
  4. 抽样（每5条取1条）打印人工可读快照，供目测确认

Exit 0 = 全部通过，Exit 1 = 有失败。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── 复用 retriever 的 frontmatter 解析（局部定义，不依赖私有函数）────────────

_FM_RE = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n', re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    try:
        import yaml as _yaml
        meta = _yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[m.end():]


# ── Chapter parsing（与 build_train_samples.py 完全相同逻辑）────────────────

_CH_HDR_RE = re.compile(r'\n(第[^\n]{1,20}章[^\n]*)\n')


def parse_chapters(raw_text: str) -> list[dict]:
    matches = list(_CH_HDR_RE.finditer(raw_text))
    chapters: list[dict] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        chapters.append({'chapter': i + 1, 'heading': heading, 'body': body})
    return chapters


# ── 检查函数 ────────────────────────────────────────────────────────────────

def check_sample(
    sample: dict,
    chapters: list[dict],
    bible_dir: Path,
) -> list[str]:
    """返回 error 列表。空列表 = PASS。"""
    errors: list[str] = []
    N = sample.get('target_chapter')
    max_chap = sample.get('max_chapter_used')

    # ── 检查 1: 时序口径 ──────────────────────────────────────────────────
    if N is None or max_chap is None:
        errors.append(f'sample missing target_chapter or max_chapter_used fields')
        return errors

    expected_max = N - 1
    if max_chap != expected_max:
        errors.append(
            f'ch{N}: max_chapter_used={max_chap} but expected {expected_max} '
            f'(should equal target_chapter - 1)'
        )

    # ── 检查 2: completion 章节对齐 ───────────────────────────────────────
    if N < 1 or N > len(chapters):
        errors.append(f'ch{N}: chapter index out of range (total={len(chapters)})')
    else:
        target_body = chapters[N - 1]['body']
        completion = sample.get('completion', '')
        snippet = completion[:50]
        if snippet and snippet not in target_body:
            errors.append(
                f'ch{N}: completion snippet not found in chapter body. '
                f'Snippet={snippet!r:.40}...'
            )
        if not completion:
            errors.append(f'ch{N}: completion is empty')

    # ── 检查 3: bible_hits 的 revealed_in <= max_chap ─────────────────────
    bible_hits = sample.get('_meta', {}).get('bible_hits', [])
    for source in bible_hits:
        candidates = list(bible_dir.rglob(f'{source}.md'))
        if not candidates:
            # 文件不存在于 story_bible（可能已清理），跳过
            continue
        raw = candidates[0].read_text(encoding='utf-8')
        meta, _ = _parse_frontmatter(raw)
        ri = meta.get('revealed_in')
        vf = meta.get('valid_from')
        if ri is None or vf is None:
            errors.append(
                f'ch{N}: bible hit "{source}" missing revealed_in/valid_from '
                f'— should not have passed temporal filter'
            )
        elif not isinstance(ri, int) or not isinstance(vf, int):
            errors.append(f'ch{N}: bible hit "{source}" revealed_in/valid_from not int')
        elif ri > max_chap:
            errors.append(
                f'ch{N}: bible hit "{source}" revealed_in={ri} > max_chap={max_chap} '
                f'— temporal filter LEAK!'
            )
        elif vf > max_chap:
            errors.append(
                f'ch{N}: bible hit "{source}" valid_from={vf} > max_chap={max_chap} '
                f'— temporal filter LEAK!'
            )

    return errors


def _snapshot(sample: dict, chapters: list[dict]) -> str:
    """生成人工可读快照（抽样目视检查用）。"""
    N = sample['target_chapter']
    max_chap = sample['max_chapter_used']
    heading = sample.get('heading', '')
    bible_hits = sample.get('_meta', {}).get('bible_hits', [])
    prior_c = sample.get('_meta', {}).get('prior_summary_chars', 0)
    completion_preview = sample.get('completion', '')[:60].replace('\n', '<NL>')

    # 从 messages 提取 user 消息预览
    messages = sample.get('messages', [])
    user_content = next((m['content'] for m in messages if m['role'] == 'user'), '')
    user_preview = user_content[:120].replace('\n', '<NL>')

    return (
        f'  ch{N:02d} ({heading}) max_chap={max_chap}\n'
        f'    bible_hits  : {bible_hits}\n'
        f'    prior_chars : {prior_c}c\n'
        f'    user_preview: {user_preview!r:.100}\n'
        f'    completion  : {completion_preview!r:.60}\n'
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description='Validate train_samples.jsonl')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--samples', default='data/processed/train_samples.jsonl')
    parser.add_argument('--raw-file', default=None,
                        help='Path to raw novel .txt (overrides auto-detection; '
                             'required when multiple .txt files exist in raw_data)')
    parser.add_argument('--snapshot-every', type=int, default=5,
                        help='Print human-readable snapshot every N samples (default: 5)')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f'ERROR: {cfg_path} not found', file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f'ERROR: {samples_path} not found — run build_train_samples.py first',
              file=sys.stderr)
        return 1

    # Load raw novel
    raw_dir = Path(cfg['paths']['raw_data'])
    if args.raw_file:
        raw_path = Path(args.raw_file)
        if not raw_path.exists():
            print(f'ERROR: --raw-file not found: {raw_path}', file=sys.stderr)
            return 1
        raw_text = raw_path.read_text(encoding='utf-8')
        print(f'Loaded chapters from {raw_path.name} (--raw-file)')
    else:
        txts = sorted(raw_dir.glob('*.txt'))
        if not txts:
            print(f'ERROR: no .txt in {raw_dir}', file=sys.stderr)
            return 1
        if len(txts) > 1:
            print(f'WARNING: {len(txts)} .txt files in {raw_dir}; using {txts[0].name}. '
                  f'Use --raw-file to specify explicitly.', file=sys.stderr)
        raw_text = txts[0].read_text(encoding='utf-8')
        print(f'Loaded chapters from {txts[0].name}')
    chapters = parse_chapters(raw_text)
    print(f'  {len(chapters)} chapters parsed')

    bible_dir = Path(cfg['paths']['story_bible'])

    # Load samples
    samples: list[dict] = []
    with open(samples_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f'Loaded {len(samples)} samples from {samples_path}\n')

    # ── Run checks ──────────────────────────────────────────────────────────
    print('=== 自动化验证 ===')
    total_errors = 0
    failed_samples: list[int] = []

    for i, sample in enumerate(samples):
        errors = check_sample(sample, chapters, bible_dir)
        N = sample.get('target_chapter', '?')
        if errors:
            total_errors += len(errors)
            failed_samples.append(N)
            for e in errors:
                print(f'  [FAIL] {e}')
        else:
            print(f'  [PASS] ch{N:02d}', end='')
            # Show bible hits inline for quick review
            hits = sample.get('_meta', {}).get('bible_hits', [])
            print(f'  bible={hits}')

    print()

    # ── Snapshot（抽样人工目视）──────────────────────────────────────────────
    print('=== 抽样快照（每5条取1条，供人工确认）===')
    for i, sample in enumerate(samples):
        if i % args.snapshot_every == 0:
            print(_snapshot(sample, chapters))

    # ── Summary ──────────────────────────────────────────────────────────────
    print('=== 汇总 ===')
    print(f'总样本数  : {len(samples)}')
    print(f'检查项目  : 时序口径 + 章节对齐 + bible revealed_in 各 {len(samples)} 项')
    print(f'失败样本  : {failed_samples if failed_samples else "无"}')
    print(f'总错误数  : {total_errors}')

    if total_errors == 0:
        print('\n[ALL PASS] 所有检查通过。')
        return 0
    else:
        print(f'\n[FAIL] {total_errors} 个错误，见上方明细。')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
