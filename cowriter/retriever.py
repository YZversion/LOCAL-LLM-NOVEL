import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import jieba
import jieba.posseg as pseg
import yaml
from rank_bm25 import BM25Okapi

# ── Frontmatter ──────────────────────────────────────────────────────────────

_FM_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from Markdown body. Returns ({meta}, body)."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[m.end():]


# ── 聚合文件：从 BM25 索引中排除，这些文件含全书跨章节信息 ─────────────────
_SKIP_FROM_INDEX = {
    "characters", "relationships", "timeline",
    "plot_threads", "chapter_summaries",
}

# ── chapter_summaries.md 解析正则 ────────────────────────────────────────────
_CS_SECTION_RE = re.compile(r'\n(?=## )')
_CS_NUM_RE = re.compile(r'\*\*章节编号\*\*\s*[:：]\s*(\d+)')
_CS_SUMMARY_RE = re.compile(r'\*\*本章概要\*\*\s*[:：]\s*(.+?)(?=\n\s*-\s*\*\*|\Z)', re.DOTALL)
_CS_HEADER_RE = re.compile(r'^##\s+(.+)')

# 小说场景高频无效词（形容词、副词、状态词等，不应作为检索实体）
_STOPWORDS = {
    "没有", "一个", "这个", "那个", "什么", "怎么", "因为", "所以",
    "但是", "然后", "如果", "虽然", "可以", "一种", "一些", "这些", "那些",
    "时候", "地方", "东西", "自己", "知道", "感觉", "觉得", "看到", "听到",
    "片刻", "声音", "湖水", "衣衫", "眼睛", "心中", "身上", "手中",
    "之中", "之间", "之下", "之上", "一番", "一阵", "一道", "一股", "一丝",
    "这时", "此时", "同时", "随即", "突然", "忽然", "顿时", "于是", "此刻",
    "的确", "果然", "竟然", "居然", "不禁", "不由", "似乎", "仿佛",
    "淡淡", "缓缓", "慢慢", "轻轻", "静静", "默默", "悄悄", "幽幽",
}

# jieba posseg 词性：人名、地名、机构名、其他专名
_ENTITY_POS = {"nr", "ns", "nt", "nz"}

_BIBLE_TEXT_MAX = 1200  # 单条设定集返回最大字数，避免撑爆 prompt


class Retriever:
    def __init__(self, config: dict):
        self.cfg = config
        self.bible_path = Path(config["paths"]["story_bible"])
        self.raw_path = Path(config["paths"]["raw_data"])
        self._docs: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._entity_names: set[str] = set()  # story bible 文件名作为已知实体
        self._load_bible()

    # ── 统一 Tokenizer ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """BM25 建索引和查询共用同一套分词：jieba 切词，过滤单字和空词。"""
        return [w for w in jieba.cut(text) if len(w) > 1 and w.strip()]

    # ── 设定集 BM25 ──────────────────────────────────────────────────────────

    def _load_bible(self):
        self._docs = []
        self._entity_names = set()
        for f in sorted(self.bible_path.rglob("*.md")):
            rel = f.relative_to(self.bible_path)
            stem = f.stem
            # 聚合文件排除出 BM25 索引（含全书跨章节信息，由专用函数单独处理）
            if stem in _SKIP_FROM_INDEX and len(rel.parts) == 1:
                continue
            self._entity_names.add(stem)
            jieba.add_word(stem, freq=10000)
            raw = f.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(raw)
            self._docs.append({
                "source": stem,
                "path": f,
                "text": body,          # BM25 不索引 frontmatter 原文
                "meta": meta,          # 完整 frontmatter dict，供 _visible() 使用
            })
        if self._docs:
            tokenized = [self._tokenize(d["text"]) for d in self._docs]
            self._bm25 = BM25Okapi(tokenized)

    def reload_bible(self):
        self._load_bible()

    def _stem_priority(self, path: Path) -> int:
        """根目录手写卡片=0（最高），generated/=1，其他子目录=2。"""
        rel = path.relative_to(self.bible_path)
        if len(rel.parts) == 1:
            return 0
        return 1 if rel.parts[0] == "generated" else 2

    def search_bible(self, query: str, top_k: int | None = None,
                     entities: list[str] | None = None,
                     max_chapter: int | None = None) -> list[dict]:
        """返回 {source, text, score}，text 截断至 _BIBLE_TEXT_MAX 字。

        max_chapter 不为 None 时启用时序过滤：
          - revealed_in <= max_chapter → 可见
          - revealed_in > max_chapter  → 不可见
          - 无 frontmatter（revealed_in=None） → 不可见（防意外泄漏）
        max_chapter 为 None 时不过滤，行为与旧版一致。

        提供 entities 时，文件名精确匹配的文档强制排在 BM25 结果之前：
          根目录手写卡片 > generated/characters/ > stem 含 entity > BM25
        """
        if not self._bm25:
            return []
        k = top_k or self.cfg["retrieval"]["bible_top_k"]
        seen: set[int] = set()
        results: list[dict] = []

        def _visible(idx: int) -> bool:
            if max_chapter is None:
                return True
            meta = self._docs[idx].get("meta", {})
            ri = meta.get("revealed_in")
            vf = meta.get("valid_from")
            vt = meta.get("valid_to")
            # 两个字段都必须存在，任一缺失则视为不可见（防意外泄漏）
            if not isinstance(ri, int) or not isinstance(vf, int):
                return False
            return (
                ri <= max_chapter
                and vf <= max_chapter
                and (vt is None or (isinstance(vt, int) and vt >= max_chapter))
            )

        # ── 文件名精确 / 弱匹配提权 ──────────────────────────────────────
        if entities:
            exact: list[tuple[int, int]] = []   # (path_priority, doc_idx)
            weak: list[int] = []
            for entity in entities:
                for i, doc in enumerate(self._docs):
                    if not _visible(i):
                        continue
                    stem = doc["source"]
                    p = doc.get("path")
                    if stem == entity:
                        prio = self._stem_priority(p) if p else 2
                        exact.append((prio, i))
                    elif entity in stem:
                        weak.append(i)
            exact.sort(key=lambda x: x[0])
            for _, i in exact:
                if i not in seen:
                    seen.add(i)
                    doc = self._docs[i]
                    results.append({"source": doc["source"],
                                    "text": doc["text"][:_BIBLE_TEXT_MAX],
                                    "score": 9999.0})
            for i in weak:
                if i not in seen:
                    seen.add(i)
                    doc = self._docs[i]
                    results.append({"source": doc["source"],
                                    "text": doc["text"][:_BIBLE_TEXT_MAX],
                                    "score": 9998.0})

        # ── BM25 结果（去重后追加）──────────────────────────────────────
        tokens = self._tokenize(query)
        if tokens:
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in ranked:
                if scores[i] <= 0:
                    break
                if not _visible(i):
                    continue
                if i not in seen:
                    seen.add(i)
                    doc = self._docs[i]
                    results.append({"source": doc["source"],
                                    "text": doc["text"][:_BIBLE_TEXT_MAX],
                                    "score": round(float(scores[i]), 4)})

        return results[:k]

    # ── 原文 grep ────────────────────────────────────────────────────────────

    def grep_raw(self, keyword: str, top_k: int | None = None) -> list[str]:
        k = top_k or self.cfg["retrieval"]["top_k"]
        ctx = self.cfg["retrieval"]["grep_context_lines"]
        results: list[str] = []

        try:
            cmd = [
                "rg", "-F", "--encoding", "utf-8", "-n",
                f"--context={ctx}", "--", keyword, str(self.raw_path),
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=10
            )
            if proc.returncode == 2:
                # rg 报错（路径不存在、权限等），静默忽略
                return []
            if proc.returncode == 0:
                blocks = [b.strip() for b in proc.stdout.split("--\n") if b.strip()]
                results = blocks[:k]
        except FileNotFoundError:
            # ripgrep 未安装，退化到 Python 递归搜索
            for txt in sorted(self.raw_path.rglob("*.txt")):
                lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                for i, line in enumerate(lines):
                    if keyword in line:
                        lo = max(0, i - ctx)
                        hi = min(len(lines), i + ctx + 1)
                        results.append("\n".join(lines[lo:hi]))
                        if len(results) >= k:
                            return results
        except subprocess.TimeoutExpired:
            pass

        return results

    # ── 实体抽取 ─────────────────────────────────────────────────────────────

    def extract_entities(self, text: str, top_n: int = 5) -> list[str]:
        """
        第一优先：story bible 文件名直接命中（人名/地名/功法名最可靠）。
        第二：jieba.posseg 识别的命名实体词性（nr/ns/nt/nz）。
        过滤停用词，按词频排序（稳定排序保证 bible 实体优先在前）。
        """
        seen: set[str] = set()
        entities: list[str] = []

        # 优先：已知 bible 实体名
        for name in self._entity_names:
            if name in text and name not in seen:
                entities.append(name)
                seen.add(name)

        # 兜底：posseg 命名实体
        for word, flag in pseg.cut(text):
            if (
                flag in _ENTITY_POS
                and len(word) >= 2
                and word not in seen
                and word not in _STOPWORDS
            ):
                entities.append(word)
                seen.add(word)

        # 按在文本中的词频降序（稳定排序：同频时保持 bible 实体在前）
        freq = Counter(w for w in jieba.cut(text) if w in seen)
        entities.sort(key=lambda w: freq.get(w, 0), reverse=True)

        return entities[:top_n]

    # ── 前情提要（按章节时序，直接解析 chapter_summaries.md）───────────────

    def get_prior_summaries(self, up_to_chapter: int, max_chars: int = 400) -> str:
        """从 chapter_summaries.md 解析章节摘要，只返回 chapter_number <= up_to_chapter 的条目。

        不依赖 _merged_data.json（该文件结构已损坏）。

        支持格式：
          ## 第N章 <标题>
          - **章节编号**: N
          - **本章概要**: ...
          - **出场人物**: ...
          ---
        """
        summaries_path = self.bible_path / "chapter_summaries.md"
        if not summaries_path.exists():
            return ""
        try:
            text = summaries_path.read_text(encoding="utf-8")
        except Exception:
            return ""

        sections = _CS_SECTION_RE.split(text)
        filtered: list[tuple[int, str, str]] = []  # (chapter_num, title, summary)

        for section in sections:
            section = section.strip()
            if not section.startswith("##"):
                continue
            m_num = _CS_NUM_RE.search(section)
            if not m_num:
                continue
            ch_num = int(m_num.group(1))
            if ch_num > up_to_chapter:
                continue
            m_sum = _CS_SUMMARY_RE.search(section)
            what = m_sum.group(1).strip() if m_sum else ""
            if not what or what == "未明确":
                continue
            m_hdr = _CS_HEADER_RE.match(section)
            title = m_hdr.group(1).strip() if m_hdr else f"第{ch_num}章"
            filtered.append((ch_num, title, what))

        filtered.sort(key=lambda x: x[0])

        parts: list[str] = []
        total = 0
        for _, title, what in filtered:
            line = f"[{title}] {what}"
            if total + len(line) + 1 > max_chars:
                break
            parts.append(line)
            total += len(line) + 1

        return "\n".join(parts)

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def retrieve(self, context: str, max_chapter: int | None = None) -> dict:
        """检索 story_bible 和原文。max_chapter 不为 None 时启用时序过滤。

        注意：grep_raw 搜索原始 txt，无法按章节过滤——grep 结果不受 max_chapter 约束，
        仅作文风参考，不应携带关键设定信息。
        """
        recent = context[-500:]
        entities = self.extract_entities(recent)

        bible_query = " ".join(entities) + " " + context[-150:]
        bible_hits = self.search_bible(bible_query, entities=entities, max_chapter=max_chapter)

        seen_grep: set[str] = set()
        grep_hits: list[str] = []
        for entity in entities[:3]:
            for hit in self.grep_raw(entity, top_k=2):
                if hit not in seen_grep:
                    seen_grep.add(hit)
                    grep_hits.append(hit)

        return {
            "entities": entities,
            "bible": bible_hits[: self.cfg["retrieval"]["bible_top_k"]],
            "grep": grep_hits[: self.cfg["retrieval"]["top_k"]],
        }
