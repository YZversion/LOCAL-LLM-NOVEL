import subprocess
from collections import Counter
from pathlib import Path

import jieba
import jieba.posseg as pseg
from rank_bm25 import BM25Okapi

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
        for f in sorted(self.bible_path.glob("*.md")):
            stem = f.stem
            self._entity_names.add(stem)
            # 把文件名（人名/地名/功法名等）加入 jieba 词典，提高分词准确性
            jieba.add_word(stem, freq=10000)
            self._docs.append({
                "source": stem,
                "text": f.read_text(encoding="utf-8"),
            })
        if self._docs:
            tokenized = [self._tokenize(d["text"]) for d in self._docs]
            self._bm25 = BM25Okapi(tokenized)

    def reload_bible(self):
        self._load_bible()

    def search_bible(self, query: str, top_k: int | None = None) -> list[dict]:
        """返回 {source, text, score}，text 截断至 _BIBLE_TEXT_MAX 字。"""
        if not self._bm25:
            return []
        k = top_k or self.cfg["retrieval"]["bible_top_k"]
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            doc = self._docs[i]
            results.append({
                "source": doc["source"],
                "text": doc["text"][:_BIBLE_TEXT_MAX],
                "score": round(float(scores[i]), 4),
            })
        return results

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

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def retrieve(self, context: str) -> dict:
        recent = context[-500:]
        entities = self.extract_entities(recent)

        # bible query = 实体名拼接 + 最近 150 字（让 BM25 有足够上下文）
        bible_query = " ".join(entities) + " " + context[-150:]
        bible_hits = self.search_bible(bible_query)

        # grep 前 3 个实体，结果去重
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
