#!/usr/bin/env python3
"""
系统A时序过滤回归测试

运行：python _test_temporal_filter.py
成功：全绿，无 FAIL；退出码 0。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# 辅助：在临时目录里建 story_bible 结构，返回 Retriever 实例
# ────────────────────────────────────────────────────────────────────────────

def _make_config(bible_dir: Path, raw_dir: Path) -> dict:
    return {
        "model": {"ollama_model": "dummy"},
        "paths": {"story_bible": str(bible_dir), "raw_data": str(raw_dir)},
        "retrieval": {"bible_top_k": 5, "top_k": 3, "grep_context_lines": 2},
        "session": {"output_tokens": 500, "max_recent_chars": 1000,
                    "summary_trigger_chars": 3000},
        "generation": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
    }


def _write_md(path: Path, content: str,
              revealed_in: int | None = None,
              valid_from: int | None = None,
              valid_to: int | None = None) -> None:
    """写 Markdown 文件；提供 revealed_in 时同时写 valid_from（默认同值）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if revealed_in is not None:
        vf = valid_from if valid_from is not None else revealed_in
        vt_str = "null" if valid_to is None else str(valid_to)
        fm = (f"---\nrevealed_in: {revealed_in}\n"
              f"valid_from: {vf}\nvalid_to: {vt_str}\n---\n\n")
        content = fm + content
    path.write_text(content, encoding="utf-8")


def _make_retriever(bible_dir: Path, raw_dir: Path):
    from cowriter.retriever import Retriever
    return Retriever(_make_config(bible_dir, raw_dir))


# ────────────────────────────────────────────────────────────────────────────
# 单元：max_chapter_for_target
# ────────────────────────────────────────────────────────────────────────────

class TestMaxChapterForTarget(unittest.TestCase):
    def test_chapter_1(self):
        from cowriter.chapter import max_chapter_for_target
        self.assertEqual(max_chapter_for_target(1), 0)

    def test_chapter_5(self):
        from cowriter.chapter import max_chapter_for_target
        self.assertEqual(max_chapter_for_target(5), 4)

    def test_chapter_10(self):
        from cowriter.chapter import max_chapter_for_target
        self.assertEqual(max_chapter_for_target(10), 9)


# ────────────────────────────────────────────────────────────────────────────
# 单元：_parse_frontmatter
# ────────────────────────────────────────────────────────────────────────────

class TestParseFrontmatter(unittest.TestCase):
    def setUp(self):
        from cowriter.retriever import _parse_frontmatter
        self._parse = _parse_frontmatter

    def test_with_frontmatter(self):
        text = "---\nrevealed_in: 3\nvalid_from: 3\nvalid_to: null\n---\n\nBody text"
        meta, body = self._parse(text)
        self.assertEqual(meta.get("revealed_in"), 3)
        self.assertEqual(meta.get("valid_from"), 3)
        self.assertIsNone(meta.get("valid_to"))
        self.assertIn("Body text", body)
        self.assertNotIn("revealed_in", body)

    def test_no_frontmatter(self):
        text = "# Title\nSome content"
        meta, body = self._parse(text)
        self.assertEqual(meta, {})
        self.assertIn("Title", body)

    def test_invalid_yaml(self):
        text = "---\n: :\n---\nBody"
        meta, body = self._parse(text)
        self.assertEqual(meta, {})

    def test_non_dict_yaml(self):
        text = "---\n- item1\n- item2\n---\nBody"
        meta, body = self._parse(text)
        self.assertEqual(meta, {})


# ────────────────────────────────────────────────────────────────────────────
# 集成：search_bible 时序过滤（requires revealed_in + valid_from）
# ────────────────────────────────────────────────────────────────────────────

class TestSearchBibleFilter(unittest.TestCase):
    """
    注意：BM25 查询使用文件 stem 作为 entity 参数，因为 jieba 会把 stem
    注册为高频词（整体分词），导致纯文本 BM25 查询无法精确匹配。
    实际系统中也是先 extract_entities 再传 entities 给 search_bible。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bible_dir = Path(self._tmp.name) / "story_bible"
        self.raw_dir = Path(self._tmp.name) / "raw"
        self.raw_dir.mkdir(parents=True)

        # 角色甲：第1章出现（有完整 frontmatter）
        _write_md(self.bible_dir / "角色甲.md",
                  "### 角色甲\n角色甲是第一章的主角。", revealed_in=1)
        # 角色乙：第5章出现（有完整 frontmatter）
        _write_md(self.bible_dir / "角色乙.md",
                  "### 角色乙\n角色乙是第五章登场的人物。", revealed_in=5)
        # 无 frontmatter 的文件（缺少 valid_from → 时序过滤下不可见）
        _write_md(self.bible_dir / "无标注.md",
                  "### 无标注\n这是没有章节标注的人物。")

    def tearDown(self):
        self._tmp.cleanup()

    def _retriever(self):
        return _make_retriever(self.bible_dir, self.raw_dir)

    def _find(self, r, name: str, max_chapter=None) -> bool:
        """通过 entity 匹配查找指定名称的文档（不依赖 BM25 分词结果）。"""
        hits = r.search_bible("", entities=[name], max_chapter=max_chapter)
        return name in {h["source"] for h in hits}

    def test_no_filter_all_visible_via_entity(self):
        r = self._retriever()
        self.assertTrue(self._find(r, "角色甲", max_chapter=None))
        self.assertTrue(self._find(r, "角色乙", max_chapter=None))
        self.assertTrue(self._find(r, "无标注", max_chapter=None))

    def test_filter_ch3_visible_甲_hidden_乙(self):
        r = self._retriever()
        self.assertTrue(self._find(r, "角色甲", max_chapter=3))
        self.assertFalse(self._find(r, "角色乙", max_chapter=3))
        # 无 valid_from → 不可见
        self.assertFalse(self._find(r, "无标注", max_chapter=3))

    def test_filter_ch5_visible_甲乙(self):
        r = self._retriever()
        self.assertTrue(self._find(r, "角色甲", max_chapter=5))
        self.assertTrue(self._find(r, "角色乙", max_chapter=5))

    def test_filter_ch0_nothing_visible(self):
        r = self._retriever()
        self.assertFalse(self._find(r, "角色甲", max_chapter=0))
        self.assertFalse(self._find(r, "角色乙", max_chapter=0))

    def test_only_revealed_in_no_valid_from_invisible(self):
        """只有 revealed_in 没有 valid_from 时，temporal filter 下不可见。"""
        tmp2 = tempfile.TemporaryDirectory()
        bd = Path(tmp2.name) / "sb"
        rd = Path(tmp2.name) / "raw"
        rd.mkdir(parents=True)
        p = bd / "半截卡.md"
        p.parent.mkdir(parents=True)
        p.write_text("---\nrevealed_in: 1\n---\n\n### 半截角色\n只有 revealed_in。",
                     encoding="utf-8")
        r = _make_retriever(bd, rd)
        hits = r.search_bible("", entities=["半截卡"], max_chapter=5)
        self.assertNotIn("半截卡", {h["source"] for h in hits})
        tmp2.cleanup()

    def test_entity_boost_respects_filter(self):
        r = self._retriever()
        self.assertFalse(self._find(r, "角色乙", max_chapter=3))

    def test_entity_boost_visible(self):
        r = self._retriever()
        self.assertTrue(self._find(r, "角色甲", max_chapter=3))

    def test_aggregate_files_excluded_from_index(self):
        """聚合文件不进入 BM25 索引，即使 max_chapter=None 也搜不到。"""
        tmp2 = tempfile.TemporaryDirectory()
        bd = Path(tmp2.name) / "sb"
        rd = Path(tmp2.name) / "raw"
        rd.mkdir(parents=True)
        bd.mkdir()
        (bd / "characters.md").write_text("### 全书人物\n林甲 林乙 林丙", encoding="utf-8")
        _write_md(bd / "普通卡.md", "### 普通卡\n普通卡是测试角色。", revealed_in=1)
        r = _make_retriever(bd, rd)
        # 聚合文件不可被 entity 匹配找到
        hits_agg = r.search_bible("", entities=["characters"], max_chapter=None)
        self.assertNotIn("characters", {h["source"] for h in hits_agg})
        # 普通卡可以被 entity 匹配找到
        hits_ok = r.search_bible("", entities=["普通卡"], max_chapter=None)
        self.assertIn("普通卡", {h["source"] for h in hits_ok})
        tmp2.cleanup()


# ────────────────────────────────────────────────────────────────────────────
# 集成：retrieve() 时序过滤端到端
# ────────────────────────────────────────────────────────────────────────────

class TestRetrieveEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bible_dir = Path(self._tmp.name) / "story_bible"
        self.raw_dir = Path(self._tmp.name) / "raw"
        self.raw_dir.mkdir(parents=True)
        _write_md(self.bible_dir / "林清雪.md",
                  "### 林清雪\n来源章节：第1章\n主角。", revealed_in=1)
        _write_md(self.bible_dir / "反派.md",
                  "### 反派\n来源章节：第10章\n大反派。", revealed_in=10)

    def tearDown(self):
        self._tmp.cleanup()

    def test_retrieve_with_chapter_filters(self):
        r = _make_retriever(self.bible_dir, self.raw_dir)
        result = r.retrieve("林清雪在湖边徘徊", max_chapter=3)
        sources = {h["source"] for h in result["bible"]}
        self.assertIn("林清雪", sources)
        self.assertNotIn("反派", sources)


# ────────────────────────────────────────────────────────────────────────────
# 集成：get_prior_summaries（现在解析 chapter_summaries.md）
# ────────────────────────────────────────────────────────────────────────────

_SAMPLE_SUMMARIES_MD = """\
# 测试章节摘要

---

## 第一章 开篇

- **章节编号**: 1
- **本章概要**: 主角登场，离开故乡
- **出场人物**: 主角

---

## 第三章 转机

- **章节编号**: 3
- **本章概要**: 遭遇危机，结识友人
- **出场人物**: 主角、友人

---

## 第五章 高潮

- **章节编号**: 5
- **本章概要**: 转折点，反派现身
- **出场人物**: 主角、反派

---
"""


class TestGetPriorSummaries(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bible_dir = Path(self._tmp.name) / "story_bible"
        self.bible_dir.mkdir(parents=True)
        self.raw_dir = Path(self._tmp.name) / "raw"
        self.raw_dir.mkdir(parents=True)
        (self.bible_dir / "chapter_summaries.md").write_text(
            _SAMPLE_SUMMARIES_MD, encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _retriever(self):
        return _make_retriever(self.bible_dir, self.raw_dir)

    def test_summaries_up_to_ch3(self):
        r = self._retriever()
        text = r.get_prior_summaries(up_to_chapter=3)
        self.assertIn("主角登场", text)
        self.assertIn("遭遇危机", text)
        self.assertNotIn("转折点", text)

    def test_summaries_up_to_ch0(self):
        r = self._retriever()
        text = r.get_prior_summaries(up_to_chapter=0)
        self.assertEqual(text.strip(), "")

    def test_summaries_full(self):
        r = self._retriever()
        text = r.get_prior_summaries(up_to_chapter=5)
        self.assertIn("转折点", text)

    def test_summaries_excludes_future(self):
        r = self._retriever()
        text = r.get_prior_summaries(up_to_chapter=4)
        self.assertNotIn("转折点", text)   # ch5 内容不应出现
        self.assertIn("遭遇危机", text)

    def test_missing_summaries_file(self):
        r = self._retriever()
        (self.bible_dir / "chapter_summaries.md").unlink()
        text = r.get_prior_summaries(up_to_chapter=5)
        self.assertEqual(text, "")

    def test_returns_nonempty_for_valid_range(self):
        r = self._retriever()
        text = r.get_prior_summaries(up_to_chapter=3)
        self.assertGreater(len(text), 0)

    def test_does_not_use_merged_json(self):
        """确认即使 _merged_data.json 不存在也能正常工作。"""
        r = self._retriever()
        merged = self.bible_dir / "_merged_data.json"
        merged.write_text("{}", encoding="utf-8")  # 存在但为空
        text = r.get_prior_summaries(up_to_chapter=3)
        self.assertIn("主角登场", text)


# ────────────────────────────────────────────────────────────────────────────
# 单元/集成：story_bible 生成脚本必须写完整 temporal frontmatter
# ────────────────────────────────────────────────────────────────────────────

class TestStoryBibleGenerators(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))

    def test_build_story_bible_static_files_get_full_frontmatter(self):
        import build_story_bible as bsb

        samples = {
            "world": bsb.generate_world_md({}),
            "style": bsb.generate_style_md({}),
            "glossary": bsb.generate_glossary_md([]),
        }
        for name, content in samples.items():
            with self.subTest(name=name):
                self.assertTrue(content.startswith("---\n"))
                self.assertIn(f"title: {name}", content)
                self.assertIn("revealed_in: 1", content)
                self.assertIn("valid_from: 1", content)
                self.assertIn("valid_to: null", content)

    def test_split_characters_writes_visible_full_frontmatter(self):
        import split_characters as sc

        tmp = tempfile.TemporaryDirectory()
        try:
            bible_dir = Path(tmp.name) / "story_bible"
            raw_dir = Path(tmp.name) / "raw"
            raw_dir.mkdir(parents=True)
            src = bible_dir / "characters.md"
            out_dir = bible_dir / "generated" / "characters"
            src.parent.mkdir(parents=True)
            src.write_text(
                "# characters\n\n"
                "## 人物设定\n\n"
                "### 林清雪（女神医）\n\n"
                "- **别名/称呼**: 女神医、林神医\n"
                "- **来源章节**: 第3章、第7章\n"
                "- **身份**: 林家传人\n",
                encoding="utf-8",
            )

            names = sc.split(src, out_dir)
            self.assertEqual(names, ["林清雪"])
            content = (out_dir / "林清雪.md").read_text(encoding="utf-8")
            self.assertIn("title: 林清雪", content)
            self.assertIn("type: character", content)
            self.assertIn("aliases:", content)
            self.assertIn("revealed_in: 3", content)
            self.assertIn("valid_from: 3", content)
            self.assertIn("valid_to: null", content)
            self.assertIn("source_chapters:", content)
            self.assertIn("  - 7", content)

            r = _make_retriever(bible_dir, raw_dir)
            visible = r.search_bible("", entities=["林清雪"], max_chapter=3)
            hidden = r.search_bible("", entities=["林清雪"], max_chapter=2)
            self.assertIn("林清雪", {h["source"] for h in visible})
            self.assertNotIn("林清雪", {h["source"] for h in hidden})
        finally:
            tmp.cleanup()


# ────────────────────────────────────────────────────────────────────────────
# 单元：add_frontmatter.py 辅助函数（更新后接口）
# ────────────────────────────────────────────────────────────────────────────

class TestAddFrontmatterHelpers(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        import add_frontmatter as af
        self.af = af

    def test_already_has_frontmatter(self):
        text = "---\nrevealed_in: 1\nvalid_from: 1\nvalid_to: null\n---\n\nBody"
        self.assertTrue(self.af.already_has_frontmatter(text))

    def test_no_frontmatter(self):
        text = "# Title\nBody"
        self.assertFalse(self.af.already_has_frontmatter(text))

    def test_extract_min_chapter_arabic(self):
        text = "- **来源章节**: 第3章、第7章"
        self.assertEqual(self.af.extract_min_chapter(text), 3)

    def test_extract_min_chapter_chinese(self):
        text = "- **来源章节**: 第一章"
        self.assertEqual(self.af.extract_min_chapter(text), 1)

    def test_extract_min_chapter_none(self):
        text = "没有来源章节信息"
        self.assertIsNone(self.af.extract_min_chapter(text))

    def test_extract_chapters_multiple(self):
        text = "- **来源章节**: 第3章、第7章、第12章"
        chs = self.af.extract_chapters(text)
        self.assertEqual(chs, [3, 7, 12])

    def test_make_frontmatter_basic(self):
        data = {"revealed_in": 3, "valid_from": 3, "valid_to": None}
        fm = self.af.make_frontmatter(data)
        self.assertTrue(fm.startswith("---\n"))
        self.assertIn("revealed_in: 3", fm)
        self.assertIn("valid_from: 3", fm)
        self.assertIn("valid_to: null", fm)

    def test_make_frontmatter_with_aliases(self):
        data = {
            "title": "林清雪",
            "type": "character",
            "aliases": ["女神医", "林神医"],
            "revealed_in": 1,
            "valid_from": 1,
            "valid_to": None,
            "source_chapters": [1],
        }
        fm = self.af.make_frontmatter(data)
        self.assertIn("title: 林清雪", fm)
        self.assertIn("type: character", fm)
        self.assertIn("  - 女神医", fm)
        self.assertIn("  - 1", fm)

    def test_extract_aliases(self):
        text = "- **别名/称呼**: 女神医、林神医、姐姐"
        aliases = self.af._extract_aliases(text)
        self.assertIn("女神医", aliases)
        self.assertIn("林神医", aliases)
        self.assertIn("姐姐", aliases)

    def test_extract_handwritten_type_character(self):
        text = "# 小女童\n\n## 类型\n人物\n\n## 内容\n..."
        t = self.af._extract_handwritten_type(text)
        self.assertEqual(t, "character")

    def test_extract_handwritten_type_location(self):
        text = "# 帝湖\n\n## 类型\n地点\n\n## 内容\n..."
        t = self.af._extract_handwritten_type(text)
        self.assertEqual(t, "location")


# ────────────────────────────────────────────────────────────────────────────
# 集成：add_frontmatter.py process_bible_dir（临时目录）
# ────────────────────────────────────────────────────────────────────────────

class TestAddFrontmatterScript(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        import add_frontmatter as af
        self.af = af
        self._tmp = tempfile.TemporaryDirectory()
        self.bible_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dry_run_no_writes(self):
        (self.bible_dir / "world.md").write_text("# world\n内容", encoding="utf-8")
        self.af.process_bible_dir(self.bible_dir, dry_run=True)
        content = (self.bible_dir / "world.md").read_text(encoding="utf-8")
        self.assertFalse(self.af.already_has_frontmatter(content))

    def test_world_gets_revealed_in_and_valid_from_1(self):
        (self.bible_dir / "world.md").write_text("# world\n内容", encoding="utf-8")
        self.af.process_bible_dir(self.bible_dir, dry_run=False)
        content = (self.bible_dir / "world.md").read_text(encoding="utf-8")
        self.assertTrue(self.af.already_has_frontmatter(content))
        self.assertIn("revealed_in: 1", content)
        self.assertIn("valid_from: 1", content)
        self.assertIn("valid_to: null", content)

    def test_aggregate_files_skipped(self):
        for name in ["characters.md", "relationships.md", "chapter_summaries.md"]:
            (self.bible_dir / name).write_text(f"# {name}\n内容", encoding="utf-8")
        self.af.process_bible_dir(self.bible_dir, dry_run=False)
        for name in ["characters.md", "relationships.md", "chapter_summaries.md"]:
            content = (self.bible_dir / name).read_text(encoding="utf-8")
            self.assertFalse(self.af.already_has_frontmatter(content),
                             f"{name} should not get frontmatter")

    def test_character_file_gets_chapter_number_and_aliases(self):
        gen = self.bible_dir / "generated" / "characters"
        gen.mkdir(parents=True)
        (gen / "林清雪.md").write_text(
            "### 林清雪\n- **别名/称呼**: 女神医、林神医\n- **来源章节**: 第3章",
            encoding="utf-8"
        )
        self.af.process_bible_dir(self.bible_dir, dry_run=False)
        content = (gen / "林清雪.md").read_text(encoding="utf-8")
        self.assertIn("revealed_in: 3", content)
        self.assertIn("valid_from: 3", content)
        self.assertIn("valid_to: null", content)
        self.assertIn("女神医", content)

    def test_idempotent(self):
        (self.bible_dir / "world.md").write_text("# world\n内容", encoding="utf-8")
        self.af.process_bible_dir(self.bible_dir, dry_run=False)
        self.af.process_bible_dir(self.bible_dir, dry_run=False)
        content = (self.bible_dir / "world.md").read_text(encoding="utf-8")
        # frontmatter 只有一个 block：开头 --- 和结尾 --- 各一个
        self.assertEqual(content.count("---\n"), 2)


# ────────────────────────────────────────────────────────────────────────────
# 集成：第8章 prompt 口径验证（端到端 N-1 确认）
# ────────────────────────────────────────────────────────────────────────────

class TestChapter8Wiring(unittest.TestCase):
    """验证 target_chapter=8 时，max_chapter=7，第8章内容不泄漏。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bible_dir = Path(self._tmp.name) / "story_bible"
        self.bible_dir.mkdir(parents=True)
        self.raw_dir = Path(self._tmp.name) / "raw"
        self.raw_dir.mkdir(parents=True)

        # 第1章角色（应可见）
        _write_md(self.bible_dir / "角色甲.md",
                  "### 角色甲\n第一章主角。", revealed_in=1)
        # 第8章角色（应不可见）
        _write_md(self.bible_dir / "角色乙.md",
                  "### 角色乙\n第八章新登场。", revealed_in=8)
        # 章节摘要文件
        (self.bible_dir / "chapter_summaries.md").write_text(
            "## 第七章 前章\n\n"
            "- **章节编号**: 7\n"
            "- **本章概要**: 第七章事件发生\n"
            "- **出场人物**: 角色甲\n\n"
            "---\n\n"
            "## 第八章 本章\n\n"
            "- **章节编号**: 8\n"
            "- **本章概要**: 第八章新事件\n"
            "- **出场人物**: 角色乙\n\n",
            encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_max_chapter_is_7_for_target_8(self):
        from cowriter.chapter import max_chapter_for_target
        self.assertEqual(max_chapter_for_target(8), 7)

    def test_search_bible_ch7_sees_甲_not_乙(self):
        r = _make_retriever(self.bible_dir, self.raw_dir)
        hits_甲 = r.search_bible("", entities=["角色甲"], max_chapter=7)
        hits_乙 = r.search_bible("", entities=["角色乙"], max_chapter=7)
        self.assertIn("角色甲", {h["source"] for h in hits_甲})
        self.assertNotIn("角色乙", {h["source"] for h in hits_乙})

    def test_prior_summaries_ch7_no_ch8(self):
        r = _make_retriever(self.bible_dir, self.raw_dir)
        text = r.get_prior_summaries(up_to_chapter=7)
        self.assertIn("第七章事件发生", text)
        self.assertNotIn("第八章新事件", text)

    def test_prior_summaries_nonempty(self):
        r = _make_retriever(self.bible_dir, self.raw_dir)
        text = r.get_prior_summaries(up_to_chapter=7)
        self.assertGreater(len(text.strip()), 0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
