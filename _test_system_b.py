#!/usr/bin/env python3
"""System B MVP regression test.

Verifies: reviewed facts -> kg.json -> Markdown projection -> Retriever BM25
with temporal visibility.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_config(bible_dir: Path, raw_dir: Path) -> dict:
    return {
        "model": {"ollama_model": "dummy"},
        "paths": {"story_bible": str(bible_dir), "raw_data": str(raw_dir)},
        "retrieval": {"bible_top_k": 5, "top_k": 3, "grep_context_lines": 2},
        "session": {"output_tokens": 500, "max_recent_chars": 1000,
                    "summary_trigger_chars": 3000},
        "generation": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
    }


class TestSystemBMVP(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bible_dir = self.root / "story_bible"
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True)
        self.kg_path = self.bible_dir / "kg.json"
        self.cards_dir = self.bible_dir / "generated" / "system_b"
        self.facts_path = self.root / "facts.json"
        self.facts_path.write_text(json.dumps({
            "facts": [
                {
                    "type": "event",
                    "title": "帝湖托付",
                    "summary": "林清雪在帝湖扁舟上向颜儿托付心事，水珠滑入衣袖，如泪一般。",
                    "chapter": 59,
                    "entities": ["林清雪", "颜儿", "帝湖"],
                    "valid_from": 59,
                    "valid_to": None,
                    "confidence": 0.8,
                    "evidence": [{"chapter": 59, "quote": "颜儿，你跟在我身边也三年了"}],
                },
                {
                    "type": "plot_thread",
                    "title": "林清雪的怀念",
                    "summary": "林清雪说自己医绝天下，却找不出一时片刻不觉得怀念。",
                    "chapter": 59,
                    "entities": ["林清雪"],
                    "valid_from": 59,
                    "valid_to": None,
                    "confidence": 0.7,
                },
            ]
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_update_render_and_retrieve(self):
        from cowriter.retriever import Retriever
        from scripts.kg_render import render_kg
        from scripts.kg_update import load_facts, load_kg, merge_facts, write_kg

        kg = load_kg(self.kg_path)
        facts = load_facts(self.facts_path)
        kg, stats = merge_facts(kg, facts)
        self.assertEqual(stats["inserted"], 2)
        write_kg(self.kg_path, kg)

        render_stats = render_kg(self.kg_path, self.cards_dir, prune=True)
        self.assertEqual(render_stats["facts"], 2)
        self.assertEqual(render_stats["entities"], 3)
        self.assertTrue((self.cards_dir / "entities" / "林清雪.md").exists())
        self.assertTrue((self.cards_dir / "entities" / "颜儿.md").exists())

        retriever = Retriever(_make_config(self.bible_dir, self.raw_dir))
        visible = retriever.retrieve("林清雪在帝湖托付颜儿", max_chapter=59)
        visible_sources = {hit["source"] for hit in visible["bible"]}
        self.assertIn("林清雪", visible_sources)
        self.assertIn("颜儿", visible_sources)

        hidden = retriever.retrieve("林清雪在帝湖托付颜儿", max_chapter=58)
        hidden_sources = {hit["source"] for hit in hidden["bible"]}
        self.assertNotIn("林清雪", hidden_sources)
        self.assertNotIn("颜儿", hidden_sources)


if __name__ == "__main__":
    unittest.main(verbosity=2)
