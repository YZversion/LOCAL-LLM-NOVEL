from datetime import datetime
from pathlib import Path

import ollama

from cowriter.prompts import build_prompt, build_summary_prompt
from cowriter.retriever import Retriever


class Session:
    def __init__(self, config: dict):
        self.cfg = config
        self.retriever = Retriever(config)
        self.model: str = config["model"]["ollama_model"]
        self.summary: str = ""
        self.accepted_text: str = ""  # 本次会话已确认的全部正文
        self.output_dir = Path(config["paths"]["outputs"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Ollama 调用 ──────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        g = self.cfg["generation"]
        options = {
            "temperature": g["temperature"],
            "top_p": g["top_p"],
            "repeat_penalty": g["repeat_penalty"],
            "num_predict": max_tokens or self.cfg["session"]["output_tokens"],
        }
        resp = ollama.chat(model=self.model, messages=messages, options=options)
        return resp["message"]["content"].strip()

    # ── 滚动摘要 ─────────────────────────────────────────────────────────────

    def _maybe_compress(self):
        trigger = self.cfg["session"]["summary_trigger_chars"]
        keep = self.cfg["session"]["max_recent_chars"]
        if len(self.accepted_text) <= trigger:
            return

        old_text = self.accepted_text[:-keep]
        self.accepted_text = self.accepted_text[-keep:]

        try:
            new_chunk_summary = self._chat(
                build_summary_prompt(old_text), max_tokens=200
            )
        except Exception:
            new_chunk_summary = ""
        # 小模型常返回空或乱码，兜底用原文前 200 字保留真实剧情上下文
        if not new_chunk_summary or len(new_chunk_summary) < 20:
            new_chunk_summary = f"[前情节选] {old_text[:200].strip()}"
        self.summary = (
            (self.summary + "\n" + new_chunk_summary).strip()
            if self.summary
            else new_chunk_summary
        )

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def seed(self, initial_text: str):
        """提供初始上文，不生成，只入库。"""
        self.accepted_text = initial_text
        self._maybe_compress()

    def generate(self, instruction: str = "", target_chars: int | None = None) -> str:
        chars = target_chars or self.cfg["session"]["output_tokens"]
        context_window = self.accepted_text[-self.cfg["session"]["max_recent_chars"]:]
        retrieval = self.retriever.retrieve(context_window)
        messages = build_prompt(
            recent_text=context_window,
            summary=self.summary,
            retrieval=retrieval,
            instruction=instruction,
            target_chars=chars,
        )
        return self._chat(messages)

    def accept(self, text: str):
        """将文本（模型续写或用户自写）纳入会话，触发摘要压缩。"""
        self.accepted_text += text
        self._maybe_compress()

    def save_draft(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"draft_{ts}.txt"
        path.write_text(self.accepted_text, encoding="utf-8")
        return path
