import json
import re
from datetime import datetime
from pathlib import Path

import ollama


def _dedup_output(text: str) -> str:
    """
    两层重复保险丝：
    段落层：同一段落（换行切分，≥50字）出现 ≥2 次 → 截断（防大段复读）。
    句段层：同一句段（句末标点切分，≥8字）出现 ≥3 次 → 截断（防短句循环）。
    50字阈值：排除排比句/短对话误伤；8字阈值：排除极短叹词。
    """
    # ── 段落层（大段复读，≥50字出现2次即截断）──────────────────────────────
    lines = text.splitlines(keepends=True)
    para_seen: dict[str, int] = {}
    para_out: list[str] = []
    for line in lines:
        key = line.strip()
        if len(key) >= 50:
            para_seen[key] = para_seen.get(key, 0) + 1
            if para_seen[key] >= 2:
                break
        para_out.append(line)
    text = ''.join(para_out).strip()
    # ── 句段层（短句循环，≥8字出现3次即截断）────────────────────────────────
    parts = re.split(r'(?<=[。！？\n])', text)
    sent_seen: dict[str, int] = {}
    sent_out: list[str] = []
    for part in parts:
        key = part.strip()
        if len(key) >= 8:
            sent_seen[key] = sent_seen.get(key, 0) + 1
            if sent_seen[key] >= 3:
                break
        sent_out.append(part)
    return ''.join(sent_out).strip()


def _strip_think(text: str) -> str:
    """剥 think 块（完整/未闭合/孤立闭标签）；剥 /no_think 及残片（任意行首）；剥行首助手语。"""
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<think>[\s\S]*', '', text)
    text = re.sub(r'</think>\s*', '', text)
    text = re.sub(r'\s*/no_think\s*', '', text)          # 完整 /no_think
    text = re.sub(r'^[ \t]*/no(?:_think|_thin|_thi|_th|_t|_)?\b[ \t]*\n?', '', text, flags=re.MULTILINE)  # /no_think 截断残片（任意行首）
    text = text.lstrip()                                 # 只清前导空白，让助手语锚定到首字
    text = re.sub(r'^好的[。，：]?\s*\n', '', text)       # 行首助手语：独占一行才删，不误伤对话句
    return text.strip()

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
        self._debug_dir = self.output_dir / "debug"
        self._debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Debug dump ────────────────────────────────────────────────────────────

    def _dump_debug(self, messages: list[dict], options: dict, thinking: str, content: str):
        ts = datetime.now().isoformat(timespec="seconds")
        req = {
            "timestamp": ts,
            "endpoint": "/api/chat (ollama.chat SDK)",
            "model": self.model,
            "messages": messages,
            "options": options,
        }
        (self._debug_dir / "last_request.json").write_text(
            json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        prompt_text = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        (self._debug_dir / "last_prompt.txt").write_text(prompt_text, encoding="utf-8")
        sep = "\n" + "=" * 60 + "\n"
        (self._debug_dir / "last_response.txt").write_text(
            f"[THINKING]\n{thinking}" + sep + f"[CONTENT]\n{content}",
            encoding="utf-8",
        )

    # ── Ollama 调用 ──────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        g = self.cfg["generation"]
        options: dict = {
            "temperature": g["temperature"],
            "top_p": g["top_p"],
            "repeat_penalty": g["repeat_penalty"],
            "num_predict": max_tokens or self.cfg["session"]["output_tokens"],
        }
        for key in ("top_k", "repeat_last_n", "presence_penalty", "frequency_penalty",
                    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n"):
            if key in g:
                options[key] = g[key]
        resp = ollama.chat(model=self.model, messages=messages, options=options, think=False)
        msg = resp["message"]
        thinking = msg.get("thinking") or ""
        content  = _strip_think(msg.get("content") or "")
        content  = _dedup_output(content)
        self._dump_debug(messages, options, thinking, content)
        return content.strip()

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
