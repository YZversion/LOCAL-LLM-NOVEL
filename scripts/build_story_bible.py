#!/usr/bin/env python3
"""
Build story_bible/*.md from raw novel txt files.

The script is intentionally self-contained so it can be run before the normal
cowriter app starts. It reuses the project's existing Ollama configuration when
available, chunks long txt files, caches each chunk extraction, and writes
Markdown files that the existing Retriever can index without any changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on local env
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROMPT_VERSION = "story-bible-extract-v1"
OUTPUT_FILES = [
    "characters",
    "world",
    "timeline",
    "plot_threads",
    "chapter_summaries",
    "relationships",
    "style",
    "glossary",
]

MISSING_VALUES = {
    "",
    "无",
    "暂无",
    "未知",
    "不详",
    "未提及",
    "未明确",
    "不明确",
    "none",
    "null",
    "n/a",
    "[待抽取]",
}

CHAPTER_RE = re.compile(
    r"(?im)^[ \t]*(?:"
    r"第[ \t]*([零〇一二三四五六七八九十百千万两0-9]{1,8})[ \t]*(?:章|章节|节|卷|回)"
    r"|Chapter[ \t]+([0-9]{1,8})"
    r")[^\r\n]{0,80}$"
)

_ACTIVE_CONFIG: dict[str, Any] = {}


class BuildError(RuntimeError):
    """User-facing build error."""


@dataclass
class Chunk:
    index: int
    file_name: str
    file_path: str
    chapter_number: int | None
    chapter_title: str
    chunk_number: int
    part_number: int | None
    start_char: int
    end_char: int
    text: str

    @property
    def label(self) -> str:
        part = f" part {self.part_number}" if self.part_number else ""
        return f"{self.chapter_title}{part} [{self.file_name}:{self.start_char}-{self.end_char}]"


def log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise BuildError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise BuildError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_config_path(value: str | None, config_path: Path, *, prefer_existing: bool) -> Path:
    if not value:
        raise BuildError("Missing required path in config.")
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw

    cwd = Path.cwd()
    candidates = unique_paths(
        [
            config_path.parent / raw,
            config_path.parent.parent / raw,
            cwd / raw,
        ]
    )
    if prefer_existing:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return candidates[0]


def find_txt_files(raw_data_dir: Path) -> list[Path]:
    if not raw_data_dir.exists():
        return []
    return sorted(p for p in raw_data_dir.rglob("*.txt") if p.is_file())


def compatible_raw_dirs(config_path: Path) -> list[Path]:
    cwd = Path.cwd()
    return unique_paths(
        [
            cwd / "data" / "raw_data",
            cwd / "raw_data",
            config_path.parent / "raw_data",
            config_path.parent.parent / "raw_data",
        ]
    )


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise BuildError(f"Cannot decode txt file as UTF-8/UTF-8-SIG/GB18030: {path}")


CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def chinese_number_to_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    total = 0
    section = 0
    number = 0
    used = False
    for char in text:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
            used = True
        elif char in CN_UNITS:
            unit = CN_UNITS[char]
            used = True
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number if used else None


def extract_chapter_number(title: str) -> int | None:
    chapter_match = re.search(r"第\s*([零〇一二三四五六七八九十百千万两0-9]{1,8})\s*(?:章|章节|节|卷|回)", title)
    if chapter_match:
        return chinese_number_to_int(chapter_match.group(1))
    english_match = re.search(r"(?i)Chapter\s+([0-9]{1,8})", title)
    if english_match:
        return int(english_match.group(1))
    return None


def split_segment_by_chars(
    *,
    file_path: Path,
    file_text: str,
    title: str,
    chapter_number: int | None,
    segment_start: int,
    segment_end: int,
    chunk_size: int,
    overlap: int,
    next_chunk_number: int,
) -> tuple[list[Chunk], int]:
    segment = file_text[segment_start:segment_end]
    chunks: list[Chunk] = []
    local_start = 0
    part = 1
    step = max(1, chunk_size - overlap)

    while local_start < len(segment):
        local_end = min(len(segment), local_start + chunk_size)
        text = segment[local_start:local_end]
        if text.strip():
            chunks.append(
                Chunk(
                    index=0,
                    file_name=file_path.name,
                    file_path=str(file_path),
                    chapter_number=chapter_number,
                    chapter_title=title,
                    chunk_number=next_chunk_number,
                    part_number=part if len(segment) > chunk_size else None,
                    start_char=segment_start + local_start,
                    end_char=segment_start + local_end,
                    text=text.strip(),
                )
            )
            next_chunk_number += 1
            part += 1
        if local_end >= len(segment):
            break
        local_start += step
    return chunks, next_chunk_number


def split_file_into_chunks(file_path: Path, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    matches = list(CHAPTER_RE.finditer(text))
    chunks: list[Chunk] = []
    next_chunk_number = 1

    if matches:
        if text[: matches[0].start()].strip():
            preface, next_chunk_number = split_segment_by_chars(
                file_path=file_path,
                file_text=text,
                title="前言/开篇",
                chapter_number=None,
                segment_start=0,
                segment_end=matches[0].start(),
                chunk_size=chunk_size,
                overlap=overlap,
                next_chunk_number=next_chunk_number,
            )
            chunks.extend(preface)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = match.group(0).strip()
            chapter_number = extract_chapter_number(title)
            chapter_chunks, next_chunk_number = split_segment_by_chars(
                file_path=file_path,
                file_text=text,
                title=title,
                chapter_number=chapter_number,
                segment_start=start,
                segment_end=end,
                chunk_size=chunk_size,
                overlap=overlap,
                next_chunk_number=next_chunk_number,
            )
            chunks.extend(chapter_chunks)
    else:
        chunks, _ = split_segment_by_chars(
            file_path=file_path,
            file_text=text,
            title="按字数切分",
            chapter_number=None,
            segment_start=0,
            segment_end=len(text),
            chunk_size=chunk_size,
            overlap=overlap,
            next_chunk_number=1,
        )

    return chunks


def assign_global_indexes(chunks: list[Chunk]) -> list[Chunk]:
    for index, chunk in enumerate(chunks, 1):
        chunk.index = index
    return chunks


def chunk_hash(chunk: Chunk) -> str:
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "file": chunk.file_path,
            "start": chunk.start_char,
            "end": chunk.end_char,
            "title": chunk.chapter_title,
            "text": chunk.text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_cache_path(cache_dir: Path, chunk: Chunk) -> Path:
    return cache_dir / "chunks" / f"{chunk.index:05d}_{chunk_hash(chunk)[:16]}.json"


def empty_extraction() -> dict[str, Any]:
    return {
        "characters": [],
        "world": {"locations": [], "organizations": [], "rules": []},
        "timeline": [],
        "plot_threads": {
            "main_thread": [],
            "sub_threads": [],
            "foreshadowing_planted": [],
            "foreshadowing_resolved": [],
            "unresolved_conflicts": [],
            "character_motives": [],
            "consistency_points": [],
        },
        "relationships": [],
        "glossary": [],
        "style": {
            "narrative_perspective": "未明确",
            "sentence_length": "未明确",
            "description_preference": "未明确",
            "dialogue_style": "未明确",
            "pacing": "未明确",
            "common_imagery": [],
            "emotional_tone": "未明确",
            "protected_features": [],
            "sample_sentences": [],
        },
        "chapter_summary": {},
    }


def build_extraction_prompt(chunk: Chunk) -> str:
    return f"""你是小说续写项目的设定整理助手。请只根据给定原文片段抽取信息，禁止编造。

硬性要求：
1. 只输出一个合法 JSON 对象，不要 Markdown，不要解释。
2. 所有事实必须来自【原文片段】。不能确定的内容写“未明确”；基于上下文判断但没有直说的内容写“推测：...”。
3. evidence / quote / sample_sentences 只能使用原文片段里的短句或关键词，不要改写成长段。
4. 如果某类信息没有出现，返回空数组 [] 或“未明确”。
5. source 使用章节标题或 chunk 元数据，便于后续检索。

请严格按这个 JSON 结构输出：
{{
  "characters": [
    {{
      "name": "姓名",
      "aliases": ["别名/称呼"],
      "identity": "身份",
      "appearance": "外貌",
      "personality": "性格",
      "abilities": "能力/修为/技能",
      "relationships": "与其他人物关系",
      "experiences": "已知经历",
      "current_status": "当前状态",
      "first_appearance": "首次出现位置",
      "source_chapters": ["来源章节"],
      "evidence": ["简短原文证据"],
      "uncertainties": ["未明确或推测信息"]
    }}
  ],
  "world": {{
    "locations": [
      {{"name": "地点名", "aliases": [], "description": "描述", "region": "所在区域", "importance": "重要性", "source": "来源章节", "evidence": []}}
    ],
    "organizations": [
      {{"name": "组织/势力名", "aliases": [], "nature": "性质", "scope": "范围", "key_characters": [], "source": "来源章节", "evidence": []}}
    ],
    "rules": [
      {{"name": "规则/体系名", "category": "类别", "description": "描述", "source": "来源章节", "evidence": []}}
    ]
  }},
  "timeline": [
    {{"time_point": "时间点", "event": "事件", "characters": [], "location": "地点", "impact": "后续影响", "source": "来源章节", "evidence": []}}
  ],
  "plot_threads": {{
    "main_thread": ["主线目标"],
    "sub_threads": [{{"name": "支线", "description": "描述", "status": "状态", "source": "来源章节", "evidence": []}}],
    "foreshadowing_planted": [{{"hint": "伏笔", "expected_resolution": "未明确或推测", "source": "来源章节", "evidence": []}}],
    "foreshadowing_resolved": [{{"hint": "原伏笔", "resolution": "回收方式", "source": "来源章节", "evidence": []}}],
    "unresolved_conflicts": [{{"conflict": "冲突", "parties": [], "source": "来源章节", "evidence": []}}],
    "character_motives": [{{"character": "人物", "motive": "动机", "source": "来源章节", "evidence": []}}],
    "consistency_points": ["续写时必须保持一致的点"]
  }},
  "relationships": [
    {{"character_a": "人物A", "character_b": "人物B", "relationship_type": "关系类型", "details": "详情", "dynamics": "动态", "source": "来源章节", "evidence": []}}
  ],
  "glossary": [
    {{"term": "名词", "category": "类别", "definition": "定义", "aliases": [], "related_terms": [], "source": "来源章节", "evidence": []}}
  ],
  "style": {{
    "narrative_perspective": "叙事视角",
    "sentence_length": "句子长度",
    "description_preference": "描写偏好",
    "dialogue_style": "对话风格",
    "pacing": "节奏",
    "common_imagery": ["常见意象"],
    "emotional_tone": "情绪基调",
    "protected_features": ["禁止破坏的风格特征"],
    "sample_sentences": ["原文典型短句"]
  }},
  "chapter_summary": {{
    "chapter_title": "章节名或 chunk 编号",
    "chapter_number": "章节编号或未明确",
    "what_happened": "本章发生了什么",
    "characters_present": [],
    "new_settings": [],
    "plot_progression": "情节推进",
    "foreshadowing": [],
    "mood_atmosphere": "情绪/氛围",
    "key_keywords": []
  }}
}}

chunk 元数据：
- 文件名：{chunk.file_name}
- 章节标题：{chunk.chapter_title}
- 章节编号：{chunk.chapter_number if chunk.chapter_number is not None else "未明确"}
- chunk 编号：{chunk.chunk_number}
- 字符范围：{chunk.start_char}-{chunk.end_char}

【原文片段开始】
{chunk.text}
【原文片段结束】
"""


def call_llm(prompt: str) -> str:
    """
    Call the configured LLM and return raw text.

    This project already uses Ollama in cowriter.session.Session._chat, so this
    function reuses the same config keys: model.provider, model.ollama_model,
    model.ollama_base_url, and generation options. If you later switch providers,
    replace or extend the blocks below; the rest of the build pipeline expects
    only this simple call_llm(prompt: str) -> str contract.
    """
    cfg = _ACTIVE_CONFIG
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm", {}), dict) else {}
    mode = (llm_cfg.get("mode") or model_cfg.get("provider") or "placeholder").lower()

    if mode in {"mock", "placeholder", "none"}:
        return json.dumps(empty_extraction(), ensure_ascii=False)

    if mode == "ollama":
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise BuildError("Missing dependency: ollama. Install requirements.txt first.") from exc

        model = model_cfg.get("ollama_model") or llm_cfg.get("model_name")
        if not model:
            raise BuildError("Ollama model is not configured. Set model.ollama_model in config.yaml.")

        base_url = model_cfg.get("ollama_base_url") or llm_cfg.get("ollama_base_url")
        client = ollama.Client(host=base_url) if base_url else ollama
        generation = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), dict) else {}
        options = {
            "temperature": llm_cfg.get("temperature", 0.2),
            "top_p": generation.get("top_p", 0.9),
            "repeat_penalty": generation.get("repeat_penalty", 1.05),
            "num_predict": llm_cfg.get("max_tokens", 4096),
        }
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你只输出合法 JSON。所有内容必须来自用户提供的原文片段，禁止编造。",
                },
                {"role": "user", "content": prompt},
            ],
            options=options,
        )
        return response["message"]["content"].strip()

    if mode in {"api", "openai", "local"}:
        raise BuildError(
            f"LLM mode `{mode}` is not wired yet. Connect it inside call_llm(prompt: str)."
        )

    raise BuildError(f"Unknown LLM mode: {mode}")


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


def parse_json_response(raw: str) -> dict[str, Any]:
    candidates: list[str] = []
    stripped = strip_code_fence(raw)
    candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    raise BuildError(f"LLM did not return valid JSON: {last_error}")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    if isinstance(value, list):
        return all(is_missing(v) for v in value)
    if isinstance(value, dict):
        return not value
    return False


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        result: list[Any] = []
        for item in value:
            result.extend(as_list(item) if isinstance(item, list) else [item])
        return result
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        pieces = []
        for key, val in value.items():
            if not is_missing(val):
                pieces.append(f"{key}: {stringify(val)}")
        return "；".join(pieces)
    if isinstance(value, list):
        return "、".join(stringify(v) for v in value if not is_missing(v))
    return str(value).strip()


def unique_strings(values: list[Any], *, limit: int | None = None, clip: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if is_missing(value):
            continue
        text = stringify(value)
        if not text or is_missing(text):
            continue
        if clip and len(text) > clip:
            text = text[:clip].rstrip() + "..."
        if text not in seen:
            seen.add(text)
            result.append(text)
            if limit and len(result) >= limit:
                break
    return result


def source_label(chunk: Chunk) -> str:
    return f"{chunk.chapter_title} | {chunk.file_name}:{chunk.start_char}-{chunk.end_char}"


def ensure_item_source(item: dict[str, Any], chunk: Chunk, *, character: bool = False) -> dict[str, Any]:
    label = source_label(chunk)
    if character:
        if is_missing(item.get("source_chapters")):
            item["source_chapters"] = [label]
    elif is_missing(item.get("source")):
        item["source"] = label
    return item


def normalize_plot_threads(value: Any, chunk: Chunk) -> dict[str, Any]:
    source = source_label(chunk)
    plot = value if isinstance(value, dict) else {}
    result = empty_extraction()["plot_threads"]
    result["main_thread"] = unique_strings(as_list(plot.get("main_thread")))
    result["consistency_points"] = unique_strings(as_list(plot.get("consistency_points")))

    for key in [
        "sub_threads",
        "foreshadowing_planted",
        "foreshadowing_resolved",
        "unresolved_conflicts",
        "character_motives",
    ]:
        items = []
        for raw_item in as_list(plot.get(key)):
            if isinstance(raw_item, dict):
                if is_missing(raw_item.get("source")):
                    raw_item["source"] = source
                items.append(raw_item)
            elif not is_missing(raw_item):
                items.append({"description": stringify(raw_item), "source": source})
        result[key] = items
    return result


def normalize_world(value: Any, chunk: Chunk) -> dict[str, list[dict[str, Any]]]:
    world = value if isinstance(value, dict) else {}
    result = {"locations": [], "organizations": [], "rules": []}
    for key in result:
        for item in as_list(world.get(key)):
            if isinstance(item, dict):
                result[key].append(ensure_item_source(item, chunk))
    return result


def normalize_timeline(data: dict[str, Any], chunk: Chunk) -> list[dict[str, Any]]:
    raw = data.get("timeline")
    if isinstance(raw, dict):
        raw_events = raw.get("events")
    else:
        raw_events = raw
    if raw_events is None:
        raw_events = data.get("events")
    events: list[dict[str, Any]] = []
    for item in as_list(raw_events):
        if isinstance(item, dict):
            events.append(ensure_item_source(item, chunk))
    return events


def normalize_chapter_summary(value: Any, chunk: Chunk) -> dict[str, Any]:
    summary = value.copy() if isinstance(value, dict) else {}
    if is_missing(summary.get("chapter_title")):
        summary["chapter_title"] = chunk.chapter_title
    if is_missing(summary.get("chapter_number")):
        summary["chapter_number"] = chunk.chapter_number if chunk.chapter_number is not None else chunk.chunk_number
    if is_missing(summary.get("source")):
        summary["source"] = source_label(chunk)
    summary["_order"] = chunk.index
    summary["_file_name"] = chunk.file_name
    return summary


def normalize_extraction(data: dict[str, Any], chunk: Chunk) -> dict[str, Any]:
    result = empty_extraction()

    for item in as_list(data.get("characters")):
        if isinstance(item, dict):
            result["characters"].append(ensure_item_source(item, chunk, character=True))

    result["world"] = normalize_world(data.get("world"), chunk)
    result["timeline"] = normalize_timeline(data, chunk)
    result["plot_threads"] = normalize_plot_threads(data.get("plot_threads"), chunk)

    for item in as_list(data.get("relationships")):
        if isinstance(item, dict):
            result["relationships"].append(ensure_item_source(item, chunk))

    glossary = data.get("glossary")
    if isinstance(glossary, dict):
        glossary = glossary.get("terms")
    for item in as_list(glossary):
        if isinstance(item, dict):
            result["glossary"].append(ensure_item_source(item, chunk))

    style = data.get("style")
    if isinstance(style, dict):
        result["style"].update(style)
        result["style"]["source"] = source_label(chunk)

    result["chapter_summary"] = normalize_chapter_summary(data.get("chapter_summary"), chunk)
    return result


def extract_chunk(
    chunk: Chunk,
    cfg: dict[str, Any],
    cache_dir: Path,
    *,
    force: bool,
    verbose: bool,
) -> dict[str, Any]:
    cache_path = chunk_cache_path(cache_dir, chunk)
    if cache_path.exists() and not force:
        log(f"[cache] {chunk.index:05d} {chunk.label}", verbose=verbose)
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached["result"]

    log(f"[llm] {chunk.index:05d} {chunk.label}", verbose=True)
    prompt = build_extraction_prompt(chunk)
    raw = call_llm(prompt)

    debug_dir = cache_dir / "debug"
    try:
        parsed = parse_json_response(raw)
        result = normalize_extraction(parsed, chunk)
    except BuildError as exc:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{chunk.index:05d}_{chunk_hash(chunk)[:16]}_raw.txt"
        debug_path.write_text(raw, encoding="utf-8")
        log(f"[warn] JSON parse failed for chunk {chunk.index}; raw saved to {debug_path}", verbose=True)
        result = normalize_extraction({}, chunk)
        result["_errors"] = [str(exc), f"raw_response: {debug_path}"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_version": PROMPT_VERSION,
                "chunk": asdict(chunk),
                "result": result,
                "cached_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return result


def normalized_key(value: Any) -> str:
    text = stringify(value)
    text = re.sub(r"\s+", "", text)
    text = text.strip("《》“”\"'`[]()（）")
    return text.lower()


def collect_record_values(record: dict[str, Any], field: str) -> list[str]:
    return unique_strings(as_list(record.get(field)), clip=220 if field in {"evidence", "sample_sentences"} else None)


def merge_named_records(
    records: list[dict[str, Any]],
    *,
    name_key: str,
    fields: list[str],
    conflict_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    conflict_fields = conflict_fields or set()
    merged: dict[str, dict[str, Any]] = {}

    for record in records:
        name = stringify(record.get(name_key))
        if is_missing(name):
            continue
        key = normalized_key(name)
        if not key:
            continue
        bucket = merged.setdefault(
            key,
            {
                name_key: name,
                "_values": defaultdict(list),
                "_conflicts": [],
            },
        )
        for field in fields:
            bucket["_values"][field].extend(collect_record_values(record, field))

    output: list[dict[str, Any]] = []
    for bucket in merged.values():
        item: dict[str, Any] = {name_key: bucket[name_key]}
        conflicts: list[str] = []
        for field in fields:
            values = unique_strings(bucket["_values"].get(field, []))
            if not values:
                item[field] = [] if field in {"aliases", "evidence", "source_chapters", "related_terms"} else "未明确"
                continue
            if field in {"aliases", "evidence", "source_chapters", "related_terms", "key_characters"}:
                item[field] = values
            else:
                item[field] = "；".join(values)
            if field in conflict_fields and len(values) > 1:
                conflicts.append(f"{field}: 存在不一致记录或多版本记录 -> {'；'.join(values)}")
        if conflicts:
            item["conflicts"] = conflicts
        output.append(item)

    return sorted(output, key=lambda x: normalized_key(x.get(name_key)))


def merge_relationships(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    fields = ["relationship_type", "details", "dynamics", "source", "evidence"]
    for record in records:
        a = stringify(record.get("character_a"))
        b = stringify(record.get("character_b"))
        if is_missing(a) or is_missing(b):
            continue
        rel_type = stringify(record.get("relationship_type")) or "未明确"
        pair = "||".join(sorted([normalized_key(a), normalized_key(b)]))
        key = f"{pair}||{normalized_key(rel_type)}"
        bucket = buckets.setdefault(
            key,
            {
                "character_a": a,
                "character_b": b,
                "_values": defaultdict(list),
            },
        )
        for field in fields:
            bucket["_values"][field].extend(collect_record_values(record, field))

    output: list[dict[str, Any]] = []
    for bucket in buckets.values():
        item = {"character_a": bucket["character_a"], "character_b": bucket["character_b"]}
        for field in fields:
            values = unique_strings(bucket["_values"].get(field, []), clip=220 if field == "evidence" else None)
            item[field] = values if field == "evidence" else ("；".join(values) if values else "未明确")
        output.append(item)
    return output


def merge_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        event = stringify(record.get("event"))
        if is_missing(event):
            continue
        source = stringify(record.get("source"))
        key = normalized_key(event + source)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "time_point": stringify(record.get("time_point")) or "未明确时间点",
                "event": event,
                "characters": unique_strings(as_list(record.get("characters"))),
                "location": stringify(record.get("location")) or "未明确",
                "impact": stringify(record.get("impact")) or "未明确",
                "source": source or "未明确",
                "evidence": unique_strings(as_list(record.get("evidence")), limit=3, clip=180),
            }
        )
    return output


def merge_plot_threads(records: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "main_thread": [],
        "sub_threads": [],
        "foreshadowing_planted": [],
        "foreshadowing_resolved": [],
        "unresolved_conflicts": [],
        "character_motives": [],
        "consistency_points": [],
    }
    for record in records:
        merged["main_thread"].extend(as_list(record.get("main_thread")))
        merged["consistency_points"].extend(as_list(record.get("consistency_points")))
        for key in [
            "sub_threads",
            "foreshadowing_planted",
            "foreshadowing_resolved",
            "unresolved_conflicts",
            "character_motives",
        ]:
            merged[key].extend(item for item in as_list(record.get(key)) if isinstance(item, dict))

    merged["main_thread"] = unique_strings(merged["main_thread"])
    merged["consistency_points"] = unique_strings(merged["consistency_points"])
    merged["sub_threads"] = merge_named_records(
        merged["sub_threads"],
        name_key="name",
        fields=["description", "status", "source", "evidence"],
        conflict_fields={"status"},
    )
    merged["foreshadowing_planted"] = merge_named_records(
        merged["foreshadowing_planted"],
        name_key="hint",
        fields=["expected_resolution", "source", "evidence"],
    )
    merged["foreshadowing_resolved"] = merge_named_records(
        merged["foreshadowing_resolved"],
        name_key="hint",
        fields=["resolution", "source", "evidence"],
    )
    merged["unresolved_conflicts"] = merge_named_records(
        merged["unresolved_conflicts"],
        name_key="conflict",
        fields=["parties", "source", "evidence"],
    )
    merged["character_motives"] = merge_named_records(
        merged["character_motives"],
        name_key="character",
        fields=["motive", "source", "evidence"],
        conflict_fields={"motive"},
    )
    return merged


def merge_style(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "narrative_perspective",
        "sentence_length",
        "description_preference",
        "dialogue_style",
        "pacing",
        "emotional_tone",
    ]
    merged: dict[str, Any] = {}
    for field in fields:
        values: list[Any] = []
        for record in records:
            values.extend(as_list(record.get(field)))
        unique = unique_strings(values)
        merged[field] = "；".join(unique) if unique else "未明确"

    for field in ["common_imagery", "protected_features", "sample_sentences"]:
        values = []
        for record in records:
            values.extend(as_list(record.get(field)))
        merged[field] = unique_strings(values, limit=30, clip=180 if field == "sample_sentences" else None)
    return merged


def merge_chapter_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = sorted(records, key=lambda item: item.get("_order", 0))
    output: list[dict[str, Any]] = []
    for item in summaries:
        output.append(
            {
                "chapter_title": stringify(item.get("chapter_title")) or "未明确",
                "chapter_number": stringify(item.get("chapter_number")) or "未明确",
                "what_happened": stringify(item.get("what_happened")) or "未明确",
                "characters_present": unique_strings(as_list(item.get("characters_present"))),
                "new_settings": unique_strings(as_list(item.get("new_settings"))),
                "plot_progression": stringify(item.get("plot_progression")) or "未明确",
                "foreshadowing": unique_strings(as_list(item.get("foreshadowing"))),
                "mood_atmosphere": stringify(item.get("mood_atmosphere")) or "未明确",
                "key_keywords": unique_strings(as_list(item.get("key_keywords"))),
                "source": stringify(item.get("source")) or "未明确",
            }
        )
    return output


def merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    characters: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    organizations: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    plot_threads: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    glossary: list[dict[str, Any]] = []
    style: list[dict[str, Any]] = []
    chapter_summaries: list[dict[str, Any]] = []

    for result in results:
        characters.extend(result.get("characters", []))
        world = result.get("world", {})
        locations.extend(world.get("locations", []))
        organizations.extend(world.get("organizations", []))
        rules.extend(world.get("rules", []))
        timeline.extend(result.get("timeline", []))
        plot_threads.append(result.get("plot_threads", {}))
        relationships.extend(result.get("relationships", []))
        glossary.extend(result.get("glossary", []))
        style.append(result.get("style", {}))
        chapter_summaries.append(result.get("chapter_summary", {}))

    return {
        "characters": merge_named_records(
            characters,
            name_key="name",
            fields=[
                "aliases",
                "identity",
                "appearance",
                "personality",
                "abilities",
                "relationships",
                "experiences",
                "current_status",
                "first_appearance",
                "source_chapters",
                "evidence",
                "uncertainties",
            ],
            conflict_fields={"identity", "appearance", "abilities", "current_status"},
        ),
        "world": {
            "locations": merge_named_records(
                locations,
                name_key="name",
                fields=["aliases", "description", "region", "importance", "source", "evidence"],
                conflict_fields={"region", "importance"},
            ),
            "organizations": merge_named_records(
                organizations,
                name_key="name",
                fields=["aliases", "nature", "scope", "key_characters", "source", "evidence"],
                conflict_fields={"nature", "scope"},
            ),
            "rules": merge_named_records(
                rules,
                name_key="name",
                fields=["category", "description", "source", "evidence"],
                conflict_fields={"category"},
            ),
        },
        "timeline": merge_timeline(timeline),
        "plot_threads": merge_plot_threads(plot_threads),
        "chapter_summaries": merge_chapter_summaries(chapter_summaries),
        "relationships": merge_relationships(relationships),
        "style": merge_style(style),
        "glossary": merge_named_records(
            glossary,
            name_key="term",
            fields=["category", "definition", "aliases", "related_terms", "source", "evidence"],
            conflict_fields={"category", "definition"},
        ),
    }


def md_value(value: Any, default: str = "未明确") -> str:
    values = unique_strings(as_list(value))
    return "、".join(values) if values else default


def md_evidence(values: Any) -> list[str]:
    return unique_strings(as_list(values), limit=6, clip=180)


def md_header(title: str) -> list[str]:
    return [
        f"# {title}",
        "",
        "> 本文件由 raw_data 小说原文自动抽取生成。所有内容应来自原文；不确定信息标注为“未明确”或“推测”。",
        "",
    ]


def append_evidence(lines: list[str], values: Any) -> None:
    evidence = md_evidence(values)
    if evidence:
        lines.append("- **相关原文证据/短摘录**:")
        for item in evidence:
            lines.append(f"  - {item}")
    else:
        lines.append("- **相关原文证据/短摘录**: 未明确")


def append_conflicts(lines: list[str], item: dict[str, Any]) -> None:
    conflicts = unique_strings(as_list(item.get("conflicts")))
    if conflicts:
        lines.append("- **不确定信息/冲突记录**:")
        for conflict in conflicts:
            lines.append(f"  - 存在不一致记录：{conflict}")


def generate_characters_md(characters: list[dict[str, Any]]) -> str:
    lines = md_header("characters")
    lines.extend(["## 人物设定", ""])
    for char in characters:
        lines.extend([f"### {md_value(char.get('name'), '未知人物')}", ""])
        fields = [
            ("aliases", "别名/称呼"),
            ("identity", "身份"),
            ("appearance", "外貌"),
            ("personality", "性格"),
            ("abilities", "能力/修为/技能"),
            ("relationships", "与其他人物关系"),
            ("experiences", "已知经历"),
            ("current_status", "当前状态"),
            ("first_appearance", "首次出现位置"),
            ("source_chapters", "来源章节"),
            ("uncertainties", "不确定信息"),
        ]
        keywords = [char.get("name"), *as_list(char.get("aliases"))]
        lines.append(f"- **关键词**: {md_value(keywords)}")
        for key, label in fields:
            lines.append(f"- **{label}**: {md_value(char.get(key))}")
        append_evidence(lines, char.get("evidence"))
        append_conflicts(lines, char)
        lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_world_md(world: dict[str, Any]) -> str:
    lines = md_header("world")
    lines.extend(["## 世界观 / 地点 / 势力 / 规则", ""])

    sections = [
        ("地点", world.get("locations", []), [("aliases", "别名"), ("description", "描述"), ("region", "所在区域"), ("importance", "重要性"), ("source", "来源章节")]),
        ("势力/组织", world.get("organizations", []), [("aliases", "别名"), ("nature", "性质"), ("scope", "势力范围"), ("key_characters", "关键人物"), ("source", "来源章节")]),
        ("规则/体系", world.get("rules", []), [("category", "类别"), ("description", "描述"), ("source", "来源章节")]),
    ]
    for title, items, fields in sections:
        lines.extend([f"## {title}", ""])
        for item in items:
            lines.extend([f"### {md_value(item.get('name'), '未命名')}", ""])
            keywords = [item.get("name"), *as_list(item.get("aliases")), item.get("category")]
            lines.append(f"- **关键词**: {md_value(keywords)}")
            for key, label in fields:
                lines.append(f"- **{label}**: {md_value(item.get(key))}")
            append_evidence(lines, item.get("evidence"))
            append_conflicts(lines, item)
            lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_timeline_md(events: list[dict[str, Any]]) -> str:
    lines = md_header("timeline")
    lines.extend(["## 剧情时间线", ""])
    for index, event in enumerate(events, 1):
        title = md_value(event.get("time_point"), f"事件 {index}")
        lines.extend([f"### {index:03d}. {title}", ""])
        fields = [
            ("event", "事件"),
            ("characters", "涉及人物"),
            ("location", "地点"),
            ("impact", "对后续剧情的影响"),
            ("source", "来源章节"),
        ]
        for key, label in fields:
            lines.append(f"- **{label}**: {md_value(event.get(key))}")
        append_evidence(lines, event.get("evidence"))
        lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_plot_threads_md(plot: dict[str, Any]) -> str:
    lines = md_header("plot_threads")
    lines.extend(["## 主线 / 支线 / 伏笔 / 冲突", ""])

    lines.extend(["### 主线目标", ""])
    main_thread = unique_strings(as_list(plot.get("main_thread")))
    lines.extend([f"- {item}" for item in main_thread] or ["- 未明确"])
    lines.append("")

    def record_section(title: str, items: list[dict[str, Any]], name_key: str, fields: list[tuple[str, str]]) -> None:
        lines.extend([f"### {title}", ""])
        if not items:
            lines.extend(["- 未明确", ""])
            return
        for item in items:
            lines.extend([f"#### {md_value(item.get(name_key), '未命名')}", ""])
            for key, label in fields:
                lines.append(f"- **{label}**: {md_value(item.get(key))}")
            append_evidence(lines, item.get("evidence"))
            append_conflicts(lines, item)
            lines.append("")

    record_section(
        "进行中的支线",
        plot.get("sub_threads", []),
        "name",
        [("description", "描述"), ("status", "状态"), ("source", "来源章节")],
    )
    record_section(
        "已埋伏笔",
        plot.get("foreshadowing_planted", []),
        "hint",
        [("expected_resolution", "预期回收"), ("source", "来源章节")],
    )
    record_section(
        "已回收伏笔",
        plot.get("foreshadowing_resolved", []),
        "hint",
        [("resolution", "回收方式"), ("source", "来源章节")],
    )
    record_section(
        "未解决冲突",
        plot.get("unresolved_conflicts", []),
        "conflict",
        [("parties", "冲突方"), ("source", "来源章节")],
    )
    record_section(
        "人物动机",
        plot.get("character_motives", []),
        "character",
        [("motive", "动机"), ("source", "来源章节")],
    )

    lines.extend(["### 后续续写时必须保持一致的点", ""])
    consistency = unique_strings(as_list(plot.get("consistency_points")))
    lines.extend([f"- {item}" for item in consistency] or ["- 未明确"])
    return "\n".join(lines).rstrip() + "\n"


def generate_chapter_summaries_md(summaries: list[dict[str, Any]]) -> str:
    lines = md_header("chapter_summaries")
    lines.extend(["## 章节摘要", ""])
    for summary in summaries:
        title = md_value(summary.get("chapter_title"), "未明确章节")
        lines.extend([f"### {title}", ""])
        fields = [
            ("chapter_number", "章节名或章节编号"),
            ("what_happened", "本章发生了什么"),
            ("characters_present", "出场人物"),
            ("new_settings", "新增设定"),
            ("plot_progression", "情节推进"),
            ("foreshadowing", "伏笔"),
            ("mood_atmosphere", "情绪/氛围"),
            ("key_keywords", "重要原文关键词"),
            ("source", "来源章节"),
        ]
        for key, label in fields:
            lines.append(f"- **{label}**: {md_value(summary.get(key))}")
        lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_relationships_md(relationships: list[dict[str, Any]]) -> str:
    lines = md_header("relationships")
    lines.extend(["## 人物关系", ""])
    for rel in relationships:
        title = f"{md_value(rel.get('character_a'), '?')} <-> {md_value(rel.get('character_b'), '?')}"
        lines.extend([f"### {title}", ""])
        fields = [
            ("relationship_type", "关系类型"),
            ("details", "关系详情"),
            ("dynamics", "关系动态"),
            ("source", "来源章节"),
        ]
        for key, label in fields:
            lines.append(f"- **{label}**: {md_value(rel.get(key))}")
        append_evidence(lines, rel.get("evidence"))
        lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_style_md(style: dict[str, Any]) -> str:
    lines = md_header("style")
    lines.extend(["## 文风与叙事习惯", ""])
    fields = [
        ("narrative_perspective", "叙事视角"),
        ("sentence_length", "句子长度"),
        ("description_preference", "描写偏好"),
        ("dialogue_style", "对话风格"),
        ("pacing", "节奏"),
        ("emotional_tone", "情绪基调"),
    ]
    for key, label in fields:
        lines.extend([f"### {label}", "", md_value(style.get(key)), ""])
    lines.extend(["### 常见意象", ""])
    lines.extend([f"- {item}" for item in unique_strings(as_list(style.get("common_imagery")))] or ["- 未明确"])
    lines.extend(["", "### 禁止破坏的风格特征", ""])
    lines.extend([f"- {item}" for item in unique_strings(as_list(style.get("protected_features")))] or ["- 未明确"])
    lines.extend(["", "### 典型原文短句", ""])
    samples = unique_strings(as_list(style.get("sample_sentences")), limit=20, clip=180)
    lines.extend([f"> {item}" for item in samples] or ["> 未明确"])
    return "\n".join(lines).rstrip() + "\n"


def generate_glossary_md(glossary: list[dict[str, Any]]) -> str:
    lines = md_header("glossary")
    lines.extend(["## 专有名词表", ""])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in glossary:
        by_category[md_value(item.get("category"), "未分类")].append(item)

    for category in sorted(by_category):
        lines.extend([f"## {category}", ""])
        for item in by_category[category]:
            lines.extend([f"### {md_value(item.get('term'), '未知名词')}", ""])
            fields = [
                ("aliases", "别名/简称"),
                ("definition", "定义/描述"),
                ("related_terms", "关联名词"),
                ("source", "来源章节"),
            ]
            keywords = [item.get("term"), item.get("category"), *as_list(item.get("aliases"))]
            lines.append(f"- **关键词**: {md_value(keywords)}")
            for key, label in fields:
                lines.append(f"- **{label}**: {md_value(item.get(key))}")
            append_evidence(lines, item.get("evidence"))
            append_conflicts(lines, item)
            lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


GENERATORS = {
    "characters": generate_characters_md,
    "world": generate_world_md,
    "timeline": generate_timeline_md,
    "plot_threads": generate_plot_threads_md,
    "chapter_summaries": generate_chapter_summaries_md,
    "relationships": generate_relationships_md,
    "style": generate_style_md,
    "glossary": generate_glossary_md,
}


def write_markdown_files(story_bible_dir: Path, merged: dict[str, Any]) -> None:
    story_bible_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        content = GENERATORS[name](merged[name])
        (story_bible_dir / f"{name}.md").write_text(content, encoding="utf-8")

    (story_bible_dir / "_merged_data.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_chunks(txt_files: list[Path], chunk_size: int, overlap: int, *, verbose: bool) -> list[Chunk]:
    chunks: list[Chunk] = []
    for txt_file in txt_files:
        text = read_text_file(txt_file)
        file_chunks = split_file_into_chunks(txt_file, text, chunk_size, overlap)
        log(f"[plan] {txt_file} -> {len(file_chunks)} chunks ({len(text)} chars)", verbose=verbose)
        chunks.extend(file_chunks)
    return assign_global_indexes(chunks)


def build_story_bible(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = cfg

    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    raw_value = args.raw_data or paths_cfg.get("raw_data")
    story_value = args.story_bible or paths_cfg.get("story_bible")
    if not raw_value:
        raise BuildError("Missing paths.raw_data in config. Use --raw-data to override.")
    if not story_value:
        raise BuildError("Missing paths.story_bible in config. Use --story-bible to override.")

    raw_data_dir = resolve_config_path(raw_value, config_path, prefer_existing=True)
    story_bible_dir = resolve_config_path(story_value, config_path, prefer_existing=True)

    txt_files = find_txt_files(raw_data_dir)
    if not txt_files and not args.raw_data:
        for candidate in compatible_raw_dirs(config_path):
            candidate_txt = find_txt_files(candidate)
            if candidate_txt:
                log(
                    f"[warn] No txt found under configured raw_data `{raw_data_dir}`; using compatible path `{candidate}`.",
                    verbose=True,
                )
                raw_data_dir = candidate
                txt_files = candidate_txt
                break

    if not txt_files:
        raise BuildError(
            f"No .txt files found under raw_data: {raw_data_dir}. "
            "Put UTF-8 txt files there or pass --raw-data."
        )

    if args.chunk_size <= 0:
        raise BuildError("--chunk-size must be positive.")
    if args.chunk_overlap < 0:
        raise BuildError("--chunk-overlap cannot be negative.")
    if args.chunk_overlap >= args.chunk_size:
        raise BuildError("--chunk-overlap must be smaller than --chunk-size.")

    chunks = build_chunks(txt_files, args.chunk_size, args.chunk_overlap, verbose=True)
    if args.limit_chunks is not None:
        chunks = chunks[: args.limit_chunks]

    log("", verbose=True)
    log("[plan] Story Bible build plan", verbose=True)
    log(f"  config       : {config_path}", verbose=True)
    log(f"  raw_data     : {raw_data_dir}", verbose=True)
    log(f"  story_bible  : {story_bible_dir}", verbose=True)
    log(f"  txt files    : {len(txt_files)}", verbose=True)
    log(f"  chunks       : {len(chunks)}", verbose=True)
    log(f"  output md    : {', '.join(name + '.md' for name in OUTPUT_FILES)}", verbose=True)
    log(f"  cache dir    : {story_bible_dir / '.build_cache'}", verbose=True)

    if args.verbose:
        for chunk in chunks[:20]:
            log(f"  - {chunk.index:05d} {chunk.label} ({len(chunk.text)} chars)", verbose=True)
        if len(chunks) > 20:
            log(f"  ... {len(chunks) - 20} more chunks", verbose=True)

    if args.dry_run:
        log("[dry-run] No LLM calls and no files were written.", verbose=True)
        return 0

    story_bible_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = story_bible_dir / ".build_cache"
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        result = extract_chunk(chunk, cfg, cache_dir, force=args.force, verbose=args.verbose)
        results.append(result)

    log("[merge] merging extracted records", verbose=True)
    merged = merge_results(results)
    write_markdown_files(story_bible_dir, merged)
    log(f"[done] Wrote Story Bible markdown to {story_bible_dir}", verbose=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build story_bible markdown from raw novel txt files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--raw-data", default=None, help="Override paths.raw_data.")
    parser.add_argument("--story-bible", default=None, help="Override paths.story_bible.")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=500, help="Overlapping characters between split chunks.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Only process first N chunks for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; do not call LLM or write files.")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-run LLM extraction.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed chunk/cache logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return build_story_bible(args)
    except BuildError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
