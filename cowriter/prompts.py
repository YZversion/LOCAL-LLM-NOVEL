import re as _re

_SENT_END = _re.compile(r'[。！？]')
_PAUSE    = _re.compile(r'[，,、；]')


def _extract_prefill(recent_text: str, max_chars: int = 15) -> str:
    """
    prefill 截取规则（固定，不随 prompt 调整）：
      1. 找最后一句：从 recent_text 末尾往前找最后一个句末标点（。！？），
         其后到文本末尾为"最后一句"。若文本以句末标点结尾（last_sent 为空），
         则退到倒数第二个句末标点之间的完整句作为"最后一句"。
      2. 截取开头：在最后一句内找第一个句内停顿（逗号/顿号/分号），
         取其之前的部分（不含停顿符），上限 max_chars 字。
      3. 短句整句：若最后一句本身 ≤ max_chars 且无内部停顿，整句拿来（去末尾标点）。
      4. 无停顿长句：无内部停顿且 > max_chars，截取前 max_chars 字。
    目的：给模型"句子已起头、只能顺着写"的状态，不在句号后重启任务腔。
    """
    text = recent_text.rstrip()
    if not text:
        return ""
    ends = [m.end() for m in _SENT_END.finditer(text)]
    last_sent = text[ends[-1]:].strip() if ends else text.strip()
    if not last_sent:
        start = ends[-2] if len(ends) >= 2 else 0
        last_sent = text[start:ends[-1] - 1].strip()
    if not last_sent:
        return ""
    # 有停顿就截到停顿前（无论句子长短）
    pm = _PAUSE.search(last_sent)
    if pm:
        return last_sent[:pm.start()]
    # 无停顿：短句整句（去末尾标点），长句截 max_chars
    if len(last_sent) <= max_chars:
        return last_sent.rstrip('。！？，,、；')
    return last_sent[:max_chars]


def _clip(text: str, limit: int) -> str:
    return (text or "").strip()[:limit]


SYSTEM_PROMPT = """\
你是一位长篇中文小说续写助手，你的输出就是小说正文本身。

延续文风：沿用作者原有的叙事节奏、遣词习惯、句式长短和人物口吻，不主动优化或改良文笔。
延续场景：只延续当前场景的情绪、动作和节奏，不主动跳转剧情，不快速推进主线。
延续人物：性格、关系、称谓和行为与设定一致；设定与上文有差异时，以上文为准；不引入从未出现的新命名人物。
延续视角：跟随当前上文的叙事视角，不切换。
输出格式：直接输出正文，不带任何标题、说明、括号注释、分隔线或前导语。

/no_think\
"""


def build_prompt(
    recent_text: str,
    summary: str,
    retrieval: dict,
    instruction: str = "",
    target_chars: int = 600,  # 保留签名兼容性；字数由 num_predict 控制，不写入 prompt
) -> list[dict]:
    blocks: list[str] = []

    # 1. 设定集
    if retrieval.get("bible"):
        parts = []
        for i, r in enumerate(retrieval["bible"][:5], 1):
            source = r.get("source", "未知来源")
            text = _clip(r.get("text", ""), 400)
            parts.append(f"【相关设定{i}：{source}】\n{text}")
        blocks.append(
            "【相关设定】\n"
            "以下内容只用于约束人物、地点、物品、关系和背景，不要复述，不要总结。\n\n"
            + "\n\n".join(parts)
        )

    # 2. 剧情摘要
    if summary:
        blocks.append(
            "【剧情摘要】\n"
            "以下内容只用于理解前情，不要在正文中解释或概括。\n"
            + _clip(summary, 500)
        )

    # 3. 原文命中段落
    if retrieval.get("grep"):
        parts = []
        for i, hit in enumerate(retrieval["grep"][:3], 1):
            parts.append(f"【原文命中段落{i}】\n{_clip(hit, 700)}")
        blocks.append(
            "【原文命中段落】\n"
            "以下段落用于参考作者原文的语气、节奏、称谓和描写方式，不是续写起点。\n\n"
            + "\n\n".join(parts)
        )

    # 4. 续写方向（仅 /重试 等带附加指令时有值）
    if instruction.strip():
        blocks.append(f"续写方向：{instruction.strip()}")

    # 5. 当前上文（置最后，紧贴下方 prefill）
    blocks.append("【当前上文】\n" + recent_text.strip())

    # prefill：截取最后一句开头，让模型处于"句子已起头"状态
    prefill = _extract_prefill(recent_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n\n".join(blocks)},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def build_summary_prompt(text_to_compress: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是一位小说编辑。请只根据给定原文概括剧情事实，禁止编造。"
                "保留人物当前状态、所在地点、重要物品、人物关系、未解决冲突、伏笔和正在发生的动作。"
                "不要分析文风，不要评价人物，不要写写作建议。"
                "输出一段简洁中文，150字以内。"
            ),
        },
        {"role": "user", "content": text_to_compress.strip()},
    ]
