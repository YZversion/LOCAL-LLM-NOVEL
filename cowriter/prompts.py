SYSTEM_PROMPT = """\
你是一位长篇中文小说续写助手。

你会收到相关设定、原文命中段落、剧情摘要、当前上文和续写要求。
这些资料只用于保持事实、人物、地点、视角和文风一致，不是正文的一部分。

你的任务是：从【当前上文】最后一句之后，直接续写小说正文。

必须遵守：
1. 严格延续作者原有文风、叙事节奏、遣词习惯和人物口吻，不主动优化文风。
2. 只延续当前场景、当前情绪和当前动作，不主动跳转剧情，不快速推进主线。
3. 人物性格、关系、称谓和行为必须符合设定；若当前上文与设定有差异，以当前上文为准。
4. 不引入上文、摘要或设定中从未出现的全新命名人物。
5. 严格跟随当前上文的叙事视角，不切换视角。
6. 只输出小说正文。第一个字必须是正文，不要有任何前缀、解释、标题、列表、分隔线、括号注释或结尾标记。
7. 不要使用"好的""当然""明白了""以下是"等助手语气。
8. 即使信息不足，也必须自然承接上文续写，禁止空输出。\
"""


def _clip(text: str, limit: int) -> str:
    return (text or "").strip()[:limit]


def build_prompt(
    recent_text: str,
    summary: str,
    retrieval: dict,
    instruction: str = "",
    target_chars: int = 600,
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

    # 4. 当前上文（唯一续写起点，紧靠要求）
    blocks.append(
        "【当前上文】\n"
        "以下是唯一的续写起点，请从最后一句之后自然接下去。\n\n"
        + recent_text.strip()
    )

    # 5. 续写要求
    requirement = (
        f"【续写要求】\n"
        f"请直接续写约 {target_chars} 字小说正文。\n"
        f"第一个字必须是正文，不要标题，不要解释，不要总结，不要空输出。\n"
        f"不要为了凑字数强行转场，宁可略短，也要自然。"
    )
    extra = instruction.strip()
    if extra:
        requirement += f"\n额外要求：{extra}"
    blocks.append(requirement)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


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
