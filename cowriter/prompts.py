SYSTEM_PROMPT = """\
你是一位协助作者续写长篇中文小说的写作助手。

【核心规则】
1. 严格模仿作者的文风、叙事节奏与行文习惯，不做任何风格上的「优化」
2. 顺着上文的势头自然延伸，不主动推进或跳跃主线剧情
3. 人物的言行、性格、口吻必须严格符合设定，不得随意拔高或软化
4. 直接续写正文，禁止总结、解释、旁白性评论
5. 不引入上文从未提及的全新命名人物

【严禁输出以下内容——违反即为错误】
- 禁止以"好的""当然""明白了""我来帮你""我将为你""以下是"等 AI 助手语气开头
- 禁止输出 Markdown 标题（## / ### 等）或分割线（---）
- 禁止夹杂英文单词、拼音或任何非正文注释
- 禁止输出写作建议、风格分析、情节说明
- 禁止切换叙事视角——严格跟随上文的视角人物
- 禁止在正文外添加括号注释、方括号标注或任何元信息
- 禁止在结尾附加"（完）""——END——"或章节编号

【输出格式要求】
直接从上文最后一句话之后续写。输出的第一个字就是正文，无需任何前缀或说明。\
"""


def build_prompt(
    recent_text: str,
    summary: str,
    retrieval: dict,
    instruction: str = "",
    target_chars: int = 600,
) -> list[dict]:
    blocks: list[str] = []

    if retrieval.get("bible"):
        parts = [f"【{r['source']}】\n{r['text'][:400]}" for r in retrieval["bible"]]
        blocks.append("## 相关设定\n" + "\n\n".join(parts))

    if retrieval.get("grep"):
        blocks.append(
            "## 原文检索（人物/地点命中段落）\n"
            + "\n---\n".join(retrieval["grep"][:3])
        )

    if summary:
        blocks.append(f"## 剧情摘要\n{summary}")

    blocks.append(f"## 当前上文（最近约 {len(recent_text)} 字）\n{recent_text}")

    suffix = f"请直接续写约 {target_chars} 字正文，第一个字就是正文，不要有任何前缀、说明或解释。{instruction}".strip()
    blocks.append(f"## 续写要求\n{suffix}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def build_summary_prompt(text_to_compress: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是一位小说编辑。请用简洁中文概括以下段落的剧情要点，"
                "保留人物当前状态、所在地点、重要物品、未解决的冲突与伏笔，"
                "约 150 字以内，不需要分析，只要事实。"
            ),
        },
        {"role": "user", "content": text_to_compress},
    ]
