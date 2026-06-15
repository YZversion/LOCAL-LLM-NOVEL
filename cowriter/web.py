#!/usr/bin/env python3
"""本地小说续写助手 — Gradio Web UI。用法：python -m cowriter.web"""
import yaml
import gradio as gr

from cowriter.session import Session
from cowriter.prompts import build_prompt

with open("config.yaml", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_sess = Session(_cfg)
_gen: list[str] = [""]  # last generated text; mutable so closures can write it


# ── 操作函数 ─────────────────────────────────────────────────────────────────

def do_generate(ctx: str, inst: str):
    if not _sess.accepted_text and ctx.strip():
        _sess.seed(ctx.strip())
    try:
        out = _sess.generate(instruction=inst.strip())
    except Exception as e:
        return "", f"[错误] {e}\n确认 Ollama 正在运行：ollama serve"
    _gen[0] = out
    return out, f"[生成完成] {len(out)} 字"


def do_accept():
    t = _gen[0]
    if not t:
        return gr.update(), "[提示] 无待接受的续写"
    _sess.accept(t)
    _gen[0] = ""
    ctx = _sess.accepted_text[-_cfg["session"]["max_recent_chars"]:]
    return ctx, f"[已接受] +{len(t)} 字，累计 {len(_sess.accepted_text)} 字"


def do_reject():
    _gen[0] = ""
    return "[已拒绝] 续写未写入 session"


def do_save():
    if not _sess.accepted_text:
        return "[提示] session 为空，无内容可保存"
    return f"[已保存] {_sess.save_draft()}"


def do_view_ctx():
    win = _sess.accepted_text[-_cfg["session"]["max_recent_chars"]:]
    if not win:
        return "[提示] session 为空，请先粘贴上文并点击生成"
    ret = _sess.retriever.retrieve(win)
    msgs = build_prompt(win, _sess.summary, ret,
                        target_chars=_cfg["session"]["output_tokens"])
    s, u = msgs[0]["content"], msgs[1]["content"]
    bc = sum(len(r["text"][:400]) for r in ret.get("bible", []))
    gc = sum(len(g) for g in ret.get("grep", [])[:3])
    stats = (
        f"[上下文统计]\n"
        f"  writing_rules  : {len(s)} 字\n"
        f"  session.summary: {len(_sess.summary)} 字\n"
        f"  retrieved_bible: {bc} 字\n"
        f"  retrieved_grep : {gc} 字\n"
        f"  recent_context : {len(win)} 字\n"
        f"  prompt 总计    : {len(s) + len(u)} 字\n\n"
    )
    preview = "【写作规则】\n" + s + "\n\n" + u + "\n\n【续写正文】\n（← 模型输出区）"
    return stats + preview[:2000]


def do_search(kw: str):
    if not kw.strip():
        return "[提示] 请输入检索关键词"
    r = _sess.retriever.retrieve(kw.strip())
    lines = [f"抽取实体：{r['entities']}"]
    for b in r["bible"]:
        lines.append(f"\n[设定:{b['source']}  score={b['score']}]\n{b['text'][:300]}")
    for g in r["grep"][:3]:
        lines.append(f"\n[原文命中]\n{g}")
    return "\n".join(lines) if len(lines) > 1 else "[无命中]"


# ── UI 布局 ──────────────────────────────────────────────────────────────────

def main():
    with gr.Blocks(title="本地小说续写助手") as demo:
        gr.Markdown("# 本地小说续写助手")

        with gr.Row():
            with gr.Column(scale=2):
                ctx_box = gr.Textbox(
                    label="当前上文（首次粘贴原文；接受续写后自动更新）",
                    lines=10, max_lines=20,
                    placeholder="在此粘贴小说上文…",
                )
                inst_box = gr.Textbox(
                    label="续写要求（可留空）",
                    lines=2,
                    placeholder="例：聚焦心理描写 / 加快节奏",
                )
                with gr.Row():
                    b_gen = gr.Button("生成", variant="primary")
                    b_acc = gr.Button("接受")
                    b_rej = gr.Button("拒绝")
                    b_sav = gr.Button("保存")
                    b_ctx = gr.Button("查看上下文")
                out_box = gr.Textbox(
                    label="模型续写结果",
                    lines=10, max_lines=20,
                    interactive=False,
                )

            with gr.Column(scale=1):
                kw_box   = gr.Textbox(label="检索关键词", placeholder="人名 / 地名…")
                b_srch   = gr.Button("检索 story_bible")
                srch_out = gr.Textbox(label="检索结果", lines=10, interactive=False)
                st_box   = gr.Textbox(label="状态日志", lines=4, interactive=False)
                ctx_prev = gr.Textbox(label="上下文预览", lines=14, interactive=False)

        b_gen.click(do_generate, [ctx_box, inst_box], [out_box, st_box])
        b_acc.click(do_accept,   [],                   [ctx_box, st_box])
        b_rej.click(do_reject,   [],                   [st_box])
        b_sav.click(do_save,     [],                   [st_box])
        b_ctx.click(do_view_ctx, [],                   [ctx_prev])
        b_srch.click(do_search,  [kw_box],             [srch_out])

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
