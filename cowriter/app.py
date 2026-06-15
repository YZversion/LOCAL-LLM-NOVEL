#!/usr/bin/env python3
"""
本地小说续写助手 — 命令行入口
用法：python -m cowriter.app  或  python cowriter/app.py
"""
import sys

# Windows 控制台中文支持
sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

import yaml
from cowriter.prompts import build_prompt
from cowriter.session import Session

HELP = """
命令列表：
  直接回车              接受模型续写，进入下一轮
  输入文字+回车         用你的版本替换模型续写，写入 session
  /拒绝                 丢弃当前模型输出，不写入 session，重新生成
  /重试 [指令]          丢弃当前输出并立即重新生成（可附加额外指令）
  /重新生成 [指令]      同 /重试
  /上下文               不调用模型，仅打印当前 prompt 结构与字数统计
  /摘要                 查看/手动编辑当前剧情摘要
  /保存                 保存当前草稿到 outputs/
  /检索 <关键词>        手动检索设定集和原文
  /帮助                 显示本帮助
  /退出  q  退出        退出（可选保存）
"""

DIVIDER = "─" * 50


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_multiline(prompt: str) -> str:
    """读取多行输入，连续两次空行结束。"""
    print(prompt)
    lines: list[str] = []
    empty = 0
    while empty < 2:
        line = input()
        if line == "":
            empty += 1
        else:
            empty = 0
            lines.append(line)
    return "\n".join(lines)


def main():
    print("=== 本地小说续写助手 ===")
    print("输入 /帮助 查看命令\n")

    config = load_config()
    session = Session(config)

    # 初始上文
    initial = read_multiline("请粘贴当前章节上文（连续两次空行结束）：")
    if initial.strip():
        session.seed(initial)
        print(f"[已载入上文，共 {len(initial)} 字]")

    last_generated = ""

    while True:
        print(f"\n{DIVIDER}")
        print("[正在生成…]")
        try:
            last_generated = session.generate()
        except Exception as e:
            print(f"[错误] {e}")
            print("请确认 Ollama 正在运行：ollama serve")
            cmd = input("输入 /退出 离开，或按回车重试：").strip()
            if cmd == "/退出":
                break
            continue

        print("\n【模型续写】")
        print(last_generated)

        cmd = input(f"\n{DIVIDER}\n回应（回车=接受 / 文字=替换 / 命令）：").strip()

        if cmd == "":
            session.accept(last_generated)
            print("[已接受续写]")

        elif cmd == "/摘要":
            print(f"\n当前摘要：\n{session.summary or '（空）'}")
            new = input("输入新摘要（回车跳过）：").strip()
            if new:
                session.summary = new
                print("[摘要已更新]")

        elif cmd == "/保存":
            path = session.save_draft()
            print(f"[已保存] {path}")

        elif cmd.startswith("/检索"):
            kw = cmd[3:].strip()
            if not kw:
                print("用法：/检索 关键词")
                continue
            r = session.retriever.retrieve(kw)
            print(f"\n抽取实体：{r['entities']}")
            for b in r["bible"]:
                print(f"\n[设定:{b['source']}]\n{b['text'][:300]}")
            for g in r["grep"][:3]:
                print(f"\n[原文命中]\n{g}")

        elif cmd == "/帮助":
            print(HELP)

        elif cmd == "/拒绝":
            print("[已丢弃，重新生成…]")
            continue

        elif cmd.startswith("/重试") or cmd.startswith("/重新生成"):
            prefix_len = 3 if cmd.startswith("/重试") else 5
            instruction = cmd[prefix_len:].strip()
            print("[重新生成…]")
            try:
                last_generated = session.generate(instruction=instruction)
            except Exception as e:
                print(f"[错误] {e}")
                continue
            print("\n【重新生成】")
            print(last_generated)
            sub = input(f"\n{DIVIDER}\n回应（回车=接受 / /拒绝=丢弃）：").strip()
            if sub == "":
                session.accept(last_generated)
                print("[已接受续写]")
            elif sub == "/拒绝":
                print("[已丢弃]")
            elif not sub.startswith("/"):
                session.accept(sub)
                print("[已记录你的版本]")

        elif cmd == "/上下文":
            ctx_win = session.accepted_text[-session.cfg["session"]["max_recent_chars"]:]
            retrieval = session.retriever.retrieve(ctx_win)
            msgs = build_prompt(
                recent_text=ctx_win,
                summary=session.summary,
                retrieval=retrieval,
                target_chars=session.cfg["session"]["output_tokens"],
            )
            sys_txt = msgs[0]["content"]
            usr_txt = msgs[1]["content"]
            full = sys_txt + "\n\n" + usr_txt
            bible_chars = sum(len(r["text"][:400]) for r in retrieval.get("bible", []))
            grep_chars = sum(len(g) for g in retrieval.get("grep", [])[:3])
            print("\n[上下文统计]")
            print(f"  writing_rules (system prompt) : {len(sys_txt)} 字")
            print(f"  session.summary               : {len(session.summary)} 字")
            print(f"  retrieved_bible               : {bible_chars} 字")
            print(f"  retrieved_grep                : {grep_chars} 字")
            print(f"  recent_context                : {len(ctx_win)} 字")
            print(f"  final_prompt 总计             : {len(full)} 字")
            print(f"\n[Prompt 前 2000 字预览]\n{DIVIDER}")
            print(full[:2000])

        elif cmd in ("/退出", "q", "退出", "exit", "quit"):
            if input("退出前保存草稿？(y/N)：").strip().lower() == "y":
                path = session.save_draft()
                print(f"[已保存] {path}")
            print("再见")
            break

        else:
            if cmd.startswith("/"):
                print(f"[未知命令] {cmd}，输入 /帮助 查看命令")
            else:
                session.accept(cmd)
                print("[已记录你的版本]")


if __name__ == "__main__":
    main()
