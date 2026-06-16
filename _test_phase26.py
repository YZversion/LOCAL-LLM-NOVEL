"""阶段2.6 快速校验脚本（不调模型，只验代码结构）"""
from cowriter.prompts import build_prompt, _extract_prefill
from cowriter.session import _strip_think

# 1. prefill 截取规则
cases = [
    # 典型：多句，以句号结尾 → 取倒数第二句开头到首逗号
    ("过了好半晌，林清雪才缓缓睁开美眸，荡漾的湖水映在她的眸心深处，悠悠的，仿佛她心底流转的思绪。",
     "过了好半晌"),
    # 短句，无内部停顿
    ("她皱了皱眉。", "她皱了皱眉"),
    # 无句末标点，末尾文本就是最后一句
    ("远处传来一声鸟鸣", "远处传来一声鸟鸣"),
    # 空
    ("", ""),
    # 多句，最后一句有逗号
    ("月光如水，洒满湖面。她忽然站起身，望向远处。",
     "她忽然站起身"),
]
print("=== prefill 截取 ===")
all_ok = True
for text, expected in cases:
    got = _extract_prefill(text)
    ok = "OK" if got == expected else "FAIL"
    if got != expected:
        all_ok = False
    print(f"  {ok} tail={repr(text[-30:])} -> {repr(got)} (expect {repr(expected)})")

# 2. 字数要求不出现在 prompt 任何位置
msgs = build_prompt("她缓缓睁开眼睛。", "", {"bible": [], "grep": []})
full_text = " ".join(m["content"] for m in msgs)
assert "续写约" not in full_text, "字数要求泄漏！"
assert "请直接续写" not in full_text, "任务句泄漏！"
assert "禁止空输出" not in full_text, "负向指令泄漏！"
assert "禁止" not in full_text, "负向指令泄漏！"
assert "/no_think" not in msgs[0]["content"], "/no_think 不应在 SYSTEM 中（已移除，改用 think=False）"
print("\n=== prompt 结构 ===")
print(f"  OK 字数要求 / 负向指令 均未出现")
print(f"  OK /no_think 不在 SYSTEM 中（已移除）")
print(f"  messages 数量: {len(msgs)}，最后 role: {msgs[-1]['role']}")
print(f"  prefill: {repr(msgs[-1]['content'])}")

# 3. prefill 紧贴上文（最后一条是 assistant）
assert msgs[-1]["role"] == "assistant", "prefill 未追加或 role 不对"

# 4. _strip_think
cases_think = [
    ("<think>内部推理</think>正文", "正文"),
    ("<think>未闭合推理\n多行", ""),
    ("正常正文", "正常正文"),
    ("<think>block1</think>中间<think>block2</think>尾", "中间尾"),
    # /no_think 截断残片（行首应删）
    ("/no\n正文开始", "正文开始"),
    ("/no_\n正文开始", "正文开始"),
    ("/no_thin\n正文开始", "正文开始"),
    ("/no_think\n正文开始", "正文开始"),
    ("第一行正文\n/no\n第三行正文", "第一行正文\n第三行正文"),
    ("正文中嵌入/no字样不误删", "正文中嵌入/no字样不误删"),
    # 行首助手语（独占一行才删）
    ("好的。\n林清雪缓缓抬眸。", "林清雪缓缓抬眸。"),
    ("好的，\n林清雪缓缓抬眸。", "林清雪缓缓抬眸。"),
    ("好的：\n正文", "正文"),
    ("好的，我同意。", "好的，我同意。"),
    ('林清雪说"好的，我明白了。"\n她转身而去。', '林清雪说"好的，我明白了。"\n她转身而去。'),
]
print("\n=== _strip_think ===")
for inp, expected in cases_think:
    got = _strip_think(inp)
    ok = "OK" if got == expected else "FAIL"
    if got != expected:
        all_ok = False
    print(f"  {ok} {repr(inp[:40])} -> {repr(got)}")

print("\nRESULT: " + ("ALL PASS" if all_ok else "SOME FAILED"))
