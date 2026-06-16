# -*- coding: utf-8 -*-
"""
阶段2.6 实机生成测试（调用真实 Ollama）
修复 /no_think 泄漏后复测：连续 8 次生成，逐项对照清单。
"""
import re
import sys
import yaml
from cowriter.session import Session

SEED = (
    "林清雪盘腿坐在一叶扁舟之上"
    "，月白衣衫垂至湖面，衣袖随"
    "风微动。小女童站在船尾轻摇"
    "长篹，湖水荡开一圈圈绳色波"
    "纹。过了好半晌，林清雪才缓"
    "缓睁开美眸，荡漾的湖水映在"
    "她的眸心深处，悠悠的，仿佛"
    "她心底流转的思绪。"
)

BAD_PATTERNS = [
    (re.compile(r"/no_think"),          "/no_think 泄漏"),
    (re.compile(r"</?think>"),          "think 标签残留"),
    (re.compile(r"好的，|接下来我将|以下是"), "助手语气"),
    (re.compile(r"\d+[.．、]"),         "1.2.3. 列表"),
    (re.compile(r"[A-Za-z]{4,}"),       "英文乱码(4字母+)"),
]

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(f"[模型] {cfg['model']['ollama_model']}")
print(f"[参数] temp={cfg['generation']['temperature']} top_p={cfg['generation']['top_p']} "
      f"top_k={cfg['generation']['top_k']} repeat_penalty={cfg['generation']['repeat_penalty']}")

sess = Session(cfg)
sess.seed(SEED)

results = []
empty_count = 0
for i in range(8):
    print(f"\n--- 第 {i+1} 次生成 ---")
    out = sess.generate()
    print(out[:300])
    if out:
        sess.accept(out)
    else:
        empty_count += 1

    issues = []
    if not out.strip():
        issues.append("空输出")
    for pat, label in BAD_PATTERNS:
        m = pat.search(out)
        if m:
            issues.append(f"{label}: {repr(m.group())}")
    results.append({"out": out, "issues": issues})
    status = "CLEAN" if not issues else "ISSUES: " + "; ".join(issues)
    print(f"  [{status}]")

print("\n" + "=" * 60)
print("=== 阶段2.6 复测清单 ===")
all_clean = all(not r["issues"] for r in results)
print(f"  连续 8 次无空输出: {'OK' if empty_count == 0 else f'FAIL -- 空输出 {empty_count} 次'}")
print(f"  /no_think 泄漏:    {'OK 未出现' if all(('/no_think 泄漏' not in str(r['issues'])) for r in results) else 'FAIL -- 见上方'}")
print(f"  无坏模式 (全8次):  {'OK' if all_clean else 'FAIL -- 见上方详情'}")
print(f"  config 模型名:     {'OK' if 'qwen3' in cfg['model']['ollama_model'] else 'FAIL'}")
print(f"\n  /检索 /保存 摘要压缩: 需手动验证（见 CLAUDE.md 第6项）")
print("=" * 60)
print("RESULT:", "ALL PASS" if all_clean and empty_count == 0 else "SOME FAILED")
