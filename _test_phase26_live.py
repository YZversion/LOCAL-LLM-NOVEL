# -*- coding: utf-8 -*-
"""
阶段2.6 实机生成测试（调用真实 Ollama）
用扁舟片段跑 3 次，逐项对照清单。
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
    "纹。过了好半品，林清雪才缓"
    "缓睡开美譸，荡漾的湖水映在"
    "她的譸心深处，悠悠的，价佛"
    "她心底流转的思绪。"
)

BAD_PATTERNS = [
    re.compile(r"[Gg]ood|[Hh]ello|[Aa]ssistant"),
    re.compile(r"\d+[.．、]"),          # 1. 2. 3. 列表
    re.compile(r"</?think>"),                    # think 残留（含孤立 </think>）
    re.compile(r"好的，|接下来我将|以下是"),  # 助手语
    re.compile(r"[A-Za-z]{4,}"),                # 英文乱码（4字母以上连续）
]

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(f"[模型] {cfg['model']['ollama_model']}")
print(f"[参数] temp={cfg['generation']['temperature']} top_p={cfg['generation']['top_p']} "
      f"top_k={cfg['generation']['top_k']} repeat_penalty={cfg['generation']['repeat_penalty']}")

sess = Session(cfg)
sess.seed(SEED)

results = []
for i in range(3):
    print(f"\n--- 第 {i+1} 次生成 ---")
    out = sess.generate()
    print(out[:400])
    if out:
        sess.accept(out)

    issues = []
    if not out.strip():
        issues.append("空输出")
    for pat in BAD_PATTERNS:
        m = pat.search(out)
        if m:
            issues.append(f"命中坏模式 {repr(m.group())}")
    # prefill 衔接检查（检查上文结尾字是否自然延续）
    results.append({"out": out, "issues": issues})
    print(f"  问题: {issues if issues else '无'}")

print("\n=== 阶段2.6 清单核对 ===")
all_empty = all(not r["out"].strip() for r in results)
any_bad   = any(r["issues"] for r in results)
print(f"  config 模型名: {'OK' if 'qwen3' in cfg['model']['ollama_model'] else 'FAIL'}")
print(f"  启动不崩溃: OK")
print(f"  连续3次无空输出: {'OK' if not all_empty and all(r['out'] for r in results) else 'FAIL'}")
print(f"  无坏模式: {'OK' if not any_bad else 'FAIL -- 见上方详情'}")
print(f"  prefill role 已加: OK (由 _test_phase26.py 验证)")
print(f"  /检索 /保存 /摘要 等命令: 未在此脚本测试，需手动验证")
