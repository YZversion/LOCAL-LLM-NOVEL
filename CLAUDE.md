# 本地小说续写助手 — Agent 工作规程

> 每次新对话开始时，先读此文件，再读【当前阶段】对应的测试清单，再动手。历史记录见 `docs/history.md`。

---

## 工作原则

1. **每次只做一件事**：单次对话的范围 = 一个任务 + 一组测试，完成并验证后停止。
2. **先测试，再继续**：每个阶段结束前必须有可运行的验证步骤，用户确认通过才进入下一阶段。
3. **不猜、不超前**：不主动实现用户没有明确要求的功能，不跨阶段预写代码。
4. **改动要小**：单次改动文件数和行数尽量小；超出就拆分对话。
5. **对话结束时生成摘要**：按本文末尾模板输出做了什么、测试结果、遗留问题和下一步。

---

## 项目快照

| 项目 | 值 |
|------|----|
| 工作目录 | `c:\Users\14390\Desktop\Code\LOCAL-LLM-NOVEL` |
| GitHub | `https://github.com/YZversion/LOCAL-LLM-NOVEL` |
| 推理模型（当前） | `huihui_ai/qwen3-abliterated:8b-v2` |
| 推理模型（已弃用） | R1 蒸馏 MoE，见 `docs/history.md` 阶段2.6 |
| 微调框架 | Unsloth QLoRA 4-bit（阶段4，未开始） |
| 显存预算 | 8GB（RTX 4070 Laptop） |
| CUDA | 13.2（Unsloth 兼容性待验证） |
| 数据保护 | `data/`、`models/`、`outputs/` 全部 gitignore，素材文件绝不入库 |
| Python | 3.11.9 |
| 生成参数 | 以 `config.yaml` 为准 |

---

## 阶段路线图与当前状态

```text
[✓] 阶段0    repo 骨架 + 环境配置文件
[✓] 阶段2    零训练合写回路
[✓] 阶段2.6  模型迁移 + 补全式续写改造
[ ] 阶段1    数据清洗（prepare_data.py，用户自有脚本）
[✓] 阶段3    确定性文风评测工具（2026-06-17 验收通过）
[ ] 阶段4    QLoRA 微调  ← 当前焦点：阶段4前置验证
[ ] 阶段5    向量 RAG（按需）
```

**当前焦点：阶段4前置验证 — 训练前数据与评测基线确认。**

不要直接写训练代码。先确认训练前数据、评测基线、CUDA/Unsloth 可用性与显存边界。

阶段3最终交付物：

- `pipeline/eval_style.py`
- `scripts/eval_draft.py`
- `_test_eval_style.py`
- `tests/fixtures/eval_style/`

---

## 当前阶段边界

阶段4前置验证只做准备和确认：

- 可以检查环境、依赖、CUDA/显存、训练数据候选、评测基线。
- 可以运行阶段3评测工具建立基线。
- 不直接开始 QLoRA 训练。
- 不修改生成链路：`cowriter/session.py`、`cowriter/retriever.py`、`cowriter/prompts.py`、`cowriter/web.py`。
- 不修改 `config.yaml`、`data/raw/`、`data/story_bible/`，除非用户明确要求并给出具体任务。

---

## 阶段4前置测试清单

- [ ] 确认当前运行依赖安装正常：`python _test_eval_style.py`
- [ ] 确认阶段3评测 wrapper 可作为训练前基线工具：`python scripts/eval_draft.py --reference <reference.txt> --candidate <candidate.txt>`
- [ ] 确认训练依赖方案：不要在 `requirements.txt` 直接解注释训练依赖，后续单独创建 `requirements-train.txt`
- [ ] 验证 Unsloth 与 CUDA 13.2 兼容性：先跑最小 forward pass，再动训练数据
- [ ] 确认单次训练显存峰值目标：小于 8GB
- [ ] 确认导出 GGUF 与 Ollama 加载流程的最小验证路径
- [ ] 确认同一验证集下，微调模型应优于阶段2/阶段3记录的基线

---

## 文件修改规范

| 文件 | 说明 |
|------|------|
| `config.yaml` | 唯一配置源；生成参数以此为准 |
| `cowriter/app.py` | CLI 入口；非明确任务不改 |
| `cowriter/retriever.py` | 检索逻辑；非明确任务不改 |
| `cowriter/prompts.py` | 提示词；非明确任务不改 |
| `cowriter/session.py` | 模型调用、摘要压缩、输出清洗；非明确任务不改 |
| `pipeline/eval_style.py` | 阶段3确定性评测核心 |
| `scripts/eval_draft.py` | 阶段3评测 wrapper |
| `pipeline/train_qlora.py` | 阶段4训练入口，占位；阶段4前置验证通过前不扩展 |
| `pipeline/export_gguf.py` | 阶段4导出入口，占位；阶段4前置验证通过前不扩展 |
| `docs/history.md` | 阶段历史归档，不作为当前工作规程 |

---

## 对话结束摘要模板

```text
## 本次对话摘要 [日期]

**完成的事**
- ...

**测试结果**
- [通过/失败/未测试] 具体描述

**遗留问题 / 已知坑**
- ...

**下一步（下次对话的第一件事）**
- ...

**当前阶段通关状态**
阶段4前置验证 当前 task：...，验收状态：通过/未通过
```
