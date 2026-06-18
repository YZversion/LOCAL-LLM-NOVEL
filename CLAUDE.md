# 本地小说续写助手 — Agent 工作规程

> 每次新对话开始时，先读此文件，再读 `docs/current_state.md`，再动手。

---

## 工作原则

1. **每次只做一件事**：单次对话的范围 = 一个任务 + 一组测试，完成并验证后停止。
2. **先测试，再继续**：每个阶段结束前必须有可运行的验证步骤，用户确认通过才进入下一阶段。
3. **不猜、不超前**：不主动实现用户没有明确要求的功能，不跨阶段预写代码。
4. **改动要小**：单次改动文件数和行数尽量小；超出就拆分对话。
5. **对话结束时生成摘要**：按本文末尾模板输出做了什么、测试结果、遗留问题和下一步。

---

## 当前阶段：阶段4 QLoRA 微调

详细环境/实验记录见 `docs/phase4_qlora.md`，项目全局状态见 `docs/current_state.md`。

**当前唯一任务：扩大样本量，验证 style_score 能否进一步提升。**

小样本验证链路已通过（2026-06-18）：
- LoRA v2（20条，5步）style_score 60.48 vs 基线 50.92，**+9.56，验收通过**
- repetition_risk: high → medium，主驱动力是 repetition_penalty +6.0
- 方法有效，瓶颈是样本量

下一步选项（用户决定）：
- A. 扩大样本量（50-200条）再训一轮，目标 style_score > 65
- B. 接入 Ollama，把 LoRA adapter 转成可在合写主回路使用的格式（需导出 GGUF）
- C. 先做 System B 知识图谱，补全 ch22-58 缺失角色后，再做更大规模微调

---

## 禁止修改的文件（非明确任务不碰）

`cowriter/session.py` · `cowriter/prompts.py` · `cowriter/retriever.py` · `cowriter/web.py` · `cowriter/chapter.py` · `config.yaml` · `data/raw/` · `data/story_bible/` · `pipeline/eval_style.py`

---

## 常用测试命令

```powershell
# 文风评测基线
python scripts/eval_draft.py --candidate <file> --config config.yaml --out-json <out.json>

# 时序过滤回归测试
python _test_temporal_filter.py

# 训练样本验证
python _test_train_samples.py

# LoRA 多轮生成（.venv-train 激活后）
python pipeline/generate_lora_multi.py
```

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
阶段4 当前 task：...，验收状态：通过/未通过
```
