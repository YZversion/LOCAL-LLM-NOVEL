# 本地小说续写助手 — Agent 工作规程

_最后更新：2026-06-22_

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

**当前唯一任务：定位 v3 扩样训练失败原因，再决定是否修 checkpoint 保存逻辑。**

已确认结果：
- 零微调基线：style_score 50.92，repetition_risk high，contamination_risk low。
- LoRA v2（第一本 20 条，5 optimizer steps）：style_score 60.48，repetition_risk medium，contamination_risk low，阶段4小样本链路验收通过。
- novel2 已切分/打标：524 条，`main=418 / extras=21 / vol4=85`；`explicit_sensitive=290 / mature_nonexplicit=156 / general=78`。
- 合并数据集已生成：`data/processed/merged_train_samples.jsonl`，共 544 条，novel1 20 + novel2 524；novel1 原始 20 条逐条字段级比对通过。
- LoRA v3（544 条，1 epoch，136 steps，warmup_steps=7）：训练完成，adapter 在 `outputs/qlora_run_v3/`；forward/backward peak 约 6.82GB。
- v3 首次评测失败：style_score 46.05，低于 v2 60.48 和基线 50.92；repetition_risk 仍为 medium 且重复率 0.0%，contamination_risk low。

已完成诊断：
- v3 主要扣分来自 `sentence_profile`：average_sentence_length 83.39，longest_sentence_length 471。
- 471 字长句不是 `eval_style.py` 分句漏切，候选文本本身缺少正常标点。
- 训练 completion 的标点密度按 `content_sensitivity` 分组差距很小：explicit_sensitive 句末标点约 3.05/100字，general 约 3.18/100字，mature_nonexplicit 约 3.22/100字；不足以解释 v3 生成端 66%-86% 的标点密度退化。
- 因此 explicit_sensitive 内容占比不是当前主要解释，问题范围收窄到训练/保存/生成稳定性。

当前卡点：
- 正在用同一个 v3 adapter 做重复生成诊断，确认标点退化是 adapter 稳定特征还是单次采样异常。
- `outputs/lora_candidate_v3_repeat1.txt` 已生成（约 3336c），尚需统计并生成第二份 repeat 候选后再下结论。
- 不要在重复生成诊断完成前直接修改 checkpoint 保存逻辑或重新训练。

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
