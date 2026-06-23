# 阶段4：QLoRA 微调实验记录

_最后更新：2026-06-22_

---

## 环境

| 项目 | 值 |
|------|----|
| venv | `.venv-train/`（与主 venv 隔离） |
| torch | 2.10.0+cu130 |
| unsloth | 2026.6.7 |
| CUDA Toolkit | 13.0（sm_89） |
| 显卡 | RTX 4070 Laptop，约 8GB VRAM |
| 基座模型 | `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` |
| 激活命令 | `.venv-train\Scripts\Activate.ps1` |

---

## 训练参数

稳定保留参数：

| 参数 | 值 |
|------|----|
| lora_r | 16 |
| lora_alpha | 16 |
| lora_dropout | 0 |
| target_modules | q/k/v/o/gate/up/down_proj |
| use_gradient_checkpointing | `"unsloth"` |
| optim | adamw_8bit |
| learning_rate | 2e-4 |
| batch_size | 1 |
| grad_accum | 4 |
| max_seq_length（训练） | 1024 |
| bf16 | True |

版本差异：

| 版本 | 数据 | epoch | optimizer steps | warmup_steps | output_dir |
|------|------|-------|-----------------|--------------|------------|
| v1 | novel1 20 条 | 1 | 5 | 5 | `outputs/qlora_run/` |
| v2 | novel1 20 条 | 1 | 5 | 1 | `outputs/qlora_run_v2/` |
| v3 | merged 544 条 | 1 | 136 | 7 | `outputs/qlora_run_v3/` |

备注：
- v1 的 `warmup_steps=5` 等于总步数，LR 从未达到峰值。
- v2 的 `warmup_steps=1` 是针对 5 步小样本实验的特例。
- v3 样本数扩大到 544 后，按总步数约 5% 设置 `warmup_steps=7`。

---

## 显存实验记录

| 实验 | max_seq_length | 结果 | 备注 |
|------|---------------|------|------|
| 推理（8B 4-bit） | 8192 | peak 5.80 GB | 余量约 2.2 GB |
| 训练 v1（OOM） | 2048 | peak 7.44 GB，fused CE OOM | warmup bug 同时存在 |
| 训练 v2（通过） | 1024 | peak 约 6.74 GB | 20 条样本，5 steps |
| 训练 v3（通过） | 1024 | peak 约 6.82 GB | 544 条样本，136 steps |

结论：样本数增加主要影响训练时长，不显著改变单步显存峰值。

---

## 数据集

### novel1

- 文件：`data/processed/train_samples.jsonl`
- 规模：20 条，ch2-21
- 构造入口：`pipeline/build_train_samples.py`
- 时序口径：`target_chapter=N -> max_chapter=N-1`
- 验证：`_test_train_samples.py` 20/20 全绿
- 用途：同时保留真实推理结构相关字段（`messages`、bible/prior summary 等）

### novel2

- 原文：`data/raw/novel2_raw.txt`
- 构造入口：`pipeline/build_novel2_labeled_samples.py`
- 样本：`data/processed/novel2_samples.jsonl`
- 标签：`data/processed/novel2_labels.jsonl`
- 规模：524 条
- 切分参数：`context_chars=1000`，`completion_chars=250`，`min_completion_chars=150`
- 用途：只做文风学习，不接入 story_bible 检索，不要求时序 frontmatter

source_section 分布：

| source_section | 样本数 |
|---|---:|
| main | 418 |
| extras | 21 |
| vol4 | 85 |

content_sensitivity 分布：

| content_sensitivity | 样本数 | 占比 |
|---|---:|---:|
| explicit_sensitive | 290 | 55.34% |
| mature_nonexplicit | 156 | 29.77% |
| general | 78 | 14.89% |

### merged

- 合并入口：`pipeline/merge_train_samples.py`
- 输出：`data/processed/merged_train_samples.jsonl`
- 总数：544 条
- 追踪字段：`merged_sample_id`、`source_book`、`source_sample_id`、`source_section`、`source_section_confidence`、`content_sensitivity`、`content_sensitivity_confidence`
- novel1 标签：`source_book=novel1`，`content_sensitivity=unlabeled`，confidence `0.0`
- novel2 标签：从 `novel2_labels.jsonl` 合并
- 验证：novel1 20 条 `messages/completion` 与原文件逐条字段级一致

---

## LoRA Adapter 版本

| 版本 | 路径 | 状态 | 备注 |
|------|------|------|------|
| v1 | `outputs/qlora_run/` | 归档 | warmup bug，loss 3.593 |
| v2 | `outputs/qlora_run_v2/` | 已通过 | 20 条小样本，style_score 60.48 |
| v3 | `outputs/qlora_run_v3/` | 未通过 | 544 条扩样，style_score 46.05 |

v2 loss：

```text
step 1: loss=3.795  LR=0
step 2: loss=4.022  LR=2e-4
step 3: loss=3.757  LR=1.5e-4
step 4: loss=3.530  LR=1e-4
step 5: loss=3.450  LR=5e-5
```

v3 loss 快照：

```text
step 1:   loss=3.565  LR=0
step 7:   loss=3.300  LR=0.0001714
step 8:   loss=3.300  LR=0.0002000
step 15:  loss=2.917  LR=0.0001891
step 30:  loss=2.651  LR=0.0001659
step 45:  loss=2.516  LR=0.0001426
step 60:  loss=2.519  LR=0.0001194
step 75:  loss=2.267  LR=0.0000961
step 90:  loss=2.295  LR=0.0000729
step 105: loss=2.443  LR=0.0000496
step 120: loss=2.295  LR=0.0000264
step 136: loss=2.488  LR=0.0000016
train_loss=2.599
```

---

## 评测结果

| 指标 | 基线 | v2 | v3 |
|------|------|----|----|
| style_score | 50.92 | 60.48 | 46.05 |
| repetition_risk | high (55.6%) | medium (16.7%) | medium (0.0%) |
| contamination_risk | low | low | low |

v2 结论：
- style_score 比基线提升 +9.56，阶段4小样本链路通过。
- 主驱动力是 repetition_penalty 提升。

v3 结论：
- 扩样训练没有带来进一步提升，style_score 低于 v2 和基线。
- repetition 继续改善，但 sentence_profile 明显恶化。
- 关键异常：average_sentence_length `83.39`，longest_sentence_length `471`。

---

## v3 诊断记录

### 最长句诊断

- 471 字最长句不是评测分句逻辑漏切：内部没有正常句末标点，只有末尾被识别为句子结束。
- v3 候选整体标点密度显著低于 v2：

| 候选 | 句末标点/100字 | 逗号类/100字 | 平均句长 | 最长句 |
|------|--------------:|------------:|---------:|------:|
| v2 | 3.38 | 5.72 | 29.06 | 56 |
| v3 原始 | 1.14 | 0.79 | 83.39 | 471 |

### 训练数据标点密度

按 completion 统计，`content_sensitivity` 组间差异不足以解释 v3 退化：

| content_sensitivity | n | 句末标点/100字 | 逗号类/100字 | 平均句长 |
|---|---:|---:|---:|---:|
| unlabeled | 20 | 3.04 | 6.19 | 30.05 |
| general | 78 | 3.18 | 6.76 | 29.31 |
| mature_nonexplicit | 156 | 3.22 | 6.60 | 29.36 |
| explicit_sensitive | 290 | 3.05 | 6.19 | 31.43 |

结论：explicit_sensitive 比其他组只低约 4%-8%，远不足以解释 v3 生成端 66%-86% 的标点密度下降。

### 当前待诊断

- 用同一个 v3 adapter 重复生成 2 次，判断标点退化是否稳定。
- `outputs/lora_candidate_v3_repeat1.txt` 已生成（约 3336c）。
- `outputs/lora_candidate_v3_repeat2.txt` 尚未生成。
- repeat1/repeat2 统计完成前，不直接进入 checkpoint 保存逻辑修改。

---

## 下一步

第一优先级：完成 v3 重复生成诊断。

判断口径：
- 如果 repeat1/repeat2 都接近 v3 原始低标点密度，说明 adapter 本身稳定退化，下一步再讨论 best checkpoint / 训练轮次 / 保存逻辑。
- 如果 repeat1/repeat2 接近 v2 正常标点密度，说明 v3 原始候选可能是采样偶发异常，下一步应查生成脚本稳定性。
- 如果结果参差不齐，应继续增加重复生成次数，而不是立刻重训。
