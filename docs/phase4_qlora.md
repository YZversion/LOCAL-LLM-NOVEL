# 阶段4：QLoRA 微调实验记录

_最后更新：2026-06-23_

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
| bf16 | True |

版本差异：

| 版本 | 数据 | max_seq_length | context_chars | epoch | optimizer steps | warmup_steps | output_dir | 状态 |
|------|------|---------------|--------------|-------|-----------------|--------------|------------|------|
| v1 | novel1 20 条 | 2048 | 1000 | 1 | 5 | 5 | `outputs/qlora_run/` | 归档（warmup bug） |
| v2 | novel1 20 条 | 1024 | 60 | 1 | 5 | 1 | `outputs/qlora_run_v2/` | 已通过 ✅ |
| v3 | merged 544 条 | 1024 | 60/1000* | 1 | 136 | 7 | `outputs/qlora_run_v3/` | 放弃 ✗ |
| v4 | 风丝引 57 条 | **1536** | **700** | **3**（TBD） | **~42**（TBD） | **2**（TBD） | `outputs/qlora_run_v4/` | 待训练 🔜 |

*v3 数据：novel1 用 context_chars=60，novel2 用 context_chars=1000（两种口径混入，是 v3 放弃原因之一）。

备注：
- v1 的 `warmup_steps=5` 等于总步数，LR 从未达到峰值。
- v2 的 `warmup_steps=1` 是针对 5 步小样本实验的特例。
- v3 样本数扩大到 544 后，按总步数约 5% 设置 `warmup_steps=7`，但 novel2 数据分布与风丝引不兼容，根本方向错误。
- v4 的 warmup_steps=2（5% of 42 总步数）、num_train_epochs=3 为建议值，待用户确认后写入训练脚本。

---

## 显存实验记录

| 实验 | max_seq_length | 实际序列长度 | 结果 | 备注 |
|------|---------------|------------|------|------|
| 推理（8B 4-bit） | 8192 | — | peak 5.80 GB | 余量约 2.2 GB |
| 训练 v1（OOM） | 2048 | ~1500t | peak 7.44 GB，fused CE OOM | warmup bug 同时存在 |
| 训练 v2（通过） | 1024 | ~880t avg | peak 约 6.74 GB | 20 条样本，5 steps，context=60c |
| 训练 v3（通过） | 1024 | ~900t avg | peak 约 6.82 GB | 544 条样本，136 steps，context=60c/1000c |
| **1536 探针（通过）** | **1536** | **~876t avg**（旧 60c 样本） | **peak 6.73 GB** | 57 条，3 steps，padding-free 自动启用 |

结论：
- 样本数增加主要影响训练时长，不显著改变单步显存峰值（Unsloth padding-free 按实际 token 数计算 VRAM）。
- 1536 探针的 6.73GB 是**下界估算**：用的是旧 60c 样本（最长 942t）；新 700c 样本（最长 1460t）实际峰值估算约 7.3GB，仍在 8GB 预算内。
- **建议**：v4 full-run 前用 `train_samples_full_57.jsonl` 再跑一次探针确认。

---

## 数据集

### 风丝引（v4 训练集，当前主线）

- 合并文件：`data/processed/train_samples_full_57.jsonl`（**v4 训练用**）
- 组成：
  - `data/processed/train_samples.jsonl`（ch2-21，20 条）
  - `data/processed/train_samples_ch22_58.jsonl`（ch22-58，37 条）
- 规模：57 条，ch2-58
- 构造入口：`pipeline/build_train_samples.py --raw-file data/raw/风丝引_原文.txt`
- 切分参数：`context_chars=700`，`completion_chars=200`，`bible_top_k=2`，`bible_max_chars=250`，`prior_max_chars=120`
- Token 分布（实测，Qwen3 tokenizer）：min=1335t，max=1460t，mean=1398t，0 条超过 1536t
- 时序口径：`target_chapter=N -> max_chapter=N-1`，57/57 ALL PASS
- 新增角色卡：`宁楚珣`（revealed_in=42，ch43+ 可见）、`大理相`（revealed_in=52，ch53+ 可见）
- 已知限制：大理相 ch53-55 BM25 miss（`_stem_priority` 机制，接受现状）

context_chars 历史对比：

| 版本 | context_chars | 推理场景覆盖率 | 备注 |
|------|--------------|-------------|------|
| v2 样本 | 60c | 3%（vs 2000c） | train/inference 分布严重不一致 |
| v4 样本 | **700c** | **35%** | 1536 max_seq_length 内最大化 context |

### novel2（已放弃，仅备档）

- 原文：`data/raw/novel2_raw.txt`
- 样本：`data/processed/novel2_samples.jsonl`（524 条）
- 合并：`data/processed/merged_train_samples.jsonl`（544 条）
- **状态**：v3 使用后角色错乱，内容分布与风丝引不兼容，**不再用于训练**，文件保留备档。

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

## 下一步（v4 训练前检查清单）

1. **用户确认超参**：`num_train_epochs=3`，`warmup_steps=2`（已写入 `pipeline/train_qlora.py`，标注 TBD）。
2. **v4 VRAM 探针**：用 `train_samples_full_57.jsonl`（最长 1460t）再跑一次 3-step 探针，预期峰值 ~7.3GB < 8GB。
   ```powershell
   .venv-train\Scripts\Activate.ps1
   python pipeline/train_qlora.py   # max_steps=3 探针
   ```
3. **full-run 训练**（探针通过后）：
   ```powershell
   python pipeline/train_qlora.py --full-run   # → outputs/qlora_run_v4/
   ```
4. **后训练评测**：
   ```powershell
   python pipeline/generate_lora_multi.py
   python scripts/eval_draft.py --candidate <候选文件> --config config.yaml --out-json v4_eval.json
   ```
5. **验收目标**：style_score > 60.48（v2 基准）。
