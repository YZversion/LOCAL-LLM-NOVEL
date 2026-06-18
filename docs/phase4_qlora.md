# 阶段4：QLoRA 微调实验记录

_最后更新：2026-06-18_

---

## 环境

| 项目 | 值 |
|------|----|
| venv | `.venv-train/`（与主 venv 隔离） |
| torch | 2.10.0+cu130 |
| unsloth | 2026.6.7 |
| CUDA Toolkit | 13.0（sm_89） |
| 基座模型 | `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` |
| 激活命令 | `.venv-train\Scripts\Activate.ps1` |

---

## 训练参数

| 参数 | 值 | 来源 |
|------|----|------|
| lora_r | 16 | Unsloth README |
| lora_alpha | 16 | Unsloth（alpha=r） |
| lora_dropout | 0 | Unsloth（优化要求 0） |
| target_modules | q/k/v/o/gate/up/down_proj | Unsloth Qwen3 标准 |
| use_gradient_checkpointing | "unsloth" | Unsloth 专用，比 True 省显存 |
| optim | adamw_8bit | 节省优化器状态 |
| learning_rate | 2e-4 | Unsloth README 标准起步 |
| grad_accum | 4 | 有效 batch=4 |
| warmup_steps | 1 | v2 修复（v1 bug：warmup=5 = total steps，LR 从未达到峰值） |
| max_seq_length（训练）| 1024 | OOM 降级；2048 在 fused CE 阶段 OOM |
| batch_size | 1 | |
| bf16 | True | sm_89 支持 |

---

## 显存实验记录

| 实验 | max_seq_length | 结果 | 备注 |
|------|---------------|------|------|
| 推理（8B 4-bit）| 8192 | peak 5.80 GB ✅ | 余量 2.2 GB |
| 训练 v1（OOM）| 2048 | peak 7.44 GB，fused CE OOM | warmup bug，LR 从未达到 2e-4 |
| 训练 v2（通过）| 1024 | peak 6.74 GB ✅ | warmup_steps=1 修复 |

---

## 训练样本

- 规模：20 条，ch2-21
- 构造入口：`pipeline/build_train_samples.py`
- 时序口径：`target_chapter=N → max_chapter=N-1`
- 验证：`_test_train_samples.py` 20/20 全绿
- 样本参数：`context_chars=1000, completion_chars=600`
- 存储：`data/processed/train_samples.jsonl`（gitignore）

---

## LoRA Adapter 版本

| 版本 | 路径 | 备注 |
|------|------|------|
| v1（归档）| `outputs/qlora_run/` | warmup bug，LR 从未达 2e-4，loss 3.593 |
| v2（当前）| `outputs/qlora_run_v2/` | warmup_steps=1 修复，loss 3.795→3.450 |

v2 训练 loss 曲线：
```
step 1: loss=3.795  LR=0       (warmup)
step 2: loss=4.022  LR=2e-4    (峰值)
step 3: loss=3.757  LR=1.5e-4
step 4: loss=3.530  LR=1e-4
step 5: loss=3.450  LR=5e-5
```

---

## 小样本验证结果（2026-06-18）

生成脚本：`pipeline/generate_lora_multi.py`（多轮，真实推理 prompt 链路）
候选文件：`outputs/lora_candidate_v2.txt`（2395c，4 轮生成）
评测命令：`python scripts/eval_draft.py --candidate outputs/lora_candidate_v2.txt --config config.yaml --out-json outputs/lora_v2_eval.json`
脱敏指标：`baselines/phase4_pre/lora_v2_metrics.json`

| 指标 | 基线（零微调）| LoRA v2 | 差值 |
|------|------------|---------|------|
| style_score | 50.92 | **60.48** | **+9.56** |
| length_profile | 0.15/20 | 0.16/20 | +0.01 |
| sentence_profile | 19.94/25 | 21.90/25 | +1.96 |
| paragraph_profile | 3.75/15 | 4.41/15 | +0.66 |
| dialogue_profile | 14.08/15 | 15.00/15 | +0.92 |
| **repetition_penalty** | 3.00/15 | **9.00/15** | **+6.00** |
| contamination_penalty | 10.00/10 | 10.00/10 | — |
| repetition_risk | high (55.6%) | medium (16.7%) | ↑ 改善 |
| contamination_risk | low | low | 维持 |

**验收结论**：style_score +9.56，超过基线，方法有效。
**主驱动力**：repetition_penalty +6.0，重复段落大幅减少。
**归因**：瓶颈是样本量（20条），不是方法或配置问题。

---

## 下一步选项

- **A. 扩大样本**：50-200条，目标 style_score > 65
- **B. 导出 GGUF**：`pipeline/export_gguf.py`（占位），接入 Ollama 合写主回路
- **C. System B 优先**：补全 ch22-58 缺失角色，再做更大规模微调
