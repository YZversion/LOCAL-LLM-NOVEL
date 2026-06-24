# 阶段4：QLoRA 微调实验记录

_最后更新：2026-06-24_

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
| v4 | 风丝引 57 条 | **1536** | **700** | **3** | **45** | **2** | `outputs/qlora_run_v4/` | 已训练，评测未通过 ✗ |

*v3 数据：novel1 用 context_chars=60，novel2 用 context_chars=1000（两种口径混入，是 v3 放弃原因之一）。

备注：
- v1 的 `warmup_steps=5` 等于总步数，LR 从未达到峰值。
- v2 的 `warmup_steps=1` 是针对 5 步小样本实验的特例。
- v3 样本数扩大到 544 后，按总步数约 5% 设置 `warmup_steps=7`，但 novel2 数据分布与风丝引不兼容，根本方向错误。
- v4 实际总步数 45（ceil(57/4)=15 steps/epoch × 3 epochs），warmup_steps=2 占 4.4%。

---

## 显存实验记录

| 实验 | max_seq_length | 实际序列长度 | 结果 | 备注 |
|------|---------------|------------|------|------|
| 推理（8B 4-bit） | 8192 | — | peak 5.80 GB | 余量约 2.2 GB |
| 训练 v1（OOM） | 2048 | ~1500t | peak 7.44 GB，fused CE OOM | warmup bug 同时存在 |
| 训练 v2（通过） | 1024 | ~880t avg | peak 约 6.74 GB | 20 条样本，5 steps，context=60c |
| 训练 v3（通过） | 1024 | ~900t avg | peak 约 6.82 GB | 544 条样本，136 steps，context=60c/1000c |
| **1536 探针（通过）** | **1536** | ~876t avg（旧 60c 样本） | peak 6.73 GB | 57 条，3 steps，padding-free 自动启用 |
| **v4 显存探针（通过）** | **1536** | **max 1460t**（新 700c 样本） | **peak 7.38 GB** | 57 条，3 steps，UNSLOTH_CE_LOSS_TARGET_GB=0.5 |
| **v4 full-run（通过）** | **1536** | max 1460t | **peak 7.38 GB** | 57 条，45 steps（3 epochs），adapter 已保存 |

结论：
- 样本数增加主要影响训练时长，不显著改变单步显存峰值（Unsloth padding-free 按实际 token 数计算 VRAM）。
- 1536 探针的 6.73GB 是**下界估算**：用的是旧 60c 样本（最长 942t）；新 700c 样本（最长 1460t）实际峰值估算约 7.3GB，仍在 8GB 预算内。
- v4 full-run 已完成，1536 + 700c 在训练态可用；推理态长上下文另见“推理入口与显存”。

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
| v4 | `outputs/qlora_run_v4/` | 未通过 | 57 条风丝引，45 steps，train_loss≈2.69，但生成不稳定 |

v2 loss：

```text
step 1: loss=3.795  LR=0
step 2: loss=4.022  LR=2e-4
step 3: loss=3.757  LR=1.5e-4
step 4: loss=3.530  LR=1e-4
step 5: loss=3.450  LR=5e-5
```

v4 loss（45 steps，3 epochs，风丝引 57 条）：

```text
epoch 1 末（step 15):  loss=2.993  epoch=1.00
epoch 2 末（step 30):  loss=2.682  epoch=2.00
epoch 3 末（step 45):  loss=2.223  epoch=3.00
train_loss ≈ 2.69  （45步均值）

特殊事件：step 12 grad_norm=1.774（约正常 3-4x），次步恢复正常，未影响收敛。
VRAM fix：UNSLOTH_CE_LOSS_TARGET_GB=0.5（绕开梯度 offload 导致的 _get_chunk_multiplier 误测）。
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

| 指标 | 基线 | v2 | v3 | v4 坏触发点 | v4 干净 ch1 起点 |
|------|------|----|----|-------------|------------------|
| 文件 | `draft_baseline_phase4.txt` | `lora_candidate_v2.txt` | `lora_candidate_v3.txt` | `adapter_candidate_v4_eval.txt` | `adapter_candidate_20260624_1114.txt` |
| style_score | 50.92 | 60.48 | 46.05 | 39.41 | 48.7361 |
| repetition_risk | high (55.6%) | medium (16.7%) | medium (0.0%) | medium | medium |
| contamination_risk | low | low | low | low | low |

v2 结论：
- style_score 比基线提升 +9.56，阶段4小样本链路通过。
- 主驱动力是 repetition_penalty 提升。

v3 结论：
- 扩样训练没有带来进一步提升，style_score 低于 v2 和基线。
- repetition 继续改善，但 sentence_profile 明显恶化。
- 关键异常：average_sentence_length `83.39`，longest_sentence_length `471`。

---

## v4 评测结论

### 坏触发点：凰后/凤倾汐 -> 叶欢

- 候选：`outputs/adapter_candidate_v4_eval.txt`
- 指标：`outputs/v4_eval_result.json`
- style_score：`39.41`
- 现象：从 ch58 附近“凰后/凤倾汐”结尾上文续写时，模型触发“凰后 -> 叶欢”强关联，切入叶欢修仙子线。
- 根因判断：训练集中叶欢子线占 57 条 completion 的约 33%，但对话占比过高、修仙细节描写稀薄；模型用基座通用玄幻词汇补细节。
- 训练数据外词汇：金丹、凝气、昆仑山脉等已确认不在训练 completion 中。

### 干净 ch1 起点复测

- 起始上文：`outputs/debug/test_context_ch1_clean.txt`
- 候选：`outputs/adapter_candidate_20260624_1114.txt`
- 指标：`outputs/v4_ch1_clean_eval.json`
- style_score：`48.7361`，仍低于基线 `50.92` 和 v2 `60.48`
- 人工核查问题：
  - 开头出现标题样文本 `【太阳穴】`，违反正文补全格式。
  - 擅自引入“帝后娘娘去世”、白衣仙子吹笛、落水少女、围观路人等剧情。
  - 出现“戴着眼镜的老者”等明显不合原著语境的现代感词汇。
  - 句子偏短、对白比例偏高，整体更像通用网文续写而非风丝引。

### v4 总结

v4 的训练 loss 与显存表现健康，但生成端不稳定。排除已知叶欢触发点后，基础续写仍不达标，因此 v4 当前配置不是“局部修叶欢线即可落地”的问题，而是 57 条 / 700c / 1536 / 3 epoch 方案整体需要回退或重规划。

---

## 推理入口与显存

为避免 Web 会话历史污染和内网/proxy 限制，已新增独立评测入口：

```powershell
.venv-train\Scripts\Activate.ps1
python pipeline/adapter_cli.py --adapter outputs/qlora_run_v4/ --context-file <context.txt> --max-seq-length 4096
python pipeline/adapter_cli.py --adapter outputs/qlora_run_v4/ --raw-prompt-file outputs/eval_anchors/ch1_clean.txt --max-seq-length 4096 --seed 1101
```

已知结论：

- 多轮 Web 会话会累积污染 `outputs/debug/last_prompt.txt`，评测必须用全新干净起点。
- 无 FA2 环境下，长上下文推理会遇到真实 O(n^2) 显存压力；此前“FA2 缺失影响很小”的判断只适合短序列。
- 当前安全推理配置：`max_seq_length=4096` + `MAX_RECENT_CHARS=800`。
- `/reject` 路径含 `gc.collect()` + `torch.cuda.empty_cache()`，用于释放显存后重试。
- `adapter_cli.py` 当前支持 `--raw-prompt-file` 和 `--seed`；raw 模式不经 Retriever / `build_prompt`，只用于 Stage 0 隔离 adapter 贡献。
- Stage 0 评测必须显式传固定 reference：`scripts/eval_draft.py --reference data/raw/风丝引_原文.txt ...`。`data/raw/` 下有多个 txt，不传 reference 会报错。
- Codex sandbox 用户对默认 HF cache `C:\Users\14390\.cache\huggingface\hub` 只有读权限，Unsloth import 会卡在 cache writable probe。Stage 0 命令需用进程级 `outputs/hf_stage0_proxy/` 作为 `HF_HOME/HF_HUB_CACHE/HF_XET_CACHE`；该 proxy 的 base model 目录用 junction 指向真实 HF cache。

---

## Stage 0 重锚评测（当前阻塞）

目的：用同一条 `adapter_cli` raw 管线重锚纯基座 / v2 / v4，隔离“adapter 本身贡献了什么”。baseline(Ollama 生产链)单独读，不和 adapter_cli 三元组求差。

### 冻结 anchor

`outputs/eval_anchors/` 下 4 个 anchor 已冻结并用 `anchors_manifest.json` sha 锁：

| id | source_chapter | register | in_training_range | 备注 |
|----|----------------|----------|-------------------|------|
| `ch1_clean` | 1 | clean_narration | false | 800c，训练区间外泛化点 |
| `ch58_bad_trigger` | 58 | palace_dialogue | true | 790c，结尾靠近“凰后/凤倾汐”触发点 |
| `mid_court_dialogue` | 56 | palace_dialogue | true | 798c，宫廷对白 |
| `yehuan_controlled` | 35 | xuanhuan_edge | true | 761c，叶欢线控制点，不含金丹/凝气/昆仑 |

这些文件不得重选、改字或追加；每次开跑先校验 sha。

### 固定运行口径

- `--max-seq-length 4096`
- `MAX_RECENT_CHARS=800`
- `--seed <int>` 必填并记录
- 采样参数读 `config.yaml`，HF generate 实际只生效 `temperature/top_p/top_k/repetition_penalty`
- 生成目标长度约 2226-2800 字，不够按既有多轮方式续接
- 每个候选必须跑 `scripts/eval_draft.py --reference data/raw/风丝引_原文.txt`

### 已跑 smoke / 判别实验

1. 纯基座 × `ch1_clean` × seed1101：
   - 候选：`outputs/adapter_candidate_20260624_1359.txt`
   - 生成成功，3 轮，2565c；加载 peak 5.77GB，生成 peak 约 6.06GB。
   - 加固定 reference 后 eval 成功：`style_score=60.8257`、`repetition_risk=medium`、`contamination_risk=low`。
   - 人工通读失败：输出退化成 instruct 助手腔，出现“这是一段...”、标题、Markdown 菜单和写作建议。

2. v2 adapter × `ch1_clean` × seed1101：
   - 只跑 1 轮用于判别，不保存候选文件。
   - 同样退化成说明文/助手腔：开头“这是一段充满诗意与情感的画面描写...”，含 `**《浮生一梦》**`，结尾“如果需要继续发展剧情...”。
   - 结论：退化不是纯基座固有行为，而是 raw prompt 构造问题。

### raw prompt 阻塞

当前 raw 模式实际送入模型的 prompt 结构为：

```text
<|im_start|>user
[800 字 anchor 原文]
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

这会让 instruct 模型把 anchor 当成待点评/扩写素材，而不是小说断点。下一步必须先最小修改 raw prompt 构造，加入明确任务：直接从断点续写小说正文，不要标题、分析、解释、写作建议或向用户提问。修复后先重跑 `v2 × ch1_clean × seed1101`；v2 恢复正文续写后才允许完整 3×4×2 fan-out。

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

### 最终决策

- v3 已被用户明确放弃，不再继续 repeat2 或相关评测。
- 放弃原因不是单纯 explicit_sensitive 占比，而是 novel2 分布与风丝引不兼容，并在 round2/3 出现角色错乱和区块链/AI 等技术术语。
- `outputs/qlora_run_v3/` 与相关输出保留备档，不删除，不作为后续主线。

---

## 下一步（QLoRA 重规划）

1. 停止 v3 repeat 诊断，不继续 novel2 合并数据线。
2. 不导出 v4，不接生产。
3. 先修 `adapter_cli.py` raw prompt 构造；修复后只重跑 `v2 × ch1_clean × seed1101` 判别。
4. 判别通过后再 fan-out Stage 0 三元组，重锚纯基座/v2/v4；旧的单点 v2 `60.48` 仍是历史可用锚点，但新判断以修 prompt 后同口径结果为准。
5. Stage 0 后再决定 QLoRA 回退/重规划是数据重构还是旋钮单变量阶梯。
6. 补训练/评测实验管理：固定 reference、prompt、采样参数、候选长度区间；每个 adapter 至少 repeat 2-3 次；记录 seed 或独立运行编号；纳入标点密度、平均句长、最长句 quick eval。
7. 在 System B 前可先做轻量 `retrieval_manifest.json`，提高“模型为什么知道这件事”的可见性。
