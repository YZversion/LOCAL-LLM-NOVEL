# 项目当前状态

_最后更新：2026-06-23_

---

## 项目快照

| 项目 | 值 |
|------|----|
| 工作目录 | `c:\Users\14390\Desktop\Code\LOCAL-LLM-NOVEL` |
| GitHub | `https://github.com/YZversion/LOCAL-LLM-NOVEL` |
| 推理模型 | `huihui_ai/qwen3-abliterated:8b-v2` |
| 微调基座 | `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` |
| 微调框架 | Unsloth QLoRA 4-bit（阶段4） |
| 显存预算 | 8GB（RTX 4070 Laptop） |
| CUDA / torch | CUDA Toolkit 13.x；torch 2.10.0+cu130；`.venv-train/` 隔离安装 |
| 数据保护 | `data/`、`models/`、`outputs/` 全部 gitignore |
| Python | 3.11.9 |
| 生成参数 | 以 `config.yaml` 为准 |

---

## 阶段路线图

```text
[✓] 阶段0    repo 骨架 + 环境配置文件
[✓] 阶段2    零训练合写回路
[✓] 阶段2.6  模型迁移 + 补全式续写改造
[ ] 阶段1    数据清洗（prepare_data.py，用户自有脚本）
[✓] 阶段3    确定性文风评测工具（2026-06-17 验收通过）
[✓] 系统A    时序过滤数据层（2026-06-17，45/45 测试全绿）
[✓] 系统A数据  chapter_summaries.md ch1-58 补全（2026-06-17）
[✓] 阶段4    小样本 QLoRA 验证链路（v2：style_score 60.48 > 基线 50.92）
[✗] 阶段4    v3（novel2 合并 544 条）放弃，角色错乱+分布不兼容，不再追究根因
[→] 阶段4    v4 准备中：57 条风丝引样本（ch2-58，700c context）待训练
[ ] 系统B    知识图谱 + story_bible 动态写回（延后）
[ ] 阶段5    向量 RAG（按需）
```

---

## 阶段4 当前结论

### 已通过

- 零微调基线：style_score `50.92`，repetition_risk `high`，contamination_risk `low`。
- LoRA v2：第一本 20 条样本，5 optimizer steps，style_score `60.48`，repetition_risk `medium`，contamination_risk `low`。
- v2 验收结论：小样本 LoRA 方法有效，主要改善来自 repetition_penalty。

### v3（已放弃）

- v3 adapter：`outputs/qlora_run_v3/`（保留备档，不删除）。
- v3 训练：novel2 合并 544 条，1 epoch，136 optimizer steps，`warmup_steps=7`，`max_seq_length=1024`。
- v3 首次评测：style_score `46.05`（低于 v2 和基线）；后续 round2/3 生成严重崩溃（角色错乱+混入技术术语）。
- **放弃决策**：novel2 内容分布与风丝引不兼容，根因不再追究，合并数据线彻底放弃。

### v4 准备中（当前主线）

**训练数据**：
- 数据集：`data/processed/train_samples_full_57.jsonl`，57 条，ch2-58，纯风丝引原文。
- 新增角色卡：`宁楚珣`（revealed_in=42）、`大理相`（revealed_in=52），时序验证全部通过。
- 样本参数：`context_chars=700`，`completion_chars=200`，`bible_top_k=2`，`bible_max_chars=250`，`prior_max_chars=120`。
- Token 分布（实测）：min=1335t，max=1460t，mean=1398t，**0 条超过 1536t**。

**训练参数**（已更新到 `pipeline/train_qlora.py`）：

| 参数 | 值 | 说明 |
|------|----|----|
| max_seq_length | **1536** | 1536 显存探针通过（峰值 6.73GB）|
| context_chars（样本） | **700** | 推理场景 2000c 的 35%（v2 时为 60c = 3%）|
| batch_size | 1 | 不变 |
| grad_accum | 4 | 不变 |
| steps/epoch | ~14 | floor(57/4) |
| num_train_epochs | **3**（TBD） | 建议值，42 steps 总量，待确认 |
| warmup_steps | **2**（TBD） | 建议值，~5% of 42 steps，待确认 |
| output_dir | `outputs/qlora_run_v4/` | |

**显存状态**：
- 1536 探针（旧 60c 数据，序列最长 942t）：峰值 6.73GB ✅
- 新 700c 数据（序列最长 1460t）：估算峰值 ~7.3GB，建议在 full-run 前再跑一次探针确认。

### 已知限制

- `generated/characters/` 下的角色卡（如大理相）在与 2 个以上根目录主角卡同时竞争 top-k 时会被系统性排出（`_stem_priority` 机制）。大理相 ch53-55 三条样本受影响（BM25 分数排名第 1 但仍被截断），已接受现状不处理。详见 `CLAUDE.md`。

---

## 系统 A（续写 prompt 结构）✅ 已实现

每次构造续写 prompt 时，按顺序包含：

1. **人物与设定**：从 `story_bible` 检索，按 `max_chapter` 过滤。
2. **前情提要**：前几章摘要，同样按 `max_chapter` 过滤。
3. **本章大纲**：用户手写的续写指令。
4. **上文**：紧邻续写点的最后一两段原文。
5. **续写**：模型从这里开始生成正文。

**章节时序口径（硬约束）**：续写第 N 章时 `max_chapter = N - 1`（`cowriter/chapter.py::max_chapter_for_target(N)`）。

可见性规则（`cowriter/retriever.py`）：

```python
revealed_in <= max_chapter and valid_from <= max_chapter
and (valid_to is None or valid_to >= max_chapter)
# 缺少 revealed_in 或 valid_from -> 不可见
```

当前数据状态：

- `chapter_summaries.md` 覆盖 1-58 章。
- `generated/characters/` 有 23 个单人物文件（含新增 宁楚珣、大理相），均含完整 frontmatter。
- `data/story_bible/` 可检索文件已补齐 frontmatter，共 31 个 BM25 索引文档。

---

## 系统 B（知识图谱，延后）

核心链路（待实现）：

```text
续写第 N 章 -> kg_extract.py -> kg.json -> kg_render.py -> .md 卡片 -> 第N+1章可检索
```

待实现脚本：`kg_extract.py` / `kg_update.py` / `kg_render.py` / `update_kg.py`

已补充角色（手写卡）：宁楚珣（ch42）、大理相（ch52）。
接受 bible=[] 的角色（出场 1 章、影响有限）：洛老太太、洛安、老太监、余挚。

详细设计见 `architecture.md`。

---

## 文件修改规范

| 文件 | 说明 |
|------|------|
| `config.yaml` | 唯一配置源；生成参数以此为准 |
| `cowriter/retriever.py` | 检索逻辑（含时序过滤） |
| `cowriter/prompts.py` | 提示词 |
| `cowriter/session.py` | 模型调用、摘要压缩、输出清洗 |
| `cowriter/chapter.py` | `max_chapter_for_target(N)` 时序口径 |
| `pipeline/eval_style.py` | 阶段3确定性评测核心 |
| `scripts/eval_draft.py` | 评测 wrapper |
| `pipeline/build_train_samples.py` | 第一本训练样本构造；必须遵守时序口径 |
| `pipeline/build_novel2_labeled_samples.py` | novel2 切分、source_section 与 content_sensitivity 打标 |
| `pipeline/merge_train_samples.py` | 合并 novel1/novel2 样本并补统一追踪字段 |
| `pipeline/train_qlora.py` | QLoRA 训练入口 |
| `pipeline/generate_lora_multi.py` | 多轮 LoRA 生成（真实推理链路） |
| `pipeline/export_gguf.py` | 导出入口，占位 |
| `architecture.md` | 系统架构文档 |
| `docs/history.md` | 阶段历史归档 |
