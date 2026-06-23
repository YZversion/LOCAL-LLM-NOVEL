# 项目当前状态

_最后更新：2026-06-22_

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
[!] 阶段4    扩样 QLoRA v3 评测未通过，正在定位训练/保存/生成稳定性问题
[ ] 系统B    知识图谱 + story_bible 动态写回（延后）
[ ] 阶段5    向量 RAG（按需）
```

---

## 阶段4 当前结论

### 已通过

- 零微调基线：style_score `50.92`，repetition_risk `high`，contamination_risk `low`。
- LoRA v2：第一本 20 条样本，5 optimizer steps，style_score `60.48`，repetition_risk `medium`，contamination_risk `low`。
- v2 验收结论：小样本 LoRA 方法有效，主要改善来自 repetition_penalty。

### 扩样数据

- novel2 已转 UTF-8：`data/raw/novel2_raw.txt`。
- novel2 结构：主线 418 条样本、番外 21 条、卷四 85 条，共 524 条。
- novel2 标签：`explicit_sensitive=290`、`mature_nonexplicit=156`、`general=78`。
- 合并数据集：`data/processed/merged_train_samples.jsonl`，共 544 条。
- 合并验证：novel1 20 条 `messages/completion` 与原始 `train_samples.jsonl` 逐条字段级一致。

### v3 训练与评测

- v3 adapter：`outputs/qlora_run_v3/`。
- v3 训练：544 条，1 epoch，136 optimizer steps，`warmup_steps=7`，`max_seq_length=1024`。
- v3 显存：forward/backward peak 约 `6.82GB`，与 v2 单步显存同量级。
- v3 首次评测：style_score `46.05`，低于 v2 `60.48` 和基线 `50.92`；repetition_risk `medium`，contamination_risk `low`。
- 主要退化：平均句长 `83.39`，最长句 `471`，标点密度明显低于 v2。

### 已排除的假设

- 471 字最长句不是 `pipeline/eval_style.py` 分句错误；候选文本本身缺少正常标点。
- 训练 completion 按 `content_sensitivity` 分组后，explicit_sensitive 标点密度只比其他组低约 4%-8%，不足以解释 v3 生成端 66%-86% 的标点密度退化。
- 当前不能把 v3 失败简单归因为 novel2 explicit_sensitive 占比。

### 当前卡点

- 需要用同一个 v3 adapter 重复生成 2 次，判断标点退化是否稳定重现。
- `outputs/lora_candidate_v3_repeat1.txt` 已生成（约 3336c），repeat2 尚未完成。
- 在重复生成诊断完成前，不应直接修改 checkpoint 保存逻辑或重新训练。

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
- `generated/characters/` 有 21 个单人物文件，均含完整 frontmatter。
- `data/story_bible/` 可检索文件已补齐 frontmatter。

---

## 系统 B（知识图谱，延后）

核心链路（待实现）：

```text
续写第 N 章 -> kg_extract.py -> kg.json -> kg_render.py -> .md 卡片 -> 第N+1章可检索
```

待实现脚本：`kg_extract.py` / `kg_update.py` / `kg_render.py` / `update_kg.py`

待补充角色（ch22-58）：宁楚珣、洛老太太、大理相、洛安、老太监、余挚

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
