# 项目当前状态

_最后更新：2026-06-18_

---

## 项目快照

| 项目 | 值 |
|------|----|
| 工作目录 | `c:\Users\14390\Desktop\Code\LOCAL-LLM-NOVEL` |
| GitHub | `https://github.com/YZversion/LOCAL-LLM-NOVEL` |
| 推理模型 | `huihui_ai/qwen3-abliterated:8b-v2` |
| 微调框架 | Unsloth QLoRA 4-bit（阶段4） |
| 显存预算 | 8GB（RTX 4070 Laptop） |
| CUDA | 13.2；torch cu130 wheel，`.venv-train/` 隔离安装 |
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
[✓] 阶段4    小样本验证链路（2026-06-18，style_score 60.48 > 基线 50.92）
[ ] 阶段4    扩大样本量训练 / 导出GGUF / 接入Ollama（下一步，用户待决策）
[ ] 系统B    知识图谱 + story_bible 动态写回（延后）
[ ] 阶段5    向量 RAG（按需）
```

---

## 系统 A（续写 prompt 结构）✅ 已实现

每次构造续写 prompt 时，按顺序包含五块：
1. **人物与设定**：从 `story_bible` 检索，按 `max_chapter` 过滤
2. **前情提要**：前几章摘要，同样按 `max_chapter` 过滤
3. **本章大纲**：用户手写的续写指令
4. **上文**：紧邻续写点的最后一两段原文
5. **续写**：模型从这里开始生成正文

**章节时序口径（硬约束）**：续写第 N 章时 `max_chapter = N - 1`（`cowriter/chapter.py::max_chapter_for_target(N)`）

可见性规则（`cowriter/retriever.py`）：
```python
revealed_in <= max_chapter and valid_from <= max_chapter
and (valid_to is None or valid_to >= max_chapter)
# 缺少 revealed_in 或 valid_from → 不可见
```

当前数据状态：
- `chapter_summaries.md` 覆盖 1-58 章
- `generated/characters/` 有 21 个单人物文件，均含完整 frontmatter
- `data/story_bible/` 可检索文件已补齐 frontmatter

---

## 系统 B（知识图谱，延后）

核心链路（待实现）：
```text
续写第 N 章 → kg_extract.py → kg.json → kg_render.py → .md 卡片 → 第N+1章可检索
```

待实现脚本：`kg_extract.py` / `kg_update.py` / `kg_render.py` / `update_kg.py`

待补充角色（ch22-58）：宁楚珣、洛老太太、大理相、洛安、老太监、余挚

详细设计见 `architecture.md`。

---

## 文件修改规范

| 文件 | 说明 |
|------|------|
| `config.yaml` | 唯一配置源；生成参数以此为准 |
| `cowriter/retriever.py` | 检索逻辑（含时序过滤）|
| `cowriter/prompts.py` | 提示词 |
| `cowriter/session.py` | 模型调用、摘要压缩、输出清洗 |
| `cowriter/chapter.py` | `max_chapter_for_target(N)` 时序口径 |
| `pipeline/eval_style.py` | 阶段3确定性评测核心 |
| `scripts/eval_draft.py` | 评测 wrapper |
| `pipeline/build_train_samples.py` | 训练样本构造；必须遵守时序口径 |
| `pipeline/train_qlora.py` | QLoRA 训练入口 |
| `pipeline/generate_lora_multi.py` | 多轮 LoRA 生成（真实推理链路） |
| `pipeline/export_gguf.py` | 导出入口，占位 |
| `architecture.md` | 系统架构文档 |
| `docs/history.md` | 阶段历史归档 |
