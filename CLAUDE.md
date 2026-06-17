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
| 推理模型 | `huihui_ai/qwen3-abliterated:8b-v2` |
| 微调框架 | Unsloth QLoRA 4-bit（阶段4，前置验证中） |
| 显存预算 | 8GB（RTX 4070 Laptop） |
| CUDA 驱动 | 13.2；torch cu130 wheel，`.venv-train/` 隔离安装 |
| 数据保护 | `data/`、`models/`、`outputs/` 全部 gitignore，素材文件绝不入库 |
| Python | 3.11.9 |
| 生成参数 | 以 `config.yaml` 为准 |

---

## 项目终极目标（必须先理解）

本项目最终目标是做一个**本地小说续写助手**。系统由两条线配合完成，但绝不能混淆：

### 系统 A：续写时的 prompt 结构（静态链路）✅ 时序过滤已实现

每次构造续写 prompt 时，按顺序包含五块：

1. **人物与设定**：从 `story_bible` 检索，按 `max_chapter` 过滤掉未来才揭晓的信息。
2. **前情提要**：读取前几章摘要，同样按 `max_chapter` 过滤。
3. **本章大纲**：用户手写的续写指令。
4. **上文**：紧邻续写点的最后一两段原文。
5. **续写**：模型从这里开始生成正文。

### 系统 B：会成长的记忆（动态闭环，未实现）

系统 B 在新章节写完后运行，用来把新增事实写回记忆：

```text
续写第 N 章
→ 抽取第 N 章中新出现的人物、关系、背景、情节
→ 标记章节号与证据
→ 写回 story_bible
→ 第 N+1 章开始可检索
```

系统 B 第一版形态（未来实现）：

```powershell
python scripts/update_kg.py --chapter 8 --input outputs/chapter_008.txt
```

### 章节时序口径（硬约束）

续写第 N 章时，`max_chapter = N - 1`（由 `cowriter/chapter.py` 的 `max_chapter_for_target(N)` 返回）。

`story_bible` 中每个可检索 `.md` 文件必须有 YAML frontmatter 声明章节可见范围：

```md
---
title: 林清雪
type: character
revealed_in: 1
valid_from: 1
valid_to: null
---
```

可见性规则（`cowriter/retriever.py` 已实现）：

```python
revealed_in <= max_chapter
and valid_from <= max_chapter
and (valid_to is None or valid_to >= max_chapter)

# 缺少 revealed_in 或 valid_from → 不可见（防意外泄漏）
```

存量文件补 frontmatter 使用一次性脚本：`scripts/add_frontmatter.py`。
当前 `data/story_bible/` 中可检索文件已补齐 `revealed_in` / `valid_from` / `valid_to`。
聚合文件（characters/relationships/timeline/plot_threads/chapter_summaries）不加 frontmatter，temporal filter 下自动不可见。

⚠️ 已知坑：`scripts/build_story_bible.py` 的 world/style/glossary 生成逻辑、`scripts/split_characters.py` 的单人物生成逻辑仍可能写出旧版最小 frontmatter（只有 `revealed_in`）。重跑这些脚本前，先修复它们生成完整 frontmatter，否则生成文件会在时序过滤下不可见。

---

## 阶段路线图与当前状态

```text
[✓] 阶段0    repo 骨架 + 环境配置文件
[✓] 阶段2    零训练合写回路
[✓] 阶段2.6  模型迁移 + 补全式续写改造
[ ] 阶段1    数据清洗（prepare_data.py，用户自有脚本）
[✓] 阶段3    确定性文风评测工具（2026-06-17 验收通过）
[✓] 系统A    时序过滤数据层（2026-06-17 验收通过，43/43 测试全绿）
[✓] 系统A数据  chapter_summaries.md ch1-58 补全（2026-06-17 完成）
[ ] 系统B    知识图谱 + story_bible 动态写回  ← 当前焦点
[ ] 阶段4    QLoRA 微调（系统B稳定后）
[ ] 阶段5    向量 RAG（按需）
```

**当前焦点：系统B — 知识图谱驱动的 story_bible 动态更新。**

目标：续写第59章时 story_bible 里有完整的 ch22-58 角色信息；写完后自动更新。

当前项目体检（2026-06-17）：
- `chapter_summaries.md` 已覆盖 1-58 章。
- `generated/characters/` 当前有 21 个单人物文件。
- 当前可检索 `.md` 已检查具备 `revealed_in` / `valid_from` / `valid_to`。
- `data/story_bible/kg.json` 尚不存在；`scripts/kg_extract.py`、`scripts/kg_update.py`、`scripts/kg_render.py`、`scripts/update_kg.py` 尚未创建。

---

## 系统B 设计（知识图谱）

详见 `architecture.md`。核心链路：

```text
① 补存量（ch22-58 缺失角色）
   原文 + JSON分析 → kg_extract.py → kg.json → kg_render.py → .md 卡片

② 续写后更新（写完第N章后运行）
   python scripts/update_kg.py --chapter N --input outputs/chapter_N.txt
   → LLM 抽取新角色/关系变化/状态变更 → 合并进 kg.json
   → 重新渲染受影响角色 .md → 第N+1章检索器自动可见

③ 续写前查询（System A，已实现）
   retrieve(max_chapter=N-1) → BM25 检索 .md 文件（无需改动）
```

数据文件：`data/story_bible/kg.json`（gitignore，不入库）

系统B 当前任务清单：
- [ ] 修复 `build_story_bible.py` / `split_characters.py` 的完整 frontmatter 生成，避免重跑后让卡片不可见
- [ ] `scripts/kg_extract.py` — LLM 从章节文本抽取实体和关系
- [ ] `scripts/kg_update.py` — 合并新实体进 kg.json，处理冲突与状态时间线
- [ ] `scripts/kg_render.py` — 从 kg.json 渲染 .md 卡片（含 frontmatter）
- [ ] `scripts/update_kg.py` — 三步合一入口：extract → update → render
- [ ] 补全 ch22-58 缺失角色（宁楚珣、洛老太太、大理相、洛安、老太监、余挚）
- [ ] 验证：检索器能找到补全后的角色

---

## 阶段4前置测试清单（暂缓，系统B完成后恢复）

- [x] 确认当前运行依赖安装正常：`python _test_eval_style.py`（2026-06-17 通过）
- [x] 确认阶段3评测 wrapper 可作为训练前基线工具（2026-06-17 通过；style_score 50.92/100）
- [x] 确认训练依赖方案：`requirements-train.txt` 已创建，`.venv-train/` 隔离
- [x] `scripts/add_frontmatter.py` 运行完毕，29 个文件已补 frontmatter（2026-06-17）
- [x] `python _test_temporal_filter.py` 43/43 全绿（2026-06-17）
- [ ] 修复 `.venv-train/` torch CPU 降级（`pip install torch==2.10.0 ... --index-url https://download.pytorch.org/whl/cu130`）
- [ ] 跑 `_test_unsloth_forward.py`（Qwen2.5-0.5B）确认 CUDA + Unsloth 可用
- [ ] 跑 `_test_unsloth_forward.py --model Qwen/Qwen3-8B-Instruct` 确认显存峰值 < 8GB
- [ ] 确认导出 GGUF 与 Ollama 加载流程的最小验证路径
- [ ] 确认微调模型 style_score > 50.92

---

## 文件修改规范

| 文件 | 说明 |
|------|------|
| `config.yaml` | 唯一配置源；生成参数以此为准 |
| `cowriter/app.py` | CLI 入口；非明确任务不改 |
| `cowriter/web.py` | Gradio UI 入口；非明确任务不改 |
| `cowriter/retriever.py` | 检索逻辑（含时序过滤）；非明确任务不改 |
| `cowriter/prompts.py` | 提示词；非明确任务不改 |
| `cowriter/session.py` | 模型调用、摘要压缩、输出清洗；非明确任务不改 |
| `cowriter/chapter.py` | `max_chapter_for_target(N)` 时序口径工具函数 |
| `pipeline/eval_style.py` | 阶段3确定性评测核心 |
| `scripts/eval_draft.py` | 阶段3评测 wrapper |
| `scripts/add_frontmatter.py` | 存量 story_bible 补完整 frontmatter（已运行完毕，勿重复）|
| `scripts/build_story_bible.py` | 从 raw txt 构建 story_bible；重跑前需修复 world/style/glossary 的完整 frontmatter |
| `scripts/split_characters.py` | 拆分人物 Markdown；重跑前需修复单人物完整 frontmatter |
| `scripts/gen_chapter_summaries.py` | 从原文批量生成章节摘要追加到 chapter_summaries.md |
| `scripts/kg_extract.py` | 【系统B】LLM 从章节文本抽取实体/关系（待实现） |
| `scripts/kg_update.py` | 【系统B】合并新实体进 kg.json（待实现） |
| `scripts/kg_render.py` | 【系统B】从 kg.json 渲染 .md 卡片（待实现） |
| `scripts/update_kg.py` | 【系统B】三步合一入口（待实现） |
| `data/story_bible/kg.json` | 【系统B】知识图谱主文件（gitignore，不入库） |
| `pipeline/train_qlora.py` | 阶段4训练入口，占位 |
| `pipeline/export_gguf.py` | 阶段4导出入口，占位 |
| `architecture.md` | 系统架构文档 |
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
系统B 当前 task：...，验收状态：通过/未通过
```
