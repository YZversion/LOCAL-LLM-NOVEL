# 本地小说续写助手架构

_最后更新：2026-06-22_

本文档记录当前小说续写项目的目录职责、运行链路、路径约定和阶段边界。当前运行基线是阶段3完成后、系统A时序过滤落地后的状态：Ollama + Qwen3-8B 非思考模式、补全式 prompt、story_bible 时序过滤检索、前情提要注入、输出清洗与去重。当前训练主线处于阶段4 QLoRA 扩样诊断中：v2 小样本通过，v3 扩样训练未通过，正在定位训练/保存/生成稳定性问题。

核心原则：

- `config.yaml` 是唯一默认配置入口。
- `cowriter/` 放交互式续写应用，不承载训练或评测流水线。
- `pipeline/` 放阶段1/3/4相关的数据、评测、训练、导出脚本。
- `data/` 只保存素材、中间产物和生成设定集，不放运行脚本。

## Top-Level Layout

```text
LOCAL-LLM-NOVEL/
├─ CLAUDE.md                   # Agent 工作规程、阶段状态和验收口径
├─ architecture.md             # 当前架构说明（本文件）
├─ config.yaml                 # 项目唯一默认配置入口
├─ requirements.txt            # 主运行依赖
├─ requirements-train.txt      # 阶段4训练依赖（.venv-train/ 隔离安装）
├─ cowriter/                   # 续写应用核心代码
│  ├─ app.py                   # CLI 入口
│  ├─ web.py                   # Gradio Web UI 入口
│  ├─ session.py               # 会话、生成（含 target_chapter）、摘要压缩
│  ├─ retriever.py             # story_bible BM25 + 时序过滤 + raw grep 检索
│  ├─ prompts.py               # 补全式续写 prompt 与摘要 prompt（含前情提要块）
│  └─ chapter.py               # max_chapter_for_target(N) 时序口径工具函数
├─ scripts/
│  ├─ build_story_bible.py     # 从 raw txt 构建 story_bible；world/style/glossary 写完整 frontmatter
│  ├─ split_characters.py      # 将人物汇总拆成单人物 Markdown；单人物卡片写完整 frontmatter
│  ├─ add_frontmatter.py       # 一次性：给存量 story_bible .md 补完整 frontmatter
│  ├─ gen_chapter_summaries.py # 从原文生成缺失章节摘要，追加到 chapter_summaries.md
│  └─ eval_draft.py            # 阶段3：对已有草稿运行确定性评测的 wrapper
├─ data/
│  ├─ raw/                     # 原始小说 txt（gitignore）
│  ├─ processed/               # 后续预处理产物（gitignore）
│  ├─ story_bible/             # Retriever 读取的设定集 Markdown（gitignore）
│  └─ *.json                   # 临时分析结果（gitignore）
├─ models/                     # 本地模型或导出权重（gitignore）
├─ outputs/                    # 续写草稿输出（gitignore）
├─ baselines/
│  └─ phase4_pre/
│     ├─ baseline_metrics.json # 阶段4前置评测基线（无原文，可入库）
│     ├─ lora_v2_metrics.json  # v2 脱敏指标
│     └─ lora_v3_metrics.json  # v3 脱敏指标
├─ pipeline/                   # 数据清洗、评估、训练、导出相关脚本
│  ├─ prepare_data.py          # 阶段1占位：用户自有清洗脚本入口
│  ├─ eval_style.py            # 阶段3：确定性文风评测 CLI
│  ├─ build_train_samples.py   # 阶段4：构造遵守章节口径的 QLoRA 训练样本
│  ├─ build_novel2_labeled_samples.py # 阶段4：novel2 切分、source_section 与 content_sensitivity 打标
│  ├─ merge_train_samples.py   # 阶段4：合并 novel1/novel2 样本并补统一追踪字段
│  ├─ train_qlora.py           # 阶段4：训练态显存实测与小样本 QLoRA 训练
│  ├─ generate_lora.py         # 阶段4：用 LoRA adapter 生成候选文本供评测
│  ├─ generate_lora_multi.py   # 阶段4：多轮 LoRA 生成，复用真实推理 prompt 链路
│  └─ export_gguf.py           # 阶段4占位：GGUF 导出
├─ tests/
│  └─ fixtures/eval_style/     # 阶段3固定假文本回归样本，不含真实小说原文
└─ _test_*.py                  # 阶段性回归/实机验证脚本
   # _test_eval_style.py       阶段3
   # _test_temporal_filter.py  系统A时序过滤（45 case）
   # _test_unsloth_forward.py  阶段4前置：CUDA + Unsloth 实机验证
   # _test_train_samples.py    阶段4：训练样本时序过滤与章节对齐
```

## Data Directory Contract

```text
data/
├─ raw/
│  ├─ .gitkeep
│  └─ <novel_source>.txt
├─ processed/
│  ├─ .gitkeep
│  ├─ train_samples.jsonl          # novel1 20 条训练样本（gitignore）
│  ├─ novel2_samples.jsonl         # novel2 524 条样本（gitignore）
│  ├─ novel2_labels.jsonl          # novel2 标签；只含 ID/标签/confidence（gitignore）
│  └─ merged_train_samples.jsonl   # novel1+novel2 合并训练集，544 条（gitignore）
└─ story_bible/
   ├─ .gitkeep
   ├─ _merged_data.json        # build_story_bible.py 生成；当前仅作保留产物
   ├─ characters.md            # 聚合文件，无 frontmatter，temporal filter 下不可见
   ├─ world.md                 # 当前数据含 revealed_in/valid_from/valid_to
   ├─ timeline.md              # 聚合文件，无 frontmatter，temporal filter 下不可见
   ├─ plot_threads.md          # 聚合文件，无 frontmatter，temporal filter 下不可见
   ├─ chapter_summaries.md     # 聚合文件；get_prior_summaries() 直接解析并按章节过滤
   ├─ relationships.md         # 聚合文件，无 frontmatter，temporal filter 下不可见
   ├─ style.md                 # 当前数据含 revealed_in/valid_from/valid_to
   ├─ glossary.md              # 当前数据含 revealed_in/valid_from/valid_to
   ├─ <手写卡片>.md             # 当前数据含 revealed_in/valid_from/valid_to
   └─ generated/
      └─ characters/           # split_characters.py 生成
         └─ <人物名>.md         # 当前数据含 revealed_in/valid_from/valid_to
```

目录职责：

- `data/raw/`: 原始小说正文。`config.yaml` 的 `paths.raw_data` 指向这里，Retriever 的原文 grep 也从这里找。阶段4训练素材也放在这里，但不入库。
- `data/processed/`: 预处理后的训练数据、切片数据、标签和合并数据集。训练脚本可显式读取这里的 JSONL，但交互式续写运行时不依赖这里。
- `data/story_bible/`: 设定集 Markdown。Retriever 会读取这里的 `*.md` 建 BM25 索引。`chapter_summaries.md` 由 `get_prior_summaries()` 直接解析并按章节号过滤；`_merged_data.json` 目前只是构建保留产物。

## Config Paths

默认配置来自根目录 [config.yaml](config.yaml)：

```yaml
model:
  provider: ollama
  ollama_model: "huihui_ai/qwen3-abliterated:8b-v2"

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  story_bible: "data/story_bible"
  models: "models"
  outputs: "outputs"

generation:
  temperature: 0.8
  top_p: 0.8
  top_k: 20
  repeat_penalty: ...  # 以 config.yaml 为准
```

这些路径被以下代码使用：

- `cowriter.retriever.Retriever`: 读取 `paths.story_bible` 下的 `*.md`，直接解析 `chapter_summaries.md`，并用 `paths.raw_data` 做原文 grep。
- `cowriter.session.Session`: 使用 `paths.outputs` 保存草稿。
- `scripts/build_story_bible.py`: 读取 `paths.raw_data` 下的 txt，写入 `paths.story_bible`。

## Runtime Flow

```text
CLI: python -m cowriter.app
Web: python -m cowriter.web
  │
  ▼
cowriter.session.Session.generate(instruction, target_chapter=N)
  │
  ├─ cowriter.chapter.max_chapter_for_target(N) → max_chap = N-1
  │
  ├─ cowriter.retriever.Retriever.retrieve(context, max_chapter=max_chap)
  │    ├─ search_bible(query, max_chapter=max_chap)
  │    │    ├─ _visible(i): revealed_in/valid_from/valid_to 窗口检查
  │    │    ├─ 实体名精确/弱匹配提权（过滤后）
  │    │    └─ BM25 排序（过滤后）
  │    └─ grep_raw(entity)  ← 不受 max_chap 约束，仅文风参考
  │
  ├─ cowriter.retriever.Retriever.get_prior_summaries(max_chap)
  │    └─ 解析 chapter_summaries.md，只返回 chapter_number <= max_chap 的摘要
  │
  ▼
cowriter.prompts.build_prompt(
    recent_text, summary, retrieval,
    instruction, prior_summary=prior_summary
)
  │  ① 【相关设定】（已时序过滤）
  │  ② 【前情提要】（已时序过滤）
  │  ③ 【剧情摘要】（会话内滚动压缩）
  │  ④ 【原文命中段落】
  │  ⑤ 续写方向（可选）
  │  ⑥ 【当前上文】+ assistant prefill
  │
  ▼
Ollama /api/chat
  │  model: huihui_ai/qwen3-abliterated:8b-v2
  │  think=False
  │
  ▼
_strip_think() → _dedup_output()
  │
  ▼
用户接受 / 替换 / 拒绝 / 保存
```

`target_chapter` 不提供时，`max_chap=None`，时序过滤不启用，行为与旧版一致。

`outputs/debug/` 会保存最近一次请求、prompt 和响应拆分结果，用于排查 prompt、thinking、content 或采样参数问题。

## Temporal Filtering（系统A，已实现）

### 章节时序口径

```python
# cowriter/chapter.py
def max_chapter_for_target(target_chapter: int) -> int:
    return target_chapter - 1
```

写第 N 章 → 只能看见 `revealed_in <= N-1` 的 story_bible 条目和章节摘要。

### Frontmatter 规范

每个可检索的 `.md` 文件应在文件头写：

```yaml
---
title: <名称>
type: <character/location/worldbuilding/style/glossary/misc>
revealed_in: <int>   # 该信息最早在第几章揭晓
valid_from: <int>    # 该信息从第几章开始有效
valid_to: null       # 若后续失效，写最后有效章节；否则 null
---
```

缺少 `revealed_in` 或 `valid_from` 的文件，`max_chapter` 启用时默认不可见（防意外泄漏）。

### 文件类别与 frontmatter 策略

| 文件类型 | frontmatter | 原因 |
|----------|-------------|------|
| `world.md` / `style.md` / `glossary.md` | `revealed_in: 1` / `valid_from: 1` / `valid_to: null` | 全时段世界观设定 |
| `generated/characters/<人物名>.md` | `revealed_in: <来源章节最小值>` / `valid_from: <同值>` / `valid_to: null` | 自动从 `来源章节` 字段提取 |
| 手写根目录卡片 | `revealed_in: 1` / `valid_from: 1` / `valid_to: null` | 保守默认（用户可手动调整） |
| 聚合文件（characters/relationships/timeline/plot_threads/chapter_summaries） | 无 | 全量信息，temporal filter 下不可见；前情提要由 `get_prior_summaries()` 单独处理 |

### 已知限制

- `grep_raw` 搜索全量原文 txt，不受 `max_chapter` 约束；返回结果仅供文风参考，不含关键设定。

## Generation Contract

阶段2.6 后，生成链路的稳定性约束集中在 `cowriter/session.py` 和 `cowriter/prompts.py`：

- `prompts.py` 负责把任务改造成"正文补全"：设定、前情提要、摘要、原文命中段落在前，当前上文在最后，末尾追加 assistant prefill。
- `session.py` 调用 Ollama 时固定 `think=False`，只取 `message.content`，不拼接 `message.thinking`。
- `_strip_think()` 兜底剥离 `<think>`、`/no_think`、`/no` 残片和独占行助手语。
- `_dedup_output()` 在输出端截断短句循环和大段单次复读。
- 续写结果只有在用户接受或手动替换后才进入 `accepted_text`，随后可能触发滚动摘要压缩。

## Stage 3 Evaluation（已完成）

阶段3已于 2026-06-17 验收完成，目标是实现确定性文风评测工具，不接入 LLM 评审、不训练模型、不修改生成链路。

当前 CLI：

```powershell
python pipeline/eval_style.py --reference ref.txt --candidate cand.txt
python scripts/eval_draft.py --reference data/raw/<novel>.txt --candidate outputs/draft_xxx.txt
python scripts/eval_draft.py --config config.yaml --candidate outputs/draft_xxx.txt
```

JSON 顶层字段（稳定）：`meta` / `inputs` / `reference_stats` / `candidate_stats` / `segmentation` / `repetition` / `contamination` / `diff` / `summary` / `style_score`

回归测试：

```powershell
python _test_eval_style.py
```

## Stage 4 QLoRA（当前优先级）

阶段4目标是验证本地 8B QLoRA 是否能稳定改善形式文风指标，并明确微调与记忆系统的边界。训练依赖放在 `requirements-train.txt` 与 `.venv-train/`，不污染主运行环境。

当前基线与版本：

- 零微调基线：style_score `50.92`，repetition_risk `high`，contamination_risk `low`。
- v2：novel1 20 条样本，5 optimizer steps，`outputs/qlora_run_v2/`，style_score `60.48`，小样本链路通过。
- v3：merged 544 条样本，136 optimizer steps，`outputs/qlora_run_v3/`，style_score `46.05`，扩样训练未通过。
- v3 显存：`max_seq_length=1024` 时 forward/backward peak 约 `6.82GB`；样本数增加主要影响训练时长，不改变单步显存量级。

阶段4数据流：

```text
novel1:
  pipeline/build_train_samples.py
  -> data/processed/train_samples.jsonl

novel2:
  data/raw/novel2_raw.txt
  -> pipeline/build_novel2_labeled_samples.py
  -> data/processed/novel2_samples.jsonl
  -> data/processed/novel2_labels.jsonl

merge:
  pipeline/merge_train_samples.py
  -> data/processed/merged_train_samples.jsonl

train:
  pipeline/train_qlora.py
  -> outputs/qlora_run_v*/

evaluate:
  pipeline/generate_lora_multi.py
  -> outputs/lora_candidate_*.txt
  -> scripts/eval_draft.py
  -> outputs/*_eval.json
  -> baselines/phase4_pre/*_metrics.json（脱敏纯指标）
```

样本契约：

- novel1 样本保留真实推理结构相关字段，必须遵守 `target_chapter=N -> max_chapter=N-1`。
- novel2 只用于文风学习，不接入 story_bible 检索，不要求时序 frontmatter。
- 合并样本不压平原始字段，只新增统一追踪字段：`merged_sample_id`、`source_book`、`source_sample_id`、`source_section`、`source_section_confidence`、`content_sensitivity`、`content_sensitivity_confidence`。
- 标签文件不得保存原文片段，只保存 ID、标签和 confidence。

当前诊断状态：

- v3 失败主要来自句长异常与标点密度退化，而不是 repetition 或 contamination。
- 训练数据按 `content_sensitivity` 分组的标点密度差异很小，已排除 explicit_sensitive 占比作为主要解释。
- 当前需要完成同一个 v3 adapter 的重复生成诊断；若低标点密度稳定重现，再讨论 best checkpoint / 训练轮次 / 保存逻辑。

验收边界：

- QLoRA 负责文风、节奏、表达习惯和大纲遵循。
- QLoRA 不负责动态记住新角色；记忆缺口由 `story_bible` / System B 单独验收。
- 微调评估时，不把人物记忆缺失当作文风微调失败；也不把文风指标失败归因给记忆系统。

## Story Bible Build

```powershell
# 构建 story_bible（world/style/glossary 写完整 frontmatter）
python scripts/build_story_bible.py --config config.yaml

# 拆分人物（单人物卡片写完整 frontmatter）
python scripts/split_characters.py

# 给存量手写卡片补 frontmatter（一次性，之后幂等）
python scripts/add_frontmatter.py --dry-run   # 先预览
python scripts/add_frontmatter.py             # 确认后执行
```

构建后目录结构见 [Data Directory Contract](#data-directory-contract)。

`Retriever._load_bible()` 使用 `rglob("*.md")` 递归扫描，`generated/` 下的文件会自动进入 BM25 索引。

当前实际 `data/story_bible/` 中可检索 `.md` 已补齐 `revealed_in` / `valid_from` / `valid_to`。`build_story_bible.py` 的 world/style/glossary 生成函数、`split_characters.py` 的单人物生成函数也已修复，重跑后仍会写完整 temporal frontmatter。

## Retrieval Flow

`cowriter/retriever.py` 的当前行为：

1. 启动时扫描 `config["paths"]["story_bible"]` 下的所有 `*.md`，解析 frontmatter，记录 `revealed_in`。
2. 用 Markdown body（不含 frontmatter）建立 BM25 索引。
3. 续写时根据当前上文提取实体，带 `max_chapter` 参数检索相关设定（时序过滤）。
4. 同时用 `rg` 或 Python fallback 在 `config["paths"]["raw_data"]` 下搜索原文命中段落（不受 `max_chapter` 约束）。
5. 另外从 `chapter_summaries.md` 中解析章节摘要，按 `chapter_number <= max_chapter` 过滤，返回前情提要文本。

## System B Memory（延后）

系统B是记忆闭环：用知识图谱或结构化写回驱动 `story_bible` 动态更新，而不是靠微调“记住”新事实。当前不作为主线推进；等阶段4微调链路确认能跑通、能提升文风后，再决定是否实现完整 `kg.json` 方案，或退回更轻的 BM25 + 结构化卡片方案。

当前状态：

- `data/story_bible/kg.json` 尚不存在。
- `scripts/kg_extract.py`、`scripts/kg_update.py`、`scripts/kg_render.py`、`scripts/update_kg.py` 尚未创建。
- 后续目标是补齐 ch22-58 缺失角色，再支持写完第 N 章后把新增事实写回，并在第 N+1 章可检索。

计划链路：

```text
补存量：
  原文 / 分析 JSON
  → kg_extract.py
  → data/story_bible/kg.json
  → kg_render.py
  → data/story_bible/generated/.../*.md（含完整 frontmatter）

续写后：
  python scripts/update_kg.py --chapter N --input outputs/chapter_N.txt
  → 抽取新增人物 / 关系 / 状态变化 / 证据
  → 合并 kg.json
  → 重新渲染受影响 Markdown 卡片
  → 下一章 retrieve(max_chapter=N) 自动可见
```

系统B脚本可以写入 `data/story_bible/kg.json` 和受影响的 story_bible Markdown 卡片，但必须保持 `revealed_in` / `valid_from` / `valid_to` 完整，并继续遵守 `data/` 不入 git 的素材保护规则。

## Change Boundaries

- `cowriter/app.py` 和 `cowriter/web.py` 是 UI 层；除非任务明确要求，不在评测或训练阶段改动。
- `cowriter/session.py` 是模型调用、摘要压缩、输出清洗的边界；后续评测结果即使接入，也应保持可选。
- `cowriter/retriever.py` 只依赖 `paths.story_bible` 与 `paths.raw_data`，不应读取 `outputs/` 或评测产物。
- `pipeline/` 脚本可以读取 `data/processed`、`outputs` 或用户指定文件，但不应隐式修改 `data/raw`。
- 阶段4训练依赖放在 `requirements-train.txt` + `.venv-train/`，不污染 `requirements.txt` 和主 venv。

## Git Hygiene

`.gitignore` 已保护小说原文、生成设定集、模型权重和输出草稿。通常应提交代码、配置、文档和 `.gitkeep`，不要提交：

- `data/raw/*`
- `data/processed/*`
- `data/story_bible/*`
- `data/*.json`
- `models/*`
- `outputs/*`
- `.venv-train/`
- `unsloth_compiled_cache/`
