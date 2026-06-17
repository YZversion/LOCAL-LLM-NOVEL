# 本地小说续写助手架构

本文档记录当前小说续写项目的目录职责、运行链路、路径约定和阶段边界。当前基线是阶段2.6完成后的零训练本地合写系统：Ollama + Qwen3-8B 非思考模式、补全式 prompt、story_bible 检索、输出清洗与去重。

核心原则：

- `config.yaml` 是唯一默认配置入口。
- `cowriter/` 放交互式续写应用，不承载训练或评测流水线。
- `pipeline/` 放阶段1/3/4相关的数据、评测、训练、导出脚本。
- `data/` 只保存素材、中间产物和生成设定集，不放运行脚本。

## Top-Level Layout

```text
LOCAL-LLM-NOVEL/
├─ CLAUDE.md                   # Agent 工作规程、阶段状态和验收口径
├─ architecture.md             # 当前架构说明
├─ config.yaml                 # 项目唯一默认配置入口
├─ cowriter/                   # 续写应用核心代码
│  ├─ app.py                   # CLI 入口
│  ├─ web.py                   # Gradio Web UI 入口
│  ├─ session.py               # 会话、生成、摘要压缩
│  ├─ retriever.py             # story_bible BM25 + raw grep 检索
│  └─ prompts.py               # 补全式续写 prompt 与摘要 prompt
├─ scripts/
│  ├─ build_story_bible.py     # 从 raw txt 构建 story_bible
│  ├─ split_characters.py      # 将人物汇总拆成单人物 Markdown
│  └─ eval_draft.py            # 阶段3：对已有草稿运行确定性评测的 wrapper
├─ data/
│  ├─ raw/                     # 原始小说 txt
│  ├─ processed/               # 后续预处理产物
│  ├─ story_bible/             # Retriever 读取的设定集 Markdown
│  └─ *.json                   # 临时分析结果或人工分析导出
├─ models/                     # 本地模型或导出权重
├─ outputs/                    # 续写草稿输出
├─ pipeline/                   # 数据清洗、评估、训练、导出相关脚本
│  ├─ prepare_data.py          # 阶段1占位：用户自有清洗脚本入口
│  ├─ eval_style.py            # 阶段3：确定性文风评测 CLI
│  ├─ train_qlora.py           # 阶段4占位：QLoRA 训练
│  └─ export_gguf.py           # 阶段4占位：GGUF 导出
├─ tests/
│  └─ fixtures/eval_style/     # 阶段3固定假文本回归样本，不含真实小说原文
└─ _test_*.py                  # 阶段性回归/实机验证脚本，含 _test_eval_style.py
```

## Data Directory Contract

`data/` 不再放运行脚本或独立配置。当前约定如下：

```text
data/
├─ raw/
│  ├─ .gitkeep
│  └─ <novel_source>.txt
├─ processed/
│  └─ .gitkeep
├─ story_bible/
│  ├─ .gitkeep
│  ├─ characters.md
│  ├─ world.md
│  ├─ timeline.md
│  ├─ plot_threads.md
│  ├─ chapter_summaries.md
│  ├─ relationships.md
│  ├─ style.md
│  ├─ glossary.md
│  └─ _merged_data.json
└─ <analysis_result>.json
```

目录职责：

- `data/raw/`: 原始小说正文。`config.yaml` 的 `paths.raw_data` 指向这里，Retriever 的原文 grep 也从这里找。
- `data/processed/`: 预处理后的训练数据、切片数据或清洗结果。当前代码未强依赖。
- `data/story_bible/`: 设定集 Markdown。Retriever 会读取这里的 `*.md` 建 BM25 索引。
- `data/*.json`: 人工分析、模型分析或调试导出。默认不入 git。

已清理的历史目录：

- `data/raw_data/`: 原文已移入 `data/raw/`，与根配置保持一致。
- `data/config/`: 未被代码引用，默认配置统一使用根 `config.yaml`。
- `data/scripts/`: 未被代码引用，脚本统一放在根 `scripts/`。

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

- `cowriter.retriever.Retriever`: 读取 `paths.story_bible` 下的 `*.md`，并用 `paths.raw_data` 做原文 grep。
- `cowriter.session.Session`: 使用 `paths.outputs` 保存草稿。
- `scripts/build_story_bible.py`: 读取 `paths.raw_data` 下的 txt，写入 `paths.story_bible`。

生成参数由 `cowriter.session.Session._chat()` 读取，并传给 `ollama.chat(..., think=False)`。DRY、presence/frequency penalty、repeat penalty 等重复抑制参数以 `config.yaml` 为准；本文档不固定具体数值。

## Runtime Flow

```text
CLI: python -m cowriter.app
Web: python -m cowriter.web
  │
  ▼
cowriter.session.Session
  │  accepted_text / summary / outputs/debug
  ▼
cowriter.retriever.Retriever
  │  story_bible BM25
  │  raw txt grep
  ▼
cowriter.prompts.build_prompt
  │  system prompt
  │  retrieved context
  │  current context
  │  assistant prefill
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

`outputs/debug/` 会保存最近一次请求、prompt 和响应拆分结果，用于排查 prompt、thinking、content 或采样参数问题。

## Generation Contract

阶段2.6后，生成链路的稳定性约束集中在 `cowriter/session.py` 和 `cowriter/prompts.py`：

- `prompts.py` 负责把任务改造成“正文补全”：相关设定、摘要、原文命中段落在前，当前上文在最后，末尾追加 assistant prefill。
- `session.py` 调用 Ollama 时固定 `think=False`，只取 `message.content`，不拼接 `message.thinking`。
- `_strip_think()` 兜底剥离 `<think>`、`/no_think`、`/no` 残片和独占行助手语。
- `_dedup_output()` 在输出端截断短句循环和大段单次复读。
- 续写结果只有在用户接受或手动替换后才进入 `accepted_text`，随后可能触发滚动摘要压缩。

## Stage 3 Evaluation Boundary（已完成）

阶段3已于 2026-06-17 验收完成，目标是实现确定性文风评测工具，不接入 LLM 评审、不训练模型、不修改生成链路。核心实现是 `pipeline/eval_style.py`；便捷入口是 `scripts/eval_draft.py`；固定回归样本位于 `tests/fixtures/eval_style/`，全部为人工构造假文本，不含真实小说原文或真实输出草稿。

阶段3工具保持与生成链路解耦：

- 输入：reference 文本与 candidate 文本，默认 UTF-8。
- 输出：Markdown 报告、JSON 报告、终端短摘要。
- 核心指标：基础统计、文本切分、重复风险、污染风险、`diff`、`style_score`。
- 边界：不修改 `cowriter.session`、`cowriter.retriever`、`cowriter.prompts`、`cowriter.web`；`scripts/eval_draft.py` 只复用评测工具读取已有文件，不调用 Ollama，不改变 accepted_text，不保存草稿。

当前 CLI：

```powershell
python pipeline/eval_style.py --reference path/to/reference.txt --candidate path/to/candidate.txt
python pipeline/eval_style.py --reference ref.txt --candidate cand.txt --out-json outputs/eval_style_report.json --out-md outputs/eval_style_report.md
python scripts/eval_draft.py --reference data/raw/<novel_source>.txt --candidate outputs/draft_xxx.txt
python scripts/eval_draft.py --config config.yaml --candidate outputs/draft_xxx.txt
```

当前 JSON 顶层字段保持稳定：

```text
meta
inputs
reference_stats
candidate_stats
segmentation
repetition
contamination
diff
summary
style_score
```

已实现能力：

- `reference_stats` / `candidate_stats`: 字符数、非空白字符数、行数、段落数、句子数、平均/中位/最长句长、对话行数与比例。
- `segmentation`: reference/candidate 的段落数、句子数、对话行数、平均每段句子数。
- `repetition`: 重复行、重复段落、连续重复句、近似相邻句、短句循环、char 2/3/4-gram 与 `low/medium/high` 风险等级。
- `contamination`: 精确/归一化/近似句子重合、char shingle、最长连续重合片段、段落级重合和风险等级。
- `diff`: 字符数、句数、段落数、句长、对话比例、每段句数等 reference/candidate 差异指标。
- `style_score`: 0-100 的形式风格接近度评分，level 为 `close/moderate/far/invalid`，不代表文学质量。

回归测试入口：

```powershell
python _test_eval_style.py
python -m py_compile pipeline/eval_style.py
python -m py_compile scripts/eval_draft.py
```

## Story Bible Build

Dry run：

```powershell
python scripts/build_story_bible.py --config config.yaml --dry-run
```

实际构建：

```powershell
python scripts/build_story_bible.py --config config.yaml
```

测试少量 chunk：

```powershell
python scripts/build_story_bible.py --config config.yaml --limit-chunks 3 --verbose
```

构建脚本会生成：

```text
data/story_bible/
├─ characters.md              # 所有人物汇总（build_story_bible.py 生成）
├─ world.md
├─ timeline.md
├─ plot_threads.md
├─ chapter_summaries.md
├─ relationships.md
├─ style.md
├─ glossary.md
├─ _merged_data.json
├─ .build_cache/              # 断点续跑缓存，不参与检索
└─ generated/
   └─ characters/             # split_characters.py 生成的单人物文件
      ├─ <character_a>.md
      ├─ <character_b>.md
      └─ （共21个，每人一文件）
```

`Retriever._load_bible()` 使用 `rglob("*.md")` 递归扫描，`generated/` 下的文件会自动进入 BM25 索引。
注意：`characters.md` 保留不删，单人物拆分文件与之共存于同一索引，对出现频率较低的配角查询精度有提升。

## Retrieval Flow

`cowriter/retriever.py` 的当前行为：

1. 启动时扫描 `config["paths"]["story_bible"]` 下的所有 `*.md`。
2. 用 Markdown 文件内容建立 BM25 索引。
3. 续写时根据当前上文提取实体，检索相关设定。
4. 同时用 `rg` 或 Python fallback 在 `config["paths"]["raw_data"]` 下搜索原文命中段落。

因此，只要 `config.yaml` 中的 `paths.story_bible` 和 `paths.raw_data` 不变，调整其它目录不会影响 Retriever。

## Change Boundaries

当前阶段的主要边界：

- 阶段3只实现评测工具，不改变生成链路。
- `cowriter/app.py` 和 `cowriter/web.py` 是 UI 层；除非任务明确要求，不在评测阶段改动。
- `cowriter/session.py` 是模型调用、摘要压缩、输出清洗的边界；后续评测结果即使接入，也应保持可选。
- `cowriter/retriever.py` 只依赖 `paths.story_bible` 与 `paths.raw_data`，不应读取 `outputs/` 或评测产物。
- `pipeline/` 脚本可以读取 `data/processed`、`outputs` 或用户指定文件，但不应隐式修改 `data/raw`。

## Git Hygiene

`.gitignore` 已保护小说原文、生成设定集、模型权重和输出草稿。通常应提交代码、配置、文档和 `.gitkeep`，不要提交：

- `data/raw/*`
- `data/processed/*`
- `data/story_bible/*`
- `data/*.json`
- `models/*`
- `outputs/*`
