# Project Architecture

本文档记录当前小说续写项目的目录职责、数据流和路径约定。重点原则：代码入口放在项目根目录或包目录中，`data/` 只保存小说素材、中间产物和生成的设定集。

## Top-Level Layout

```text
LOCAL-LLM-NOVEL/
├─ config.yaml                 # 项目唯一默认配置入口
├─ cowriter/                   # 续写应用核心代码
│  ├─ app.py                   # CLI 入口
│  ├─ session.py               # 会话、生成、摘要压缩
│  ├─ retriever.py             # story_bible BM25 + raw grep 检索
│  └─ prompts.py               # 续写 prompt
├─ scripts/
│  └─ build_story_bible.py     # 从 raw txt 构建 story_bible
├─ data/
│  ├─ raw/                     # 原始小说 txt
│  ├─ processed/               # 后续预处理产物
│  ├─ story_bible/             # Retriever 读取的设定集 Markdown
│  └─ *.json                   # 临时分析结果或人工分析导出
├─ models/                     # 本地模型或导出权重
├─ outputs/                    # 续写草稿输出
└─ pipeline/                   # 训练/评估/导出相关脚本
```

## Data Directory Contract

`data/` 不再放运行脚本或独立配置。当前约定如下：

```text
data/
├─ raw/
│  ├─ .gitkeep
│  └─ 风丝引_原文.txt
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
└─ 分析结果_31-58章.json
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
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  story_bible: "data/story_bible"
  models: "models"
  outputs: "outputs"
```

这些路径被以下代码使用：

- `cowriter.retriever.Retriever`: 读取 `paths.story_bible` 下的 `*.md`，并用 `paths.raw_data` 做原文 grep。
- `cowriter.session.Session`: 使用 `paths.outputs` 保存草稿。
- `scripts/build_story_bible.py`: 读取 `paths.raw_data` 下的 txt，写入 `paths.story_bible`。

## Runtime Flow

```text
raw txt
  │
  ▼
scripts/build_story_bible.py
  │  章节/字符切分
  │  LLM 抽取 JSON
  │  合并去重
  ▼
data/story_bible/*.md
  │
  ▼
cowriter.retriever.Retriever
  │  BM25 检索设定集
  │  grep 检索原文
  ▼
cowriter.session.Session
  │
  ▼
cowriter.prompts.build_prompt
  │
  ▼
Ollama 续写
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
      ├─ 林清雪.md
      ├─ 凤倾汐.md
      └─ （共21个，每人一文件）
```

`Retriever._load_bible()` 使用 `rglob("*.md")` 递归扫描，`generated/` 下的文件会自动进入 BM25 索引。
注意：`characters.md` 保留不删，单人物拆分文件与之共存于同一索引，对出现频率较低的配角查询精度有提升。

## Retrieval Flow

`cowriter/retriever.py` 的行为保持不变：

1. 启动时扫描 `config["paths"]["story_bible"]` 下的所有 `*.md`。
2. 用 Markdown 文件内容建立 BM25 索引。
3. 续写时根据当前上文提取实体，检索相关设定。
4. 同时用 `rg` 或 Python fallback 在 `config["paths"]["raw_data"]` 下搜索原文命中段落。

因此，只要 `config.yaml` 中的 `paths.story_bible` 和 `paths.raw_data` 不变，整理 `data/` 内部的历史重复目录不会影响 Retriever。

## Impact Check

本次整理对运行代码没有破坏性影响：

- 根配置仍指向 `data/raw` 和 `data/story_bible`。
- 原始 txt 已放入 `data/raw/`，与 `paths.raw_data` 一致。
- story bible Markdown 仍保留在 `data/story_bible/`。
- `data/config/` 和 `data/scripts/` 没有任何代码引用；配置和脚本已有根目录版本。
- `scripts/build_story_bible.py` 仍保留对历史 `raw_data` 目录名的兼容探测，但当前主路径已经是 `data/raw`。

## Git Hygiene

`.gitignore` 已保护小说原文、生成设定集、模型权重和输出草稿。通常应提交代码、配置、文档和 `.gitkeep`，不要提交：

- `data/raw/*`
- `data/processed/*`
- `data/story_bible/*`
- `data/*.json`
- `models/*`
- `outputs/*`
