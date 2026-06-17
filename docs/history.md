# 项目阶段历史归档

本文档保存已完成阶段的决策、测试与代码改动记录。当前工作规程见根目录 `CLAUDE.md`。

---

## 阶段2.6 模型迁移决策记录（2026-06-16，已完成）

### 为什么弃用旧模型（R1 蒸馏 MoE 2X1.5B）

| 症状 | 根因（非调参可救） |
|------|-------------------|
| "好的，接下来我将按以下步骤编写" + 1./2./3. 列表 | R1 蒸馏的目标行为本身就是“先列计划再执行”，不适合小说续写 |
| 文风/人物/连贯性塌 | MoE 仅激活约 1.5B 参数，长篇叙事吃稠密参数量 |
| 英文乱码 token | Q4 量化 + 小模型 + 混合蒸馏导致词表/数值不稳 |
| 偶发空输出 | 同属模型不稳定性 |

结论：任务错配（推理型模型 ≠ 写作型模型），换模型，不在旧模型上继续调参。

### 新方向

- 模型切换到 `huihui_ai/qwen3-abliterated:8b-v2`。
- Qwen3 必须用 `think=False` 关思考，并在输出端兜底清洗 `<think>` 残留。
- prompt 从任务式改为补全式，用 assistant prefill 让模型顺着上文写正文。
- 生成参数最终以 `config.yaml` 为准；历史记录不固定 `repeat_penalty` 数值。

### 阶段2.6 测试清单（6/6 通过）

- [✓] `config.yaml` 模型名已切换为 `huihui_ai/qwen3-abliterated:8b-v2`
- [✓] `python -m cowriter.app` 用新模型可启动，不崩溃
- [✓] 单次生成输出为纯中文小说正文，无助手语、列表、`<think>`、英文乱码
- [✓] prefill 生效，续写与上文衔接自然
- [✓] 连续 5 次生成无空输出
- [✓] `/检索`、`/保存`、摘要压缩等阶段2命令正常

### 阶段2.6 已执行改动概要

- `cowriter/session.py`：增强 `_strip_think()`，新增 `_dedup_output()`，`_chat()` 接入输出清洗。
- `cowriter/prompts.py`：精简 `SYSTEM_PROMPT`，改为补全式续写 prefill。
- `config.yaml`：切换模型并调整写作采样参数，具体值以文件为准。
- `_test_phase26.py`：更新阶段2.6 回归断言。
- `_test_dedup.py`：补充去重实机与单元验证。

---

## 阶段3 确定性文风评测工具（2026-06-17，已完成）

阶段3目标是实现不调用 LLM 的确定性评测工具，用于比较 reference 与 candidate 的形式风格差异、重复风险和污染风险。

### 阶段3.1-3.3（2026-06-16）

- `pipeline/eval_style.py`：实现确定性文风评测 CLI，可读取 UTF-8 reference/candidate，输出 JSON、Markdown 或终端报告。
- 3.1：实现基础文本统计、重复风险基础检测、污染检测基础版、reference/candidate 差异摘要。
- 3.2：增强中文小说文本切分，支持中文/英文句末标点、引号闭合、多标点、段落空行边界和对话行识别；新增 `segmentation` 顶层字段。
- 3.3：增强 `repetition` 字段，支持重复行、重复段落、连续重复句、近似相邻句、短句循环、char 2/3/4-gram 与 `low/medium/high` 风险等级。
- `_test_eval_style.py`：新增轻量回归测试，覆盖切分、重复句、ABAB 短句循环、重复段落、近似相邻句、空 candidate、JSON 可解析和 Markdown 输出。

### 阶段3.4-3.8（2026-06-17）

- `pipeline/eval_style.py`：完成确定性文风评测工具，输出 `meta`、增强 `inputs`、基础统计、文本切分、重复风险、污染风险、`diff`、`style_score`、JSON/Markdown/终端摘要。
- 3.4：增强污染检测，覆盖精确/归一化/近似句子重合、char shingle、最长重合片段和段落级重合。
- 3.5：新增 reference vs candidate 形式风格差异评分 `style_score`，分为 `close/moderate/far/invalid`。
- 3.6：完善报告输出，稳定 JSON schema，优化 Markdown 结构，新增 `--verbose` 与 `--quiet`。
- 3.7：新增 `tests/fixtures/eval_style/` 固定回归样本，全部为人工假文本，不含真实小说原文或真实输出。
- 3.8：新增 `scripts/eval_draft.py` 独立 wrapper，可对已有草稿一键评测；不调用 LLM，不修改生成链路，不接入训练。
- `_test_eval_style.py`：改为优先读取 fixtures，并覆盖 schema、报告输出、wrapper、错误路径和嵌套输出目录。

### 阶段3最终交付物

- `pipeline/eval_style.py`
- `scripts/eval_draft.py`
- `_test_eval_style.py`
- `tests/fixtures/eval_style/`

### 阶段3常用命令

```powershell
python _test_eval_style.py
python -m py_compile pipeline\eval_style.py
python -m py_compile scripts\eval_draft.py
python scripts\eval_draft.py --reference <reference.txt> --candidate <candidate.txt>
python scripts\eval_draft.py --config config.yaml --candidate <candidate.txt>
```

---

## 系统A 时序过滤数据层（2026-06-17，代码完成，待用户验收）

续写 prompt 的五块结构已于阶段2.6落地，本次任务在此基础上加入章节可见性约束，防止未来信息泄漏进 prompt。

### 核心设计决策

- `max_chapter_for_target(N) = N-1`：写第 N 章时，最多可见 N-1 章信息。
- frontmatter 最小化：只用 `revealed_in: N`，不引入 `valid_from`/`valid_to`（过度工程）。
- "无 frontmatter → 不可见"：启用时序过滤时，缺标注的文件默认不可见，防意外泄漏。
- 聚合文件（characters/relationships/timeline/plot_threads/chapter_summaries）不加 frontmatter，temporal filter 下自动不可见；`get_prior_summaries` 直接读 `_merged_data.json` 不走 BM25。
- `grep_raw` 已知限制：搜全量 txt 无法按章节过滤，文档已注明，仅作文风参考。

### 交付物

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `cowriter/chapter.py` | 新建 | `max_chapter_for_target(N)` |
| `cowriter/retriever.py` | 修改 | frontmatter 解析、`search_bible(max_chapter)`、`get_prior_summaries()` |
| `cowriter/prompts.py` | 修改 | `build_prompt(prior_summary=)` 增加【前情提要】块 |
| `cowriter/session.py` | 修改 | `generate(target_chapter=N)` 接入时序口径 |
| `scripts/add_frontmatter.py` | 新建 | 一次性补存量 frontmatter，支持 `--dry-run` |
| `scripts/build_story_bible.py` | 修改 | world/style/glossary 生成函数自动前置 `revealed_in: 1` |
| `scripts/split_characters.py` | 修改 | 拆分时自动提取 `来源章节` 写入 `revealed_in: N` |
| `_test_temporal_filter.py` | 新建 | 7 个测试类，24 个 case，覆盖全闭环 |

### 待用户操作

1. `python scripts/add_frontmatter.py --dry-run` 预览，确认后去掉 `--dry-run` 执行一次。
2. `python _test_temporal_filter.py` 全绿即验收通过。

---

## 阶段4前置验证 — 评测基线建立（2026-06-17）

`huihui_ai/qwen3-abliterated:8b-v2` 零微调基线：`style_score 50.92/100`（level: far），`repetition_risk: high`，`contamination_risk: low`。候选文本为 5 次连续续写（2226 字），参考为 `data/raw/风丝引_原文.txt`（364151 非空白字符）。无文本内容的指标文件提交至 `baselines/phase4_pre/baseline_metrics.json`。微调后模型需在同一参考文本上超过此分数。
