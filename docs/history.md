# 项目阶段历史归档

本文档保存已完成阶段的决策、测试与代码改动记录。当前工作规程见根目录 `CLAUDE.md`。

---

## 本地成品化：v2 终端 UI + System B MVP（2026-06-24）

- 决定以历史可用 adapter `outputs/qlora_run_v2/` 作为今天的本地成品入口；v3/v4 保留为实验记录，不导出、不接生产。
- 新增 `scripts/run_v2_local_ui.ps1`：内网环境下不走 Web / Gradio，直接调用 `.venv-train` + `pipeline/adapter_cli.py`，默认 adapter 为 v2，并设置进程级 HF cache proxy `outputs/hf_stage0_proxy/`。
- 新增 System B MVP 脚本：
  - `scripts/kg_extract.py`：从已接受章节文本生成可人工编辑的 facts 草稿。
  - `scripts/kg_update.py`：把已审核 facts 合并进 `data/story_bible/kg.json`。
  - `scripts/kg_render.py`：把 `kg.json` 渲染成 Retriever 可读的 Markdown cards。
  - `scripts/update_kg.py`：合并 + 渲染的一键入口。
- 新增 `_test_system_b.py`，验证 `kg.json -> Markdown cards -> Retriever BM25 + frontmatter` 链路，以及 `target_chapter=N -> max_chapter=N-1` 的时序隔离。
- 文档同步：`README.md`、`CLAUDE.md`、`architecture.md`、`docs/current_state.md`、`docs/phase4_qlora.md`。

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

---

## 阶段4 QLoRA 小样本验证与扩样诊断（2026-06-18 至 2026-06-22）

### v2 小样本验证通过

- 数据：第一本 20 条样本，`data/processed/train_samples.jsonl`。
- 训练：1 epoch，5 optimizer steps，`warmup_steps=1`，`max_seq_length=1024`。
- adapter：`outputs/qlora_run_v2/`。
- 评测：`outputs/lora_candidate_v2.txt`，脱敏指标 `baselines/phase4_pre/lora_v2_metrics.json`。
- 结果：style_score `60.48`，高于基线 `50.92`；repetition_risk 从 `high` 改善为 `medium`；contamination_risk 保持 `low`。
- 结论：QLoRA 方法链路有效，下一步瓶颈转向样本规模。

### novel2 数据扩充

- 原始文件为 GBK/GB18030 编码，已转换为 UTF-8：`data/raw/novel2_raw.txt`。
- 结构确认：主线、番外、卷四延续部分均保留。
- 切分入口：`pipeline/build_novel2_labeled_samples.py`。
- 输出：`data/processed/novel2_samples.jsonl` 与 `data/processed/novel2_labels.jsonl`。
- 样本数：524 条。
- source_section：main 418、extras 21、vol4 85。
- content_sensitivity：explicit_sensitive 290、mature_nonexplicit 156、general 78。
- 标签文件只存 ID、标签与 confidence，不存原文片段。

### 合并数据集

- 合并入口：`pipeline/merge_train_samples.py`。
- 输出：`data/processed/merged_train_samples.jsonl`。
- 总数：544 条，novel1 20 + novel2 524。
- 统一追踪字段：`merged_sample_id`、`source_book`、`source_sample_id`、`source_section`、`source_section_confidence`、`content_sensitivity`、`content_sensitivity_confidence`。
- 验证：novel1 20 条 `messages/completion` 与原始文件逐条字段级一致；novel2 标签无缺失、无错配。

### v3 扩样训练未通过

- 数据：合并数据集 544 条。
- 训练：1 epoch，136 optimizer steps，`warmup_steps=7`，`max_seq_length=1024`。
- adapter：`outputs/qlora_run_v3/`。
- 显存：forward/backward peak 约 6.82GB。
- 结果：style_score `46.05`，低于 v2 `60.48` 和基线 `50.92`。
- repetition_risk 保持 `medium`，重复率指标继续改善；contamination_risk 保持 `low`。
- 主要问题：sentence_profile 大幅下降，average_sentence_length `83.39`，longest_sentence_length `471`。

### v3 诊断结论

- 最长句异常不是 `pipeline/eval_style.py` 分句逻辑漏切；候选文本本身缺少正常标点。
- 训练数据 completion 按 `content_sensitivity` 分组统计后，explicit_sensitive 的句末标点密度约 `3.05/100字`，general 约 `3.18/100字`，mature_nonexplicit 约 `3.22/100字`。
- 组间差异只有 4%-8%，不足以解释 v3 生成端 66%-86% 的标点密度退化。
- 当前未能确认 v3 失败来自训练保存时机还是生成采样偶发异常；当时曾计划完成同 adapter 的重复生成诊断。此计划已被后续 v3 放弃决策覆盖，见下一节。

---

## 阶段4 v4 训练与评测结论（2026-06-23 至 2026-06-24）

### v3 最终放弃

- v3 后续 round2/3 生成严重崩溃，表现为角色错乱、混入区块链/AI 等与两本小说都无关的技术术语，以及超长无标点堆叠句。
- 用户明确决定不再追究 v3 根因，停止 repeat 诊断，不继续 novel2 合并数据线。
- `outputs/qlora_run_v3/` 和 `data/processed/merged_train_samples.jsonl` 保留备档，不删除、不再作为主线。

### v4 训练完成

- 数据：57 条风丝引样本，ch2-21 原有 20 条 + ch22-58 新增 37 条，统一使用 `context_chars=700`。
- 新增角色卡：宁楚珣（revealed_in=42）、大理相（revealed_in=52），训练样本时序验证 57/57 通过。
- 配置：`max_seq_length=1536`，`num_train_epochs=3`，`warmup_steps=2`。
- 显存修复：设置 `UNSLOTH_CE_LOSS_TARGET_GB=0.5`，绕开 fused CE 的 mem_get_info 测量时机导致的 OOM。
- 结果：45 optimizer steps 完成，loss 从 3.758 降到 2.223，adapter 保存到 `outputs/qlora_run_v4/`。

### v4 评测未通过

- 坏触发点评测：`outputs/adapter_candidate_v4_eval.txt` / `outputs/v4_eval_result.json`，style_score `39.41`。从 ch58 附近“凰后/凤倾汐”上文续写时触发“凰后 -> 叶欢”强关联，切入叶欢修仙子线，并出现训练数据外的通用玄幻词汇。
- 干净 ch1 起点复测：`outputs/adapter_candidate_20260624_1114.txt` / `outputs/v4_ch1_clean_eval.json`，style_score `48.7361`，仍低于基线 `50.92` 和 v2 `60.48`。
- 人工核查：候选出现标题样文本、擅自跳剧情、乱加人物、现代感词汇，说明 v4 不是单点叶欢线问题，而是当前配置整体不稳定。

### 新路线

1. 先回退或重规划 QLoRA，让基础续写稳定到至少接近 v2。
2. 同时可先加轻量 debug：`outputs/debug/retrieval_manifest.json`。
3. 再做 System B MVP：`kg.json -> Markdown cards -> BM25`。
4. 等生成模型稳定后，再让 System B 承担长篇记忆闭环。

---

## 阶段4 Stage 0 重锚评测准备与 raw prompt 阻塞（2026-06-24）

### 固定评测资产

- 已创建并冻结 `outputs/eval_anchors/` 下 4 个 anchor：`ch1_clean`、`ch58_bad_trigger`、`mid_court_dialogue`、`yehuan_controlled`。
- `anchors_manifest.json` 记录每个 anchor 的 source chapter、register、训练区间标记和 sha256；后续运行前必须校验 sha，不得重选或修改 anchor。
- `pipeline/adapter_cli.py` 已具备 Stage 0 raw 管线能力：`--raw-prompt-file`、`--seed`、`MAX_RECENT_CHARS=800`，raw 模式跳过 Retriever / `build_prompt`。
- Stage 0 评测 reference 锁定为 `data/raw/风丝引_原文.txt`。由于 `data/raw/` 下有多个 txt，`scripts/eval_draft.py` 必须显式传 `--reference data/raw/风丝引_原文.txt`。

### HF cache 运行规避

- 诊断发现 Codex sandbox 用户对默认 HF cache `C:\Users\14390\.cache\huggingface\hub` 只有读权限。
- Unsloth import 会在 `unsloth_zoo/hf_cache.py::_is_writable()` 中用 `tempfile.NamedTemporaryFile(dir=hub_cache)` 探测可写性，因此会卡在默认 cache。
- 用进程级 `outputs/hf_stage0_proxy/` 作为 `HF_HOME/HF_HUB_CACHE/HF_XET_CACHE` 后，repo id 可离线解析；base model 目录用 junction 指向真实 HF cache，避免复制 16GB 权重。
- 最小加载诊断通过：`huihui-ai/Huihui-Qwen3-8B-abliterated-v2` 可由 `.venv-train` / Unsloth 离线加载 4-bit，加载约 15s，VRAM peak 约 5.77GB。

### smoke 与判别实验

- 纯基座 × `ch1_clean` × seed1101 raw smoke：
  - 生成成功，候选 `outputs/adapter_candidate_20260624_1359.txt`，3 轮，2565c。
  - 显存安全：加载 peak 5.77GB，生成 peak 约 6.06GB。
  - 加固定 reference 后 eval 成功：`style_score=60.8257`、`repetition_risk=medium`、`contamination_risk=low`。
  - 人工通读失败：候选明显是 instruct 助手腔，出现“这是一段...”、标题、Markdown 菜单和写作建议。
- v2 adapter × `ch1_clean` × seed1101 raw 判别：
  - 只跑 1 轮用于判别。
  - 同样输出说明文/助手腔，开头“这是一段充满诗意与情感的画面描写...”，含 `**《浮生一梦》**`，结尾“如果需要继续发展剧情...”。
  - 结论：退化不是纯基座固有行为，而是 raw prompt 构造问题。

### 当前停点

Stage 0 暂停，不允许 fan-out 到纯基座/v2/v4 × 4 anchors × 2 repeats。下一步必须先最小修改 raw prompt 构造：raw 模式仍不走 retrieval，但要明确要求模型从断点直接续写小说正文，禁止标题、分析、解释、写作建议和向用户提问。修复后先重跑 `v2 × ch1_clean × seed1101`；只有 v2 恢复正文续写后，才继续 Stage 0 完整评测。
