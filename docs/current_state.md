# 项目当前状态

_最后更新：2026-06-24_

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
[✗] 阶段4    v4（风丝引 57 条）训练完成但评测未通过，基础续写不稳定
[→] 阶段4    Stage 0 重锚评测：先修 adapter_cli raw prompt 构造
[ ] 阶段4    修好 raw prompt 后再 fan-out 纯基座/v2/v4 × 4 anchors × 2 repeats
[✓] 本地UI    `scripts/run_v2_local_ui.ps1` 使用 `outputs/qlora_run_v2/`，不走 Web
[ ] Debug    retrieval_manifest.json（检索注入可视化，System B 前置）
[✓] 系统B MVP kg.json -> Markdown cards -> BM25（人工审核 facts，已可测试）
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

### v4（已训练，未通过）

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
| num_train_epochs | **3** | 实际 45 steps（ceil(57/4)=15 steps/epoch × 3）|
| warmup_steps | **2** | 4.4% of 45 steps |
| output_dir | `outputs/qlora_run_v4/` | |

**显存状态**：
- v4 探针（700c 数据，序列最长 1460t）：峰值 **7.38 GB** ✅（UNSLOTH_CE_LOSS_TARGET_GB=0.5）
- v4 full-run：峰值同探针，**7.38 GB** ✅，adapter 已保存。

**评测结论**：
- 坏触发点评测：`outputs/adapter_candidate_v4_eval.txt` / `outputs/v4_eval_result.json`，style_score `39.41`。从 ch58 附近“凰后/凤倾汐”上文续写时触发“凰后 -> 叶欢”强关联，切入叶欢修仙子线，并用训练数据外的通用玄幻词汇（金丹/凝气/昆仑山脉等）填充。
- 干净 ch1 起点复测：`outputs/adapter_candidate_20260624_1114.txt` / `outputs/v4_ch1_clean_eval.json`，style_score `48.7361`，仍低于基线 `50.92` 和 v2 `60.48`。人工核查发现标题样文本、擅自跳剧情、乱加人物、现代感词汇等问题。
- **结论**：v4 当前配置整体不稳定，不能导出 GGUF 或接生产；下一步不是局部修叶欢线，而是回退或重规划 QLoRA。

### 当前下一步

1. 先修 `pipeline/adapter_cli.py` 的 raw prompt 构造：raw 模式仍不走 retrieval，但必须明确要求模型从断点直接续写小说正文，禁止标题、分析、解释、写作建议和向用户提问。
2. 修复后只重跑 `v2 × ch1_clean × seed1101` 判别实验；若 v2 恢复正常正文续写，再用固定 reference 跑 `eval_draft.py`。
3. 判别通过后才进入 Stage 0 fan-out：纯基座/v2/v4 × 4 anchors × 2 repeats。只比较 adapter_cli 三元组内部，不和 Ollama baseline 求差。
4. Stage 0 重锚完成后，再决定 QLoRA 是数据重构还是旋钮单变量阶梯。
5. 同时可以先加轻量 debug：`outputs/debug/retrieval_manifest.json`，记录检索注入来源、原因、槽位占用和 temporal filter 排除数量。
6. System B MVP 已可测试：`kg_extract.py` 生成可编辑 facts 草稿，`update_kg.py` 合并 `kg.json` 并渲染 Markdown cards；抽取仍需人工审核，不自动相信模型。

### 本地 v2 UI

当前成品入口不走 Web / Gradio：

```powershell
.\scripts\run_v2_local_ui.ps1
.\scripts\run_v2_local_ui.ps1 -ContextFile outputs\debug\test_context_ch1_clean.txt
```

默认 adapter：`outputs/qlora_run_v2/`。脚本会设置 `outputs/hf_stage0_proxy/` 作为进程级 HF cache，避免 Unsloth import 被默认 HF cache 权限卡住。

### Stage 0 重锚评测现状

**冻结资产**：
- Anchors：`outputs/eval_anchors/` 下 4 个 txt + `anchors_manifest.json` sha 锁。已多次校验一致，不得重选或修改。
- `ch1_clean`：ch1，800c，训练区间外，泛化点。
- `ch58_bad_trigger`：ch58，790c，训练区间内，结尾靠近“凰后/凤倾汐”坏触发点。
- `mid_court_dialogue`：ch56，798c，训练区间内，宫廷对白。
- `yehuan_controlled`：ch35，761c，训练区间内，叶欢线控制点，不含金丹/凝气/昆仑等泛玄幻词。

**adapter_cli 状态**：
- 已支持 `--raw-prompt-file`，raw 模式跳过 Retriever / `build_prompt`。
- 已支持 `--seed`，传入后调用 `transformers.set_seed(seed)`。
- 当前安全推理口径：`--max-seq-length 4096` + `MAX_RECENT_CHARS=800`。
- 实际生效采样参数只有 `temperature/top_p/top_k/repetition_penalty`，均来自 `config.yaml`。

**评测 reference 锁定**：
- `scripts/eval_draft.py` 必须显式传：
  `--reference data/raw/风丝引_原文.txt`
- 该 reference 是 Stage 0 的固定参照系，中途不得更换。
- 上一轮纯基座 smoke 候选 `outputs/adapter_candidate_20260624_1359.txt` 用该 reference 已能评测，结果 `style_score=60.8257`、`repetition_risk=medium`、`contamination_risk=low`。该分数不能作为可用性结论，因为候选文本是助手腔退化输出。

**当前阻塞：raw prompt 构造问题**：
- 纯基座 `ch1_clean` seed1101 raw smoke 能生成，显存安全（加载 peak 5.77GB，生成 peak 约 6.06GB），但输出退化成“这是一段...”、标题、Markdown 菜单和写作建议。
- 同一 `ch1_clean`、同 seed1101、挂 v2 adapter 后也退化成助手腔/说明文（含 `**《浮生一梦》**` 和“如果需要继续发展剧情...”）。
- 因 v2 也退化，判定为 raw prompt 构造问题，不是纯基座固有行为。当前 raw prompt 实际结构只是把 anchor 当 user message，随后接 assistant 起始标记，缺少续写任务约束。
- 判别通过前禁止跑完整 3×4×2 fan-out；否则三元组都会被同一个 prompt 构造缺陷污染。

**运行环境规避**：
- 默认 HF cache `C:\Users\14390\.cache\huggingface\hub` 对 Codex sandbox 只有读权限，Unsloth import 会卡在 cache writable probe。
- Stage 0 使用进程级 proxy：`outputs/hf_stage0_proxy/`。设置 `HF_HOME/HF_HUB_CACHE/HF_XET_CACHE` 到该目录后，repo id 可离线解析；其中 base model 目录用 junction 指向真实 HF cache。

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

## 系统 B（知识图谱 MVP）

核心链路（已实现 MVP）：

```text
已接受章节文本 -> kg_extract.py -> kg_update.py -> data/story_bible/kg.json
-> kg_render.py -> Markdown cards -> 下一章 Retriever 用 BM25 + frontmatter 检索
```

已实现脚本：`scripts/kg_extract.py` / `scripts/kg_update.py` / `scripts/kg_render.py` / `scripts/update_kg.py`

第一版只做朴素可控闭环，不做复杂 GraphRAG。`kg.json` 是事实源，Markdown 是投影层，方便未来升级 vector/KG 时不推翻数据。`kg_extract.py` 只生成可编辑草稿，事实必须人工确认后再 `update_kg.py`。

回归测试：

```powershell
python _test_system_b.py
```

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
