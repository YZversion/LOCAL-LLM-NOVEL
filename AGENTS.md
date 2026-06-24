# 本地小说续写助手 — Agent 工作规程

_最后更新：2026-06-24_

> 每次新对话开始时，先读此文件，再读 `docs/current_state.md`，再动手。

---

## 工作原则

1. **每次只做一件事**：单次对话的范围 = 一个任务 + 一组测试，完成并验证后停止。
2. **先测试，再继续**：每个阶段结束前必须有可运行的验证步骤，用户确认通过才进入下一阶段。
3. **不猜、不超前**：不主动实现用户没有明确要求的功能，不跨阶段预写代码。
4. **改动要小**：单次改动文件数和行数尽量小；超出就拆分对话。
5. **对话结束时生成摘要**：按本文末尾模板输出做了什么、测试结果、遗留问题和下一步。

---

## 当前阶段：本地成品化（v2 + System B MVP）

详细环境/实验记录见 `docs/phase4_qlora.md`，项目全局状态见 `docs/current_state.md`。

**当前任务：用历史可用的 `outputs/qlora_run_v2/` 作为本地终端续写入口，并落地 System B 第一版 `kg.json -> Markdown cards -> BM25`。今天不走 Web / Gradio，不训练、不导出、不合并 LoRA。**

已确认决策：
- v3（novel2 合并 544 条）已明确放弃：round2/3 严重崩溃（角色错乱 + 区块链/AI 等技术术语），novel2 分布与风丝引不兼容，不再追究根因，不再做 repeat 诊断。
- v4 已完整训练完成，但评测未通过：57 条风丝引样本，`context_chars=700`，`max_seq_length=1536`，3 epoch，adapter 保存在 `outputs/qlora_run_v4/`；生成质量整体不稳定，不能导出或接生产。
- 今天的可用模型选择是 **v2**：`outputs/qlora_run_v2/` 是历史可用锚点，曾在同一 reference 下得到 style_score `60.48`，优先用于本地成品入口。
- 内网环境不使用 Web UI。当前入口是 PowerShell 终端脚本：`scripts/run_v2_local_ui.ps1`。
- System B 第一版只做朴素闭环：人工审核 facts -> `kg.json` 事实源 -> Markdown cards 投影层 -> 现有 Retriever 的 BM25 + frontmatter 检索。暂不上 GraphRAG / LightRAG。

本地 v2 UI：

```powershell
.\scripts\run_v2_local_ui.ps1
.\scripts\run_v2_local_ui.ps1 -ContextFile outputs\debug\test_context_ch1_clean.txt
```

System B MVP：

```powershell
python scripts\kg_extract.py --chapter 59 --input outputs\chapter_059.txt --out outputs\system_b\ch59_facts.draft.json --entities 林清雪,颜儿
python scripts\update_kg.py --facts outputs\system_b\ch59_facts.reviewed.json --kg data\story_bible\kg.json --out-dir data\story_bible\generated\system_b --prune
python _test_system_b.py
```

System B 数据口径：
- `data/story_bible/kg.json` 是事实源；`data/story_bible/generated/system_b/` 只是投影层。
- 状态变化不覆盖旧事实：旧状态用 `valid_to` 关闭，新状态用新的 `valid_from` 开启。
- 支持类型：`event`、`character_state`、`relationship_delta`、`location_state`、`plot_thread`。
- `kg_extract.py` 只生成低置信度草稿，不自动相信模型抽取；写入 `kg.json` 前必须人工确认。

Stage 0 raw-prompt 重锚评测仍是未完成实验线，不作为今天成品主线。若以后恢复，先修 `pipeline/adapter_cli.py` 的 raw prompt 构造，再重跑 `v2 × ch1_clean × seed1101`；通过后才 fan-out 到 3×4×2。

运行环境已知坑：
- 默认 HF cache `C:\Users\14390\.cache\huggingface\hub` 对 Codex sandbox 只有读权限，Unsloth import 会卡在 cache writable probe。
- 本地 v2 UI 和 Stage 0 运行时使用进程级 HF cache proxy：`outputs/hf_stage0_proxy/`，其中 `hub/models--huihui-ai--Huihui-Qwen3-8B-abliterated-v2` 是指向真实 HF cache 的 junction。设置 `HF_HOME/HF_HUB_CACHE/HF_XET_CACHE` 到该 proxy 后，repo id 可离线解析，4-bit 基座加载约 15s，峰值约 5.77GB。

当前优先级：
1. **验收本地 v2 终端 UI**：用 `scripts/run_v2_local_ui.ps1` 跑一段真实上文，人工确认不出现 Web 依赖、HF cache 权限卡死或助手腔退化。
2. **验收 System B MVP**：用一份人工审核 facts 更新 `kg.json`，确认 Markdown cards 能被 `Retriever.search_bible(..., max_chapter=N)` 按时序召回。
3. **补轻量检索 debug**：实现 `outputs/debug/retrieval_manifest.json`，记录 target/max chapter、命中文件、注入原因、槽位字符数和 temporal filter 排除数量。
4. **之后再回到 QLoRA 重规划**：基础成品可用后，再决定是重构样本还是单变量旋钮阶梯。

## 已知限制 / 设计权衡

**`_stem_priority()` 机制对 generated/characters/ 卡片的系统性排出**

`cowriter/retriever.py` 的 `_stem_priority()` 给根目录卡片赋 priority=0、给 `generated/` 子目录卡片赋 priority=1。当多个实体在同一查询上下文中被精确匹配，精确匹配按 priority 排序后填满 top-k，priority=1 的 generated/ 卡片在与 2 个以上根目录主角卡同时竞争时会被系统性截断出局，与 BM25 相关性分数无关。

已用大理相案例验证：ch53 查询中大理相 BM25 分数 50.46（排名第 1），但因叶欢（priority=0）+ 洛诗（priority=0）占满 top-k=2 两个 slot，大理相（priority=1）被截断。当前受影响：大理相卡片在 ch53-55 三条训练样本里无法被召回。

这是 `_stem_priority` 设计的权衡副作用（手写卡 > 自动生成卡），不是 bug。当前阶段接受现状，不处理。

若未来需要修复，三个方向：
- ① 把 generated/ 卡片移到根目录（破坏目录语义）
- ② 提高 bible_top_k（增加 token 预算压力，当前 1536 预算已较紧）
- ③ 修改 `_stem_priority` 让两类卡片平权（影响所有 generated/ 卡片的全局检索行为，需评估）

---

## 禁止修改的文件（非明确任务不碰）

`cowriter/session.py` · `cowriter/prompts.py` · `cowriter/retriever.py` · `cowriter/web.py` · `cowriter/chapter.py` · `config.yaml` · `data/raw/` · `pipeline/eval_style.py` · `data/processed/merged_train_samples.jsonl`（novel2 544 条，已放弃，保留备档）

`data/story_bible/` 默认不碰；只有明确执行 System B 写入任务时，才允许写 `data/story_bible/kg.json` 和 `data/story_bible/generated/system_b/`。

---

## 常用测试命令

```powershell
# 文风评测基线
python scripts/eval_draft.py --candidate <file> --reference data/raw/风丝引_原文.txt --config config.yaml --out-json <out.json>

# 时序过滤回归测试
python _test_temporal_filter.py

# 训练样本验证
python _test_train_samples.py

# LoRA 多轮生成（.venv-train 激活后）
python pipeline/generate_lora_multi.py

# 本地 v2 终端 UI（不走 Web）
.\scripts\run_v2_local_ui.ps1

# System B MVP 回归测试
python _test_system_b.py
```

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
阶段4 当前 task：...，验收状态：通过/未通过
```
