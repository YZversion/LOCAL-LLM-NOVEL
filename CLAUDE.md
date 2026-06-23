# 本地小说续写助手 — Agent 工作规程

_最后更新：2026-06-23_

> 每次新对话开始时，先读此文件，再读 `docs/current_state.md`，再动手。

---

## 工作原则

1. **每次只做一件事**：单次对话的范围 = 一个任务 + 一组测试，完成并验证后停止。
2. **先测试，再继续**：每个阶段结束前必须有可运行的验证步骤，用户确认通过才进入下一阶段。
3. **不猜、不超前**：不主动实现用户没有明确要求的功能，不跨阶段预写代码。
4. **改动要小**：单次改动文件数和行数尽量小；超出就拆分对话。
5. **对话结束时生成摘要**：按本文末尾模板输出做了什么、测试结果、遗留问题和下一步。

---

## 当前阶段：阶段4 QLoRA 微调

详细环境/实验记录见 `docs/phase4_qlora.md`，项目全局状态见 `docs/current_state.md`。

**当前任务：确认 num_train_epochs 和 warmup_steps，然后用 57 条风丝引样本训练 v4 adapter。**

已确认决策（v3 方向已放弃）：
- v3（novel2 合并 544 条）放弃：round2/3 严重崩溃（角色错乱 + 技术术语），根因为 novel2 分布与风丝引不兼容，不再追究。
- 新方向：只用风丝引自身数据，从 20 条（ch2-21）扩充至 57 条（ch2-58）。
- 补写两张新角色卡：宁楚珣（revealed_in=42）、大理相（revealed_in=52），时序验证全部通过。

当前训练数据（v4）：
- 57 条，`data/processed/train_samples_full_57.jsonl`（ch2-58 风丝引原文）。
- 参数：`context_chars=700`，`completion_chars=200`，`bible_top_k=2`，`bible_max_chars=250`，`prior_max_chars=120`。
- Token 分布：min=1335t，max=1460t，mean=1398t，0 条超过 1536t 上限。
- `_test_train_samples.py`：57/57 ALL PASS（含宁楚珣 ch42/ch43 边界、大理相 ch52 边界）。

当前卡点：
- `pipeline/train_qlora.py` 已更新：`MAX_SEQ_LENGTH=1536`，`--samples` 默认指向 `train_samples_full_57.jsonl`，`--full-run` 输出目录为 `outputs/qlora_run_v4/`。
- `WARMUP_STEPS=2`，`NUM_TRAIN_EPOCHS=3` 为建议值（TBD），等用户确认后开始训练。
- 建议在 full-run 前用新 57 条 700c 样本再跑一次 1536 显存探针（当前探针用的是旧 60c 数据，实际序列最长 942t；新数据最长 1460t，VRAM 会略高，估算 ~7.3GB）。

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

`cowriter/session.py` · `cowriter/prompts.py` · `cowriter/retriever.py` · `cowriter/web.py` · `cowriter/chapter.py` · `config.yaml` · `data/raw/` · `data/story_bible/` · `pipeline/eval_style.py` · `data/processed/merged_train_samples.jsonl`（novel2 544 条，已放弃，保留备档）

---

## 常用测试命令

```powershell
# 文风评测基线
python scripts/eval_draft.py --candidate <file> --config config.yaml --out-json <out.json>

# 时序过滤回归测试
python _test_temporal_filter.py

# 训练样本验证
python _test_train_samples.py

# LoRA 多轮生成（.venv-train 激活后）
python pipeline/generate_lora_multi.py
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
