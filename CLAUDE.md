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

## 当前阶段：阶段4 QLoRA 重规划

详细环境/实验记录见 `docs/phase4_qlora.md`，项目全局状态见 `docs/current_state.md`。

**当前任务：Stage 0 先修 `adapter_cli.py` raw prompt 构造，再重跑 `v2 × ch1_clean × seed1101` 判别实验。判别通过前禁止 fan-out 到 3×4×2。**

已确认决策：
- v3（novel2 合并 544 条）已明确放弃：round2/3 严重崩溃（角色错乱 + 区块链/AI 等技术术语），novel2 分布与风丝引不兼容，不再追究根因，不再做 repeat 诊断。
- v4 已完整训练完成：57 条风丝引样本（ch2-58），`context_chars=700`，`max_seq_length=1536`，`num_train_epochs=3`，`warmup_steps=2`，`UNSLOTH_CE_LOSS_TARGET_GB=0.5`，adapter 保存在 `outputs/qlora_run_v4/`。
- v4 训练曲线健康但评测未通过：loss 从 3.758 降到 2.223，显存峰值约 7.38GB；生成质量仍不稳定，不能导出或接生产。
- Stage 0 评测集已冻结在 `outputs/eval_anchors/`（4 个 anchor txt + `anchors_manifest.json` sha 锁），不得重选或修改 anchor。
- `pipeline/adapter_cli.py` 已支持 `--raw-prompt-file`、`--seed`、`MAX_RECENT_CHARS=800`，并打印实际生效的 `temperature/top_p/top_k/repetition_penalty`。
- 评测 reference 已锁定为 `data/raw/风丝引_原文.txt`；所有 `scripts/eval_draft.py` 调用必须显式传 `--reference data/raw/风丝引_原文.txt`，中途不得更换。

v4 已知评测结论：
- 坏触发点评测：`outputs/adapter_candidate_v4_eval.txt` / `outputs/v4_eval_result.json`，style_score `39.41`。从 ch58 附近“凰后/凤倾汐”上文续写时触发“凰后 -> 叶欢”强关联，切入叶欢修仙子线，并用训练数据外的通用玄幻词汇（金丹/凝气/昆仑山脉等）填充。
- 干净起点复测：`outputs/adapter_candidate_20260624_1114.txt` / `outputs/v4_ch1_clean_eval.json`，style_score `48.7361`，仍低于基线 `50.92` 和 v2 `60.48`。人工核查发现标题样文本、擅自跳剧情、乱加人物、现代感词汇等问题。
- 结论：v4 不是单点叶欢线问题，而是当前 57 条 / 700c / 1536 / 3epoch 配置整体不稳定。

Stage 0 最新阻塞：
- 纯基座 `ch1_clean` seed1101 raw smoke 能生成，候选 `outputs/adapter_candidate_20260624_1359.txt`，但退化成 instruct 助手腔、标题、Markdown 菜单和写作建议。
- 同一 `ch1_clean`、同 seed1101、挂 v2 adapter 后也出现同类退化（开头“这是一段充满诗意与情感的画面描写...”，含 `**《浮生一梦》**` 和“如果需要继续发展剧情...”）。
- 因 v2 也退化，判定为 **raw prompt 构造问题**，不是纯基座固有行为。当前 raw 模式实际 prompt 只是 `<|im_start|>user\n[800字原文]\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>`，缺少“直接续写小说正文、不要解释、不要标题、不要向用户提问”等约束。
- 下一步只允许最小修改 raw prompt 构造；修复后先重跑 `v2 × ch1_clean × seed1101`。若 v2 恢复正文续写，再用固定 reference 跑 eval，之后才考虑完整 3×4×2 fan-out。

运行环境已知坑：
- 默认 HF cache `C:\Users\14390\.cache\huggingface\hub` 对 Codex sandbox 只有读权限，Unsloth import 会卡在 cache writable probe。
- Stage 0 运行时使用进程级 HF cache proxy：`outputs/hf_stage0_proxy/`，其中 `hub/models--huihui-ai--Huihui-Qwen3-8B-abliterated-v2` 是指向真实 HF cache 的 junction。设置 `HF_HOME/HF_HUB_CACHE/HF_XET_CACHE` 到该 proxy 后，repo id 可离线解析，4-bit 基座加载约 15s，峰值约 5.77GB。

当前优先级：
1. **修 raw prompt 构造**：raw 模式仍不走 retrieval，但必须明确要求模型从断点直接续写小说正文，禁止标题、分析、解释、写作建议和提问。
2. **重跑最小判别**：只跑 `v2 × ch1_clean × seed1101`；通过后再用固定 reference 跑 eval。
3. **Stage 0 fan-out**：判别通过后才跑纯基座/v2/v4 × 4 anchors × 2 repeats；只比较 adapter_cli 三元组内部，不和 Ollama baseline 求差。
4. **QLoRA 回退或重规划**：以 v2 `style_score 60.48` 为历史可用锚点，但新 Stage 0 数字必须在修 prompt 后同口径重锚。
5. **轻量检索 debug**：实现 `outputs/debug/retrieval_manifest.json`。
6. **System B MVP**：等基础生成稳定后，再做 `kg.json -> Markdown cards -> BM25`，不要现在上复杂 GraphRAG/LightRAG。

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
python scripts/eval_draft.py --candidate <file> --reference data/raw/风丝引_原文.txt --config config.yaml --out-json <out.json>

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
