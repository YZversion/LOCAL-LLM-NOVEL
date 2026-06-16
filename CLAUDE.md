# 本地小说续写助手 — Agent 工作规程

> 每次新对话开始时，先读此文件，再读【当前阶段】对应的测试清单，再动手。

---

## 工作原则

1. **每次只做一件事**：单次对话的范围 = 一个任务 + 一组测试，完成并验证后停止。
2. **先测试，再继续**：每个阶段结束前必须有可运行的验证步骤，用户确认通过才能进入下一阶段。
3. **不猜、不超前**：不主动实现用户没有明确要求的功能，不跨阶段预写代码。
4. **改动要小**：单次改动文件数 ≤ 3，改动行数 ≤ 100，超出就拆分对话。
5. **对话结束时生成摘要**：按本文件末尾的模板，输出本次对话做了什么、测试结果、下一步是什么。

---

## 项目快照

| 项目 | 值 |
|------|----|
| 工作目录 | `c:\Users\14390\Desktop\Code\LOCAL-LLM-NOVEL` |
| GitHub | `https://github.com/YZversion/LOCAL-LLM-NOVEL` |
| 推理模型（当前） | `huihui_ai/qwen3-abliterated:8b-v2`（Qwen3-8B 去审查 instruct，标准 ChatML，已 pull） |
| 推理模型（已弃用） | `hf.co/DavidAU/Qwen2.5-MOE-2X1.5B-DeepSeek-...-4B-gguf:Q4_K_M`（R1 蒸馏 MoE，见阶段2.6 决策记录） |
| 微调框架 | Unsloth QLoRA 4-bit（阶段4，未开始） |
| 显存预算 | 8GB（RTX 4070 Laptop） |
| CUDA | 13.2（Unsloth 兼容性待验证） |
| 数据保护 | `data/`、`models/`、`outputs/` 全部 gitignore，素材文件绝不入库 |
| Python | 3.11.9 |

---

## 阶段路线图与当前状态

```
[✓] 阶段0    repo 骨架 + 环境配置文件
[✓] 阶段2    零训练合写回路（7/7 通过，基于已弃用模型）
[✓] 阶段2.6  模型迁移（弃用 R1 蒸馏 MoE → Qwen3-8B）+ 补全式续写改造（6/6 通过）
[ ] 阶段1    数据清洗（prepare_data.py，用户自有脚本）
[ ] 阶段3    文风评测（eval_style.py）  ← 当前焦点
[ ] 阶段4    QLoRA 微调
[ ] 阶段5    向量 RAG（按需）
```

**当前焦点：阶段3 — 实现确定性文风评测工具 `pipeline/eval_style.py`。**
阶段3 按 3.0 到 3.8 小任务推进；3.1-3.3 已通过，下一步等待用户明确开始 3.4。

---

## 阶段2.6 模型迁移决策记录（已完成）

### 为什么弃用旧模型（R1 蒸馏 MoE 2X1.5B）

| 症状 | 根因（非调参可救） |
|------|-------------------|
| "好的，接下来我将按以下步骤编写" + 1./2./3. 列表 | R1 蒸馏的目标行为本身就是"先列计划再执行"，模型在正确地做被训练的事，只是那件事不是写小说 |
| 文风/人物/连贯性塌 | MoE 仅激活 ~1.5B 参数，长篇叙事吃稠密参数量，1.5B 激活不够 |
| 英文乱码 token（`themngthtson` 等） | Q4 量化 + 小模型 + 混合蒸馏导致词表/数值不稳，作者已标 Known Issue，无解 |
| 偶发空输出 | 同属上述不稳定性 |

结论：**任务错配（推理型模型 ≠ 写作型模型），换模型，不在旧模型上继续调参。**

### 新方向

- **模型**：`huihui_ai/qwen3-abliterated:8b-v2`。标准 ChatML（system 有专用边界，约束力强）；abliterated 去审查，利于偏暗情节。
- **必须关思考**：Qwen3 是混合推理模型，默认输出 `<think>` 链。不关会重现旧模型的列表/助手语问题。用 `think=False` 关思考，并在输出端清洗残留标签或字面泄漏。
- **补全式而非任务式**：去掉"请续写约600字"这类任务句 + 负向指令（小模型对"禁止…"几乎无效）。改用 prefill 让模型顺着上文写正文，而不是"接受任务再回答"。
- **采样调回写作区间**：旧模型为压垃圾把 temp 压到 0.25，换模型后会让小说又干又重复，需调回。

### 阶段2.6 测试清单（已完成：6/6 通过）

- [✓] `config.yaml` 模型名已切换为 `huihui_ai/qwen3-abliterated:8b-v2`
- [✓] `python -m cowriter.app` 用新模型可启动，不崩溃
- [✓] 单次生成输出为**纯中文小说正文**：无"好的/接下来我将"、无 1./2./3. 列表、无 `<think>` 残留、无英文乱码
- [✓] prefill 生效：续写与上文衔接自然，不另起"任务回答"
- [✓] 连续 5 次生成无空输出（验证旧模型的偶发空输出问题已消失）
- [✓] `/检索`、`/保存`、摘要压缩等阶段2 既有命令在新模型下仍正常

---

## 阶段2.6 已执行改动概要（归档）

> 本节保留阶段2.6的落地范围，作为后续排查参考。阶段3开始前不再改动模型调用与 prompt 拼装，**不碰 Gradio UI 和 BM25/grep 检索**。

### config.yaml
- 模型名 → `huihui_ai/qwen3-abliterated:8b-v2`
- 采样参数（Qwen3 非思考模式基础上偏创作）：
  - `temperature: 0.8`
  - `top_p: 0.8`
  - `top_k: 20`
  - `repeat_penalty: 1.05`
  - 保留 `num_predict` / `output_tokens` 现有逻辑

### cowriter/session.py
- 所有 `ollama.chat()` 加 `think=False`
- 解析 `resp["message"]` 时分别取 `.content` 与 `.thinking`，**绝不把 thinking 拼进输出**
- 加正则后处理：剥掉残留 `<think>...</think>`（含只有开标签未闭合的情况）

### cowriter/prompts.py
- `SYSTEM_PROMPT` 精简为纯文风/视角/人物约束，**去掉所有负向指令**（"禁止空输出""禁止助手语气"改为正向风格描述）；去掉 `/no_think` 依赖
- 改 prefill：messages 末尾追加 `{"role":"assistant","content": <上文结尾自然截取的几个字>}`，让模型只能顺着写正文
- block 顺序：背景（设定/摘要/grep命中）在前，「当前上文」放最后紧贴 prefill；去掉"请直接续写约600字"任务句

### 兼容性提示
- 真正生效的是 `think=False`；输出端清洗用于兜底剥离残留 `<think>`、`/no_think` 或 `/no` 字面泄漏。
- fallback 预案：若 Qwen3 关思考后仍有 ChatML 残留或文风不稳，退到 `qwen2.5:7b-instruct`（纯 instruct，无推理污染）。

---

## 阶段3 测试清单（进行中：3.1-3.3 已通过）

- [✓] 3.1 CLI 骨架：`pipeline/eval_style.py` 能读取 reference/candidate，输出 JSON、Markdown 或终端报告
- [✓] 3.2 文本切分增强：中文小说句子/段落/对话行切分稳定，新增 `segmentation` JSON 字段
- [✓] 3.3 重复检测增强：重复行、重复段落、连续重复句、近似相邻句、短句循环、char 2/3/4-gram 均有确定性检测
- [ ] 3.4 污染检测增强：继续强化 candidate 复制 reference 的检测
- [ ] 3.5 reference vs candidate 差异评分
- [ ] 3.6 报告输出完善
- [ ] 3.7 建立 fixtures
- [ ] 3.8 可选接入当前流程
- [ ] 有至少 3 章「模型续写 vs 真实章节」对比记录

---

## 阶段4 测试清单（尚未开始）

- [ ] Unsloth 在 CUDA 13.2 上能跑 forward pass（先验证，再动数据）
- [ ] `train_qlora.py` 单 epoch 不 OOM（显存峰值 < 8GB）
- [ ] 导出 GGUF 并在 Ollama 加载成功
- [ ] 同一验证集下，微调模型得分 > 阶段2基线

---

## 已完成的代码改动记录

> 以下记录已完成的代码改动历史。新增阶段记录时另起小节，不覆盖旧记录。

### 阶段2.6（2026-06-16）

- `cowriter/session.py`：增强 `_strip_think()`，剥离 `/no_think` 完整字面泄漏、`/no` 残片、`好的。` 等独占行助手语；新增 `_dedup_output()`，处理短句循环和大段单次复读；`_chat()` 接入输出清洗。
- `cowriter/prompts.py`：精简 `SYSTEM_PROMPT`，改为补全式续写 prefill，去掉 `/no_think` 依赖。
- `config.yaml`：切换到 `huihui_ai/qwen3-abliterated:8b-v2`，调整写作采样参数与重复抑制参数。
- `_test_phase26.py`：更新阶段2.6 回归断言，覆盖输出清洗关键路径。
- `_test_dedup.py`：补充去重实机与单元验证，用于观察短句循环、大段复读和残片清洗。

### 阶段3.1-3.3（2026-06-16）

- `pipeline/eval_style.py`：实现确定性文风评测 CLI，可读取 UTF-8 reference/candidate，输出 JSON、Markdown 或终端报告；暂不接入 LLM、Web UI 或续写 CLI。
- 阶段3.1：实现基础文本统计、重复风险基础检测、污染检测基础版、reference/candidate 差异摘要。
- 阶段3.2：增强中文小说文本切分，支持中文/英文句末标点、引号闭合、多标点、段落空行边界和对话行识别；新增 `segmentation` 顶层字段。
- 阶段3.3：增强 `repetition` 字段，支持重复行、重复段落、连续重复句、近似相邻句、短句循环、char 2/3/4-gram 与 `low/medium/high` 风险等级。
- `_test_eval_style.py`：新增轻量回归测试，覆盖切分、重复句、ABAB短句循环、重复段落、近似相邻句、空 candidate、JSON 可解析和 Markdown 输出。

### cowriter/app.py
- 新增 `/上下文`：不调模型，打印 prompt 各段字数 + 前 2000 字预览，含 `【写作规则】` 顶部和 `【续写正文】` 底部标签
- 新增 `/拒绝`：丢弃当前输出，不写 session，自动重新生成
- `/重试 [指令]`：合并替换旧的 `/重新生成`，功能相同
- 退出命令扩展：`/退出`、`q`、`退出`、`exit`、`quit` 均可退出
- else 分支加防护：`/` 开头未知命令提示错误，不写 session

### cowriter/session.py
- `_maybe_compress()`：加 try-except 包住 `_chat()` 调用，捕获 Ollama ResponseError
- 加 fallback：LLM 返回空或 < 20 字时，用 `[前情节选] {old_text[:200]}` 作为摘要，保证 `session.summary` 非空

### cowriter/prompts.py
- `SYSTEM_PROMPT` 改为执行感写法，8 条正向规则，新增「禁止空输出」规则
  - ⚠️ 阶段2.6 将推翻此项：负向指令对小模型无效，改补全式 + prefill
- `build_prompt()` block 顺序：设定 → 摘要 → 原文检索 → 上文 → 要求；全换中文方括号标签
- `build_summary_prompt()`：加防幻觉规则（禁止编造、不要分析文风）

### cowriter/retriever.py（已重构）
- 新增 `_tokenize()`：BM25 建索引和查询共用同一套分词，过滤单字空词
- `_load_bible()`：把 `.md` 文件名注册进 jieba 词典（freq=10000），维护 `_entity_names` 集合
- `extract_entities()`：三层优先级 → bible 已知实体 → posseg 命名实体（nr/ns/nt/nz）→ 停用词过滤
- `search_bible()`：返回 `{source, text, score}`，text 截断至 1200 字
- `grep_raw()`：加 `-F` 防正则，处理 rg returncode 0/1/2，fallback 改用 `rglob`
- `retrieve()`：bible query = 实体拼接 + 末 150 字；grep 结果去重

---

## 已知坑 / 注意事项

| 问题 | 处理方式 |
|------|---------|
| 模型偶发空输出 | 旧模型问题，**预期由阶段2.6 换模型解决**；验证前仍可用 `/重试` |
| 模型输出 AI 助手语 / 1.2.3. 列表 | R1 蒸馏的目标行为，**预期由换模型 + 补全式 prefill 解决**，非 prompt 可救 |
| 英文乱码 token | 旧模型 Q4 量化 Known Issue，**由换模型解决** |
| Qwen3 思考链泄漏（新） | Qwen3 默认输出 `<think>`，必须 `think=False`；并在输出端正则剥离残留 `<think>` 块 |
| 摘要压缩：小模型无法生成合理摘要 | `_maybe_compress()` 已加 try-except + fallback：LLM 失败时用原文前 200 字作摘要节选，保证 `session.summary` 非空 |
| story_bible 文件名即实体名 | 命名须直接用人名/地名，如 `林清雪.md` |
| CUDA 13.2 Unsloth 兼容性 | 阶段4开始前必须先跑 forward pass 验证，不要直接上数据 |
| `data/story_bible/*.md` 不入 git | `.gitignore` 已保护，只有 `.gitkeep` 入库 |

---

## 文件修改规范

| 文件 | 说明 |
|------|------|
| `config.yaml` | 唯一配置源，所有脚本从这读参数 |
| `cowriter/app.py` | 主入口，改动后必须手测所有命令（含 `/上下文`、`/拒绝`、`q`） |
| `cowriter/retriever.py` | 检索逻辑，改动后必须手动跑 `/检索` 验证 |
| `cowriter/prompts.py` | 提示词，改动后必须跑一次完整生成对比前后输出 |
| `cowriter/session.py` | 主循环，改动后必须跑完整的接受→压缩→再生成流程 |
| `pipeline/` | 阶段1/3/4脚本目录；阶段3只允许改 `pipeline/eval_style.py` 及其轻量测试，不改生成链路 |

---

## 对话结束摘要模板

每次对话结束时，输出以下格式（不省略任何字段）：

```
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
阶段3 当前 task：3.x，验收状态：通过/未通过
```
