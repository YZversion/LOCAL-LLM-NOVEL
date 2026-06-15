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
| 推理模型 | `hf.co/DavidAU/Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B-gguf:Q4_K_M`（Ollama） |
| 微调框架 | Unsloth QLoRA 4-bit（阶段4，未开始） |
| 显存预算 | 8GB（RTX 4070 Laptop） |
| CUDA | 13.2（Unsloth 兼容性待验证） |
| 数据保护 | `data/`、`models/`、`outputs/` 全部 gitignore，素材文件绝不入库 |
| Python | 3.11.9 |

---

## 阶段路线图与当前状态

```
[✓] 阶段0  repo 骨架 + 环境配置文件
[✓] 阶段2  零训练合写回路（7/7 通过）
[ ] 阶段1  数据清洗（prepare_data.py，用户自有脚本）
[ ] 阶段3  文风评测（eval_style.py）
[ ] 阶段4  QLoRA 微调
[ ] 阶段5  向量 RAG（按需）
```

**当前焦点：阶段2 已完成，下一步待决策（Web UI vs 阶段2.5 原文检索增强）**

---

## 阶段2 测试清单（当前阶段必须全部通过）

- [✓] `pip install -r requirements.txt` 无报错
- [✓] `ollama pull hf.co/DavidAU/Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B-gguf:Q4_K_M` 成功，`ollama list` 可见
- [✓] `python -m cowriter.app` 可启动，不崩溃
- [~] 粘贴上文 → 生成 → 接受 → 再生成循环（模型偶发空输出，用 `/重试` 可绕过）
- [✓] `/检索 <人物名>` 在有设定集文件时能返回结果（已用《风丝引》素材验证）
- [✓] `/保存` 在 `outputs/` 生成 txt 文件，路径打印清楚
- [✓] 摘要压缩：输入超过 4000 字后 `session.summary` 非空，`【剧情摘要】`块进入下一轮 prompt

---

## 阶段3 测试清单（尚未开始）

- [ ] `eval_style.py` 能读取验证章节和模型续写，输出四维分数
- [ ] 有至少 3 章「模型续写 vs 真实章节」对比记录

---

## 阶段4 测试清单（尚未开始）

- [ ] Unsloth 在 CUDA 13.2 上能跑 forward pass（先验证，再动数据）
- [ ] `train_qlora.py` 单 epoch 不 OOM（显存峰值 < 8GB）
- [ ] 导出 GGUF 并在 Ollama 加载成功
- [ ] 同一验证集下，微调模型得分 > 阶段2基线

---

## 已完成的代码改动记录

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
| 模型偶发空输出 | 用 `/重试` 重新生成，无需重启 |
| 模型可能输出 AI 助手语 | prompt 已加强；若仍出现用 `/拒绝` 丢弃 |
| 摘要压缩：小模型无法生成合理摘要 | `_maybe_compress()` 已加 try-except + fallback：LLM 失败时用原文前 200 字作为摘要节选，保证 `session.summary` 非空 |
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
| `pipeline/` | 阶段4前不动 |

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
阶段2测试清单：X/7 通过
```
