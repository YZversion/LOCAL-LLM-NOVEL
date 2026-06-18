#!/usr/bin/env python3
"""
阶段4：用 LoRA adapter 生成候选文本，供 scripts/eval_draft.py 评测。

运行环境：.venv-train/（需激活）
前提：已跑完 train_qlora.py --full-run，adapter 在 outputs/qlora_run/

生成参数与 config.yaml generation 段对齐（temperature/top_p/top_k/repetition_penalty），
使用与基线评测相同的 prompt（outputs/debug/last_prompt.txt）——
保证只有"是否经过 LoRA 微调"这一个变量变化。

Usage:
    .venv-train\\Scripts\\Activate.ps1
    python pipeline/generate_lora.py
    # 然后：
    python scripts/eval_draft.py --candidate outputs/lora_candidate.txt --config config.yaml
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID     = "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"
ADAPTER_PATH = "outputs/qlora_run_v2"    # v2 adapter (warmup_steps=1 fixed)
PROMPT_PATH  = "outputs/debug/last_prompt.txt"
OUTPUT_PATH  = "outputs/lora_candidate_v2.txt"

# 与 config.yaml generation 段对齐
TEMPERATURE        = 0.8
TOP_P              = 0.8
TOP_K              = 20
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS     = 2500  # ~2226c target; last_prompt is ~3500 tokens so need 8192 window
MAX_SEQ_LENGTH_INFER = 8192  # inference: must hold ~3500 input + ~2000 output tokens


def parse_prompt_file(path: Path) -> tuple[str, str]:
    """Parse [SYSTEM] and [USER] sections from last_prompt.txt format."""
    text = path.read_text(encoding="utf-8")

    def extract(start_marker: str, end_marker: str) -> str:
        s = text.find(start_marker)
        if s == -1:
            return ""
        s += len(start_marker)
        e = text.find(end_marker, s)
        return text[s:e].strip() if e != -1 else text[s:].strip()

    system_content = extract("[SYSTEM]\n", "\n[USER]")
    user_content   = extract("[USER]\n",   "\n[ASSISTANT]")
    return system_content, user_content


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not found — activate .venv-train/ first", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    from unsloth import FastLanguageModel

    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}", file=sys.stderr)
        print("  → Run: python pipeline/train_qlora.py --full-run --max-seq-length 1024", file=sys.stderr)
        return 1

    prompt_path = Path(PROMPT_PATH)
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    print("--- Parsing baseline prompt ---")
    system_content, user_content = parse_prompt_file(prompt_path)
    if not system_content or not user_content:
        print("ERROR: could not parse [SYSTEM] or [USER] from prompt file", file=sys.stderr)
        return 1
    print(f"  system: {len(system_content)}c")
    print(f"  user:   {len(user_content)}c")

    print(f"\n--- Loading model + adapter from {adapter_path}/ ---")
    print(f"  max_seq_length={MAX_SEQ_LENGTH_INFER}  (inference window: input ~3500t + output ~2000t)")
    print("  (Unsloth reads base_model_name_or_path from adapter_config.json)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH_INFER,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("  Model ready.")

    print("\n--- Building input (chat template) ---")
    chat = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
    try:
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_len}")

    print(f"\n--- Generating (max_new_tokens={MAX_NEW_TOKENS}, "
          f"temp={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}, "
          f"rep_penalty={REPETITION_PENALTY}) ---")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
        )

    new_ids = output_ids[0][input_len:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Strip <think>...</think> block if present (Qwen3 thinking mode leak)
    if "<think>" in generated:
        end = generated.find("</think>")
        if end != -1:
            generated = generated[end + len("</think>"):].strip()

    print(f"  Generated: {len(generated)}c")
    print()
    print("--- Preview (first 400c) ---")
    print(generated[:400])
    print("...")

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generated, encoding="utf-8")
    print(f"\n--- Saved to {out_path} ---")
    print()
    print("Next step:")
    print("  python scripts/eval_draft.py --candidate outputs/lora_candidate_v2.txt --config config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
