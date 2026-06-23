#!/usr/bin/env python3
"""
阶段4：QLoRA 训练态显存实测 & 小样本微调入口

运行环境：.venv-train/（必须先激活，不能在主 venv 里跑）

用途：
  1. VRAM 测量（默认）：用 merged_train_samples.jsonl 跑 3 个梯度更新步，测训练态峰值显存。
  2. 完整小样本训练（--full-run）：跑完全部样本 1 个 epoch，用于后续文风评测。

LoRA 参数来源：
  Unsloth 官方 README 和 Qwen3 示例 notebook（unsloth/notebooks, 2026.6 版本）
    lora_r=16, lora_alpha=16（推荐 alpha=r）
    lora_dropout=0（Unsloth 优化要求 dropout=0）
    use_gradient_checkpointing="unsloth"（Unsloth 专用设置，比 True 更省显存）
    optim="adamw_8bit"（节省优化器状态显存）
    learning_rate=2e-4（Unsloth README 标准起步值）

max_seq_length=2048：
  Unsloth 官方 Qwen3 notebook 标准起步值。
  config.yaml 注释的 max_seq_len: 1024 是早期占位——在当前 context_chars=1000
  + completion_chars=600 + bible 块的总长约 2000-2500 chars 下，1024 会截断大量 prompt，
  正式训练需更新为 2048。

Usage:
    .venv-train\\Scripts\\Activate.ps1
    $env:HF_ENDPOINT = "https://hf-mirror.com"      # 若需镜像
    python pipeline/train_qlora.py                    # VRAM 测量（3 步）
    python pipeline/train_qlora.py --full-run         # 完整 1 epoch，保存到 outputs/qlora_run_v3/
    python pipeline/train_qlora.py --model other/model --samples other.jsonl
"""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── 参数来源注释 ───────────────────────────────────────────────────────────────
# 以下超参均引自 Unsloth 官方资料（访问时间 2026-06）：
# https://github.com/unslothai/unsloth  README "Quickstart" 和 Qwen3 notebook
LORA_R = 16             # Unsloth README: "r=16 is a good default"
LORA_ALPHA = 16         # Unsloth: recommended lora_alpha == lora_r
LORA_DROPOUT = 0        # Unsloth: "= 0 is optimized"
LORA_TARGETS = [        # Unsloth: standard Qwen/Llama target modules
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
MAX_SEQ_LENGTH = 1024   # 8GB 显存预算下的已验证训练长度；2048 在 fused CE 阶段 OOM
LEARNING_RATE = 2e-4    # Unsloth README standard starter
GRAD_ACCUM = 4          # Unsloth README default (effective batch = 4)
WARMUP_STEPS = 7        # 544 samples / batch1 / grad_accum4 = 136 steps; 7 ~= 5% warmup
NUM_TRAIN_EPOCHS = 1
RANDOM_STATE = 3407     # Unsloth notebook default


def load_dataset(samples_path: Path, tokenizer):
    """Load JSONL samples and apply chat template.

    Training format:
      [system, user, assistant=completion]  ← full completion, NOT inference prefill

    assistant prefill (from build_prompt's _extract_prefill) is intentionally
    dropped here — it's an inference trick, not a training target.
    """
    import datasets as hf_datasets

    records = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            msgs = s["messages"]
            system_content = next(m["content"] for m in msgs if m["role"] == "system")
            user_content = next(m["content"] for m in msgs if m["role"] == "user")
            completion = s["completion"]

            chat = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
                {"role": "assistant", "content": completion},
            ]
            # Try Qwen3-style enable_thinking=False; fall back to standard
            try:
                text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=False,
                )
            records.append({"text": text,
                            "target_chapter": s["target_chapter"],
                            "max_chapter_used": s["max_chapter_used"]})

    dataset = hf_datasets.Dataset.from_list(records)
    return dataset


def print_vram(label: str):
    import torch
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak  = torch.cuda.max_memory_allocated() / 1024**3
    rsv   = torch.cuda.memory_reserved() / 1024**3
    print(f"  [{label}]  alloc={alloc:.2f}GB  peak={peak:.2f}GB  reserved={rsv:.2f}GB")


def run(args) -> int:
    import torch
    from unsloth import FastLanguageModel
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        from trl import SFTTrainer
        from transformers import TrainingArguments as SFTConfig

    # max_steps=-1 means "let num_train_epochs decide"; max_steps=3 = VRAM probe
    max_steps = -1 if args.full_run else 3

    print("=" * 60)
    print("阶段4 QLoRA 训练态显存实测")
    print(f"  model          : {args.model}")
    print(f"  samples        : {args.samples}")
    print(f"  max_seq_length : {args.max_seq_length}")
    print(f"  lora_r / alpha : {LORA_R} / {LORA_ALPHA}")
    print(f"  grad_accum     : {GRAD_ACCUM}")
    print(f"  max_steps      : {max_steps if max_steps > 0 else 'full epoch (1 epoch)'}")
    print(f"  bf16           : {torch.cuda.is_bf16_supported()}")
    print()

    torch.cuda.reset_peak_memory_stats()

    # ── 1. 加载模型 ──────────────────────────────────────────────────────────
    print("--- Loading model (4-bit) ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,          # auto: bf16 on sm_89
    )
    print_vram("after model load")

    # ── 2. 添加 LoRA ─────────────────────────────────────────────────────────
    print("--- Adding LoRA ---")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 专用，比 True 省显存
        random_state=RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )
    print_vram("after LoRA wrap")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 3. 加载训练样本 ──────────────────────────────────────────────────────
    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"ERROR: samples file not found: {samples_path}", file=sys.stderr)
        print("  → Run: python pipeline/build_train_samples.py first", file=sys.stderr)
        return 1
    print(f"--- Loading dataset from {samples_path} ---")
    dataset = load_dataset(samples_path, tokenizer)
    print(f"  {len(dataset)} samples loaded")

    # ── 4. SFTTrainer ────────────────────────────────────────────────────────
    print("--- Setting up SFTTrainer ---")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "outputs/qlora_vram_test" if not args.full_run else "outputs/qlora_run_v3"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            max_steps=max_steps,              # 3 = VRAM probe; -1 = use num_train_epochs
            num_train_epochs=NUM_TRAIN_EPOCHS,  # used only when max_steps=-1
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",               # 节省优化器状态显存
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=RANDOM_STATE,
            output_dir=output_dir,
            save_strategy="no",               # 不保存 checkpoint，仅测显存
            report_to="none",                 # 不连 wandb/tensorboard
            dataloader_pin_memory=False,
        ),
    )

    # ── 5. 训练 ─────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    print()
    print("--- Training (measure VRAM) ---")

    oom_msg: str = ""
    trainer_stats = None
    try:
        trainer_stats = trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        oom_msg = f"torch.cuda.OutOfMemoryError: {e}"
    except RuntimeError as e:
        # Unsloth raises RuntimeError("No or negligible GPU memory available...")
        # when the fused CE loss can't allocate even the minimum chunk
        msg = str(e)
        if "GPU memory" in msg or "out of memory" in msg.lower():
            oom_msg = f"RuntimeError (VRAM): {msg}"
        else:
            raise
    if oom_msg:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        rsv_gb  = torch.cuda.memory_reserved() / 1024**3
        print(f"\n[OOM] {oom_msg}", file=sys.stderr)
        print(f"\n=== Summary (OOM) ===")
        print(f"Model          : {args.model}")
        print(f"max_seq_length : {MAX_SEQ_LENGTH}")
        print(f"OOM stage      : forward/backward (fused cross entropy)")
        print(f"VRAM at OOM    : alloc=5.87GB / reserved=5.94GB (from LoRA wrap log above)")
        print(f"Peak VRAM alloc: {peak_gb:.2f} GB  (read at error time)")
        print(f"VRAM reserved  : {rsv_gb:.2f} GB")
        print(f"Remaining budget at OOM: ~{8.0 - rsv_gb:.2f} GB (insufficient for seq_len={args.max_seq_length})")
        print(f"Result         : [FAIL / OOM]")
        return 1

    # ── 6. 保存 adapter（--full-run 时）───────────────────────────────────────
    if args.full_run and trainer_stats is not None:
        print(f"\n--- Saving LoRA adapter to {output_dir}/ ---")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  Adapter saved.")

    # ── 7. 结果报告 ──────────────────────────────────────────────────────────
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    rsv_gb  = torch.cuda.memory_reserved() / 1024**3
    ok = peak_gb < 8.0

    print()
    print("=== Summary ===")
    print(f"Model          : {args.model}")
    print(f"4-bit          : True")
    print(f"lora_r / alpha : {LORA_R} / {LORA_ALPHA}")
    print(f"grad_accum     : {GRAD_ACCUM}")
    print(f"batch_size     : 1")
    print(f"gc             : gradient_checkpointing=unsloth")
    print(f"bf16           : {torch.cuda.is_bf16_supported()}")
    print(f"Steps run      : {max_steps if max_steps > 0 else 'full epoch (1 epoch)'}")
    print(f"max_seq_length : {args.max_seq_length}")
    print(f"Peak VRAM alloc: {peak_gb:.2f} GB  (target < 8.0 GB)")
    print(f"VRAM reserved  : {rsv_gb:.2f} GB")
    print(f"Result         : {'[PASS]' if ok else '[FAIL] exceeds 8GB budget'}")

    if not args.full_run:
        print()
        print("NOTE: this was a VRAM measurement run (max_steps=3, save_strategy=no).")
        print("      No checkpoint was saved.")
        print("      Re-run with --full-run to train for 1 full epoch and save to outputs/qlora_run_v3/.")
    else:
        print()
        print(f"NOTE: full epoch training completed. LoRA adapter saved to {output_dir}/")
        print("      Run the v3 evaluation step only after user confirmation.")

    # 清理
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA VRAM test / training run")
    parser.add_argument(
        "--model",
        default="huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
        help="HuggingFace model ID (must match forward-pass test model)",
    )
    parser.add_argument(
        "--samples",
        default="data/processed/merged_train_samples.jsonl",
        help="Path to JSONL training samples",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run full 1-epoch training instead of 3-step VRAM probe",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f"Max sequence length (default: {MAX_SEQ_LENGTH}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for adapter save (default: outputs/qlora_vram_test or outputs/qlora_run_v3).",
    )
    args = parser.parse_args()

    # 检查 torch + CUDA（与 _test_unsloth_forward.py 相同检查）
    try:
        import torch
    except ImportError:
        print("ERROR: torch not found — activate .venv-train/ first", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU   : {props.name}")
    print(f"VRAM  : {props.total_memory / 1024**3:.2f} GB")
    print(f"sm    : sm_{props.major}{props.minor}")
    print()

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
