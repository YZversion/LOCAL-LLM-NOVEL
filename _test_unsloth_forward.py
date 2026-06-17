#!/usr/bin/env python3
"""阶段4前置验证 — Unsloth + CUDA 最小前向通过测试。

Usage:
    python _test_unsloth_forward.py                                  # 快速平台验证（0.5B）
    python _test_unsloth_forward.py --model Qwen/Qwen3-8B-Instruct   # 8B VRAM 测试
    python _test_unsloth_forward.py --model Qwen/Qwen2.5-0.5B --no-4bit

Must run inside .venv-train/ (not the phase-2 env).
Exit 0 = PASS (peak VRAM < 8 GB), Exit 1 = FAIL or error.
"""
import argparse
import gc
import sys

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
VRAM_TARGET_GB = 8.0


def check_torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed in this venv", file=sys.stderr)
        return False

    print(f"torch : {torch.__version__}")
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() = False", file=sys.stderr)
        print("  → Possible causes: wrong torch wheel (CPU-only), driver issue", file=sys.stderr)
        return False

    props = torch.cuda.get_device_properties(0)
    print(f"GPU   : {props.name}")
    print(f"VRAM  : {props.total_memory / 1024**3:.2f} GB")
    print(f"sm    : sm_{props.major}{props.minor}")
    if props.major * 10 + props.minor < 70:
        print("ERROR: Compute capability < 7.0; Unsloth requires ≥ 7.0", file=sys.stderr)
        return False
    return True


def check_unsloth() -> bool:
    try:
        import unsloth
        print(f"unsloth : {unsloth.__version__}")
        return True
    except ImportError as e:
        print(f"ERROR: unsloth not installed — {e}", file=sys.stderr)
        print("  → Run: pip install -r requirements-train.txt", file=sys.stderr)
        return False


def run_forward_pass(model_name: str, load_in_4bit: bool) -> tuple[bool, float]:
    import torch
    from unsloth import FastLanguageModel

    print(f"\n--- Loading {model_name} (4bit={load_in_4bit}) ---")
    print("    (First run downloads the model from HuggingFace Hub if not cached)")
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer("你好，世界。", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gb = torch.cuda.memory_reserved() / 1024**3

    print(f"Output (first 120 chars): {response[:120]!r}")
    print(f"Peak VRAM allocated : {peak_gb:.2f} GB")
    print(f"VRAM reserved       : {reserved_gb:.2f} GB")

    del model, tokenizer, outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()

    ok = peak_gb < VRAM_TARGET_GB
    return ok, peak_gb


def main() -> int:
    parser = argparse.ArgumentParser(description="Unsloth + CUDA forward-pass pre-validation")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Load in full precision instead of 4-bit")
    args = parser.parse_args()

    print("=== Unsloth + CUDA 阶段4前置验证 ===\n")

    if not check_torch_cuda():
        return 1
    if not check_unsloth():
        return 1

    try:
        ok, peak_gb = run_forward_pass(args.model, load_in_4bit=not args.no_4bit)
    except Exception as e:
        print(f"\nERROR during forward pass: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    status = "PASS" if ok else "FAIL"
    print(f"\n=== Summary ===")
    print(f"Model     : {args.model}")
    print(f"4-bit     : {not args.no_4bit}")
    print(f"Peak VRAM : {peak_gb:.2f} GB  (target < {VRAM_TARGET_GB} GB)")
    print(f"Result    : [{status}]")

    if args.model == DEFAULT_MODEL:
        print(f"\nNOTE: This was the quick platform check ({DEFAULT_MODEL}).")
        print(f"      Re-run with --model Qwen/Qwen3-8B-Instruct for the actual 8B VRAM measurement.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
