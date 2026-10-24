#!/usr/bin/env python3
"""Export a trained SkillGen adapter for serving (PLAN.md §44).

Pipeline (standard, in this exact order):

    adapter -> merge in 16-bit (``merged_16bit``) -> GGUF ``q4_k_m``
    quantized FROM THE MERGED model -> Ollama

NEVER merge directly to 4-bit: the GGUF quantization step always
starts from the merged 16-bit model.

This script first builds an explicit, ordered EXPORT PLAN (pure data,
importable and unit-testable via :func:`build_export_plan`). Execution
needs a GPU + ``unsloth``; without them (e.g. this CPU-only VM) use
``--dry-run`` to print the plan and the exact commands to run on
Kaggle / a local GPU machine. Without ``--dry-run`` and without a
GPU + unsloth, it prints the plan and exits non-zero with a clear
message — nothing is executed and no traceback is shown.

Usage:
    python scripts/export_model.py --adapter adapters/e1 \
        --base Qwen/Qwen2.5-7B-Instruct --out-dir exports/skillgen-v1 \
        [--quantization q4_k_m] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_export_plan(
    adapter: str,
    base: str,
    out_dir: str,
    quantization: str = "q4_k_m",
) -> list[dict]:
    """Return the ordered export plan as a list of step dicts.

    Pure function: builds data only, executes nothing. Step order is
    the §44 contract — ``merged_16bit`` strictly before ``gguf``, and
    the GGUF step's source is the merged 16-bit model directory.
    """
    out = Path(out_dir)
    merged_dir = str(out / "merged_16bit")
    gguf_dir = str(out / "gguf")
    return [
        {
            "name": "load_base_and_adapter",
            "detail": (
                f"Load base model {base!r} in 16-bit and attach the "
                f"LoRA adapter at {adapter!r} (unsloth FastLanguageModel)."
            ),
            "base": base,
            "adapter": adapter,
        },
        {
            "name": "merged_16bit",
            "detail": (
                "Merge the adapter into the base weights in 16-bit "
                f"precision and save the merged model to {merged_dir} "
                "(save_pretrained_merged(..., save_method='merged_16bit')). "
                "This is the ONLY merge step, and it merges at full 16-bit "
                "precision — quantization happens later, in the gguf step."
            ),
            "output_dir": merged_dir,
            "save_method": "merged_16bit",
        },
        {
            "name": "gguf",
            "detail": (
                f"Quantize the MERGED 16-bit model at {merged_dir} to GGUF "
                f"with quantization_method={quantization!r} and save to "
                f"{gguf_dir} (save_pretrained_gguf run on the merged model). "
                "The source of this step is merged_16bit, NOT the adapter "
                "and NOT a direct 4-bit merge."
            ),
            "source": merged_dir,
            "output_dir": gguf_dir,
            "quantization": quantization,
        },
        {
            "name": "ollama_create",
            "detail": (
                "Serve the GGUF locally with Ollama: inspect the Modelfile "
                "that Ollama/Unsloth generates (its chat template MUST match "
                f"the training template, PLAN.md §44/§21), then run: "
                f"ollama create skillgen -f {gguf_dir}/Modelfile && "
                "ollama run skillgen"
            ),
            "gguf_dir": gguf_dir,
        },
    ]


def format_plan(plan: list[dict]) -> str:
    lines = ["Export plan (PLAN.md §44):"]
    for i, step in enumerate(plan, start=1):
        lines.append(f"  {i}. {step['name']}: {step['detail']}")
    return "\n".join(lines)


def gpu_commands(adapter: str, base: str, out_dir: str, quantization: str) -> str:
    """The exact commands to execute this export on a GPU machine."""
    return (
        "To execute on Kaggle / a local GPU machine:\n"
        "  pip install unsloth  # Kaggle cell or GPU venv; NOT a local dep here\n"
        f"  python scripts/export_model.py --adapter {adapter} --base {base} "
        f"--out-dir {out_dir} --quantization {quantization}\n"
        f"  # then inspect {out_dir}/gguf/Modelfile before: ollama create skillgen "
        f"-f {out_dir}/gguf/Modelfile"
    )


def _gpu_and_unsloth_available() -> bool:
    try:
        import torch
        import unsloth  # noqa: F401
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def execute_plan(plan: list[dict]) -> None:
    """Execute the plan via the unsloth API. GPU + unsloth only."""
    from unsloth import FastLanguageModel

    load_step, merge_step, gguf_step = plan[0], plan[1], plan[2]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_step["base"],
        dtype=None,
        load_in_4bit=False,
    )
    # Attach the adapter weights saved by training.
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, load_step["adapter"])
    merged = model.merge_and_unload()
    merged.save_pretrained(merge_step["output_dir"])
    tokenizer.save_pretrained(merge_step["output_dir"])
    # GGUF quantization FROM THE MERGED model (never a direct 4-bit merge).
    merged.save_pretrained_gguf(
        gguf_step["output_dir"],
        tokenizer,
        quantization_method=gguf_step["quantization"],
    )
    print(
        "Export executed. Next: inspect the generated Modelfile, then "
        f"ollama create skillgen -f {gguf_step['output_dir']}/Modelfile"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter")
    ap.add_argument("--base", required=True, help="Base model id or path")
    ap.add_argument("--out-dir", required=True, help="Export output directory")
    ap.add_argument("--quantization", default="q4_k_m", help="GGUF quantization (default q4_k_m)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the export plan and GPU commands; execute nothing",
    )
    args = ap.parse_args(argv)

    plan = build_export_plan(
        adapter=args.adapter,
        base=args.base,
        out_dir=args.out_dir,
        quantization=args.quantization,
    )
    print(format_plan(plan))

    if args.dry_run:
        print()
        print(gpu_commands(args.adapter, args.base, args.out_dir, args.quantization))
        print("\nDry run: nothing was executed.")
        return 0

    if not _gpu_and_unsloth_available():
        print(
            "\nCannot execute: this export requires a GPU + unsloth, which "
            "are not available here. Nothing was executed.\n"
            + gpu_commands(args.adapter, args.base, args.out_dir, args.quantization),
            file=sys.stderr,
        )
        return 2

    execute_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
