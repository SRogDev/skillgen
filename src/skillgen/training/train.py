"""SkillGen training entry point (PLAN.md §17/§18/§21/§27/§28).

Usage::

    uv run python -m skillgen.training.train --config configs/qlora_e1.yaml

**Production engine: Unsloth** (Roger's explicit direction,
2026-10-23). The config key ``training.backend`` selects the engine:

- ``unsloth`` (E1 and every ablation config E2–E6): the model is
  loaded and LoRA-wrapped by :mod:`skillgen.training.unsloth_backend`
  (``FastLanguageModel.from_pretrained`` on the pre-quantized
  ``unsloth/<model>-bnb-4bit`` checkpoint + ``get_peft_model`` with
  ``use_gradient_checkpointing="unsloth"``), then trained with TRL's
  ``SFTTrainer`` — the canonical Unsloth recipe. The ``peft`` backend
  exists ONLY because Unsloth requires CUDA and the local dev VM has
  none; it is used solely by ``configs/smoke_cpu.yaml``.
- ``peft`` (CPU smoke only): the original plain path —
  ``AutoModelForCausalLM`` + ``BitsAndBytesConfig`` (only when
  ``quantization.enabled``) + a PEFT ``LoraConfig`` given to
  ``SFTTrainer``.

Import order matters: ``import unsloth`` patches transformers/trl/peft
on import, so when the backend is ``unsloth`` this module imports
``unsloth_backend`` BEFORE importing torch/transformers/trl (all heavy
imports are therefore local to :func:`run`).

What this does, in order:

1. Loads the YAML config (``skillgen.training.config``).
2. Loads the model + tokenizer through the selected backend.
3. Builds the PRE-TOKENIZED train/val datasets with
   :mod:`skillgen.training.prepare` — the labels (assistant-only
   masking, PLAN.md §8) are built and unit-tested there, never inside
   the trainer, and reach ``SFTTrainer`` untouched on both backends.
4. **§21 masking gate:** verifies the first training example with
   ``prepare.verify_masking`` and PRINTS the decoded unmasked text.
   Training refuses to start if verification fails — a broken chat
   template can mask everything in silence and produce loss 0.0 with
   zero learning.
5. Trains with ``SFTTrainer``; saves the adapter + tokenizer to the
   config's ``output_dir``. Best checkpoint is selected by val loss
   (``load_best_model_at_end``), with early stopping.

Why ``bitsandbytes`` and ``unsloth`` are NOT project dependencies
(``pyproject.toml``): bitsandbytes ships CUDA kernels — it cannot even
import on this CPU-only VM — and Unsloth pins its own torch/xformers
stack, which would fight the CPU-torch pin used for local development.
Both are installed **in the Kaggle notebook cells** instead
(``notebooks/05_qlora_training.ipynb``), where the CUDA torch they
expect already exists.

TRL API note (installed trl 1.16.0, inspected before writing this):
``SFTTrainer._prepare_dataset`` treats a dataset with an ``input_ids``
column as pre-processed and skips its own tokenization, and a dataset
that already provides ``labels`` is left as is; the default
``DataCollatorForLanguageModeling`` pads our labels with -100 and
uses them unchanged. We set ``max_length`` on ``SFTConfig`` to the
config's ``max_seq_length`` (already enforced by prepare.py's filter,
so TRL's truncation step is a no-op safeguard) and ``packing=False``
(§23). No trainer-internal masking (``assistant_only_loss`` /
``completion_only_loss``) is enabled: the labels come from prepare.py.
On the Unsloth backend the model arrives with LoRA already applied by
``get_peft_model``, so no ``peft_config`` is passed to the trainer,
and gradient checkpointing is NOT enabled in ``SFTConfig`` either —
``get_peft_model(..., use_gradient_checkpointing="unsloth")`` already
turned on Unsloth's own checkpointing. One more API adaptation: the
installed transformers (5.x, pulled in by trl 1.16) removed
``TrainingArguments.warmup_ratio``; ``warmup_steps`` now accepts a
float in [0, 1) meaning "ratio of total steps", so the config's
``warmup_ratio: 0.03`` is passed as ``warmup_steps=0.03`` — same
semantics, new spelling. And TRL 1.16's ``SFTConfig`` enables bf16 by
default, which transformers 5.x rejects on a CPU-only setup, so
``bf16``/``fp16`` are pinned explicitly from the config + hardware.

Experiment tracking (§28): ``report_to=["wandb"]`` ONLY when the
``WANDB_API_KEY`` environment variable is present; otherwise ``[]``
(offline-safe — local and Kaggle runs without a key never phone home).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from skillgen.training.config import load_config
from skillgen.training.prepare import load_training_dataset, verify_masking

REPO_ROOT = Path(__file__).resolve().parents[3]


def _torch_dtype(name: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def run(config_path: str | Path) -> dict:
    config = load_config(config_path)
    training_cfg = config["training"]
    backend = str(training_cfg.get("backend", "unsloth")).lower()
    if backend not in {"unsloth", "peft"}:
        raise ValueError(
            f"training.backend must be 'unsloth' or 'peft', got {backend!r}"
        )
    seed = int(training_cfg.get("seed", 42))

    # --- Backend first: Unsloth MUST be imported before transformers /
    # trl / peft in this process (it patches them on import). Its typed
    # guard also fails fast, before any heavy loading, when unsloth or
    # CUDA is unavailable.
    unsloth_backend = None
    if backend == "unsloth":
        from skillgen.training import unsloth_backend as _ub

        _ub.require_unsloth()
        unsloth_backend = _ub

    import torch
    from transformers import EarlyStoppingCallback, set_seed

    # Seed everywhere (§46).
    random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)

    model_name = config["model"]["name"]

    # --- Model + tokenizer through the selected backend ---
    model = None
    if backend == "unsloth":
        model, tokenizer = unsloth_backend.load_unsloth_model(config)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset (pre-tokenized; labels built + tested in prepare.py) ---
    train_dataset, val_dataset, stats, first_record = load_training_dataset(
        tokenizer, config
    )

    # --- §21 masking gate: verify + print before any training ---
    if config.get("train_on_assistant_only", True):
        first_example = {
            "input_ids": train_dataset[0]["input_ids"],
            "labels": train_dataset[0]["labels"],
        }
        decoded = verify_masking(first_example, tokenizer, first_record)
        print("[gate] §21 masking verification PASSED. Decoded unmasked "
              "tokens of the first training example (must be ONLY the "
              "assistant blueprint + EOS):")
        print(decoded)
    else:
        print("[gate] train_on_assistant_only=false (E2 ablation): "
              "labels are a copy of input_ids; masking gate skipped by design.")

    # --- Model (peft backend loads it here; unsloth already loaded it) ---
    quant_cfg = config.get("quantization", {})
    peft_config = None
    if backend == "peft":
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        model_kwargs: dict = {}
        if quant_cfg.get("enabled"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(quant_cfg.get("double_quant", True)),
                bnb_4bit_compute_dtype=_torch_dtype(quant_cfg.get("compute_dtype", "fp16")),
            )
            model_kwargs["torch_dtype"] = _torch_dtype(quant_cfg.get("compute_dtype", "fp16"))
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # --- LoRA (peft backend only; unsloth applied it in get_peft_model) ---
        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=lora_cfg.get("bias", "none"),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # --- Trainer ---
    gc_setting = training_cfg.get("gradient_checkpointing", False)
    if backend == "unsloth":
        # Unsloth's own checkpointing was enabled by get_peft_model;
        # enabling Trainer's as well would fight it.
        gradient_checkpointing = False
        gc_kwargs = None
    else:
        gradient_checkpointing = bool(gc_setting)
        gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None
    output_dir = str(REPO_ROOT / config["output_dir"])
    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else []

    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        # transformers 5.x: warmup_ratio is gone; warmup_steps accepts a
        # float in [0, 1) as a ratio of total steps (see module docstring).
        warmup_steps=float(training_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        optim=training_cfg.get("optim", "adamw_torch"),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(training_cfg.get("eval_steps", 50)),
        save_strategy="steps",
        save_steps=int(training_cfg.get("save_steps", 50)),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        data_seed=seed,
        report_to=report_to,
        # prepare.py already enforced the cap by filtering; this makes
        # TRL's own truncation a no-op safeguard (see module docstring).
        max_length=int(config["data"]["max_seq_length"]),
        packing=False,
        completion_only_loss=False,
        assistant_only_loss=False,
        # TRL 1.16's SFTConfig enables bf16 by default; transformers 5.x
        # rejects bf16 on a CPU-only setup, so pin both flags explicitly.
        bf16=False,
        fp16=bool(
            quant_cfg.get("enabled")
            and quant_cfg.get("compute_dtype") == "fp16"
            and torch.cuda.is_available()
        ),
    )

    callbacks = []
    patience = training_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=int(patience))
        )

    trainer_kwargs: dict = {}
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[done] backend: {backend}")
    print(f"[done] train loss: {train_result.training_loss}")
    print(f"[done] eval loss: {eval_result.get('eval_loss')}")
    print(f"[done] adapter + tokenizer saved to {output_dir}")
    return {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "stats": stats,
        "output_dir": output_dir,
        "backend": backend,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
