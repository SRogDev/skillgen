"""Unsloth backend — the PRODUCTION training engine for SkillGen.

Roger's explicit direction (2026-10-23): SkillGen's GPU work is done
through **Unsloth** — training here, and (via the helpers below) the
real serving backend, the Kaggle baselines and the GGUF export in
``scripts/export_model.py``. The plain transformers+PEFT path in
``train.py`` survives ONLY as the ``peft`` backend for the CPU smoke,
because Unsloth requires CUDA and the local development VM has none.

IMPORT ORDER — read this before touching any entry point
--------------------------------------------------------
``import unsloth`` must happen **before** ``transformers`` / ``trl`` /
``peft`` are imported in the same process: on import, Unsloth patches
those libraries (attention kernels, gradient checkpointing, trainer
internals). Importing it afterwards silently loses the patches. That
is why ``train.py`` imports this module *first* inside ``run()`` when
the config's backend is ``unsloth``, and only then imports torch /
transformers / trl. This module itself imports unsloth (guarded) at
module level and nothing from transformers.

What Unsloth does and does not do here
--------------------------------------
Unsloth optimises the *model side*: 4-bit NF4 loading of the
pre-quantized checkpoint, patched attention/kernels, and its own
gradient checkpointing (``use_gradient_checkpointing="unsloth"``,
~30% VRAM saving). Training still runs through TRL's ``SFTTrainer`` —
that IS the canonical Unsloth fine-tuning recipe (Unsloth model +
SFTTrainer) — fed with the **pre-tokenized dataset from
:mod:`skillgen.training.prepare`**, so the certified assistant-only
labels (§8/§21, unit-tested in ``tests/test_prepare.py``) reach the
trainer untouched. Unsloth replaces neither our data pipeline nor our
masking: Unsloth's own alternative, ``train_on_responses_only``, is
deliberately NOT used — our masking is verified by decoding the
labels, and the §21 gate in ``train.py`` prints that proof before
every run.

Failure mode
------------
On a machine without unsloth or without CUDA, :func:`require_unsloth`
raises :class:`UnslothUnavailableError` with an actionable message —
never a raw ``ImportError`` traceback for the user.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "UnslothUnavailableError",
    "unsloth_importable",
    "require_unsloth",
    "load_unsloth_model",
    "enable_inference",
    "save_gguf",
]


class UnslothUnavailableError(RuntimeError):
    """Unsloth (or the CUDA GPU it needs) is not available here."""


# Guarded import: this module must be importable on the CPU-only dev VM
# (where unsloth is not — and cannot be — installed) so that callers can
# fail with the typed error above instead of an ImportError at import
# time. When unsloth IS installed, importing it here is also what
# applies its transformers/trl/peft patches, so this import must stay
# ahead of any transformers import in the entry point (see docstring).
_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - depends on the machine
    import unsloth as _unsloth
except Exception as exc:  # ImportError, or a broken CUDA stack, etc.
    _unsloth = None
    _IMPORT_ERROR = exc


def _cuda_available() -> bool:
    """True iff a CUDA GPU is usable (Unsloth is GPU-only)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def unsloth_importable() -> bool:
    """True iff ``import unsloth`` succeeded on this machine."""
    return _unsloth is not None


def require_unsloth():
    """Return the ``unsloth`` module, or raise a typed, actionable error.

    Raises:
        UnslothUnavailableError: if unsloth is not installed (with the
            install instruction — it is a Kaggle-cell / GPU-venv
            install, never a project dependency here) or if no CUDA
            GPU is available.
    """
    if _unsloth is None:
        raise UnslothUnavailableError(
            "unsloth is not installed in this environment "
            f"(import failed: {_IMPORT_ERROR!r}). Unsloth is the "
            "production engine for SkillGen's GPU work and is installed "
            "in the Kaggle notebook cells (notebooks/05_qlora_training.ipynb) "
            "or in a GPU venv with `pip install unsloth` — it is "
            "deliberately NOT a pyproject dependency, because it pins "
            "its own CUDA torch stack. On this CPU-only machine, use "
            "the `peft` backend (configs/smoke_cpu.yaml) instead."
        )
    if not _cuda_available():
        raise UnslothUnavailableError(
            "unsloth is installed but no CUDA GPU is available on this "
            "machine. Unsloth requires CUDA; run this on Kaggle (T4) or "
            "another GPU host, or use the `peft` backend "
            "(configs/smoke_cpu.yaml) for CPU work."
        )
    return _unsloth


def load_unsloth_model(config: dict[str, Any]):
    """Load base model + tokenizer and apply LoRA, the Unsloth way.

    All values come from the YAML config — nothing is hardcoded here:

    - ``FastLanguageModel.from_pretrained(model_name=config model.name,
      max_seq_length=config data.max_seq_length, dtype=None,
      load_in_4bit=config quantization.enabled)`` — ``dtype=None``
      lets Unsloth auto-select (fp16 on the T4, which has no bf16).
      The E1 ``model.name`` is the pre-quantized
      ``unsloth/Qwen2.5-7B-Instruct-bnb-4bit`` checkpoint (PLAN §15).
    - ``FastLanguageModel.get_peft_model(...)`` with r / alpha /
      dropout / bias / target_modules from ``config lora``,
      ``use_gradient_checkpointing`` from ``config training``
      (``"unsloth"`` for E1) and ``random_state`` from the seed.

    Returns ``(model, tokenizer)`` with LoRA already applied — callers
    must NOT pass a separate ``peft_config`` to the trainer.
    """
    unsloth = require_unsloth()
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=int(config["data"]["max_seq_length"]),
        dtype=None,
        load_in_4bit=bool(config.get("quantization", {}).get("enabled", True)),
    )
    gc_setting = config.get("training", {}).get("gradient_checkpointing")
    if gc_setting == "unsloth":
        use_gc: Any = "unsloth"
    elif gc_setting:
        use_gc = True
    else:
        use_gc = False
    lora_cfg = config["lora"]
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=lora_cfg.get("bias", "none"),
        target_modules=list(lora_cfg["target_modules"]),
        use_gradient_checkpointing=use_gc,
        random_state=int(config.get("training", {}).get("seed", 42)),
    )
    return model, tokenizer


def enable_inference(model):
    """Switch a trained/loaded model to Unsloth's fast inference mode.

    Wraps ``FastLanguageModel.for_inference`` (enables Unsloth's
    native fast generation). Used by the serving backend and the
    notebook's post-training sanity check.
    """
    unsloth = require_unsloth()
    return unsloth.FastLanguageModel.for_inference(model)


def save_gguf(model, tokenizer, output_dir: str, quantization_method: str = "q4_k_m"):
    """Export GGUF via ``model.save_pretrained_gguf`` (PLAN §44).

    The GGUF is always produced **from the merged 16-bit model** by the
    export flow in ``scripts/export_model.py`` — never by merging
    directly into 4-bit. This helper exists so the call has one typed
    home; ``scripts/export_model.py`` keeps its own explicit plan.
    """
    require_unsloth()
    model.save_pretrained_gguf(
        output_dir, tokenizer, quantization_method=quantization_method
    )
