"""QLoRA training (PLAN §15-28). Requires a CUDA GPU with compute capability >= 7 (T4+).

Functions mirror PLAN §52: load_model, load_dataset, prepare_dataset, configure_lora,
configure_training, verify_masking, train, save_adapter.
"""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import yaml

from skillgen.data import filter_by_length, length_report, load_split, to_messages

# Qwen2.5 / ChatML turn markers (PLAN §21). Change together with the base model.
RESPONSE_PARTS = {
    "qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama": ("<|start_header_id|>user<|end_header_id|>\n\n",
              "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def check_environment() -> dict:
    import torch
    assert torch.cuda.is_available(), "No GPU: training needs a CUDA GPU (Kaggle T4)."
    major, minor = torch.cuda.get_device_capability()
    assert major >= 7, f"GPU sm_{major}{minor} unsupported by 4-bit kernels (P100?). Switch to T4."
    props = torch.cuda.get_device_properties(0)
    return {"gpu": props.name, "vram_gb": round(props.total_memory / 1e9, 1),
            "sm": f"{major}.{minor}", "bf16": major >= 8, "torch": torch.__version__,
            "cuda": torch.version.cuda}


def load_model(cfg: dict):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["training"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,  # auto: fp16 on T4, bf16 on Ampere+
    )
    return model, tokenizer


def configure_lora(model, cfg: dict):
    from unsloth import FastLanguageModel
    lc = cfg["lora"]
    return FastLanguageModel.get_peft_model(
        model, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        bias=lc["bias"], target_modules=lc["target_modules"],
        use_gradient_checkpointing="unsloth", random_state=cfg["training"]["seed"],
    )


def render(rec: dict, tokenizer) -> str:
    """Native chat template (PLAN §8); EOS comes from the template's end-of-turn token."""
    return tokenizer.apply_chat_template(to_messages(rec), tokenize=False)


def prepare_dataset(split: str, tokenizer, max_len: int, out_dir: Path):
    """Render with the chat template, drop over-length examples, log what was dropped."""
    from datasets import Dataset
    recs = load_split(split)
    count = lambda r: len(tokenizer(render(r, tokenizer), add_special_tokens=False)["input_ids"])
    kept, dropped = filter_by_length(recs, count, max_len)
    report = length_report(dropped, len(recs), max_len)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"length_report_{split}.json").write_text(json.dumps(report, indent=1))
    print(f"[{split}] kept {report['kept']}/{report['total']} at max_seq_length={max_len}")
    return Dataset.from_list([{"text": render(r, tokenizer)} for r in kept]), report


def configure_training(cfg: dict, out_dir: Path, bf16: bool):
    from trl import SFTConfig
    t = cfg["training"]
    kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        dataset_text_field="text",
        packing=False,  # PLAN §23
        learning_rate=float(t["lr"]),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["grad_accum"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optimizer"],
        lr_scheduler_type=t["scheduler"],
        max_grad_norm=t["max_grad_norm"],
        seed=t["seed"],
        fp16=not bf16, bf16=bf16,
        logging_steps=5,
        eval_strategy="steps", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["eval_steps"], save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=cfg.get("tracking", {}).get("report_to", "none"),
        run_name=cfg["experiment_id"],
    )
    # TRL renamed max_seq_length -> max_length; support both.
    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs["max_length" if "max_length" in params else "max_seq_length"] = t["max_seq_length"]
    return SFTConfig(**kwargs)


def verify_masking(trainer, tokenizer) -> str:
    """PLAN §21: decode labels != -100 of one example; must be ONLY the assistant turn."""
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    kept = labels[labels != -100]
    assert len(kept) > 0, "All labels masked -> loss=0, nothing learns. Check response_part."
    text = tokenizer.decode(kept)
    assert "You are SkillGen" not in text, "System prompt is NOT masked."
    assert text.lstrip().startswith("{"), f"Supervised text should start with the JSON: {text[:80]!r}"
    print(f"[mask ok] {len(kept)}/{(labels != -100).numel()} label tokens; starts: {text[:120]!r}")
    return text


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(config_path: str) -> Path:
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    cfg = load_config(config_path)
    out_dir = Path(cfg.get("output_dir", "experiments")) / cfg["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = check_environment()
    print("[env]", env)

    model, tokenizer = load_model(cfg)
    model = configure_lora(model, cfg)
    max_len = cfg["training"]["max_seq_length"]
    train_ds, train_rep = prepare_dataset("train", tokenizer, max_len, out_dir)
    val_ds, val_rep = prepare_dataset("val", tokenizer, max_len, out_dir)

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=configure_training(cfg, out_dir, env["bf16"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["training"]["early_stopping_patience"])],
    )
    instr, resp = RESPONSE_PARTS[cfg["model"]["family"]]
    trainer = train_on_responses_only(trainer, instruction_part=instr, response_part=resp)
    verify_masking(trainer, tokenizer)

    result = trainer.train(resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint"))
    adapter_dir = save_adapter(model, tokenizer, out_dir)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "config": cfg, "env": env, "git_commit": _git_commit(),
        "train_kept": train_rep["kept"], "val_kept": val_rep["kept"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "train_metrics": result.metrics,
    }, indent=1, default=str))
    return adapter_dir


def save_adapter(model, tokenizer, out_dir: Path) -> Path:
    """Save LoRA adapter + tokenizer only (PLAN §27), not the merged model."""
    d = out_dir / "adapter"
    model.save_pretrained(str(d))
    tokenizer.save_pretrained(str(d))
    return d
