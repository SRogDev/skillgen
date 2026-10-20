"""Tests for Phase 3 training data preparation (PLAN.md §8/§21/§23).

Three groups:

1. **Masking** — the §21 critical check. On a REAL train record, the
   tokens whose label is not -100 must decode to *exactly* the record's
   ``expected_blueprint`` plus the EOS token, and every prompt position
   must be masked. EOS handling (documented in ``prepare.py``): the Qwen
   chat template renders the assistant turn as ``{content}<|im_end|>\\n``;
   ``build_example`` cuts the example right after ``<|im_end|>``, so the
   decoded unmasked text is ``expected_blueprint + tokenizer.eos_token``
   with no trailing newline.
2. **Length filter** — the Phase-0 decision (filter, never truncate):
   a synthetic record rendered longer than a tiny ``max_seq_length`` is
   excluded by the loader, a short one is kept.
3. **Configs** — every E-variant config differs from E1 in EXACTLY its
   intended key(s) (PLAN.md §19: one change per experiment). Identity
   fields (``experiment_id``, ``output_dir``) are excluded from the diff:
   they name the run, they are not hyperparameters.

The tokenizer used here is Qwen2.5-0.5B-Instruct's: same vocab and chat
template family as the (provisional) 7B base, already cached locally,
and enough to certify the masking logic — which is tokenizer-agnostic
code in ``prepare.py``, re-verified on the 7B tokenizer inside notebook
05's masking gate before the real Kaggle run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from skillgen.evaluation.baseline import load_split_records
from skillgen.training.config import flatten_config, load_config
from skillgen.training.prepare import (
    build_example,
    load_training_dataset,
    verify_masking,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def shortest_train_record():
    """The shortest REAL train record (by rendered char length)."""
    records = load_split_records("train")
    return min(
        records,
        key=lambda r: len(r["instruction"]) + len(r["expected_blueprint"]),
    )


# ---------------------------------------------------------------------------
# 1. Masking (§21)
# ---------------------------------------------------------------------------

def test_masking_decodes_to_exactly_the_blueprint(tokenizer, shortest_train_record):
    example = build_example(shortest_train_record, tokenizer)

    assert len(example["labels"]) == len(example["input_ids"])

    unmasked = [
        tok for tok, lab in zip(example["input_ids"], example["labels"]) if lab != -100
    ]
    assert unmasked, "masking must leave the assistant tokens unmasked"

    # Every position before the first unmasked token is a prompt token and
    # must be masked; every unmasked label must equal its input id.
    first_unmasked = example["labels"].index(unmasked[0])
    assert all(lab == -100 for lab in example["labels"][:first_unmasked])
    for tok, lab in zip(example["input_ids"], example["labels"]):
        assert lab == -100 or lab == tok

    decoded = tokenizer.decode(unmasked)
    # EOS suffix handling: prepare.py cuts after <|im_end|>, so the decoded
    # unmasked text is exactly the blueprint plus the EOS token string.
    assert decoded == shortest_train_record["expected_blueprint"] + tokenizer.eos_token

    # The example itself must END at the EOS token (PLAN.md §21: EOS at
    # the end of each example, or the model never learns to stop).
    assert example["input_ids"][-1] == tokenizer.eos_token_id
    assert example["labels"][-1] == tokenizer.eos_token_id


def test_verify_masking_helper_agrees(tokenizer, shortest_train_record):
    example = build_example(shortest_train_record, tokenizer)
    decoded = verify_masking(example, tokenizer, shortest_train_record)
    assert decoded == shortest_train_record["expected_blueprint"] + tokenizer.eos_token


def test_masking_off_labels_are_a_copy(tokenizer, shortest_train_record):
    """E2's flag: train_on_assistant_only=False → labels == input_ids."""
    example = build_example(
        shortest_train_record, tokenizer, train_on_assistant_only=False
    )
    assert example["labels"] == example["input_ids"]


# ---------------------------------------------------------------------------
# 2. Length filter (Phase-0 decision: filter, never truncate)
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def test_loader_excludes_records_over_max_seq(tokenizer, tmp_path):
    short = {"instruction": "hi", "expected_blueprint": "{}",
             "metadata": {"id": "short"}}
    long = {"instruction": "hi",
            "expected_blueprint": json.dumps({"blob": "x" * 5000}),
            "metadata": {"id": "long"}}
    _write_jsonl(tmp_path / "train_shard_00.jsonl", [short, long])
    _write_jsonl(tmp_path / "val.jsonl", [short])

    config = {
        "train_on_assistant_only": True,
        "data": {
            "max_seq_length": 200,
            "dataset_limit": None,
            "eval_limit": None,
            "selection": "first",
        },
    }
    train_ds, val_ds, stats, first_record = load_training_dataset(
        tokenizer, config, splits_dir=tmp_path
    )
    assert first_record["metadata"]["id"] == "short"

    assert len(train_ds) == 1
    assert stats["train_total"] == 2
    assert stats["train_excluded_over_cap"] == 1
    assert stats["train_kept"] == 1
    # Nothing was truncated: the kept example is whole and ends at EOS.
    assert train_ds[0]["input_ids"][-1] == tokenizer.eos_token_id
    # Val is built for loss only; its short record is present.
    assert len(val_ds) == 1
    assert stats["val_kept"] == 1


# ---------------------------------------------------------------------------
# 3. Config diffs (§19: exactly one change per experiment)
# ---------------------------------------------------------------------------

IDENTITY_KEYS = {"experiment_id", "output_dir"}

VARIANTS = {
    "qlora_e2_no_masking.yaml": {"train_on_assistant_only"},
    "qlora_e3_epochs1.yaml": {"training.num_train_epochs"},
    "qlora_e3_epochs3.yaml": {"training.num_train_epochs"},
    "qlora_e4_rank8.yaml": {"lora.r"},
    "qlora_e4_rank32.yaml": {"lora.r"},
    "qlora_e5_attention_only.yaml": {"lora.target_modules"},
    "qlora_e6_data250.yaml": {"data.dataset_limit"},
    "qlora_e6_data500.yaml": {"data.dataset_limit"},
}


def _diff_keys(a: dict, b: dict) -> set[str]:
    fa = {k: v for k, v in flatten_config(a).items() if k not in IDENTITY_KEYS}
    fb = {k: v for k, v in flatten_config(b).items() if k not in IDENTITY_KEYS}
    return {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


def test_e1_config_matches_plan_recipe():
    cfg = load_config(CONFIGS_DIR / "qlora_e1.yaml")
    assert cfg["train_on_assistant_only"] is True
    assert cfg["model"]["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["quantization"] == {
        "enabled": True, "bits": 4, "quant_type": "nf4",
        "double_quant": True, "compute_dtype": "fp16",
    }
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["alpha"] == 32
    assert cfg["lora"]["dropout"] == 0.0
    assert cfg["lora"]["bias"] == "none"
    assert cfg["lora"]["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    t = cfg["training"]
    assert t["learning_rate"] == 2e-4
    assert t["num_train_epochs"] == 2
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 16
    assert t["warmup_ratio"] == 0.03
    assert t["weight_decay"] == 0.01
    assert t["optim"] == "adamw_8bit"
    assert t["lr_scheduler_type"] == "cosine"
    assert t["gradient_checkpointing"] == "unsloth"
    assert t["max_grad_norm"] == 1.0
    assert t["seed"] == 42
    # Phase-0 override of the provisional PLAN §18 value 4096.
    assert cfg["data"]["max_seq_length"] == 8192


@pytest.mark.parametrize("filename,expected", sorted(VARIANTS.items()))
def test_variant_configs_differ_only_as_intended(filename, expected):
    e1 = load_config(CONFIGS_DIR / "qlora_e1.yaml")
    variant = load_config(CONFIGS_DIR / filename)
    assert _diff_keys(e1, variant) == expected


def test_smoke_config_is_clearly_a_smoke():
    cfg = load_config(CONFIGS_DIR / "smoke_cpu.yaml")
    assert cfg["quantization"]["enabled"] is False  # plain LoRA, not QLoRA
    assert cfg["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["data"]["dataset_limit"] == 20
    assert cfg["data"]["selection"] == "shortest"
    assert cfg["output_dir"] == "experiments/e1_smoke/adapter"
