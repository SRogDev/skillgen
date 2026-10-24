"""Training-data preparation for SkillGen QLoRA (PLAN.md §8/§21/§23).

This module turns frozen split records into a **pre-tokenized** dataset of
``{input_ids, labels}`` where the labels implement the §8 rule: loss ONLY
on assistant tokens. The labels are built HERE — never by a trainer
internally — because this is the code the §21 masking test certifies
(``tests/test_prepare.py``): decode the tokens whose label is not -100 and
you must get exactly the record's blueprint (+ EOS).

Chat-template / EOS handling (verified against the Qwen2.5 tokenizer):

- Prompt text: ``apply_chat_template([system, user],
  add_generation_prompt=True)``; full text:
  ``apply_chat_template([system, user, assistant])``. Both are tokenized
  with ``add_special_tokens=False`` — the template already emits every
  special token (``<|im_start|>`` / ``<|im_end|>``).
- The Qwen template renders the assistant turn as
  ``{content}<|im_end|>\\n``: the rendered full text ends with the EOS
  token ``<|im_end|>`` followed by a trailing newline (scaffolding for a
  hypothetical next turn). :func:`build_example` cuts the example right
  after the EOS token, so **every example ends exactly at EOS** (§21:
  EOS at the end of each example, or the model never learns to stop) and
  the decoded unmasked text is exactly
  ``expected_blueprint + tokenizer.eos_token``.
- Prefix property: the prompt tokens must be an exact prefix of the full
  tokens. :func:`build_example` asserts it and raises otherwise — a
  template/tokenizer change that breaks it would silently mislabel every
  example, so it must fail loudly, never "approximately".

Length rule (Phase-0 decision, ``experiments/eda_v1/report.md``,
OVERRIDING the provisional PLAN.md §18 value of 4096):
``max_seq_length = 8192`` and **filter, never truncate** — train records
rendered longer than the cap are excluded whole (800 → 582 kept), because
a truncated blueprint is invalid JSON. Val records over the cap are
excluded from the eval-loss set only (counted and printed); val/test are
never excluded from *evaluation* (Phase 2/4 scoring handles them), and
the test split is never loaded here at all.

TRL note (installed trl 1.16.0, inspected 2026-10-20):
``SFTTrainer._prepare_dataset`` treats a dataset that already has an
``input_ids`` column as pre-processed and skips its own tokenization; a
dataset that already provides ``labels`` is left as is ("A dataset that
already provides a labels column is left as is"), and the default
``DataCollatorForLanguageModeling`` pads those labels with -100 and uses
them unchanged. So the labels certified here are the labels trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from skillgen.evaluation.baseline import SYSTEM_PROMPT, load_split_records

#: Default cap from the Phase-0 length decision (see module docstring).
DEFAULT_MAX_SEQ_LENGTH = 8192


def build_example(
    record: dict,
    tokenizer,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    train_on_assistant_only: bool = True,
) -> dict[str, list[int]]:
    """Build one pre-tokenized ``{input_ids, labels}`` example.

    ``max_seq_length`` is accepted for signature stability with the
    loader but deliberately NOT enforced here: filtering (never
    truncation) is the loader's job, and it needs the true rendered
    length to count exclusions honestly.
    """
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["instruction"]},
    ]
    messages_full = messages_prompt + [
        {"role": "assistant", "content": record["expected_blueprint"]}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages_full, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # EOS handling: drop trailing whitespace-only tokens after the EOS
    # token (the Qwen template's trailing "\n"), then require the example
    # to end exactly at EOS. See the module docstring.
    while (
        full_ids
        and full_ids[-1] != tokenizer.eos_token_id
        and tokenizer.decode([full_ids[-1]]).strip() == ""
    ):
        full_ids.pop()
    if not full_ids or full_ids[-1] != tokenizer.eos_token_id:
        raise ValueError(
            "rendered example does not end with the EOS token "
            f"({tokenizer.eos_token!r}); last tokens: "
            f"{tokenizer.convert_ids_to_tokens(full_ids[-3:])}"
        )

    # Prefix property: prompt tokens must be an exact prefix of the full
    # tokens, or the -100 boundary would mislabel tokens. Fail loudly.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat-template prefix property violated: tokenized prompt is "
            "not a prefix of the tokenized full conversation; refusing to "
            "build labels that would silently mislabel tokens."
        )

    if train_on_assistant_only:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    else:
        labels = list(full_ids)
    return {"input_ids": full_ids, "labels": labels}


def verify_masking(example: dict, tokenizer, record: dict) -> str:
    """The PLAN.md §21 critical check.

    Decode the tokens whose label is not -100 and assert the result is
    exactly the record's ``expected_blueprint`` plus the EOS token string
    (the trailing-EOS form documented in the module docstring). Returns
    the decoded text so callers can print it as a pre-training gate.
    """
    unmasked = [
        tok for tok, lab in zip(example["input_ids"], example["labels"]) if lab != -100
    ]
    if not unmasked:
        raise ValueError("masking verification failed: no unmasked tokens")
    decoded = tokenizer.decode(unmasked)
    expected = record["expected_blueprint"] + tokenizer.eos_token
    if decoded != expected:
        raise ValueError(
            "masking verification failed: decoded unmasked tokens do not "
            "equal expected_blueprint + EOS.\n"
            f"decoded[:200]  = {decoded[:200]!r}\n"
            f"expected[:200] = {expected[:200]!r}"
        )
    return decoded


def _build_filtered(
    records: list[dict],
    tokenizer,
    max_seq_length: int,
    train_on_assistant_only: bool,
) -> tuple[list[tuple[dict, dict]], int]:
    """Build (record, example) pairs, excluding over-cap records.

    Exclusion is by whole record (filter, never truncate — see the
    module docstring). Pairs are kept so callers can map a dataset row
    back to its source record (the §21 masking gate needs exactly that).
    """
    pairs: list[tuple[dict, dict]] = []
    excluded = 0
    for record in records:
        example = build_example(
            record,
            tokenizer,
            max_seq_length=max_seq_length,
            train_on_assistant_only=train_on_assistant_only,
        )
        if len(example["input_ids"]) > max_seq_length:
            excluded += 1
            continue
        pairs.append((record, example))
    return pairs, excluded


def load_training_dataset(
    tokenizer,
    config: dict[str, Any],
    splits_dir: str | Path | None = None,
) -> tuple[Dataset, Dataset, dict[str, int], dict | None]:
    """Load the train split (filtered) and the val eval set.

    Only the TRAIN split is ever trained on; val is returned separately
    and used for eval loss only. The frozen test split is never loaded.

    Config keys used: ``train_on_assistant_only`` and the ``data``
    section (``max_seq_length``, ``dataset_limit``, ``eval_limit``,
    ``selection`` — ``"first"`` keeps deterministic file order,
    ``"shortest"`` sorts the filtered pool by rendered length before
    limiting; the CPU smoke uses it to stay fast).

    Returns ``(train_dataset, val_dataset, stats, first_train_record)``
    where ``stats`` counts totals/kept/excluded for both splits and
    ``first_train_record`` is the source record of ``train_dataset[0]``
    (after selection/limit) — train.py's §21 gate verifies the masking
    of exactly that example.
    """
    data_cfg = config.get("data", {})
    max_seq_length = int(data_cfg.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH))
    dataset_limit = data_cfg.get("dataset_limit")
    eval_limit = data_cfg.get("eval_limit")
    selection = data_cfg.get("selection", "first")
    masked = bool(config.get("train_on_assistant_only", True))

    train_records = load_split_records("train", splits_dir=splits_dir)
    val_records = load_split_records("val", splits_dir=splits_dir)

    train_pairs, train_excluded = _build_filtered(
        train_records, tokenizer, max_seq_length, masked
    )
    if selection == "shortest":
        train_pairs.sort(key=lambda pair: len(pair[1]["input_ids"]))
    elif selection != "first":
        raise ValueError(f"unknown data.selection: {selection!r}")
    if dataset_limit is not None:
        train_pairs = train_pairs[: int(dataset_limit)]
    train_examples = [example for _, example in train_pairs]
    first_train_record = train_pairs[0][0] if train_pairs else None

    val_pairs, val_excluded = _build_filtered(
        val_records, tokenizer, max_seq_length, masked
    )
    if eval_limit is not None:
        val_pairs = val_pairs[: int(eval_limit)]
    val_examples = [example for _, example in val_pairs]

    stats = {
        "train_total": len(train_records),
        "train_excluded_over_cap": train_excluded,
        "train_kept": len(train_examples),
        "val_total": len(val_records),
        "val_excluded_over_cap": val_excluded,
        "val_kept": len(val_examples),
    }
    print(
        f"[prepare] train: {stats['train_total']} records, "
        f"{train_excluded} excluded over {max_seq_length} tokens "
        f"(filter, never truncate), {len(train_examples)} in dataset"
    )
    print(
        f"[prepare] val (eval loss only, never trained on): "
        f"{stats['val_total']} records, {val_excluded} over cap excluded "
        f"from the eval set, {len(val_examples)} in eval set"
    )
    return (
        Dataset.from_list(train_examples),
        Dataset.from_list(val_examples),
        stats,
        first_train_record,
    )
