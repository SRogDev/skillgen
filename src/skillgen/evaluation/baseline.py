"""Baseline (experiment E0) generation and scoring for SkillGen.

This module builds the prompts for the untrained base model, generates
blueprint predictions with a Hugging Face causal LM, and scores them with
the *existing* 5-layer harness in ``scripts/eval_harness.py`` — its
functions (``l1_parse``, ``l2_validate_schema``, ``l3_field_scores``,
``l5_materialize``) are imported and reused here, never reimplemented.

Prompt contract (PLAN.md §8): the model's native chat template, a fixed
system prompt, and — for few-shot — fixed exemplars taken ONLY from the
train split. The frozen test split is never used for exemplars and is
never modified by anything in this module.

SkillGen Score and the skipped L4 layer
---------------------------------------
PLAN.md §33 defines::

    SkillGen Score = 0.25*L1 + 0.25*L2 + 0.20*L3 + 0.15*L4 + 0.15*L5

where ``L3`` is the mean of the harness's three L3 sub-metrics (name
exact-match, file-path F1, description token-F1). L4 is the LLM judge,
which needs ``OPENROUTER_API_KEY``; ``scripts/eval_harness.py`` *omits*
L4 from its report when the judge is not run. Following that behaviour,
:func:`score_predictions` reports ``l4_judge == "skipped"`` and computes
the SkillGen Score **renormalised over the available layers**: the
weighted sum of L1/L2/L3/L5 is divided by their total weight
(0.25 + 0.25 + 0.20 + 0.15 = 0.85), so the score stays on a 0–1 scale
and a perfect no-judge run scores exactly 1.0. The un-renormalised
weighted sum is also returned as ``skillgen_score_weighted_sum_no_l4``
(maximum 0.85) so nobody mistakes the two.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Sequence

from skillgen.data.eda import load_jsonl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: System prompt, PLAN.md §8, EXACT — do not reword.
SYSTEM_PROMPT = (
    "You are SkillGen, an expert designer of reusable Agent Skills. "
    "Given the user request, output ONLY a valid JSON Skill Blueprint. "
    "No explanations, no markdown fences."
)

#: The real E0 configuration (PLAN.md §15/§34): the untrained 7B base model.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REAL_E0_MODEL_ID = DEFAULT_MODEL_ID

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"

#: Fixed few-shot exemplars (E0b): the first 3 records of
#: ``train_shard_00.jsonl`` whose rendered length — ``len(instruction) +
#: len(expected_blueprint)`` in characters — is "moderate", defined here
#: as 5,000–10,000 characters (the shard is sorted by size descending and
#: most records are far larger; a moderate exemplar keeps the few-shot
#: prompt usable). Their rendered lengths are 9,869 / 9,985 / 9,934 chars.
#: Reproducible: the split files are frozen (sha256 in the manifest).
FEW_SHOT_EXEMPLAR_IDS = (
    "anthropics_claude-plugins-official__writing-rules",
    "kostja94_marketing-skills__indexing",
    "affaan-m_ecc__blender-motion-state-inspection",
)

#: Weights from PLAN.md §33.
SKILLGEN_WEIGHTS = {"l1": 0.25, "l2": 0.25, "l3": 0.20, "l4": 0.15, "l5": 0.15}


# ---------------------------------------------------------------------------
# Eval harness import (scripts/eval_harness.py is not a package module)
# ---------------------------------------------------------------------------

def _load_eval_harness():
    path = REPO_ROOT / "scripts" / "eval_harness.py"
    spec = importlib.util.spec_from_file_location("skillgen_eval_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_harness = _load_eval_harness()


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------

def load_split_records(
    split: str, limit: int | None = None, splits_dir: str | Path | None = None
) -> list[dict]:
    """Load records for ``split`` in {"train", "val", "test"}.

    Train is all ``train_shard_*.jsonl`` files concatenated in filename
    order (800 records); val/test are their single files (100 each).
    ``limit`` keeps only the first ``limit`` records.
    """
    base = Path(splits_dir) if splits_dir is not None else DEFAULT_SPLITS_DIR
    if split == "train":
        files = sorted(base.glob("train_shard_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no train shards found in {base}")
        records: list[dict] = []
        for path in files:
            records.extend(load_jsonl(path))
    elif split in ("val", "test"):
        records = load_jsonl(base / f"{split}.jsonl")
    else:
        raise ValueError(f"unknown split: {split!r} (expected train/val/test)")
    if limit is not None:
        records = records[:limit]
    return records


def _record_id(record: dict) -> str:
    return record["metadata"]["id"]


# ---------------------------------------------------------------------------
# Few-shot exemplars (train split ONLY)
# ---------------------------------------------------------------------------

def get_few_shot_exemplars(splits_dir: str | Path | None = None) -> list[dict]:
    """Return the 3 fixed few-shot exemplar records, in fixed order.

    The records are looked up by :data:`FEW_SHOT_EXEMPLAR_IDS` in
    ``train_shard_00.jsonl`` and validated against every split: an
    exemplar that is missing from train, or that also appears in val or
    test, raises ``AssertionError``.
    """
    base = Path(splits_dir) if splits_dir is not None else DEFAULT_SPLITS_DIR
    shard00 = load_jsonl(base / "train_shard_00.jsonl")
    by_id = {_record_id(r): r for r in shard00}
    exemplars = []
    for exemplar_id in FEW_SHOT_EXEMPLAR_IDS:
        if exemplar_id not in by_id:
            raise AssertionError(
                f"few-shot exemplar {exemplar_id!r} not found in train_shard_00"
            )
        record = by_id[exemplar_id]
        rendered_len = len(record["instruction"]) + len(record["expected_blueprint"])
        if not 5_000 <= rendered_len <= 10_000:
            raise AssertionError(
                f"exemplar {exemplar_id!r} rendered length {rendered_len} "
                "is outside the documented moderate band (5,000-10,000 chars)"
            )
        exemplars.append(record)
    validate_exemplars_are_train(exemplars, splits_dir=splits_dir)
    return exemplars


def validate_exemplars_are_train(
    exemplars: Sequence[dict], splits_dir: str | Path | None = None
) -> None:
    """Assert every exemplar comes from train and from no other split."""
    train_ids = {_record_id(r) for r in load_split_records("train", splits_dir=splits_dir)}
    val_ids = {_record_id(r) for r in load_split_records("val", splits_dir=splits_dir)}
    test_ids = {_record_id(r) for r in load_split_records("test", splits_dir=splits_dir)}
    for exemplar in exemplars:
        exemplar_id = _record_id(exemplar)
        assert exemplar_id in train_ids, f"exemplar {exemplar_id!r} is not in train"
        assert exemplar_id not in val_ids, f"exemplar {exemplar_id!r} leaks into val"
        assert exemplar_id not in test_ids, f"exemplar {exemplar_id!r} leaks into test"


# ---------------------------------------------------------------------------
# Prompt construction (pure — unit-tested)
# ---------------------------------------------------------------------------

def build_messages(
    record: dict, mode: str = "zero_shot", exemplars: Sequence[dict] | None = None
) -> list[dict]:
    """Build chat messages for one record.

    - ``zero_shot``: ``[system, user(instruction)]``.
    - ``few_shot``: system, then the 3 exemplar ``(user, assistant)``
      pairs (assistant content is the exemplar's ``expected_blueprint``
      verbatim), then ``user(instruction)``.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "zero_shot":
        pass
    elif mode == "few_shot":
        if not exemplars:
            raise ValueError("few_shot mode requires exemplars (train records only)")
        if len(exemplars) != 3:
            raise ValueError(f"few_shot expects exactly 3 exemplars, got {len(exemplars)}")
        for exemplar in exemplars:
            messages.append({"role": "user", "content": exemplar["instruction"]})
            messages.append(
                {"role": "assistant", "content": exemplar["expected_blueprint"]}
            )
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected zero_shot/few_shot)")
    messages.append({"role": "user", "content": record["instruction"]})
    return messages


# ---------------------------------------------------------------------------
# Generation (heavy — not unit-tested)
# ---------------------------------------------------------------------------

def generate_predictions(
    model_id: str = DEFAULT_MODEL_ID,
    mode: str = "zero_shot",
    split: str = "val",
    limit: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    splits_dir: str | Path | None = None,
    exemplars: Sequence[dict] | None = None,
) -> list[dict]:
    """Generate blueprint predictions for a split with a HF causal LM (CPU).

    Prompts are rendered with ``tokenizer.apply_chat_template(...,
    add_generation_prompt=True)``; only the newly generated tokens are
    decoded. Each prediction is in the exact format
    ``scripts/eval_harness.py`` expects — ``{"id", "output"}`` — plus
    provenance fields ``model``, ``mode`` and ``split``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = load_split_records(split, limit=limit, splits_dir=splits_dir)
    if mode == "few_shot":
        exemplars = (
            list(exemplars)
            if exemplars is not None
            else get_few_shot_exemplars(splits_dir=splits_dir)
        )
        # Refuse to run few-shot with exemplars that are not from train.
        validate_exemplars_are_train(exemplars, splits_dir=splits_dir)
    elif mode != "zero_shot":
        raise ValueError(f"unknown mode: {mode!r} (expected zero_shot/few_shot)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    if temperature and temperature > 0:
        torch.manual_seed(42)

    predictions = []
    for record in records:
        messages = build_messages(record, mode, exemplars=exemplars)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=temperature)
        else:
            generate_kwargs.update(do_sample=False)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        generated_ids = outputs[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(
            {
                "id": _record_id(record),
                "output": text,
                "model": model_id,
                "mode": mode,
                "split": split,
            }
        )
    return predictions


# ---------------------------------------------------------------------------
# Scoring (reuses scripts/eval_harness.py functions — no reimplementation)
# ---------------------------------------------------------------------------

def score_predictions(predictions: Sequence[dict], records: Sequence[dict]) -> dict:
    """Score predictions against reference records with the eval harness.

    Aggregation mirrors ``eval_harness.evaluate``: L1/L2/L5 are rates over
    all predictions; the L3 sub-metrics are averaged over the predictions
    that parsed as JSON (the only ones the harness can field-score); L4
    is skipped (no judge). See the module docstring for how the SkillGen
    Score is renormalised when L4 is skipped.
    """
    refs = {_record_id(r): r for r in records}
    n = len(predictions)
    l1_ok = l2_ok = l5_ok = 0
    name_ems, path_f1s, desc_f1s = [], [], []
    per_item = []

    for prediction in predictions:
        rid = prediction["id"]
        item = {"id": rid}
        ref = refs.get(rid)
        if ref is None:
            item["error"] = "no reference"
            per_item.append(item)
            continue
        ok1, obj = eval_harness.l1_parse(prediction.get("output", ""))
        item["l1_parse"] = ok1
        l1_ok += ok1
        if ok1:
            ref_bp = json.loads(ref["expected_blueprint"])
            errs = eval_harness.l2_validate_schema(obj)
            item["l2_errors"] = errs
            item["l2_valid"] = not errs
            l2_ok += not errs
            s3 = eval_harness.l3_field_scores(obj, ref_bp)
            item["l3"] = s3
            name_ems.append(s3["name_em"])
            path_f1s.append(s3["file_path_f1"])
            desc_f1s.append(s3["desc_token_f1"])
            ok5, e5 = eval_harness.l5_materialize(obj)
            item["l5_materialize"] = ok5
            item["l5_errors"] = e5
            l5_ok += ok5
        per_item.append(item)

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    l1 = l1_ok / n if n else 0.0
    l2 = l2_ok / n if n else 0.0
    l3_name_em = avg(name_ems)
    l3_file_path_f1 = avg(path_f1s)
    l3_desc_token_f1 = avg(desc_f1s)
    l3 = (l3_name_em + l3_file_path_f1 + l3_desc_token_f1) / 3.0
    l5 = l5_ok / n if n else 0.0

    w = SKILLGEN_WEIGHTS
    weighted_sum_no_l4 = w["l1"] * l1 + w["l2"] * l2 + w["l3"] * l3 + w["l5"] * l5
    available_weight = w["l1"] + w["l2"] + w["l3"] + w["l5"]  # 0.85, L4 skipped
    skillgen_score = weighted_sum_no_l4 / available_weight if available_weight else 0.0

    return {
        "n": n,
        "l1_parse_rate": l1,
        "l2_schema_rate": l2,
        "l3_name_em": l3_name_em,
        "l3_file_path_f1": l3_file_path_f1,
        "l3_desc_token_f1": l3_desc_token_f1,
        "l3": l3,
        "l4_judge": "skipped",
        "l4_note": (
            "L4 (LLM judge) not run: eval_harness omits L4 when the judge "
            "is not run (it requires OPENROUTER_API_KEY)."
        ),
        "l5_materialize_rate": l5,
        "skillgen_score": skillgen_score,
        "skillgen_score_weighted_sum_no_l4": weighted_sum_no_l4,
        "skillgen_score_note": (
            "PLAN.md §33: 0.25*L1 + 0.25*L2 + 0.20*L3 + 0.15*L4 + 0.15*L5, "
            "with L3 = mean(name_em, file_path_f1, desc_token_f1). L4 was "
            "skipped, so skillgen_score is the weighted sum renormalised "
            "by the available weight (0.85)."
        ),
        "items": per_item,
    }
