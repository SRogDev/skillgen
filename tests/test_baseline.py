"""Tests for src/skillgen/evaluation/baseline.py — Phase 2 baseline pipeline.

Written BEFORE the implementation (TDD). Generation itself is NOT unit-tested
(too heavy for CPU unit tests); the prompt-construction logic is pure and is
tested here against the real frozen split files.

Covers:
(a) zero-shot messages are exactly [system, user] with the exact PLAN.md §8
    system prompt;
(b) few-shot messages contain exactly the 3 fixed exemplar (user, assistant)
    pairs before the final user turn, and every exemplar id belongs to the
    train split and to neither val nor test;
(c) each exemplar's assistant content equals its ``expected_blueprint``.
"""

import json

from skillgen.evaluation.baseline import (
    FEW_SHOT_EXEMPLAR_IDS,
    SYSTEM_PROMPT,
    build_messages,
    get_few_shot_exemplars,
    load_split_records,
    score_predictions,
)

EXPECTED_SYSTEM_PROMPT = (
    "You are SkillGen, an expert designer of reusable Agent Skills. "
    "Given the user request, output ONLY a valid JSON Skill Blueprint. "
    "No explanations, no markdown fences."
)


def _record(instruction="Make me a skill that tracks invoices."):
    return {
        "instruction": instruction,
        "expected_blueprint": '{"name": "invoice-tracker"}',
        "metadata": {"id": "synthetic__invoice-tracker"},
    }


# --------------------------------------------------------------------- (a)
def test_system_prompt_is_exact_plan_section_8():
    assert SYSTEM_PROMPT == EXPECTED_SYSTEM_PROMPT


def test_zero_shot_messages_are_system_then_user():
    record = _record()
    messages = build_messages(record, "zero_shot")
    assert messages == [
        {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
        {"role": "user", "content": record["instruction"]},
    ]


def test_zero_shot_ignores_exemplars():
    messages = build_messages(_record(), "zero_shot", exemplars=get_few_shot_exemplars())
    assert [m["role"] for m in messages] == ["system", "user"]


# --------------------------------------------------------------------- (b)
def test_few_shot_messages_have_three_exemplar_pairs_before_final_user():
    exemplars = get_few_shot_exemplars()
    record = _record()
    messages = build_messages(record, "few_shot", exemplars=exemplars)

    # system + 3 * (user, assistant) + final user = 8 messages
    assert len(messages) == 8
    assert messages[0] == {"role": "system", "content": EXPECTED_SYSTEM_PROMPT}
    assert [m["role"] for m in messages[1:7]] == ["user", "assistant"] * 3
    assert messages[-1] == {"role": "user", "content": record["instruction"]}

    for i, exemplar in enumerate(exemplars):
        user_msg = messages[1 + 2 * i]
        assistant_msg = messages[2 + 2 * i]
        assert user_msg["content"] == exemplar["instruction"]
        # (c) assistant content is exactly the exemplar's expected blueprint
        assert assistant_msg["content"] == exemplar["expected_blueprint"]


def test_few_shot_exemplar_ids_are_fixed_and_from_train_only():
    exemplars = get_few_shot_exemplars()
    exemplar_ids = [e["metadata"]["id"] for e in exemplars]
    assert exemplar_ids == list(FEW_SHOT_EXEMPLAR_IDS)
    assert len(exemplar_ids) == 3

    train_ids = {r["metadata"]["id"] for r in load_split_records("train")}
    val_ids = {r["metadata"]["id"] for r in load_split_records("val")}
    test_ids = {r["metadata"]["id"] for r in load_split_records("test")}
    for exemplar_id in exemplar_ids:
        assert exemplar_id in train_ids
        assert exemplar_id not in val_ids
        assert exemplar_id not in test_ids


def test_few_shot_requires_exemplars():
    try:
        build_messages(_record(), "few_shot")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("few_shot without exemplars must raise ValueError")


def test_unknown_mode_raises():
    try:
        build_messages(_record(), "many_shot")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown mode must raise ValueError")


# ------------------------------------------------------------ split loading
def test_load_split_records_counts_and_limit():
    assert len(load_split_records("train")) == 800
    assert len(load_split_records("val")) == 100
    assert len(load_split_records("test")) == 100
    limited = load_split_records("val", limit=5)
    assert len(limited) == 5
    assert limited == load_split_records("val")[:5]


# ----------------------------------------------------------------- scoring
def test_score_predictions_perfect_prediction_scores_one():
    record = load_split_records("val", limit=1)[0]
    predictions = [
        {
            "id": record["metadata"]["id"],
            "output": record["expected_blueprint"],
            "model": "test-model",
            "mode": "zero_shot",
        }
    ]
    scores = score_predictions(predictions, [record])
    assert scores["n"] == 1
    assert scores["l1_parse_rate"] == 1.0
    assert scores["l2_schema_rate"] == 1.0
    assert scores["l3"] == 1.0
    assert scores["l5_materialize_rate"] == 1.0
    # L4 is skipped (no judge), so the SkillGen Score is renormalised over
    # the remaining weights (0.25 + 0.25 + 0.20 + 0.15 = 0.85).
    assert scores["l4_judge"] == "skipped"
    assert scores["skillgen_score"] == 1.0


def test_score_predictions_garbage_scores_zero():
    record = load_split_records("val", limit=1)[0]
    predictions = [{"id": record["metadata"]["id"], "output": "not json at all"}]
    scores = score_predictions(predictions, [record])
    assert scores["l1_parse_rate"] == 0.0
    assert scores["l2_schema_rate"] == 0.0
    assert scores["l3"] == 0.0
    assert scores["l5_materialize_rate"] == 0.0
    assert scores["skillgen_score"] == 0.0
    # The reference blueprint itself must be valid JSON for the test above.
    json.loads(record["expected_blueprint"])
