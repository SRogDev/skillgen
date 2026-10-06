"""Dataset loading and prompt construction (stdlib only, testable without GPU).

PLAN §7-8: only `instruction` (+`context`) goes into the prompt; `expected_blueprint`
is the assistant target. Metadata never enters the prompt.
PLAN §23: examples longer than max_seq_length are DROPPED, never truncated
(a truncated target teaches the model to emit broken JSON).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "splits"

SYSTEM_PROMPT = (
    "You are SkillGen, an expert designer of reusable Agent Skills. Given the user "
    "request, output ONLY a valid JSON Skill Blueprint. No explanations, no markdown fences."
)


def load_split(split: str, splits_dir: Path = SPLITS) -> list[dict]:
    """Load a frozen split. Refuses 'test' unless explicitly allowed (PLAN §13)."""
    if split == "test":
        raise ValueError("Frozen test set: use load_test_for_final_eval(), never for training.")
    manifest = json.loads((splits_dir / "manifest.json").read_text())
    recs: list[dict] = []
    for fname in manifest["files"][split]:
        with open(splits_dir / fname) as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return recs


def load_test_for_final_eval(splits_dir: Path = SPLITS) -> list[dict]:
    with open(splits_dir / "test.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def user_content(rec: dict) -> str:
    ctx = rec.get("context")
    if not ctx:
        return rec["instruction"]
    if not isinstance(ctx, str):
        ctx = json.dumps(ctx, ensure_ascii=False)
    return f"{rec['instruction']}\n\n<context>\n{ctx}\n</context>"


def to_messages(rec: dict) -> list[dict]:
    target = rec["expected_blueprint"]
    if not isinstance(target, str):
        target = json.dumps(target, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(rec)},
        {"role": "assistant", "content": target},
    ]


def filter_by_length(
    recs: Iterable[dict], count_tokens: Callable[[dict], int], max_len: int
) -> tuple[list[dict], list[dict]]:
    """Split records into (kept, dropped). Each dropped record gets `_n_tokens`."""
    kept, dropped = [], []
    for r in recs:
        n = count_tokens(r)
        (kept if n <= max_len else dropped).append(r if n <= max_len else {**r, "_n_tokens": n})
    return kept, dropped


def length_report(dropped: list[dict], total: int, max_len: int) -> dict:
    return {
        "max_seq_length": max_len,
        "total": total,
        "kept": total - len(dropped),
        "dropped": len(dropped),
        "dropped_ids": [d.get("metadata", {}).get("id") for d in dropped],
    }
