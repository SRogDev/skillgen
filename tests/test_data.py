#!/usr/bin/env python3
"""Tests for src/skillgen/data.py — stdlib only. Run: python3 tests/test_data.py"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from skillgen import data as d  # noqa: E402

REC = {"instruction": "Make a skill", "expected_blueprint": '{"name": "x"}',
       "metadata": {"id": "a", "domain": "software"}}


def test_messages_shape_and_no_metadata_leak():
    m = d.to_messages(REC)
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[1]["content"] == "Make a skill"
    assert m[2]["content"] == '{"name": "x"}'
    assert "software" not in json.dumps(m[:2]), "metadata must not enter the prompt"


def test_context_appended_to_user():
    m = d.to_messages({**REC, "context": {"existing_skill": "old"}})
    assert "<context>" in m[1]["content"] and "existing_skill" in m[1]["content"]


def test_dict_target_serialized():
    m = d.to_messages({**REC, "expected_blueprint": {"name": "x"}})
    assert json.loads(m[2]["content"]) == {"name": "x"}


def test_filter_drops_never_truncates():
    recs = [{**REC, "metadata": {"id": str(i)}, "instruction": "w " * i} for i in (1, 5, 10)]
    kept, dropped = d.filter_by_length(recs, lambda r: len(r["instruction"].split()), 5)
    assert [r["metadata"]["id"] for r in kept] == ["1", "5"]
    assert dropped[0]["_n_tokens"] == 10
    assert kept[0]["instruction"] == recs[0]["instruction"], "kept records unchanged"
    rep = d.length_report(dropped, 3, 5)
    assert rep == {"max_seq_length": 5, "total": 3, "kept": 2, "dropped": 1, "dropped_ids": ["10"]}


def test_test_split_refused():
    try:
        d.load_split("test")
    except ValueError:
        return
    raise AssertionError("loading the frozen test set for training must fail")


def test_real_splits_load():
    assert len(d.load_split("train")) == 800
    assert len(d.load_split("val")) == 100


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
    print(f"{len(fns)}/{len(fns)} passed")
