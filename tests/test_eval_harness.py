#!/usr/bin/env python3
"""Tests for scripts/eval_harness.py — stdlib only. Run: python3 tests/test_eval_harness.py"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import eval_harness as eh

REF_BP = {
    "blueprint_version": "1.0",
    "name": "invoice-tracker",
    "description": "Track invoices and remind about overdue payments",
    "purpose": "p", "when_to_use": ["x"], "when_not_to_use": [],
    "files": [
        {"path": "SKILL.md", "purpose": "main",
         "content": "---\nname: invoice-tracker\ndescription: Track invoices\n---\n\n# hi\n"},
        {"path": "scripts/track.py", "purpose": "s", "content": "print(1)"},
    ],
}


def test_l1_parse():
    assert eh.l1_parse('{"a": 1}')[0] is True
    assert eh.l1_parse('not json')[0] is False
    ok, obj = eh.l1_parse('```json\n{"a": 1}\n```')
    assert ok and obj == {"a": 1}, "should strip code fences"


def test_l2_schema():
    errs = eh.l2_validate_schema(REF_BP)
    assert errs == [], f"valid blueprint flagged: {errs}"
    bad = dict(REF_BP, name="Bad_Name!")
    assert eh.l2_validate_schema(bad), "bad name not caught"
    bad2 = dict(REF_BP); bad2["files"] = []
    assert eh.l2_validate_schema(bad2), "empty files not caught"
    bad3 = {k: v for k, v in REF_BP.items() if k != "description"}
    assert eh.l2_validate_schema(bad3), "missing field not caught"


def test_l3_fields():
    pred = dict(REF_BP)  # perfect prediction
    s = eh.l3_field_scores(pred, REF_BP)
    assert s["name_em"] == 1.0 and s["file_path_f1"] == 1.0
    pred2 = dict(REF_BP, name="other-name",
                 files=[{"path": "SKILL.md", "purpose": "p", "content": "c"}])
    s2 = eh.l3_field_scores(pred2, REF_BP)
    assert s2["name_em"] == 0.0
    assert 0.0 < s2["file_path_f1"] < 1.0, "partial file overlap should score partial"


def test_l5_materialize():
    ok, errs = eh.l5_materialize(REF_BP)
    assert ok, f"materialize failed: {errs}"
    bad = dict(REF_BP, files=[{"path": "notes.md", "purpose": "p", "content": "x"}])
    ok2, _ = eh.l5_materialize(bad)
    assert not ok2, "missing SKILL.md not caught"


def test_evaluate_end_to_end(tmpdir="/tmp"):
    import tempfile
    refs = [{"instruction": "track invoices",
             "expected_blueprint": json.dumps(REF_BP),
             "metadata": {"id": "r1"}}]
    preds = [{"id": "r1", "output": json.dumps(REF_BP)},
             ]
    with tempfile.TemporaryDirectory() as d:
        rp, pp = Path(d) / "refs.jsonl", Path(d) / "preds.jsonl"
        rp.write_text("\n".join(json.dumps(r) for r in refs) + "\n")
        pp.write_text("\n".join(json.dumps(p) for p in preds) + "\n")
        rep = eh.evaluate(str(pp), str(rp))
    assert rep["n"] == 1
    assert rep["l1_parse_rate"] == 1.0
    assert rep["l2_schema_rate"] == 1.0
    assert rep["l3_name_em"] == 1.0
    assert rep["l5_materialize_rate"] == 1.0


if __name__ == "__main__":
    test_l1_parse()
    test_l2_schema()
    test_l3_fields()
    test_l5_materialize()
    test_evaluate_end_to_end()
    print("test_eval_harness: 5/5 OK")
