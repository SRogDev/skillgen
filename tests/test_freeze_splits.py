#!/usr/bin/env python3
"""Tests for scripts/freeze_splits.py — stdlib only. Run: python3 tests/test_freeze_splits.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import freeze_splits as fs


TOPICS = [
    ("Deploy a postgres backup rotation with pgbackrest and S3 lifecycle rules", "software"),
    ("Write launch tweets and a Product Hunt tagline for a habit tracker", "marketing"),
    ("Build a churn prediction notebook from stripe invoices with sklearn", "analytics"),
    ("Draft a sales discovery call script for mid-market logistics firms", "sales"),
    ("Create an onboarding checklist for new finance hires with approvals", "business"),
    ("Summarize weekly support tickets into themes and action items", "support"),
    ("Generate a meal-prep grocery list from high-protein dinner recipes", "productivity"),
    ("Outline a literature review on retrieval augmented generation evals", "research"),
    ("Write commit message guidelines enforcing conventional commits style", "writing"),
    ("Design an incident runbook for database failover with verification", "operations"),
]


DETAILS = [
    "The team is fully remote across four time zones and hates meetings",
    "We just migrated to Kubernetes and the old runbooks are obsolete",
    "The CEO wants a demo by Friday and the API is still unstable",
    "Compliance requires SOC2 audit trails on every manual change",
    "We have three junior devs who need guardrails, not freedom",
    "The budget was cut 40% so everything must run on spot instances",
    "Customers are enterprise, so SSO and audit logs are non-negotiable",
    "We ship daily and the current process blocks releases every time",
    "The data lives in five different tools that do not talk to each other",
    "Half the team is new and nobody knows where anything is documented",
]


def _toy_records(n=40):
    recs = []
    for i in range(n):
        topic, domain = TOPICS[i % len(TOPICS)]
        detail = DETAILS[(i // len(TOPICS)) % len(DETAILS)]
        recs.append({
            "instruction": f"{topic}. {detail}.",
            "expected_blueprint": json.dumps({
                "blueprint_version": "1.0",
                "name": f"thing-{i}",
                "description": f"Does thing {i}",
                "purpose": "p", "when_to_use": ["x"], "when_not_to_use": [],
                "files": [{"path": "SKILL.md", "purpose": "p", "content": "c"}],
            }),
            "metadata": {"id": f"id-{i}", "domain": domain,
                         "repo": "r", "provenance": {}},
        })
    # exact duplicate of record 0 (different id) -> must not leak across splits
    dup = json.loads(json.dumps(recs[0]))
    dup["metadata"]["id"] = "id-dup"
    recs.append(dup)
    return recs


def test_dedup_groups_stay_together():
    recs = _toy_records()
    groups = fs.cluster_near_duplicates(recs)
    # find group containing id-0 and id-dup
    g0 = next(g for g in groups if "id-0" in g)
    assert "id-dup" in g0, "exact duplicate must be in the same cluster"


def test_split_counts_and_no_leakage(tmpdir="/tmp"):
    recs = _toy_records(40)  # 41 with dup
    with tempfile.TemporaryDirectory() as d:
        train, val, test = fs.make_splits(recs, seed=7, out_dir=Path(d),
                                          sizes=(32, 4, 4))
        assert len(train) == 32 and len(val) == 4 and len(test) == 4
        ids = [r["metadata"]["id"] for r in train + val + test]
        assert len(set(ids)) == len(ids), "id leaked across splits"
        instr = [r["instruction"] for r in train + val + test]
        assert len(set(instr)) == len(instr), "instruction text leaked across splits"
        # manifest exists and hashes verify
        man = json.loads((Path(d) / "manifest.json").read_text())
        assert man["seed"] == 7 and man["counts"] == {"train": 32, "val": 4, "test": 4}
        assert fs.verify_manifest(Path(d)), "manifest hash mismatch"


def test_stratified_by_domain():
    recs = _toy_records(100)
    with tempfile.TemporaryDirectory() as d:
        train, val, test = fs.make_splits(recs, seed=7, out_dir=Path(d),
                                          sizes=(80, 10, 10))
        from collections import Counter
        global_share = Counter(r["metadata"]["domain"] for r in recs)
        n = len(recs)
        for split, name in ((train, "train"), (val, "val"), (test, "test")):
            c = Counter(r["metadata"]["domain"] for r in split)
            for dom, g in global_share.items():
                share = c[dom] / len(split)
                assert abs(share - g / n) < 0.15, \
                    f"{name}: domain {dom} share {share:.2f} vs global {g/n:.2f}"


def test_deterministic():
    recs = _toy_records(40)
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = [r["metadata"]["id"] for r in fs.make_splits(recs, seed=7, out_dir=Path(d1), sizes=(32, 4, 4))[0]]
        b = [r["metadata"]["id"] for r in fs.make_splits(recs, seed=7, out_dir=Path(d2), sizes=(32, 4, 4))[0]]
        assert a == b, "splits not deterministic"


if __name__ == "__main__":
    test_dedup_groups_stay_together()
    test_split_counts_and_no_leakage()
    test_stratified_by_domain()
    test_deterministic()
    print("test_freeze_splits: 4/4 OK")
