"""Tests for Phase 4 evaluation tooling — written FIRST (TDD).

Covers src/skillgen/evaluation/compare.py (paired counts, McNemar exact,
Wilson CI, paired bootstrap) and the failure classifier in
scripts/failure_analysis.py (harness signals -> PLAN.md §48 taxonomy).

All fixtures are hand-computed.
"""

import importlib.util
import math
from pathlib import Path

import pytest

from skillgen.evaluation.compare import (
    mcnemar_exact,
    paired_counts,
    score_diff_ci,
    wilson_ci,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_failure_module():
    path = REPO_ROOT / "scripts" / "failure_analysis.py"
    spec = importlib.util.spec_from_file_location("failure_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fa = _load_failure_module()


def _item(id_, l2_valid, l1=True, l5=False):
    return {"id": id_, "l1_parse": l1, "l2_valid": l2_valid, "l5_materialize": l5}


# A: pass,pass,fail,fail ; B: pass,fail,pass,fail  (on l2_valid)
A = [_item("a", True), _item("b", True), _item("c", False), _item("d", False)]
B = [_item("a", True), _item("b", False), _item("c", True), _item("d", False)]


def test_paired_counts_hand_computed():
    counts = paired_counts(A, B)
    assert counts == {"both_pass": 1, "only_a": 1, "only_b": 1, "both_fail": 1, "n": 4}


def test_paired_counts_supports_l1_and_l5_metrics():
    a = [{"id": "x", "l1_parse": True, "l5_materialize": False}]
    b = [{"id": "x", "l1_parse": False, "l5_materialize": True}]
    assert paired_counts(a, b, metric="l1_parse")["only_a"] == 1
    assert paired_counts(a, b, metric="l5")["only_b"] == 1


def test_paired_counts_requires_same_ids():
    with pytest.raises(ValueError):
        paired_counts([_item("x", True)], [_item("y", True)])


def test_mcnemar_exact_known_value_b9_c1():
    # n=10 discordant, two-sided exact = 2 * sum_{k=0}^{1} C(10,k)/2^10
    # = 2 * (1 + 10) / 1024 = 0.021484375
    expected = 2 * sum(math.comb(10, k) for k in range(2)) / 2**10
    assert expected == pytest.approx(0.021484375)
    assert mcnemar_exact(9, 1) == pytest.approx(expected)
    assert mcnemar_exact(1, 9) == pytest.approx(expected)  # symmetric


def test_mcnemar_exact_no_discordant_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_balanced_is_one():
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_wilson_ci_contains_true_proportion_and_shrinks_with_n():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    lo2, hi2 = wilson_ci(500, 1000)
    assert lo2 < 0.5 < hi2
    assert (hi2 - lo2) < (hi - lo)


def test_wilson_ci_bounds():
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0, abs=1e-9)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        wilson_ci(1, 0)


def test_score_diff_ci_identical_runs_is_zero():
    lo, hi, diff = score_diff_ci(A, A)
    assert diff == 0.0 and lo <= 0.0 <= hi


def test_score_diff_ci_deterministic_seed():
    assert score_diff_ci(A, B) == score_diff_ci(A, B)


# ---------------- failure classifier (scripts/failure_analysis.py) ---------------

def test_classifier_missing_field_and_invalid_name():
    item = {
        "id": "x", "l1_parse": True, "l2_valid": False,
        "l2_errors": ["missing field: files", "invalid name: 'Foo Bar'"],
        "l5_materialize": False, "l5_errors": ["SKILL.md not materialized"],
    }
    codes = fa.classify_item(item)
    assert codes == ["E2", "E4"]  # L5 failure is downstream of L2 here, not E8


def test_classifier_l1_failure_is_e10():
    item = {"id": "x", "l1_parse": False, "l2_valid": False}
    assert fa.classify_item(item) == ["E10"]


def test_classifier_unmapped_error_is_e10_other():
    item = {"id": "x", "l1_parse": True, "l2_valid": False,
            "l2_errors": ["some brand new harness error"], "l5_materialize": False}
    assert fa.classify_item(item) == ["E10"]


def test_classifier_valid_item_has_no_codes():
    item = {"id": "x", "l1_parse": True, "l2_valid": True, "l2_errors": [],
            "l5_materialize": True, "l3": {"name_em": 1.0, "file_path_f1": 1.0,
                                            "desc_token_f1": 1.0}}
    assert fa.classify_item(item) == []


def test_classifier_taxonomy_counts_on_real_e0_smoke_shape():
    items = [
        {"id": str(i), "l1_parse": True, "l2_valid": False,
         "l2_errors": ["missing field: blueprint_version", "missing field: purpose",
                       "missing field: when_to_use", "missing field: when_not_to_use",
                       "missing field: files", "invalid name: 'Foo'"],
         "l5_materialize": False, "l5_errors": ["SKILL.md not materialized"]}
        for i in range(5)
    ]
    counts = fa.taxonomy_counts(items)
    assert counts["E2"] == 5 and counts["E4"] == 5
    assert sum(counts.values()) == 10
