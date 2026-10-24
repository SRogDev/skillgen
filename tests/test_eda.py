"""Tests for src/skillgen/data/eda.py — pure stdlib EDA helpers.

Written BEFORE the implementation (TDD): the first run must fail with
ModuleNotFoundError because skillgen.data.eda does not exist yet.
"""

import json

import pytest

from skillgen.data.eda import (
    blueprint_stats,
    domain_counts,
    length_stats,
    load_jsonl,
    percentile,
    truncation_report,
)


# ---------------------------------------------------------------- percentile
def test_percentile_median_of_known_data():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([10, 20, 30], 50) == pytest.approx(20.0)


def test_percentile_extremes_and_interpolation():
    values = [0, 100]
    assert percentile(values, 0) == pytest.approx(0.0)
    assert percentile(values, 100) == pytest.approx(100.0)
    assert percentile(values, 25) == pytest.approx(25.0)
    # numpy-style linear interpolation on 1..100: p95 = 95.05
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_percentile_single_value():
    assert percentile([42], 50) == pytest.approx(42.0)
    assert percentile([42], 99) == pytest.approx(42.0)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)


# --------------------------------------------------------------- length_stats
def test_length_stats_known_data():
    stats = length_stats([10, 20, 30, 40])
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(25.0)
    assert stats["p50"] == pytest.approx(25.0)
    assert stats["max"] == 40
    assert stats["p90"] >= stats["p50"]
    assert stats["p99"] >= stats["p95"] >= stats["p90"]


def test_length_stats_empty():
    stats = length_stats([])
    assert stats["count"] == 0
    assert stats["max"] == 0


# ----------------------------------------------------------- truncation_report
def test_truncation_report_counts_and_percentages():
    lengths = [100, 500, 1500, 5000]
    report = truncation_report(lengths, [1024, 4096])
    assert report[1024]["n_over"] == 2
    assert report[1024]["pct_over"] == pytest.approx(50.0)
    assert report[4096]["n_over"] == 1
    assert report[4096]["pct_over"] == pytest.approx(25.0)


def test_truncation_report_exact_threshold_is_not_over():
    report = truncation_report([1024], [1024])
    assert report[1024]["n_over"] == 0
    assert report[1024]["pct_over"] == pytest.approx(0.0)


def test_truncation_report_empty_lengths():
    report = truncation_report([], [1024])
    assert report[1024]["n_over"] == 0
    assert report[1024]["pct_over"] == pytest.approx(0.0)


# --------------------------------------------------------------- domain_counts
def _record(domain="software", blueprint=None, instruction="do a thing"):
    if blueprint is None:
        blueprint = {
            "blueprint_version": "1.0",
            "name": "demo",
            "description": "demo skill",
            "purpose": "demo",
            "when_to_use": [],
            "when_not_to_use": [],
            "files": [{"path": "SKILL.md", "purpose": "main", "content": "# demo"}],
        }
    return {
        "instruction": instruction,
        "expected_blueprint": (
            blueprint if isinstance(blueprint, str) else json.dumps(blueprint)
        ),
        "metadata": {"id": "demo__x", "domain": domain, "repo": "a/b", "provenance": {}},
    }


def test_domain_counts():
    records = [_record("software"), _record("software"), _record("finance")]
    assert domain_counts(records) == {"software": 2, "finance": 1}


def test_domain_counts_missing_domain_is_unknown():
    rec = _record()
    del rec["metadata"]["domain"]
    assert domain_counts([rec]) == {"unknown": 1}


# ------------------------------------------------------------- blueprint_stats
def test_blueprint_stats_happy_path():
    stats = blueprint_stats([_record(), _record()])
    assert stats["total_records"] == 2
    assert stats["parse_failures"] == 0
    assert stats["missing_skill_md"] == 0
    assert stats["files_per_blueprint"] == {1: 2}


def test_blueprint_stats_files_distribution():
    bp = {
        "blueprint_version": "1.0",
        "name": "multi",
        "description": "d",
        "purpose": "p",
        "when_to_use": [],
        "when_not_to_use": [],
        "files": [
            {"path": "SKILL.md", "purpose": "main", "content": "x"},
            {"path": "scripts/run.py", "purpose": "helper", "content": "y"},
            {"path": "references/notes.md", "purpose": "ref", "content": "z"},
        ],
    }
    stats = blueprint_stats([_record(blueprint=bp)])
    assert stats["files_per_blueprint"] == {3: 1}
    assert stats["missing_skill_md"] == 0


def test_blueprint_stats_malformed_json():
    stats = blueprint_stats([_record(blueprint="{not valid json")])
    assert stats["total_records"] == 1
    assert stats["parse_failures"] == 1
    # A record that fails to parse is not counted in the files distribution
    # nor in missing_skill_md.
    assert stats["files_per_blueprint"] == {}
    assert stats["missing_skill_md"] == 0


def test_blueprint_stats_missing_skill_md():
    bp = {
        "blueprint_version": "1.0",
        "name": "no-main-file",
        "description": "d",
        "purpose": "p",
        "when_to_use": [],
        "when_not_to_use": [],
        "files": [{"path": "scripts/only.py", "purpose": "helper", "content": "y"}],
    }
    stats = blueprint_stats([_record(blueprint=bp)])
    assert stats["parse_failures"] == 0
    assert stats["missing_skill_md"] == 1
    assert stats["files_per_blueprint"] == {1: 1}


def test_blueprint_stats_mixed_records():
    bad = _record(blueprint="not json at all")
    good = _record()
    stats = blueprint_stats([bad, good])
    assert stats["total_records"] == 2
    assert stats["parse_failures"] == 1
    assert stats["files_per_blueprint"] == {1: 1}


# ------------------------------------------------------------------ load_jsonl
def test_load_jsonl_roundtrip(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [_record("software"), _record("finance")]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8"
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["metadata"]["domain"] == "software"
    assert loaded[1]["metadata"]["domain"] == "finance"
