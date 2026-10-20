#!/usr/bin/env python3
"""Tests for the §31 expert-set curation scaffold (data/expert/).

Stdlib only. Run: python3 tests/test_expert_set.py  (or pytest)

The curation itself is Roger's work and may be incomplete: tests WARN
(print, no failure) when fewer than 20 cases exist, but FAIL on any
schema violation in a present case. Template files are ignored.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import validate_expert_set as ves


def _cases():
    return sorted(
        p
        for p in ves.EXPERT_DIR.glob("E*.json")
        if not p.name.startswith("_template")
    )


def test_template_is_ignored():
    templates = list(ves.EXPERT_DIR.glob("_template*.json"))
    assert templates, "a _template file must exist as the curation example"
    for t in templates:
        assert t not in _cases()


def test_present_cases_are_schema_valid():
    errors = []
    for path in _cases():
        errors.extend(ves.validate_case(path))
    assert not errors, "\n".join(errors)


def test_ids_unique():
    ids = [p.stem for p in _cases()]
    assert len(ids) == len(set(ids)), "duplicate case ids"


def test_expected_id_space():
    for path in _cases():
        assert path.stem in ves.EXPECTED_IDS, f"{path.name} outside E001..E020"


def test_curation_progress_reported_not_failed():
    present = len(_cases())
    missing = sorted(ves.EXPECTED_IDS - {p.stem for p in _cases()})
    if present < 20:
        print(f"\n[expert-set] curation in progress: {present}/20 present; "
              f"missing: {', '.join(missing) or 'none'}")
    assert present <= 20


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
