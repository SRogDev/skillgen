#!/usr/bin/env python3
"""Failure analysis for a harness scores.json (PLAN.md §47/§48).

Classifies every failed item into the E1-E10 error taxonomy by mapping
the harness signals (l1_parse, l2_errors strings, l3, l5) to codes.
The mapping table lives in docs/ERROR_TAXONOMY.md and in TAXONOMY below.

PLAN.md §48 adopts the questionnaire taxonomy "as is" and names only
the anchors in the repo: E1 = invalid structure, E10 = format failure.
E2-E9 below are the documented local operationalisation of the harness
signals (see docs/ERROR_TAXONOMY.md, "Known gaps"). Any harness error
string that matches no rule is classified E10/other — honestly, never
silently dropped.

Usage:
    uv run python scripts/failure_analysis.py --scores experiments/e0_smoke/scores.json \
        [--baseline experiments/<other>/scores.json] [--out failures.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skillgen.evaluation.compare import load_items  # noqa: E402

TAXONOMY = {
    "E1": "Invalid structure — parsed JSON is not an object / top-level structure unusable",
    "E2": "Missing required field",
    "E3": "Wrong field type",
    "E4": "Invalid name (not kebab-case)",
    "E5": "Incomplete file set (files empty, missing SKILL.md, file entry incomplete)",
    "E6": "Unsafe path",
    "E7": "Empty or invalid content field (e.g. empty description)",
    "E8": "Materialization failure (L5 fails while L2 is valid)",
    "E9": "Field-fidelity failure (L2 valid but L3 name exact-match fails)",
    "E10": "Format failure / other (L1 parse failure, or any unmapped harness signal)",
}

# Ordered rules over a single l2_errors / l5_errors string.
_ERROR_RULES = [
    (re.compile(r"^blueprint is not an object"), "E1"),
    (re.compile(r"^missing field:"), "E2"),
    (re.compile(r"has wrong type|must be list\[str\]"), "E3"),
    (re.compile(r"^invalid name:"), "E4"),
    (re.compile(r"^files is empty|^files missing SKILL\.md|^files\[\d+\]"), "E5"),
    (re.compile(r"unsafe"), "E6"),
    (re.compile(r"^empty description"), "E7"),
]


def classify_error_string(err: str) -> str:
    for pattern, code in _ERROR_RULES:
        if pattern.search(err):
            return code
    return "E10"


def classify_item(item: dict) -> list[str]:
    """Return the sorted taxonomy codes for one scored item ([] if it passes)."""
    if not item.get("l1_parse", False):
        return ["E10"]
    codes: set[str] = set()
    for err in item.get("l2_errors", []) or []:
        codes.add(classify_error_string(err))
    l2_valid = item.get("l2_valid", not item.get("l2_errors"))
    if l2_valid:
        # L5 is only an independent failure when the blueprint itself is valid;
        # otherwise the L5 failure is a downstream consequence of the L2 codes.
        if item.get("l5_materialize") is False:
            codes.add("E8")
        l3 = item.get("l3") or {}
        if l3 and l3.get("name_em", 1.0) == 0.0:
            codes.add("E9")
    return sorted(codes, key=lambda c: int(c[1:]))


def taxonomy_counts(items: list[dict]) -> dict[str, int]:
    counts = {code: 0 for code in TAXONOMY}
    for item in items:
        for code in classify_item(item):
            counts[code] += 1
    return counts


def render_failures_md(items: list[dict], label: str, baseline_items: list[dict] | None = None) -> str:
    from skillgen.evaluation.compare import compare_runs

    counts = taxonomy_counts(items)
    failed = [(i["id"], classify_item(i)) for i in items if classify_item(i)]
    passed_ids = sorted(i["id"] for i in items if not classify_item(i))
    lines = [
        f"# Failure analysis — {label}",
        "",
        "Format: PLAN.md §47 (BEST / WORST / REGRESSION / NEW FAILURE), "
        "classified with the §48 taxonomy E1–E10 (see docs/ERROR_TAXONOMY.md).",
        "",
        "## Taxonomy counts (items per code; one item can carry several codes)",
        "",
        "| Code | Meaning | Items |",
        "|---|---|---|",
        *[f"| {c} | {TAXONOMY[c]} | {counts[c]} |" for c in TAXONOMY],
        "",
        f"Failed items: {len(failed)} of {len(items)}.",
        "",
        "## BEST (passing items)",
        "",
        f"Count: {len(passed_ids)}",
        "",
        *[f"- `{i}`" for i in passed_ids],
        "",
        "## WORST (failed items, with taxonomy codes)",
        "",
        f"Count: {len(failed)}",
        "",
        *[f"- `{i}` — {', '.join(codes)}" for i, codes in failed],
        "",
    ]
    if baseline_items is None:
        lines += [
            "## REGRESSION",
            "",
            "No comparison run provided.",
            "",
            "## NEW FAILURE",
            "",
            "No comparison run provided.",
            "",
        ]
    else:
        result = compare_runs(baseline_items, items)
        base_failed = {i["id"] for i in baseline_items if classify_item(i)}
        new_failures = [i for i in result["regressions"] if i not in base_failed]
        lines += [
            "## REGRESSION (passed in baseline, fail here)",
            "",
            f"Count: {len(result['regressions'])}",
            "",
            *[f"- `{i}`" for i in result["regressions"]],
            "",
            "## NEW FAILURE (regressions not failed in the baseline in any form)",
            "",
            f"Count: {len(new_failures)}",
            "",
            *[f"- `{i}`" for i in new_failures],
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    ap.add_argument("--label")
    args = ap.parse_args()
    items = load_items(args.scores)
    baseline = load_items(args.baseline) if args.baseline else None
    label = args.label or args.scores
    md = render_failures_md(items, label, baseline_items=baseline)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
