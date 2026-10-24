"""Pure, stdlib-only exploratory data analysis helpers for the SkillGen dataset.

Every function here is pure (no I/O except :func:`load_jsonl`, no third-party
imports — in particular no ``transformers`` import) so the statistics can be
unit-tested fast and reused from notebooks and scripts alike.

Record shape (one JSON object per line in ``data/splits/*.jsonl``)::

    {"instruction": str,
     "expected_blueprint": str,          # a JSON *string*
     "metadata": {"id": str, "domain": str, "repo": str, "provenance": {...}}}
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile (0–100) with linear interpolation.

    Uses the same method as NumPy's default (``method="linear"``): the rank is
    ``q/100 * (n - 1)`` and fractional ranks interpolate between neighbours.

    Raises:
        ValueError: if ``values`` is empty or ``q`` is outside [0, 100].
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def length_stats(values: Sequence[float]) -> dict:
    """Summary statistics for a list of lengths.

    Returns ``count, mean, p50, p90, p95, p99, max``. An empty input yields
    ``count == 0`` and zeros elsewhere (so callers never divide by zero
    accidentally — they can check ``count`` first).
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def truncation_report(
    lengths: Sequence[float], thresholds: Iterable[int]
) -> dict[int, dict]:
    """For each threshold, how many lengths are strictly over it.

    Returns ``{threshold: {"n_over": int, "pct_over": float}}`` where
    ``pct_over`` is a percentage (0–100) of all lengths. A length exactly equal
    to the threshold is *not* over it (it fits exactly).
    """
    total = len(lengths)
    report: dict[int, dict] = {}
    for threshold in thresholds:
        n_over = sum(1 for length in lengths if length > threshold)
        report[int(threshold)] = {
            "n_over": n_over,
            "pct_over": (100.0 * n_over / total) if total else 0.0,
        }
    return report


def domain_counts(records: Sequence[dict]) -> dict[str, int]:
    """Count records per ``metadata.domain`` (missing domain -> ``"unknown"``)."""
    counts: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata") or {}
        counts[metadata.get("domain") or "unknown"] += 1
    return dict(counts)


def blueprint_stats(records: Sequence[dict]) -> dict:
    """Statistics over the ``expected_blueprint`` JSON string of each record.

    Returns a dict with:

    - ``total_records``: number of records inspected.
    - ``parse_failures``: records whose blueprint JSON fails to parse.
    - ``missing_skill_md``: *parsed* blueprints with no file whose path is or
      ends with ``SKILL.md``.
    - ``files_per_blueprint``: ``{n_files: n_blueprints}`` distribution over
      parsed blueprints only (a parse failure has no trustworthy file list).
    """
    files_distribution: Counter[int] = Counter()
    parse_failures = 0
    missing_skill_md = 0
    for record in records:
        try:
            blueprint = json.loads(record.get("expected_blueprint", ""))
        except (json.JSONDecodeError, TypeError):
            parse_failures += 1
            continue
        if not isinstance(blueprint, dict):
            parse_failures += 1
            continue
        files = blueprint.get("files")
        if not isinstance(files, list):
            files = []
        files_distribution[len(files)] += 1
        has_skill_md = any(
            isinstance(f, dict)
            and isinstance(f.get("path"), str)
            and (f["path"] == "SKILL.md" or f["path"].endswith("/SKILL.md"))
            for f in files
        )
        if not has_skill_md:
            missing_skill_md += 1
    return {
        "total_records": len(records),
        "parse_failures": parse_failures,
        "missing_skill_md": missing_skill_md,
        "files_per_blueprint": dict(sorted(files_distribution.items())),
    }
