"""Paired comparison of two evaluation runs (PLAN.md §19/§34/§47).

Stdlib-only statistics for comparing two ``scores.json``-style reports
(as produced by ``scripts/eval_harness.py`` / ``baseline.score_predictions``):
each report has an ``items`` list with per-item ``l1_parse``, ``l2_valid``
and ``l5_materialize`` booleans.

The PLAN asks for paired fixed/broken counts *before* aggregate scores,
a McNemar test on binary outcomes, and 95% CIs. The difference CI here is
a paired bootstrap over the chosen binary metric (documented choice:
per-item SkillGen Score contributions are not stored in scores.json, so
we bootstrap the binary-metric difference, not the aggregate score).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

METRIC_FIELDS = {"l1_parse": "l1_parse", "l2_valid": "l2_valid", "l5": "l5_materialize",
                 "l5_materialize": "l5_materialize"}


def load_items(path: str | Path) -> list[dict]:
    """Load the per-item list from a scores.json-style file."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data["items"]


def _outcome(item: dict, field: str) -> bool:
    return bool(item.get(field, False))


def paired_counts(a_items: list[dict], b_items: list[dict], metric: str = "l2_valid") -> dict:
    """Paired pass/fail counts on a binary metric, matched by item id.

    only_a = fixed in A / broken in B when B is the candidate (regressions
    if A is the reference); only_b = the converse (new passes).
    """
    field = METRIC_FIELDS[metric]
    a_by = {i["id"]: i for i in a_items}
    b_by = {i["id"]: i for i in b_items}
    if set(a_by) != set(b_by):
        raise ValueError("item id sets differ between the two runs")
    counts = {"both_pass": 0, "only_a": 0, "only_b": 0, "both_fail": 0, "n": len(a_by)}
    for id_ in a_by:
        pa, pb = _outcome(a_by[id_], field), _outcome(b_by[id_], field)
        if pa and pb:
            counts["both_pass"] += 1
        elif pa:
            counts["only_a"] += 1
        elif pb:
            counts["only_b"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_diff_ci(a_items: list[dict], b_items: list[dict], metric: str = "l2_valid",
                  n_resamples: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI for (rate_b - rate_a) on a binary metric.

    Returns (lo, hi, observed_diff). Fixed seed -> deterministic.
    """
    field = METRIC_FIELDS[metric]
    a_by = {i["id"]: i for i in a_items}
    if set(a_by) != {i["id"] for i in b_items}:
        raise ValueError("item id sets differ between the two runs")
    b_by = {i["id"]: i for i in b_items}
    diffs_per_item = [
        (1.0 if _outcome(b_by[i], field) else 0.0) - (1.0 if _outcome(a_by[i], field) else 0.0)
        for i in a_by
    ]
    n = len(diffs_per_item)
    observed = sum(diffs_per_item) / n if n else 0.0
    rng = random.Random(seed)
    boot = []
    for _ in range(n_resamples):
        boot.append(sum(diffs_per_item[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    lo = boot[int(0.025 * n_resamples)]
    hi = boot[min(n_resamples - 1, int(0.975 * n_resamples))]
    return (lo, hi, observed)


def compare_runs(a_items: list[dict], b_items: list[dict], metric: str = "l2_valid") -> dict:
    counts = paired_counts(a_items, b_items, metric)
    lo, hi, diff = score_diff_ci(a_items, b_items, metric)
    n = counts["n"]
    rate_a = (counts["both_pass"] + counts["only_a"]) / n if n else 0.0
    rate_b = (counts["both_pass"] + counts["only_b"]) / n if n else 0.0
    return {
        "metric": metric, "n": n, "counts": counts,
        "rate_a": rate_a, "rate_b": rate_b,
        "rate_a_ci": wilson_ci(counts["both_pass"] + counts["only_a"], n) if n else (0.0, 0.0),
        "rate_b_ci": wilson_ci(counts["both_pass"] + counts["only_b"], n) if n else (0.0, 0.0),
        "diff": diff, "diff_ci": (lo, hi),
        "mcnemar_p": mcnemar_exact(counts["only_a"], counts["only_b"]),
        "regressions": sorted(i["id"] for i in a_items
                              if _outcome(i, METRIC_FIELDS[metric])
                              and not _outcome({x["id"]: x for x in b_items}[i["id"]], METRIC_FIELDS[metric])),
        "new_passes": sorted(i["id"] for i in b_items
                             if _outcome(i, METRIC_FIELDS[metric])
                             and not _outcome({x["id"]: x for x in a_items}[i["id"]], METRIC_FIELDS[metric])),
    }


def render_comparison_md(result: dict, label_a: str = "A (reference)",
                         label_b: str = "B (candidate)") -> str:
    """Render the §47-style comparison skeleton (BEST/WORST/REGRESSION/NEW FAILURE).

    Here BEST = items passing in both, WORST = failing in both,
    REGRESSION = pass in A, fail in B; NEW FAILURE lists the same
    regression ids classified as failures of B (per §47 the failure file
    groups them separately from regressions counted in the summary).
    """
    c = result["counts"]
    lines = [
        f"# Paired comparison — {label_a} vs {label_b}",
        "",
        f"Metric: `{result['metric']}` (binary, paired by item id), n = {result['n']}",
        "",
        f"- Rate A: {result['rate_a']:.4f} (Wilson 95% CI {result['rate_a_ci'][0]:.4f}–{result['rate_a_ci'][1]:.4f})",
        f"- Rate B: {result['rate_b']:.4f} (Wilson 95% CI {result['rate_b_ci'][0]:.4f}–{result['rate_b_ci'][1]:.4f})",
        f"- Difference (B − A): {result['diff']:+.4f} (paired bootstrap 95% CI {result['diff_ci'][0]:+.4f}–{result['diff_ci'][1]:+.4f})",
        f"- McNemar exact p (discordant {c['only_a']} vs {c['only_b']}): {result['mcnemar_p']:.4f}",
        "",
        "## BEST (pass in both)",
        "",
        f"Count: {c['both_pass']}",
        "",
        "## WORST (fail in both)",
        "",
        f"Count: {c['both_fail']}",
        "",
        "## REGRESSION (pass in A, fail in B)",
        "",
        f"Count: {c['only_a']}",
        "",
        *[f"- `{i}`" for i in result["regressions"]],
        "",
        "## NEW FAILURE",
        "",
        "Failures of B on items A passed (same set as REGRESSION, listed "
        "per PLAN.md §47; genuinely new failures on items neither run was "
        "expected to pass are the WORST set above):",
        "",
        *[f"- `{i}`" for i in result["regressions"]],
        "",
        "## New passes (fail in A, pass in B)",
        "",
        f"Count: {c['only_b']}",
        "",
        *[f"- `{i}`" for i in result["new_passes"]],
        "",
    ]
    return "\n".join(lines)
