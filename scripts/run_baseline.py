#!/usr/bin/env python3
"""Run the SkillGen baseline pipeline (experiment E0) and score it.

Generates blueprint predictions with an untrained base model (zero-shot,
E0a, or few-shot, E0b) on a frozen split, writes ``predictions.jsonl`` in
the exact format ``scripts/eval_harness.py`` expects, scores them by
reusing that harness (via ``skillgen.evaluation.baseline``), and writes
``scores.json`` plus a run ``README.md``.

The default configuration IS the real E0 configuration: the 7B base
model, zero-shot, on the full validation split. Anything else — a
different (smaller) model, or a limit below the full split size — is a
**SMOKE** run: pipeline proof only, NOT E0 results. The run README
labels it as such. Real E0 (7B, full val/test) runs on Kaggle GPU and
is still pending; this script's smoke runs never touch the frozen test
split unless explicitly asked to.

Usage:
    uv run python scripts/run_baseline.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --mode zero_shot \
        --split val --limit 5 --max-new-tokens 1024 \
        --out-dir experiments/e0_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skillgen.evaluation.baseline import (  # noqa: E402
    DEFAULT_MODEL_ID,
    FEW_SHOT_EXEMPLAR_IDS,
    REAL_E0_MODEL_ID,
    generate_predictions,
    load_split_records,
    resolve_loader,
    score_predictions,
)


def is_smoke(model_id: str, limit: int | None, full_split_size: int) -> bool:
    """A run is a SMOKE if it is not the real E0 model, or it is limited."""
    if model_id != REAL_E0_MODEL_ID:
        return True
    return limit is not None and limit < full_split_size


def write_run_readme(
    out_dir: Path,
    *,
    model_id: str,
    mode: str,
    split: str,
    limit: int | None,
    full_split_size: int,
    max_new_tokens: int,
    temperature: float,
    scores: dict,
    elapsed_seconds: float,
    loader: str = "transformers",
) -> bool:
    smoke = is_smoke(model_id, limit, full_split_size)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    if smoke:
        lines += [
            "# SMOKE RUN — baseline pipeline smoke (NOT E0 results)",
            "",
            "**SMOKE — pipeline proof only, NOT experiment E0 results.** "
            "This run exists to prove the baseline pipeline works end to "
            "end (prompt construction → generation → harness scoring) on "
            "CPU with a small model. Its scores are NOT comparable to E0.",
            "",
            f"Real E0 is the untrained `{REAL_E0_MODEL_ID}` on the full "
            "val/test splits; it is **still pending the Kaggle GPU run**.",
            "",
        ]
    else:
        lines += [
            "# Baseline run (E0)",
            "",
        ]
    lines += [
        "## Configuration",
        "",
        f"- Date: {date}",
        f"- Label: {'SMOKE' if smoke else 'E0'}",
        f"- Model: `{model_id}`",
        f"- Mode: `{mode}`"
        + (
            f" (exemplars, train split only: {', '.join(FEW_SHOT_EXEMPLAR_IDS)})"
            if mode == "few_shot"
            else ""
        ),
        f"- Split: `{split}` (limit: {limit if limit is not None else 'none'} "
        f"of {full_split_size} records; {scores['n']} predictions)",
        f"- Max new tokens: {max_new_tokens}",
        f"- Temperature: {temperature} (0 = greedy)",
        f"- Wall time (generation + scoring): {elapsed_seconds:.1f} s",
        f"- Loader: `{loader}`"
        + (
            " (Unsloth FastLanguageModel, 4-bit — GPU run)"
            if loader == "unsloth"
            else " (plain transformers — CPU/local run)"
        ),
        "",
        "## Scores (scripts/eval_harness.py, reused — not reimplemented)",
        "",
        f"- L1 parse rate: {scores['l1_parse_rate']:.4f}",
        f"- L2 schema rate: {scores['l2_schema_rate']:.4f}",
        f"- L3 name exact-match: {scores['l3_name_em']:.4f}",
        f"- L3 file-path F1: {scores['l3_file_path_f1']:.4f}",
        f"- L3 description token-F1: {scores['l3_desc_token_f1']:.4f}",
        f"- L3 (mean of the three above): {scores['l3']:.4f}",
        "- L4 judge: skipped (no judge run; the harness omits L4 without "
        "`OPENROUTER_API_KEY`)",
        f"- L5 materialize rate: {scores['l5_materialize_rate']:.4f}",
        f"- **SkillGen Score: {scores['skillgen_score']:.4f}** "
        "(PLAN.md §33 weights, renormalised over the available layers "
        "because L4 was skipped — see `scores.json` for the exact note "
        "and the un-renormalised weighted sum)",
        "",
        "Files: `predictions.jsonl` (harness format: `id` + `output`, plus "
        "`model`/`mode`/`split` provenance), `scores.json` (full report "
        "including per-item detail).",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return smoke


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--loader",
        choices=["transformers", "unsloth"],
        default=None,
        help="model loader (default: SKILLGEN_BASELINE_LOADER env or "
        "transformers; unsloth = Kaggle GPU path for the real E0 runs)",
    )
    ap.add_argument("--out-dir", default="experiments/e0_baseline")
    args = ap.parse_args()
    loader = resolve_loader(args.loader)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_split_records(args.split, limit=args.limit)
    full_split_size = len(load_split_records(args.split))

    start = time.time()
    predictions = generate_predictions(
        model_id=args.model,
        mode=args.mode,
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        loader=loader,
    )
    scores = score_predictions(predictions, records)
    elapsed = time.time() - start

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for prediction in predictions:
            fh.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    (out_dir / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    smoke = write_run_readme(
        out_dir,
        model_id=args.model,
        mode=args.mode,
        split=args.split,
        limit=args.limit,
        full_split_size=full_split_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        scores=scores,
        elapsed_seconds=elapsed,
        loader=loader,
    )

    summary = {k: v for k, v in scores.items() if k != "items"}
    summary["smoke"] = smoke
    summary["loader"] = loader
    summary["out_dir"] = str(out_dir)
    summary["elapsed_seconds"] = round(elapsed, 1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
