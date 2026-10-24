# SMOKE RUN — baseline pipeline smoke (NOT E0 results)

**SMOKE — pipeline proof only, NOT experiment E0 results.** This run exists to prove the baseline pipeline works end to end (prompt construction → generation → harness scoring) on CPU with a small model. Its scores are NOT comparable to E0.

Real E0 is the untrained `Qwen/Qwen2.5-7B-Instruct` on the full val/test splits; it is **still pending the Kaggle GPU run**.

## Configuration

- Date: 2026-10-20 20:22 UTC
- Label: SMOKE
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Mode: `zero_shot`
- Split: `val` (limit: 5 of 100 records; 5 predictions)
- Max new tokens: 1024
- Temperature: 0.0 (0 = greedy)
- Wall time (generation + scoring): 433.6 s
- Device: CPU (local run)

## Scores (scripts/eval_harness.py, reused — not reimplemented)

- L1 parse rate: 1.0000
- L2 schema rate: 0.0000
- L3 name exact-match: 0.0000
- L3 file-path F1: 0.0000
- L3 description token-F1: 0.3870
- L3 (mean of the three above): 0.1290
- L4 judge: skipped (no judge run; the harness omits L4 without `OPENROUTER_API_KEY`)
- L5 materialize rate: 0.0000
- **SkillGen Score: 0.3245** (PLAN.md §33 weights, renormalised over the available layers because L4 was skipped — see `scores.json` for the exact note and the un-renormalised weighted sum)

Files: `predictions.jsonl` (harness format: `id` + `output`, plus `model`/`mode`/`split` provenance), `scores.json` (full report including per-item detail).
