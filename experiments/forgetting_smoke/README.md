# Forgetting check — SMOKE run (CPU, local question set)

**SMOKE — harness proof only, NOT the PLAN.md §37 forgetting result.**
§37 requires 50 ARC-Challenge questions, base vs the real E1 (7B QLoRA)
adapter, on Kaggle. That run has not happened.

## What was run here

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (CPU), base vs the Phase-3 smoke
  adapter (`experiments/e1_smoke/adapter/` — plain LoRA, 10 steps on
  20 records).
- Questions: the **original local 20-question set** in
  `src/skillgen/evaluation/forgetting.py` (`LOCAL_QUESTIONS`; science,
  history, math, geography, language; 4 options each). **These are NOT
  ARC questions** and no number here may be cited as an ARC score.
- Scoring: loglikelihood argmax over the options (no generation).

## Results (real, from `results.json`)

| Run | Accuracy | Correct |
|---|---|---|
| Base 0.5B | 0.8000 | 16 / 20 |
| Base + smoke adapter | 0.8000 | 16 / 20 |
| Drop | 0.0 points | — |

The two runs give **identical predictions on all 20 questions** (the same
4 misses: two math items, "synonym of rapid", "plural of octopus"). A
10-step adapter trained on 20 short blueprint records showing ~no
forgetting is the expected outcome — the point of this smoke is that the
before/after harness works end to end and is ready for the real E1
adapter. §37's rule (>5% drop = documented finding) applies to the
Kaggle run with the 50-question ARC subset loaded by
`load_arc_questions()`.
