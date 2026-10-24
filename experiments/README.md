# Experiment registry (PLAN.md §19 / §39)

One change per experiment, same seed (42), same frozen test set, same
prompts — so differences are attributable. Status vocabulary:
**done (real)** = the actual experiment ran at its intended scale;
**smoke-only** = only a CPU/SMOKE-scale pipeline proof ran (numbers are
NOT experiment results); **ready-pending-GPU** = code and configs are
ready, the run needs Kaggle; **design-only** = outlined in the PLAN, not
yet integrated.

| ID | Hypothesis | Single change vs reference | Config / pipeline artifact | Status |
|---|---|---|---|---|
| E0a | The untrained base model is the honest comparison point; its zero-shot SkillGen Score is what fine-tuning must beat | Reference run (no change) — base model, zero-shot | `scripts/run_baseline.py` + `src/skillgen/evaluation/baseline.py` (`zero_shot`) | **smoke-only** — pipeline done and a 0.5B CPU smoke ran (`experiments/e0_smoke/`: L1 1.0000 / L2 0.0000 / L5 0.0000, SkillGen Score 0.3245, SMOKE only); the real 7B run on the full val/test splits is pending GPU |
| E0b | Few-shot prompting (2–5 exemplars) may already teach the format; fine-tuning must add more than format alone | Base model with 3 fixed train-split exemplars in the prompt (vs E0a zero-shot) | `src/skillgen/evaluation/baseline.py` (`few_shot`, `FEW_SHOT_EXEMPLAR_IDS`, unit-tested) | **ready-pending-GPU — no run yet, not even a smoke.** The few-shot pipeline is implemented and unit-tested, but no E0b predictions exist anywhere |
| E1 | QLoRA fine-tuning (§18 recipe) improves over both baselines | QLoRA training on the base model (vs E0a) | `configs/qlora_e1.yaml`, `src/skillgen/training/{prepare,train}.py`, `notebooks/05_qlora_training.ipynb` | **smoke-only** — code + configs ready and a CPU LoRA smoke ran (0.5B, 10 steps, 20 records; `experiments/e1_smoke/`); the real 7B QLoRA run is pending GPU (Kaggle) |
| E2 | Assistant-only masking is the largest single effect on structured tasks | Masking off (all-tokens loss) vs E1 | `configs/qlora_e2_no_masking.yaml` | **ready-pending-GPU** — config ready (differs from E1 in exactly its intended key, test-enforced); no run |
| E3 | Epochs trace an overfitting cliff: 1 may underfit, 3 may overfit vs the 2-epoch recipe | Epochs 1 and 3 vs E1's 2 | `configs/qlora_e3_epochs1.yaml`, `configs/qlora_e3_epochs3.yaml` | **ready-pending-GPU** — configs ready; no run |
| E4 | LoRA rank shows diminishing returns (8 / 16 / 32) | Rank 8 and rank 32 vs E1's 16 | `configs/qlora_e4_rank8.yaml`, `configs/qlora_e4_rank32.yaml` | **ready-pending-GPU** — configs ready; no run |
| E5 | All-linear LoRA ≥ attention-only LoRA | Target modules attention-only vs E1's all-linear | `configs/qlora_e5_attention_only.yaml` | **ready-pending-GPU** — config ready; no run |
| E6 | Data efficiency: how much of E1's gain comes from 250 / 500 / 800 records | Train size 250 and 500 (drawn from the filtered 582-record pool) vs E1 | `configs/qlora_e6_data250.yaml`, `configs/qlora_e6_data500.yaml` | **ready-pending-GPU** — configs ready; no run |
| E7 | Constrained decoding separates "learned the format" from "needs a guardrail" | Inference: `outlines` JSON-schema constrained decoding vs unconstrained (vs the promoted model) | PLAN.md §19/§41 outline only | **design-only** — outlined in the PLAN (§19, §41); not yet integrated into the inference path |

## Phase 4 evaluation tooling (this phase)

- Paired comparison: `src/skillgen/evaluation/compare.py` (paired counts,
  McNemar exact, Wilson CIs, paired bootstrap) + `tests/test_compare.py`.
- Failure analysis: `scripts/failure_analysis.py` → per-experiment
  `failures.md` in the §47 format, classified with the §48 taxonomy
  (mapping in `docs/ERROR_TAXONOMY.md`). First real output:
  `experiments/e0_smoke/failures.md` (SMOKE run: E2 missing field ×5,
  E4 invalid name ×5).
- Forgetting check (§37): `src/skillgen/evaluation/forgetting.py`
  (loglikelihood MC scoring; ARC via HF when available, original local
  20-question fallback — NOT ARC). Smoke: `experiments/forgetting_smoke/`
  (base 0.8000 vs smoke adapter 0.8000 on the local set).

Every real comparison (E0 vs E1 on the 7B model) is **pending Kaggle**;
nothing in this registry presents smoke numbers as experiment results.

## Accept / Reject gate (PLAN.md §39)

A candidate model is **promoted only if**:

- its SkillGen Score is **≥ current promoted score + 2 points**, **and**
- it shows **no regression on the expert set**.

Otherwise it is rejected (or kept as an experiment artifact only).
Comparisons are read paired first (fixed/broken counts, McNemar,
`compare.py`) before any aggregate score is trusted.
