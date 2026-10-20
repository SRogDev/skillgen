# Model Card — SkillGen

## Model identity

- **Name:** SkillGen (project / model family)
- **Base model (PROVISIONAL):** `Qwen/Qwen2.5-7B-Instruct` — pending
  verification of 2026 model releases before the base model is frozen
  (see `PLAN.md` §15 and `docs/DECISIONS.md`).
- **Adaptation method:** QLoRA adapters (r16 / alpha32 / all-linear,
  4-bit base, PLAN.md §18) trained with assistant-only loss masking.
- **Status: NOT YET TRAINED at 7B. No weights are released.** The full
  training pipeline, configs (E1–E6), serving API and export pipeline
  exist and are tested, but the real 7B QLoRA run (experiment E1) has
  not happened yet — it is pending a Kaggle GPU run. **Real 7B results
  do not exist yet.** Everything measured so far is labeled SMOKE /
  0.5B / pipeline-proof below, and none of it is evidence about the
  quality of a 7B SkillGen model.

## Intended use

- Generate a JSON **Skill Blueprint** from a natural-language request,
  which a separate harness validates (L1 parse → L2 schema) and only
  then materializes into an Agent Skill package (`SKILL.md` + files).
  The model never writes to disk directly.
- Served locally via the Phase-5 FastAPI app
  (`src/skillgen/serving/`, `POST /generate-skill`) with the §42
  post-inference validation + repair loop (max 2 retries, then a typed
  reject), and exported via the §44 pipeline (adapter → merged 16-bit
  → GGUF `q4_k_m` → Ollama) once a real adapter exists.

### Out of scope

- Executing generated code: in v1 generated scripts are never run
  (at most `ast.parse` in the harness); sandboxed execution is out of
  v1 (PLAN.md §56).
- General-purpose chat / instruction following outside skill design.
- Autonomous agents, ensembles, RLHF/DPO (PLAN.md §50: no DPO in v1).
- Any production or multi-user serving: v1 serving is local, no auth,
  concurrency 1 by design (§43).

## Training data

- **1,000 records** derived from real skills on
  [skills.sh](https://skills.sh), licenses **MIT (769) / Apache-2.0
  (231)** only, license verified per record. Targets are the real
  skills converted to Skill Blueprints; instructions are synthetic
  backtranslations of those skills (PLAN.md §11). Provenance is stored
  per record (`metadata.provenance`).
- **Splits (frozen):** 800 train / 100 val / 100 test, seed 7,
  near-duplicate clustering (token Jaccard > 0.8, one cluster per
  split), domain-stratified, zero overlap; sha256 in
  `data/splits/manifest.json`. The test split is frozen — never train
  on it, never tune against it.
- **Phase-0 length filter (honest limitation):** E1 trains on the
  **582 of 800 train records whose full rendered text is ≤ 8192
  tokens** (filter-never-truncate; the other 218 are excluded whole).
  Consequence: **very large multi-file skills are underrepresented in
  training** — the dropped tail is exactly where the biggest,
  most complex skills live (train token p95 is 34,071; max
  1,947,918). The trained model should be expected to be weakest on
  very large multi-file skills, and val/test are never filtered, so
  evaluation will show that gap rather than hide it.

## Evaluation design

- **5-layer ladder** (PLAN.md §29, `scripts/eval_harness.py`), each
  layer reported separately: L1 JSON parse rate → L2 schema validity
  rate → L3 field accuracy (name exact-match, file-path F1,
  description token-F1) → L4 LLM-judge (rubric 1–5 × 4 criteria;
  skipped without `OPENROUTER_API_KEY`) → L5 materialization rate.
- **SkillGen Score** (§33):
  `0.25·L1 + 0.25·L2 + 0.20·L3 + 0.15·L4 + 0.15·L5`.
  When L4 is skipped, the score is renormalized over the available
  layers (divided by 0.85) and the un-renormalized sum is also
  recorded — see `docs/DECISIONS.md`.
- **Accept gate (§39):** a candidate is promoted only if its SkillGen
  Score is **≥ current promoted score + 2 points AND** it shows **no
  regression on the expert set**.
- Baselines (§34): base zero-shot (E0a), base few-shot with 3 fixed
  train exemplars (E0b), QLoRA model (E1) — same frozen splits, same
  prompts; paired comparison (fixed/broken counts, McNemar, CIs) via
  `src/skillgen/evaluation/compare.py`. Ablations E2–E6 and the §37
  forgetting check (50 ARC-Challenge questions before/after; a drop
  > 5% is a finding) are designed and tooled.

## Current evidence — SMOKE ONLY

Every number below is **SMOKE / 0.5B / pipeline-proof**: it proves the
pipelines run end to end, and says nothing about 7B model quality.

| Evidence | Label | Numbers |
|---|---|---|
| Baseline smoke (`experiments/e0_smoke/scores.json`: Qwen2.5-0.5B-Instruct, zero-shot, val limit 5, CPU) | **SMOKE / 0.5B / pipeline-proof** | L1 **1.0**, L2 **0.0**, SkillGen Score **0.3245** (renormalized over 0.85, L4 skipped). All 5 outputs were coherent JSON with invented schemas — the failure fine-tuning must fix. |
| LoRA training smoke (`experiments/e1_smoke/`: 0.5B, plain LoRA, 20 shortest filtered records, 10 steps, CPU) | **SMOKE / 0.5B / pipeline-proof** | Train loss **2.45 → 1.80** over 10 steps (per-step first 2.4453, last 1.8009); masking gate passed. |
| Forgetting smoke (`experiments/forgetting_smoke/`: base 0.5B vs the smoke adapter, local 20-question set — NOT ARC) | **SMOKE / 0.5B / pipeline-proof** | **0.80 vs 0.80** (16/20 both), identical predictions, drop 0.0 points — expected for a 10-step adapter. |

## Limitations

- **No 7B model exists yet** — no capability claims can be made.
- Training data underrepresents very large multi-file skills (the
  ≤8192-token filter above).
- Instructions are synthetic backtranslations, not real user requests;
  distribution shift to real requests is unmeasured.
- The harness cannot see semantically-wrong-but-schema-valid output
  (hallucinated capabilities, wrong instructions) without the L4
  judge or the expert set — a documented blind spot
  (`docs/ERROR_TAXONOMY.md`).
- The base model choice is provisional and may change after the 2026
  model verification, which would invalidate direct comparability
  with any earlier runs.

## Pending

- Real E0 (7B, full val/test) and E1 (7B QLoRA) runs on Kaggle, and
  their paired Phase-4 comparison.
- Ablations E2–E6; ARC-50 forgetting check on the real adapter.
- Expert-set evaluation (§31).
- 2026 base-model verification before freezing the base model.
- Executing the §44 export (merged 16-bit → GGUF `q4_k_m` → Ollama)
  on a GPU machine once a real adapter exists — locally it is
  dry-run only (`scripts/export_model.py --dry-run`).
