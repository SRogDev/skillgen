# Findings

Cumulative findings from the real artifacts in this repo
(`experiments/`, `STATUS.md`, the executed notebooks). Every number
below comes from a committed artifact; SMOKE numbers are labeled
SMOKE and are pipeline proof, not model-quality results. Real 7B
results do not exist yet — the real runs are pending a Kaggle GPU run
(see the last section).

## Phase 0 — EDA and the dataset length decision

Source: `experiments/eda_v1/report.json` / `report.md`,
`notebooks/00_environment.ipynb`, `notebooks/01_dataset_eda.ipynb`.

- Token lengths of the **full rendered training text** (§8 system
  prompt + instruction + blueprint, Qwen2.5-7B-Instruct chat template)
  are extremely right-skewed. Train: **p50 3,838 / p95 34,071 /
  max 1,947,918** tokens. The longest train record even exceeds the
  model's 131,072-token context window.
- **46.75% of train records (374/800) exceed 4,096 tokens**; 27.25%
  (218/800) exceed 8,192; val/test show the same skew (42% / 53% over
  4,096).
- Dataset health is not the problem: 0 JSON parse failures and 0
  blueprints missing `SKILL.md` across all 1,000 records. The problem
  is length, not validity.
- **Decision: `max_seq_length = 8192`, filter-never-truncate** — E1
  trains on the **582** train records that fit whole; a truncated
  blueprint is invalid JSON and would teach incomplete outputs.
  Val/test are never excluded from evaluation (over-cap records only
  flagged: 29/100 in each). Known cost: very large multi-file skills
  are underrepresented in training.

## Phase 1 — PyTorch foundations and LoRA from scratch

Source: `notebooks/02_pytorch_foundations.ipynb`,
`notebooks/03_lora_from_scratch.ipynb`,
`src/skillgen/training/fundamentals.py`, `tests/test_lora.py`.

- **LoRA math / VRAM:** for a 7B model, full fine-tuning states cost
  **84 GB** (7e9 × (2 + 2 + 8) bytes: weights + gradients + Adam
  states) vs **QLoRA 3.66 GB** (4-bit base + fp16 adapters at
  r=16/all-linear/28 layers = 40,370,176 trainable params + 8-bit
  optimizer states) — ≈22.9× less. That gap is why Phase 3 uses QLoRA
  on a 16 GB T4.
- A tiny character-level LM (611,015 params) trained with a fully
  manual loop on 3 real blueprint texts: **loss 5.60 → 1.04** over
  300 steps — memorisation on a tiny corpus, used to teach the loop,
  not a capability result.
- `LoRALinear` from scratch: frozen base, kaiming A / zero B, so
  ΔW = 0 at init (verified exactly); trainable count = r·(in + out).
  A LoRA-only 10-step smoke left the base weights bit-identical
  (`torch.equal`).

## Phase 2 — Baseline pipeline (E0) + CPU SMOKE

Source: `src/skillgen/evaluation/baseline.py`,
`experiments/e0_smoke/` (scores, predictions, failures),
`notebooks/04_baseline.ipynb`.

- The pipeline imports and reuses the existing harness
  (`scripts/eval_harness.py`) — no scoring is reimplemented — and
  few-shot exemplars are 3 fixed train-split records, asserted absent
  from val/test.
- **SMOKE run (pipeline proof, NOT E0 results):** Qwen2.5-0.5B-Instruct,
  zero-shot, val limit 5, CPU. All 5 outputs were complete, coherent,
  on-topic JSON — but with **invented schemas**: every one misses
  `blueprint_version`, `purpose`, `when_to_use`, `when_not_to_use`
  and `files`, and uses a Title/CamelCase name instead of the
  kebab-case reference name. Scores: L1 1.0000 / L2 0.0000 /
  L3 0.1290 / L5 0.0000, SkillGen Score 0.3245 (renormalized, L4
  skipped). Finding: **schema invention is the dominant failure of
  the untrained model** — precisely what fine-tuning must fix.
- Failure taxonomy on the smoke (`experiments/e0_smoke/failures.md`):
  **E2 (missing required field) ×5, E4 (invalid name) ×5**, all other
  codes 0.
- **HF Xet download stall:** on this VM, Hugging Face downloads via
  Xet stalled at 0 bytes; the workaround is `HF_HUB_DISABLE_XET=1`,
  after which the ~1 GB download completed. Recorded here because it
  will recur on this network.

## Phase 3 — QLoRA training pipeline + CPU SMOKE

Source: `src/skillgen/training/{prepare,train,config}.py`,
`configs/`, `experiments/e1_smoke/`, `notebooks/05_qlora_training.ipynb`
(Kaggle production notebook, not executed locally).

- **The masking gate works:** `prepare.py` builds assistant-only
  labels (-100 on prompt positions), and `train.py`'s §21 gate
  re-verifies on the first example and prints the decoded unmasked
  text before training — in the smoke it was exactly the blueprint +
  `<|im_end|>`, nothing else.
- **TRL behaviour (inspected, not assumed):** trl 1.16's `SFTTrainer`
  treats a dataset with an `input_ids` column as pre-tokenized and
  **accepts a provided `labels` column unchanged** — so prepare.py's
  certified labels are the trained labels; no trainer-internal
  masking is relied on.
- **Transformers 5.x API changes found by running, not guessing:**
  `warmup_ratio` was removed (the config's 0.03 is passed as
  `warmup_steps=0.03`, which now accepts a float ratio), and bf16
  defaults had to be pinned off explicitly because transformers 5.x
  rejects bf16 on CPU-only setups.
- **SMOKE run (pipeline proof, NOT E1):** 0.5B, plain LoRA r8
  (quantization off — bitsandbytes needs CUDA), 20 shortest filtered
  train records, 10 steps, CPU. Train loss **2.4453 → 1.8009**
  (mean 1.8634); val eval loss 2.0292 → 2.0031. Proves the plumbing
  (filtering, masking, TRL wiring, adapter save), nothing about 7B
  quality.

## Phase 4 — Evaluation tooling + SMOKE demos

Source: `src/skillgen/evaluation/{compare,forgetting}.py`,
`scripts/failure_analysis.py`, `docs/ERROR_TAXONOMY.md`,
`experiments/forgetting_smoke/`, `experiments/README.md`,
`notebooks/06_evaluation.ipynb`.

- Paired comparison tooling works on real smoke artifacts (paired
  fixed/broken counts, exact McNemar — hand-verified p = 0.021484375
  for b=9, c=1 — Wilson CIs, paired bootstrap, seed 42). A
  clearly-labeled **synthetic** degraded copy demos `compare.py`
  end to end; no real E0-vs-E1 comparison exists yet.
- **Forgetting SMOKE (pipeline proof only):** base Qwen2.5-0.5B
  **0.80 vs 0.80** for the Phase-3 smoke adapter on the local
  20-question set (16/20 both, identical predictions, drop 0.0) —
  expected for a 10-step adapter. The local set is NOT ARC; the real
  §37 check is 50 ARC-Challenge questions on the E1 adapter.
- **Taxonomy gap, documented:** PLAN.md §48 names only the anchors
  **E1 and E10** in this repo; **E2–E9 were operationalized from
  harness signals** and are documented as a local operationalisation
  in `docs/ERROR_TAXONOMY.md` (unmapped strings fall to E10/other; an
  L5 failure downstream of an L2 failure is not double-counted as
  E8). If the original questionnaire definitions surface, the table
  must be reconciled against them.

## Phase 5 — Serving, export, docs

Source: `src/skillgen/serving/`, `scripts/export_model.py`,
`tests/test_serving.py`, `tests/test_export.py`, `MODEL_CARD.md`.

- The §42 loop works against a fake backend: valid-on-first-try
  (attempts 1), repair-after-schema-error (attempts 2), and typed
  reject after 1 + 2 attempts (HTTP 422, `blueprint_invalid`), plus
  typed 502 on backend failure — all covered by TestClient tests with
  no model download.
- The demo default backend is a FakeBackend returning one real train
  blueprint, and `/health` names it `fake-demo (NOT a trained
  model)` — the demo cannot be mistaken for the real model.
- The §44 export exists as an explicit ordered plan
  (load → `merged_16bit` → GGUF `q4_k_m` from the merged model →
  Ollama instructions), unit-tested for ordering and for the
  no-direct-4-bit-merge rule. Execution needs GPU + unsloth, so on
  this VM it is dry-run only, by design.

## Pending the Kaggle GPU run

Nothing below has run; there are no numbers for any of it yet.

- Real E0a (7B zero-shot) and E0b (7B few-shot) on the full val/test
  splits — E0b has not run at all yet, not even a smoke.
- Real E1: 7B QLoRA over the 582-record filtered pool (notebook 05).
- Ablations E2–E6 (masking, epochs, rank, target modules, data size).
- The §37 forgetting check at real scale: 50 ARC-Challenge questions,
  base vs E1 adapter.
- Expert-set evaluation (§31) and the Phase-4 paired E0-vs-E1
  comparison under the §39 accept gate.
- Executing the §44 export (merged 16-bit → GGUF → Ollama) with a
  real adapter, including Modelfile chat-template inspection.
