# SkillGen — Status

Last updated: 2026-10-20.

## Phases

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | EDA, environment notebook, dataset length decision | **Done — 2026-10-20** |
| Phase 1 | PyTorch foundations + LoRA from scratch | **Done — 2026-10-20** |
| Phase 2 | Baseline pipeline (E0): prompt construction, generation, harness scoring + CPU smoke run | **Done — 2026-10-20 (pipeline + SMOKE); real E0 (7B, full val/test) pending Kaggle GPU** |
| Phase 3 | QLoRA training pipeline (prepare/train, configs E1–E6) + CPU smoke | **Done — 2026-10-20 (code + SMOKE); real E1 (7B QLoRA) pending Kaggle GPU** |
| Phase 4 | Evaluation tooling (paired comparison, failure analysis, forgetting check) + experiment registry | **Done — 2026-10-20 (tooling + SMOKE demos); real E0-vs-E1 comparisons and ablation runs (E1–E6) pending Kaggle GPU** |
| Phase 5 | Serving / inference path, export (merged model / GGUF) + full Spanish learning guide | Pending |

## Phase 0 results (measured, not estimated)

EDA over the frozen 800/100/100 splits, tokenizing the **full rendered training
text** (system prompt from PLAN.md §8 + instruction + blueprint, via the
`Qwen/Qwen2.5-7B-Instruct` chat template). Full numbers:
`experiments/eda_v1/report.json` and `experiments/eda_v1/report.md`;
notebooks: `notebooks/00_environment.ipynb`, `notebooks/01_dataset_eda.ipynb`.

Token lengths (full rendered text):

| Split | p50 | p95 | p99 | max | over 4096 |
|---|---|---|---|---|---|
| train (800) | 3,838.5 | 34,071.2 | 120,768.0 | 1,947,918 | 374 (46.75%) |
| val (100) | 3,522.5 | 35,504.2 | 86,666.1 | 99,946 | 42 (42.00%) |
| test (100) | 4,466.0 | 51,767.3 | 319,161.9 | 571,700 | 53 (53.00%) |

Dataset health: 0 JSON parse failures and 0 blueprints missing `SKILL.md`
across all 1,000 records. The dataset's problem is length, not validity: the
longest train record (1,947,918 tokens) even exceeds the model's 131,072-token
context window.

**Length decision:** `max_seq_length = 8192`. Train rule: **filter, never
truncate** — drop the **218 of 800 train records (27.25%)** whose rendered
length is over 8192 (a truncated blueprint is invalid JSON and would teach
incomplete outputs); 582 train records are kept whole. Val/test are **never**
excluded from evaluation — over-cap records are only flagged (val 29/100,
test 29/100). Fallback: VRAM < 16 GB or compute capability < 7.0 →
`max_seq_length = 2048`, per-device batch 1, more gradient accumulation
(see `notebooks/00_environment.ipynb`).

**No training has happened yet.** Phase 0 is measurement and environment
auditing only.

## Phase 1 results (measured, executed notebooks)

Torch 2.14.1+cpu (CPU wheel via the PyTorch CPU index in `pyproject.toml`;
`torch.cuda.is_available()` = False on this VM — Kaggle supplies its own
CUDA torch). Notebooks `02_pytorch_foundations.ipynb` and
`03_lora_from_scratch.ipynb` executed in place, zero error outputs.

- Tiny character-level LM (611,015 params; embedding + position + MLP head)
  trained with a fully manual loop on 3 real blueprint texts (4,502 chars):
  loss **5.6030 → 1.0381** over 300 steps — memorisation on a tiny corpus,
  used to teach gradients, `backward()`, the optimiser, `zero_grad()` and
  overfitting.
- `LoRALinear` from scratch (`src/skillgen/training/fundamentals.py`):
  frozen base, kaiming A / zero B (ΔW = 0 at init, verified exactly),
  `trainable_parameter_count()` = r·(in + out). 5 new tests in
  `tests/test_lora.py` (RED → GREEN); full suite 32/32 pytest green.
- LoRA-only 10-step smoke on a toy classification task: loss
  **1.1446 → 0.0180**, base weights bit-identical before/after
  (`torch.equal`).
- 7B VRAM math computed in the notebook: full fine-tune states
  **84.00 GB** (7e9 × (2 + 2 + 8) bytes) vs QLoRA **3.66 GB** (4-bit base +
  fp16 adapters, 40,370,176 params at r=16/all-linear/28 layers, + 8-bit
  optimiser states) — ≈22.9× less, which is why Phase 3 uses QLoRA on a
  16 GB T4.

## Phase 2 results (baseline pipeline E0 + CPU SMOKE)

Pipeline in `src/skillgen/evaluation/baseline.py` (+ CLI
`scripts/run_baseline.py`): `build_messages` (PLAN.md §8 system prompt
verbatim; zero-shot = system + user; few-shot = system + 3 fixed
train-split exemplar pairs + user), `load_split_records`,
`generate_predictions` (HF causal LM on CPU, chat template with
`add_generation_prompt=True`, only generated tokens decoded, predictions
in the harness's `{"id", "output"}` format plus `model`/`mode`/`split`
provenance) and `score_predictions`, which **imports and reuses**
`scripts/eval_harness.py` (`l1_parse`, `l2_validate_schema`,
`l3_field_scores`, `l5_materialize`) — no scoring is reimplemented.
Few-shot exemplars are 3 fixed train records (first 3 of
`train_shard_00.jsonl` with rendered length 5,000–10,000 chars;
ids in `FEW_SHOT_EXEMPLAR_IDS`), asserted to be in train and absent
from val/test; few-shot refuses to run otherwise. The frozen test split
was **not** generated on. 10 new tests in `tests/test_baseline.py`
(RED → GREEN); full suite **42/42 pytest green**.

**SkillGen Score note (L4 skipped):** the harness omits L4 when the
judge is not run (it needs `OPENROUTER_API_KEY`). Following that,
`score_predictions` reports `l4_judge == "skipped"` and renormalises
the PLAN.md §33 weights over the available layers (÷ 0.85), with the
un-renormalised weighted sum also recorded. L3 = mean of the three
harness L3 sub-metrics.

**SMOKE run (pipeline proof, NOT E0 results)** —
`experiments/e0_smoke/`: `Qwen/Qwen2.5-0.5B-Instruct`, zero-shot, val,
limit 5, max_new_tokens 1024, greedy, CPU; wall time 433.6 s
(generation + scoring, after the ~1 GB download; on this VM the HF Xet
download stalled at 0 bytes, so the run used `HF_HUB_DISABLE_XET=1`).
All 5 outputs were complete, coherent, on-topic JSON objects — but
wrapped in markdown fences and using **invented schemas**: every one
misses `blueprint_version`, `purpose`, `when_to_use`,
`when_not_to_use` and `files`, and uses a Title/CamelCase name instead
of the kebab-case reference name. Scores (harness, exact):

| Metric | SMOKE value |
|---|---|
| L1 parse rate | 1.0000 |
| L2 schema rate | 0.0000 |
| L3 name EM / file-path F1 / desc token-F1 | 0.0000 / 0.0000 / 0.3870 |
| L3 (mean) | 0.1290 |
| L4 judge | skipped |
| L5 materialize rate | 0.0000 |
| SkillGen Score (renormalised, no L4) | 0.3245 |

Low L2/L3/L5 are the **expected** outcome for a 0.5B untrained model:
it produces plausible JSON that does not follow the Skill Blueprint
schema — precisely what fine-tuning is supposed to fix. **Real E0
(`Qwen/Qwen2.5-7B-Instruct`, full val/test) is still pending the
Kaggle GPU run.**

## Phase 3 results (QLoRA training pipeline + CPU SMOKE)

Code in `src/skillgen/training/prepare.py` + `train.py` (+ `config.py`),
deps via uv: **trl 1.16.0, peft 0.21.2, accelerate 1.15.0, datasets
5.1.0** (transformers 5.20.x / torch 2.14.1+cpu as before).
`bitsandbytes` and `unsloth` are deliberately NOT local deps (bnb needs
CUDA; unsloth pins its own torch stack) — they install in the Kaggle
notebook cells only.

- **prepare.py**: renders prompt/full text with the model's native chat
  template, tokenizes both with no extra special tokens, and builds
  labels = -100 on prompt positions + full ids after. The Qwen template
  renders the assistant turn as `{content}<|im_end|>\n`; examples are
  cut right after `<|im_end|>` so each ends exactly at EOS, and the
  prompt-prefix property is asserted (raise, never silently mislabel).
  Length rule is the Phase-0 decision (8192, filter-never-truncate).
  `train.py`'s §21 gate re-verifies masking on the first example and
  prints the decoded unmasked text before training starts.
- **TRL API (inspected, not assumed)**: trl 1.16.0's `SFTTrainer`
  treats a dataset with an `input_ids` column as pre-processed and
  leaves a provided `labels` column as is — so prepare.py's certified
  labels are the trained labels; no trainer-internal masking is used.
  Two adaptations vs. the classic recipe: transformers 5.x removed
  `warmup_ratio` (the config's 0.03 is passed as `warmup_steps=0.03`,
  which now accepts a float ratio), and TRL's bf16 default is pinned
  off explicitly (transformers 5.x rejects bf16 on CPU-only setups).
- **Configs** (`configs/`): `qlora_e1.yaml` (§18 recipe, 8192 override,
  PROVISIONAL base noted) + E2/E3×2/E4×2/E5/E6×2 one-change variants +
  `smoke_cpu.yaml`. E6 sizes (250/500) come from the **filtered 582**
  pool, noted in the config comments. 14 new tests in
  `tests/test_prepare.py` (masking on a real record, filter,
  config-diff: each variant differs from E1 in exactly its intended
  key); full suite **56/56 pytest green**.
- **SMOKE run (`experiments/e1_smoke/`, NOT E1)**: 0.5B, plain LoRA r8
  (quantization off), 20 shortest filtered train records, 10 steps,
  batch 2, 1 epoch, CPU; wall 804 s. Masking gate passed (decoded
  unmasked = blueprint + `<|im_end|>` only). Train loss per step
  **2.4453 → 1.8009** (mean 1.8634); val eval loss **2.0292 → 2.0031**.
  Adapter saved (`adapter_model.safetensors` 17,640,136 bytes,
  git-ignored; README + log committed).
- **Notebooks**: `04_baseline.ipynb` (executed; E0a/E0b rationale,
  harness layers, L4-skipped renormalisation over 0.85, one failed SMOKE
  prediction dissected: invented schema — the failure fine-tuning must
  fix) and `05_qlora_training.ipynb` (Kaggle production notebook for
  E1, Spanish, NOT executed).
- **Real E1 is still pending**: 7B QLoRA on Kaggle (notebook 05),
  ~73 steps over the 582-record pool. No E1 numbers exist yet.

## Base model

**PROVISIONAL:** `Qwen/Qwen2.5-7B-Instruct` — pending verification of 2026
model releases before the base model is frozen (see PLAN.md). Phase 0 used
its tokenizer only, on CPU.

## Frozen test set

⚠️ `data/splits/test.jsonl` is **frozen** — never train on it, never modify
any split file. Splits: 800/100/100, seed 7, sha256 in
`data/splits/manifest.json`.

## Phase 4 results (evaluation tooling + SMOKE demos)

Tooling only — **no real experiment comparison has run**; E0 (7B) and
E1 (7B QLoRA) are both still pending Kaggle, and the registry
(`experiments/README.md`) says so per experiment: E0a smoke-only,
E0b pipeline implemented + unit-tested with **no run at all yet**,
E1 smoke-only, E2–E6 configs ready-pending-GPU, E7 design-only.

- **compare.py** (`src/skillgen/evaluation/compare.py`, stdlib only):
  `paired_counts` (both/only-A/only-B/both-fail on l1_parse, l2_valid
  or l5), `mcnemar_exact` (two-sided exact binomial via `math.comb`),
  `wilson_ci`, and `score_diff_ci` — a paired bootstrap (seed 42, 2000
  resamples) over the binary-metric difference (documented choice:
  per-item SkillGen Score contributions are not stored in scores.json).
  `render_comparison_md` emits the §47 BEST/WORST/REGRESSION/NEW-FAILURE
  skeleton. 15 tests in `tests/test_compare.py` (RED → GREEN), including
  the hand-computed McNemar value for b=9, c=1: p = 0.021484375.
- **Failure analysis** (`scripts/failure_analysis.py` +
  `docs/ERROR_TAXONOMY.md`): maps harness signals to §48 codes E1–E10.
  Honest gap: §48 in the repo names only the anchors (E1 invalid
  structure, E10 format failure); E2–E9 are a documented local
  operationalisation, unmapped strings fall to E10/other, and an L5
  failure downstream of an L2 failure is not double-counted as E8.
  Real output on the e0_smoke scores (`experiments/e0_smoke/failures.md`):
  **E2 (missing required field) ×5, E4 (invalid name) ×5**, all other
  codes 0 — every smoke item invents the schema (5 missing fields +
  a non-kebab name). That is the failure SFT must fix.
- **Forgetting check** (`src/skillgen/evaluation/forgetting.py`):
  loglikelihood argmax over 4 options per question (no generation).
  `load_arc_questions()` loads ARC-Challenge from HF when available and
  falls back to the original local 20-question set (science/history/
  math/geography/language), labelled by source — the local set is
  **NOT ARC**. SMOKE run (`experiments/forgetting_smoke/`): base
  Qwen2.5-0.5B-Instruct **0.8000 (16/20)** vs the Phase-3 smoke adapter
  **0.8000 (16/20)**, identical predictions on all 20, drop 0.0 points —
  expected for a 10-step adapter; the §37 rule (>5% drop = finding)
  applies to the real run (50 ARC questions, E1 adapter, Kaggle).
- **Notebook** `06_evaluation.ipynb` (executed, zero errors, Spanish):
  the real e0_smoke failure analysis; a `compare.py` demo against a
  clearly-labelled **synthetic** degraded copy (predictions truncated
  to 200 chars — demo data, not an experiment): L1 paired counts
  only_a=5/only_b=0, McNemar p = 0.0625; L2 both_fail=5, p = 1.0000;
  the forgetting smoke; and how the Kaggle E1-vs-E0 comparison will be
  read (paired counts first, §39 gate: promote only if SkillGen
  Score ≥ current + 2 pts and no expert-set regression).
- Full suite: **72/72 pytest green** (56 previous + 16 new); the old
  stdlib test scripts still pass (eval harness 5/5, freeze splits 4/4).
