# SkillGen — Status

Last updated: 2026-10-20.

## Phases

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | EDA, environment notebook, dataset length decision | **Done — 2026-10-20** |
| Phase 1 | ML fundamentals + baseline (non-fine-tuned) pipeline | Pending |
| Phase 2 | QLoRA training code + CPU smoke tests (honestly labeled; real GPU runs on Kaggle) | Pending |
| Phase 3 | Evaluation + ablations design and runs | Pending |
| Phase 4 | Serving / inference path | Pending |
| Phase 5 | Export (merged model / GGUF) + full Spanish learning guide | Pending |

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

## Base model

**PROVISIONAL:** `Qwen/Qwen2.5-7B-Instruct` — pending verification of 2026
model releases before the base model is frozen (see PLAN.md). Phase 0 used
its tokenizer only, on CPU.

## Frozen test set

⚠️ `data/splits/test.jsonl` is **frozen** — never train on it, never modify
any split file. Splits: 800/100/100, seed 7, sha256 in
`data/splits/manifest.json`.
