# SkillGen — Status

Last updated: 2026-10-20.

## Phases

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | EDA, environment notebook, dataset length decision | **Done — 2026-10-20** |
| Phase 1 | PyTorch foundations + LoRA from scratch | **Done — 2026-10-20** |
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

## Base model

**PROVISIONAL:** `Qwen/Qwen2.5-7B-Instruct` — pending verification of 2026
model releases before the base model is frozen (see PLAN.md). Phase 0 used
its tokenizer only, on CPU.

## Frozen test set

⚠️ `data/splits/test.jsonl` is **frozen** — never train on it, never modify
any split file. Splits: 800/100/100, seed 7, sha256 in
`data/splits/manifest.json`.
