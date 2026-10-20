# Decisions (ADR-lite)

One entry per decision: **Decision / Alternatives / Why.** These
record choices already made and executed in this repo; PLAN.md holds
the full plan, `docs/FINDINGS.md` the measured evidence behind them.

## 1. Base model: Qwen2.5-7B-Instruct, PROVISIONAL

- **Decision:** Use `Qwen/Qwen2.5-7B-Instruct` as the base model,
  explicitly PROVISIONAL — pending verification of 2026 model
  releases before the base is frozen.
- **Alternatives:** Freeze a base immediately; wait for the 2026
  verification before building anything.
- **Why:** The pipeline (tokenizer, chat template, configs, harness)
  is model-shaped but not model-locked; building against a concrete,
  available 7B instruct model unblocks every phase, while the
  PROVISIONAL label keeps the door open. Freezing early risks training
  on a superseded base; waiting risks building nothing.

## 2. max_seq_length 8192 + filter-never-truncate

- **Decision:** `max_seq_length = 8192`; train records whose full
  rendered text exceeds it are dropped whole (218 of 800 dropped, 582
  kept). Val/test are never dropped.
- **Alternatives:** 4096 (drops 46.75% of train); 16384 (drops only
  13.38%); truncate long records to fit.
- **Why:** Truncation teaches invalid, incomplete JSON blueprints —
  the output format is the thing being learned. 4096 loses nearly
  half the data; 16384 sharply raises activation memory and step
  time for 7B QLoRA on a 16 GB T4. 8192 is the largest power-of-two
  length treated as a safe default on that hardware, and every kept
  example stays complete. The cost is documented honestly: very large
  multi-file skills are underrepresented in training.

## 3. QLoRA over full fine-tuning

- **Decision:** Train with QLoRA (4-bit base + fp16 LoRA adapters,
  r16/alpha32/all-linear, 8-bit optimizer) — PLAN.md §18 recipe.
- **Alternatives:** Full fine-tuning of the 7B model.
- **Why:** Measured VRAM math (Phase 1): full fine-tune states cost
  **84.00 GB**; QLoRA costs **3.66 GB** (≈22.9× less). Full FT does
  not fit the available hardware (Kaggle T4, 16 GB); QLoRA fits with
  headroom. This is a hardware-evidence decision, not a preference.

## 4. Assistant-only labels are built in prepare.py — never trainer-internal masking

- **Decision:** `src/skillgen/training/prepare.py` tokenizes prompt
  and full text itself and builds `labels` (-100 on prompt positions)
  before the trainer ever sees the data; `train.py` re-verifies the
  masking on the first example as a §21 gate.
- **Alternatives:** Rely on the trainer/framework's completion-only
  masking options.
- **Why:** Masking decides what the model learns; it must be
  inspectable, unit-testable, and identical across framework
  versions. Building labels in prepare.py made them certifiable by a
  test that decodes the unmasked tokens of a real record. (Verified
  against the installed trl 1.16 source: a provided `labels` column
  is left untouched.)

## 5. Local CPU LoRA smoke vs production QLoRA

- **Decision:** Locally, run only a CPU smoke: 0.5B model, plain LoRA
  (no quantization), few records/steps, explicitly labeled SMOKE.
  Production QLoRA (7B, 4-bit, bitsandbytes/unsloth) runs only on
  Kaggle.
- **Alternatives:** Install bitsandbytes/unsloth locally and try to
  approximate the production run on CPU; skip local smoke entirely.
- **Why:** bitsandbytes needs CUDA and unsloth pins its own torch
  stack — neither belongs in the local CPU environment. The smoke's
  job is plumbing proof (filtering, masking, TRL wiring, adapter
  save) at $0; pretending it approximates production quality would be
  dishonest. Every smoke artifact is labeled SMOKE / 0.5B /
  pipeline-proof.

## 6. L4-skipped SkillGen Score: renormalization over 0.85

- **Decision:** When the L4 judge is not run (no
  `OPENROUTER_API_KEY`), report the SkillGen Score renormalized over
  the available layers — weighted sum divided by 0.85
  (0.25 + 0.25 + 0.20 + 0.15) — and also record the un-renormalized
  weighted sum.
- **Alternatives:** Score L4 as 0 (max score 0.85); omit the combined
  score entirely when L4 is skipped.
- **Why:** Following the harness's own behaviour (it omits L4 when
  the judge is not run). Scoring L4 as 0 would punish a run for a
  missing API key, not for model quality, and would make scores
  incomparable across judge/no-judge runs. Renormalization keeps the
  score on a 0–1 scale; recording both numbers keeps the choice
  auditable.

## 7. CPU-only torch pin locally

- **Decision:** `pyproject.toml` resolves torch from the PyTorch CPU
  index (`https://download.pytorch.org/whl/cpu`); local dev carries
  no CUDA wheels.
- **Alternatives:** Default PyPI torch (multi-GB CUDA dependencies).
- **Why:** This VM has no GPU; the CUDA wheels are dead weight on
  every install and sync. Kaggle provides its own CUDA torch, so the
  pin affects only local CPU work — training semantics are unchanged
  where training actually happens.

## 8. No DPO in v1

- **Decision:** No RLHF/DPO in v1 (PLAN.md §50). Preference pairs may
  be saved as data, but no preference training is built.
- **Alternatives:** Add a DPO stage after SFT in v1.
- **Why:** PLAN §50 fixes v1's limits: the SFT hypothesis (does
  QLoRA beat the baselines on the SkillGen Score?) must be answered
  first, with the §39 accept gate, before any preference stage can be
  evaluated honestly. DPO without a measured SFT baseline would be
  complexity without an attributable effect.

## 9. Error taxonomy E2–E9 operationalized from harness signals

- **Decision:** PLAN.md §48's E1–E10 taxonomy is implemented with E1
  and E10 as named in the plan, and E2–E9 defined by the exact
  harness signal that maps to each code (`docs/ERROR_TAXONOMY.md`,
  `scripts/failure_analysis.py`).
- **Alternatives:** Leave §48 unimplemented until the original
  questionnaire definitions are recovered; invent definitions
  without tying them to signals.
- **Why:** §48 in this repo names only the anchors (E1 invalid
  structure, E10 format failure); the questionnaire itself is not in
  the repo, so E2–E9 are not recoverable verbatim. Signal-tied
  definitions make the classifier deterministic, testable, and
  honest about its provenance — and the gap is documented in
  `docs/ERROR_TAXONOMY.md` so the table can be reconciled if the
  original definitions surface.
