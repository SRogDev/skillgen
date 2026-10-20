# SkillGen

Fine-tune a small open model (Qwen2.5-7B-Instruct, QLoRA) to generate **Agent Skills** from a natural-language request.

Pipeline: `user request → model → JSON Skill Blueprint → validator/harness → filesystem`. The model never writes to disk directly; it outputs a validated blueprint.

## Dataset

`data/dataset_skillgen_1000_shard_*.jsonl` — **1000 examples in 10 shards of 100** (sharded because the GitHub API rejects single blobs > ~50 MB; reconstruct with `cat data/dataset_skillgen_1000_shard_*.jsonl > data/dataset_skillgen_1000.jsonl`). Each example:

- `instruction`: synthetic user request (backtranslation of a real skill)
- `expected_blueprint`: JSON Skill Blueprint derived from a real skill on [skills.sh](https://skills.sh) (MIT / Apache-2.0 only, license verified)
- `metadata`: id, domain, repo, provenance (license, quality checks)

11 domains (software, marketing, productivity, finance, sales, writing, research, analytics, business, operations, support). 769 MIT + 231 Apache-2.0.

⚠️ **Frozen test set — never train on it.** Splits: 800/100/100 (seed 7), `manifest.json` with sha256.

## Status

- **Dataset frozen (2026-09-27):** splits via `scripts/freeze_splits.py` — near-dup clustering (token Jaccard > 0.8, one cluster per split), domain stratification, zero overlap.
- **Eval harness** (`scripts/eval_harness.py`): 5-layer ladder — L1 JSON parse → L2 schema validity → L3 field accuracy (name EM, file-path F1, desc token-F1) → L4 LLM-judge → L5 tmpdir materialization. **9/9 tests** (stdlib).
- **Phase 0 — EDA + environment + length decision — DONE (2026-10-20).** Tokenized the full rendered training text (PLAN.md §8 system prompt + instruction + blueprint, Qwen2.5-7B-Instruct chat template) for all 1,000 records (`scripts/eda_dataset.py`, `experiments/eda_v1/report.md`): train token p50 **3,838.5** / p95 **34,071.2** / p99 **120,768.0** / max **1,947,918**; **46.75%** of train records exceed 4,096 tokens. Dataset health: 0 JSON parse failures, 0 blueprints missing `SKILL.md`. **Length decision: `max_seq_length = 8192`, filter-never-truncate — 218 of 800 train records (27.25%) excluded from training, 582 kept;** val/test are never excluded from evaluation (over-cap records only flagged). See `STATUS.md`. **No training has happened yet.**
- **Phase 1 — PyTorch foundations + LoRA from scratch — DONE (2026-10-20).** A tiny character-level language model trained with a fully manual loop (forward → cross-entropy → `backward()` → `step()` → `zero_grad()`) over real blueprint text: loss **5.6030 → 1.0381** in 300 CPU steps. `LoRALinear` implemented from scratch in `src/skillgen/training/fundamentals.py` (frozen base, kaiming A / zero B, scaling α/r) with 5 unit tests; QLoRA memory math computed in-notebook for a 7B model: **84.00 GB** full fine-tune states vs **3.66 GB** QLoRA (≈22.9× less).
- **Notebooks:** `notebooks/00_environment.ipynb` (environment audit — Python/GPU/CUDA/disk/library versions, Kaggle detection, seq-length fallback rule; executed, CPU-only WARN verdict on this VM), `notebooks/01_dataset_eda.ipynb` (dataset EDA with token/char histograms, truncation table, domain chart; executed, teaching prose in Spanish), `notebooks/02_pytorch_foundations.ipynb` (tensors/gradients explained in Spanish + the manual training loop above; executed) and `notebooks/03_lora_from_scratch.ipynb` (`LoRALinear` demo, parameter counts, 7B VRAM table, 10-step LoRA-only smoke with bit-identical frozen base; executed, teaching prose in Spanish).
- **Phase 2 — Baseline pipeline (E0) — DONE (2026-10-20), real E0 still pending.** `src/skillgen/evaluation/baseline.py` + `scripts/run_baseline.py`: PLAN.md §8 prompts (zero-shot, and few-shot with 3 fixed train-split exemplars — never from val/test), CPU generation with the model's chat template, and scoring that **imports and reuses** `scripts/eval_harness.py` (no reimplemented scoring). L4 is skipped (no judge key), so the SkillGen Score is renormalised over the available layers (see `STATUS.md`). **SMOKE run executed** (`experiments/e0_smoke/`, clearly labeled SMOKE — pipeline proof, **not** E0 results): `Qwen/Qwen2.5-0.5B-Instruct`, zero-shot, val limit 5 — all 5 outputs were complete, coherent JSON but with invented schemas (no `files`, wrong names), giving L1 **1.0000** / L2 **0.0000** / L3 **0.1290** / L5 **0.0000**, SkillGen Score **0.3245**. Real E0 (7B, full val/test) awaits the Kaggle GPU run.
- **Torch is CPU-only locally:** `pyproject.toml` resolves `torch` from the PyTorch CPU index (`https://download.pytorch.org/whl/cpu`, currently 2.14.1+cpu) so local dev carries no multi-GB CUDA wheels. Kaggle provides its own CUDA torch for real training runs.
- **QLoRA recipe** in `PLAN.md`: r16/alpha32/lr2e-4/2 epochs/batch16/warmup3%/cosine/adamw_8bit/all-linear/bnb-4bit; mask verification, early stopping on best val loss, merge-16bit → GGUF q4_k_m → Ollama; ablations E0–E7.
- **Base model is PROVISIONAL:** Qwen2.5-7B-Instruct, pending verification of 2026 models before freezing it as base.
- **Next:** Phases 3–5 (QLoRA code + CPU smokes, eval/ablations, serving, export) and the real E0 run on Kaggle — see `STATUS.md`; per-record provenance; official Agent Skills spec.

## Scripts

- `scripts/select_full.py` — stratified selection from cloned skills.sh repos
- `scripts/convert_batch.py` — selections → Skill Blueprints
- `scripts/build_dataset.py` — assemble final JSONL (validation, dedup)
- `scripts/freeze_splits.py` — freeze train/val/test splits
- `scripts/eval_harness.py` — 5-layer eval ladder (`--judge` needs `OPENROUTER_API_KEY`)
- `scripts/eda_dataset.py` — Phase 0 EDA: tokenizes the full rendered training text and writes `experiments/eda_v1/report.json` + `report.md` (run: `uv run python scripts/eda_dataset.py`)
- `scripts/run_baseline.py` — Phase 2 baseline pipeline (E0): generates predictions with a base model and scores them with the eval harness; writes `predictions.jsonl` + `scores.json` + a run README (labeled **SMOKE** unless it is the real 7B/full-split E0 config). Defaults are the real E0 config: `Qwen/Qwen2.5-7B-Instruct`, zero-shot, full val. (On this VM, set `HF_HUB_DISABLE_XET=1` — the Xet download path stalled at 0 bytes.)

Tests: `python3 tests/test_freeze_splits.py`, `python3 tests/test_eval_harness.py` (stdlib only), plus `uv run pytest` (42 tests: EDA helpers in `src/skillgen/data/eda.py`, LoRA in `src/skillgen/training/fundamentals.py` — `tests/test_lora.py`, and the baseline pipeline in `tests/test_baseline.py`).

`data/raw/` (2.4 GB of cloned repos) is intentionally **not** versioned — regenerate from provenance.

## License

MIT — see [LICENSE](LICENSE).
