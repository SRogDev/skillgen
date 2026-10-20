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
- **Notebooks:** `notebooks/00_environment.ipynb` (environment audit — Python/GPU/CUDA/disk/library versions, Kaggle detection, seq-length fallback rule; executed, CPU-only WARN verdict on this VM) and `notebooks/01_dataset_eda.ipynb` (dataset EDA with token/char histograms, truncation table, domain chart; executed, teaching prose in Spanish).
- **QLoRA recipe** in `PLAN.md`: r16/alpha32/lr2e-4/2 epochs/batch16/warmup3%/cosine/adamw_8bit/all-linear/bnb-4bit; mask verification, early stopping on best val loss, merge-16bit → GGUF q4_k_m → Ollama; ablations E0–E7.
- **Base model is PROVISIONAL:** Qwen2.5-7B-Instruct, pending verification of 2026 models before freezing it as base.
- **Next:** Phases 1–5 (baseline pipeline, QLoRA code + CPU smokes, eval/ablations, serving, export) — see `STATUS.md`; per-record provenance; official Agent Skills spec.

## Scripts

- `scripts/select_full.py` — stratified selection from cloned skills.sh repos
- `scripts/convert_batch.py` — selections → Skill Blueprints
- `scripts/build_dataset.py` — assemble final JSONL (validation, dedup)
- `scripts/freeze_splits.py` — freeze train/val/test splits
- `scripts/eval_harness.py` — 5-layer eval ladder (`--judge` needs `OPENROUTER_API_KEY`)
- `scripts/eda_dataset.py` — Phase 0 EDA: tokenizes the full rendered training text and writes `experiments/eda_v1/report.json` + `report.md` (run: `uv run python scripts/eda_dataset.py`)

Tests: `python3 tests/test_freeze_splits.py`, `python3 tests/test_eval_harness.py` (stdlib only), plus `uv run pytest tests/test_eda.py` (18 tests for the pure EDA helpers in `src/skillgen/data/eda.py`).

`data/raw/` (2.4 GB of cloned repos) is intentionally **not** versioned — regenerate from provenance.

## License

MIT — see [LICENSE](LICENSE).
