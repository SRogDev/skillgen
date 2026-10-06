# STATUS — skillgen

> Single source of truth for where this project stands. Last updated: 2026-10-06.
> Read this before starting work. Update it in the same PR when reality changes.

## Done
- 2026-09-27 — Public repo (SRogDev/skillgen), MIT.
- 2026-09-27 — Dataset FROZEN: 1000 samples in 10 shards; splits 800/100/100 (seed 7); `manifest.json` with sha256; `scripts/freeze_splits.py` (near-dup clustering Jaccard>0.8, stratified, zero overlap).
- 2026-09-27 — Eval harness `scripts/eval_harness.py`: L1 parse → L2 schema → L3 fields (name EM, file-path F1, desc token-F1) → L4 LLM-judge → L5 tmpdir materialization. 9/9 tests (stdlib).
- 2026-09-27 — PLAN.md: QLoRA recipe (r16/alpha32/lr2e-4/2 epochs/batch16/warmup3%/cosine/adamw_8bit/all-linear/bnb-4bit), mask verification, early stopping, merge-16bit → GGUF q4_k_m → Ollama, ablations E0–E7.

## In progress / blocked
- Verify 2026 models before freezing Qwen2.5-7B as the base.

## Next
- Unsloth QLoRA notebook (log truncations — one 7.9MB train record truncates hard at max_seq_length 4096); per-record provenance; individual `skills.sh` verification; official Agent Skills spec.
- ⚠️ FROZEN TEST SET: never train on it. Ever.
