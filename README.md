# SkillGen

Learning project: fine-tune a small open model (Qwen2.5-7B-Instruct, QLoRA) to
generate **Agent Skills** from a natural-language request.

Pipeline: `user request → model → JSON Skill Blueprint → validator/harness → filesystem`.
The model never writes to disk directly; it outputs a validated blueprint.

## Dataset

`data/dataset_skillgen_1000_shard_*.jsonl` — 1000 examples in 10 shards of 100
(sharded because the GitHub API rejects single blobs > ~50 MB; reconstruct with
`cat data/dataset_skillgen_1000_shard_*.jsonl > data/dataset_skillgen_1000.jsonl`).
Each example:

- `instruction`: synthetic user request (backtranslation of a real skill)
- `expected_blueprint`: JSON Skill Blueprint derived from a real skill on
  [skills.sh](https://skills.sh) (MIT / Apache-2.0 only, license verified)
- `metadata`: id, domain, repo, provenance (license, quality checks)

11 domains (software, marketing, productivity, finance, sales, writing,
research, analytics, business, operations, support). 769 MIT + 231 Apache-2.0.

⚠️ **Do not train before freezing the test/expert split** (see PLAN.md §13).
The split is not frozen yet.

## Scripts

- `scripts/select_full.py` — stratified selection from cloned skills.sh repos
  (license filter, dedup, anti-stub filter)
- `scripts/convert_batch.py` — convert selections to Skill Blueprints
  (`convert_skill.py` handles one skill: frontmatter parse, license verify,
  name normalization)
- `scripts/build_dataset.py` — assemble the final JSONL (validates names,
  blueprints, instruction coverage, duplicates, near-duplicates)
- `scripts/freeze_splits.py` — freeze train/val/test splits (800/100/100,
  seed 7): near-duplicate clustering (token Jaccard > 0.8, one cluster per
  split), domain stratification, sha256 manifest in `data/splits/`
- `scripts/eval_harness.py` — 5-layer eval ladder for predictions:
  L1 JSON parse → L2 schema validity → L3 field accuracy → L4 LLM-as-judge
  (`--judge`, needs `OPENROUTER_API_KEY`) → L5 blueprint materialization.
  Usage: `python3 scripts/eval_harness.py --preds preds.jsonl --refs refs.jsonl`

Tests (stdlib only, no pytest needed): `python3 tests/test_freeze_splits.py`
and `python3 tests/test_eval_harness.py`.

## Reproducing

1. Clone skills.sh-indexed repos into `data/raw/_repos/` (see PLAN.md §12).
2. `python3 scripts/select_full.py --quotas '{...}' --out data/selection.jsonl`
3. `python3 scripts/convert_batch.py --selection data/selection.jsonl`
4. Write synthetic instructions per blueprint → `data/instructions.json`
5. `python3 scripts/build_dataset.py --selections ... --instructions ... --out data/dataset.jsonl`

`data/raw/` (2.4 GB of cloned repos) is intentionally **not** versioned —
regenerate from provenance. Same for `data/blueprints/` intermediates.

## Status

Dataset v1 (1000) done. Splits frozen (800/100/100, `data/splits/`).
Eval harness done (`scripts/eval_harness.py`). Next: training script/notebook
(Unsloth QLoRA) — see PLAN.md. Open items: LICENSE file, per-record provenance
completion, individual skills.sh verification, official Agent Skills spec check.

## License

Code: MIT. Dataset entries keep their original skill licenses (MIT/Apache-2.0,
recorded per record in provenance).
