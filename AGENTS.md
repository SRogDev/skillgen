# AGENTS.md — skillgen

Working agreement for any AI agent operating in this repo (Muse, GPT, Codex, Cursor, etc.).
Owner: Roger (SRogDev). Discussion language: Spanish. Implementation plans for coding agents: English.

## How Roger works — read this first
- Incremental MVPs over big up-front design. Architecture emerges from concrete needs; no speculative complexity.
- "Learning by building": implement the minimum that works, learn when real problems arise, ship artifacts.
- Always inspect the repo before making architectural or file-level decisions.
- Reuse existing systems instead of creating parallel abstractions.
- Many functional, basic MVPs first; polish later.
- Explanations: only what's needed to build and solve the immediate problem. No theory dumps.
- Be an auditor, not a cheerleader: evidence-only assessments, uncomfortable findings stated plainly, always paired with concrete next options.
- Corrections are final: adopt Roger's version without debate and keep moving.
- No unprompted initiatives outside the approved scope — offer options instead.

## Engineering discipline
- ALL coding work: ODD routing (route by size) → TDD (test evidence) → RDD (diff self-review before delivery). No exceptions. (Muse: use the `gentle-ai` workspace skill.)
- ALL frontend/UI work: design-system-first — resolve style, palette, fonts, UX guidelines BEFORE implementing. No exceptions. (Muse: use the `ui-ux-pro-max` workspace skill, run its design-system search first.)
- Python: `uv` ONLY — `uv venv`, deps in `pyproject.toml [project]`, `uv sync`, `uv.lock` committed. Never `python -m venv` + pip, never bare `requirements.txt`.
- JS/TS: check the latest Next.js version before installing (16.x is current as of 2026-09); use the official codemod for upgrades. Prefer Biome over ESLint/Prettier.
- Conventional commits: `feat:` / `fix:` / `chore:` / `docs:` / `refactor:` / `test:`.

## PR workflow & history (mandatory)
- Every change ships as a PR against `main`. Muse is authorized to merge his own PRs.
- One PR = one logical change. Squash-merge so `main` stays linear and readable — one clean commit per change.
- Never push directly to `main`.
- PR description must include verifiable evidence: test counts, build results, commit SHAs. Never invent numbers — write "not measured" when unknown.
- When a phase/milestone completes, update STATUS.md in the same PR.

## Context docs (keep them current)
- `README.md` — human/contributor-facing overview.
- `docs/PROJECT_BRIEF.md` — full project context in one file, written to be handed to ANOTHER AI for planning/ideation. Keep it accurate; it is the handoff doc.
- `STATUS.md` — timeline: done / doing / next. Read it before starting work; update it when reality changes.

## Stack
- Python (uv only). Public repo, MIT.
- SkillGen trains a small model (proposed Qwen2.5-7B) to GENERATE Agent Skills (SKILL.md + resources) from natural-language descriptions.
- Dataset: 1000 samples FROZEN in 10 shards; splits 800/100/100 (seed 7); `manifest.json` with sha256; `scripts/freeze_splits.py` (near-dup clustering Jaccard>0.8, stratified, zero overlap).
- ⚠️ FROZEN TEST SET: NEVER train on it. Ever.
- Eval harness `scripts/eval_harness.py`: L1 parse → L2 schema → L3 fields (name EM, file-path F1, desc token-F1) → L4 LLM-judge → L5 tmpdir materialization. 9/9 tests (stdlib).
- PLAN.md: QLoRA recipe (r16/alpha32/lr2e-4/2 epochs/batch16/warmup3%/cosine/adamw_8bit/all-linear/bnb-4bit), mask verification (labels≠-100 on assistant tokens only), early stopping on best val loss, merge-16bit → GGUF q4_k_m → Ollama, ablations E0–E7.
- Note: one 7.9MB train record truncates hard at max_seq_length 4096 — notebook 03 must log truncations.

## Repo map
- `data/` — frozen dataset (10 shards) + `manifest.json`
- `scripts/` — `freeze_splits.py`, `eval_harness.py`
- `tests/` — 9 tests; `PLAN.md` — full training plan
- Commands: `uv sync` / `uv run pytest`

## What NOT to do
- No speculative abstractions, "just in case" features, or parallel systems.
- Don't reformat whole files for style; keep diffs reviewable.
- Never commit secrets, `.env` files, or credentials.
- Don't invent metrics, benchmarks, or test results.
