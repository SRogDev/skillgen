# skillgen — Project Brief

> Full project context in one file. Hand this to ANOTHER AI (GPT, etc.) for planning
> and ideation, then bring the refined specs back. Keep this file accurate — it is the handoff doc.
> For the current timeline see `STATUS.md`. For how to work in this repo see `AGENTS.md`.

## One-liner
SkillGen trains a small model (proposed Qwen2.5-7B) to GENERATE Agent Skills (SKILL.md + resources) from natural-language descriptions.

## Problem & audience
Writing Agent Skills by hand doesn't scale; if a small fine-tuned model can generate them reliably, skill creation becomes a commodity capability.

## Product (what it is / is not)
A training pipeline: frozen dataset → QLoRA fine-tune → GGUF/Ollama model that writes Agent Skills. NOT a generic fine-tuning framework — one task, measured ruthlessly (L1–L5 eval harness).

## Key decisions (locked)
- Public repo, MIT.
- Dataset frozen with provenance (sha256 manifest, seed 7, near-dup clustering) — frozen test NEVER used for training.
- QLoRA recipe per PLAN.md; mask verification (labels≠-100 on assistant tokens only); early stopping on best val loss.
- Evals: L1 parse → L2 schema → L3 fields → L4 LLM-judge → L5 tmpdir materialization.

## Stack
Python (uv), Unsloth, QLoRA, Ollama/GGUF. Eval harness stdlib-only.

## Business model
Open-source; feeds Roger's AI-infra story. No direct monetization.

## Open questions
- 2026 model verification before freezing Qwen2.5-7B.
- Whether L4 (LLM-judge) scores predict real skill usability.
