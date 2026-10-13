---
name: crazy-behavioral
description: Design the explicit behavioral contract for a new Agent Skill from a natural-language request, optimized for a small or weaker language model to turn into a Skill Blueprint. Use before SkillGen generation: takes a raw request or a developed skill design and produces a structured Behavioral Contract (purpose, triggers, behavior, decision rules, safety, acceptance criteria). Never authors the final SKILL.md or skill files — output is the intermediate design artifact for the generation pipeline.
---

# crazy-behavioral

Produce the **Behavioral Contract** — a complete, explicit, unambiguous specification
of how a skill must behave — for SkillGen's generation pipeline to turn into a
Skill Blueprint. This skill designs behavior; it does not build artifacts.

Pipeline position: `user request → crazy-behavioral → Behavioral Contract →
(existing SkillGen generation: model → blueprint JSON → harness → filesystem)`.
The downstream pipeline is unchanged: the model still generates, the harness
still validates, materialization stays controlled. See `docs/crazy-behavioral.md`
(repo-root-relative) for the integration.

## Input

Accept either:

- A **natural-language request** (short, long, ambiguous, multi-requirement), or
- A **developed skill design** (the user already specified the procedure and/or
  the distinctive mechanism — preserve those decisions; formalize, don't replace).

If the user already specified the procedure and Crazy Thing, keep them and sharpen
their operational precision. Never displace a good user-provided idea with process.

## Interaction model

Ask focused questions **only** when a material uncertainty cannot be resolved from
the available information — a question whose answer would change the behavior.
Never run a compulsory interview on a clear request. When the design is
sufficiently specified, output the contract and stop.

## The contract

Fill in `references/contract-template.md` completely. Every section required:

1. **Purpose and expected outcomes** — what the skill does, for whom, to what end.
2. **Trigger conditions** — exact situations that must activate the skill.
3. **Negative triggers** — situations that must NOT activate it.
4. **Inputs, context, assumptions, prerequisites.**
5. **Step-by-step behavior** — ordered steps, each with a completion criterion
   (the agent must be able to tell done from not-done).
6. **Decision rules and branches** — concrete if/then rules for every choice
   point; no "use your judgment" without an operational definition.
7. **Steering instructions** — how the agent reasons and prioritizes.
8. **Failure handling and recovery** — what can go wrong, what to do.
9. **Safety and permission boundaries** — prohibited actions, authorization needs.
10. **Evidence and verification** — what the agent must check or produce.
11. **Stopping conditions** — when the work is done.
12. **Output contract** — exact shape of what the skill delivers.
13. **Required and optional capabilities** — tools, integrations, knowledge.
14. **Resource requirements** — scripts, references, data the skill needs.
15. **Acceptance criteria and test scenarios** — how to tell the skill works.
16. **Crazy Thing** — the distinctive behavioral mechanism, if one is justified;
    as a falsifiable statement (mechanism, expected advantage, how to test it).
    Omit rather than invent a gimmick.
17. **Explicit uncertainties and unresolved questions** — what is still open.

## Provenance labels

Every contract statement carries exactly one label:

- `[USER]` — the user explicitly required it.
- `[DESIGNER]` — you decided it to make the behavior operational.
- `[ASSUMED]` — neither user nor evidence established it; flagged for review.
- `[OPEN]` — unresolved; blocks or conditions the behavior.

A contract with unlabeled statements, or with `[ASSUMED]` material presented as
`[USER]` requirements, is unfinished.

## Small-model adaptation rules

Assume the downstream model has limited reasoning ability and context:

- Make requirements **explicit** — never rely on implied knowledge.
- Use clear, unambiguous language; one meaning per term; keep terminology
  consistent across the whole contract.
- **Separate required behavior from optional enhancements** — the model must
  never have to guess what is mandatory.
- Resolve contradictions **before** producing the contract; never hand the
  model two instructions that conflict.
- Prefer concrete conditions, examples, and decision rules over abstractions.
- State dependencies and tool assumptions outright ("requires: file read
  access; web search optional").
- Delete redundant instructions and unnecessarily complex branching.
- The goal is **minimum ambiguity**, not maximum length: keep every
  behaviorally important detail, cut everything else.

Banned without an operational definition: "be smart", "be creative", "ensure
high quality", "handle edge cases" (name them or drop the phrase).

## Contract → blueprint mapping

The contract must answer everything the Skill Blueprint schema needs. Check the
mapping before finishing (field names per the SkillGen blueprint, v1.0):

| Blueprint field | Contract source |
|---|---|
| `name` | Purpose §1 → a spec-valid name (`[a-z0-9-]`, ≤64 chars, = directory name) |
| `description` | Purpose §1 + triggers §2 → what + when + trigger keywords (≤1024 chars) |
| `purpose` | §1 verbatim |
| `when_to_use[]` | §2, one entry per trigger |
| `when_not_to_use[]` | §3, one entry per negative trigger |
| `files[]` | §5 + §12 + §14 → SKILL.md content plan + each script/reference/asset with path, purpose, and content outline |
| `tools[]` | §13 required capabilities |
| `dependencies[]` | §13–§14 external requirements |
| `evaluation_criteria[]` | §15 acceptance criteria + test scenarios |

If a blueprint field cannot be determined from the contract, the contract is
incomplete — go back, do not leave it for the model to invent.

## What this skill will NOT do

- It will **not** generate the final SKILL.md.
- It will **not** create skill directories, supplementary files, or any
  filesystem artifact.
- It will **not** write to disk in any form — output is the contract text.
- It will **not** claim to have validated an artifact that does not exist yet.
- It will **not** change SkillGen's dataset, splits, training setup, model
  configuration, or evaluation baseline. Those are out of scope by design.
