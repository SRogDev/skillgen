# crazy-behavioral — SkillGen integration

`crazy-behavioral` (`skills/crazy-behavioral/`) is a behavioral-design skill that
runs **before** SkillGen's generation pipeline. It turns a raw user request into
an explicit **Behavioral Contract** optimized for a small/weaker language model,
so the model receives minimum-ambiguity input. It stops before authoring any
skill files.

## Pipeline

Existing path:

```
user request → model → JSON Skill Blueprint → harness (L1–L5) → filesystem
```

Assisted path (this integration):

```
user request → crazy-behavioral → Behavioral Contract → model → JSON Skill Blueprint → harness (L1–L5) → filesystem
                              ↑ contract IS the instruction
```

Only the input side changes. Everything downstream is untouched.

## Invocation

1. A human or a capable agent runs `crazy-behavioral` on the raw request
   (see `skills/crazy-behavioral/SKILL.md`).
2. The skill asks focused questions only for material uncertainties, then
   outputs the filled Behavioral Contract
   (`skills/crazy-behavioral/references/contract-template.md`).
3. Serialize the contract into the generation input:
   - **instruction** = the full contract text (17 sections, provenance-labeled).
   - **context** = the original raw request (preserves the user's own words).
4. Feed that into the existing pipeline unchanged (same system prompt, same
   model, same harness, same materialization).

## Contract → blueprint field mapping

The contract is designed so every REQUIRED blueprint field (harness L2) is
answerable from it — no schema change needed (`blueprint_version` stays `1.0`):

| Blueprint field | Contract source |
|---|---|
| `name` | §1 Purpose → spec-valid name |
| `description` | §1 + §2 Triggers → what + when + keywords |
| `purpose` | §1 |
| `when_to_use[]` | §2, one entry per trigger |
| `when_not_to_use[]` | §3, one entry per negative trigger |
| `files[]` | §5 behavior + §12 output + §14 resources |
| `tools[]` / `dependencies[]` | §13 capabilities + §14 |
| `evaluation_criteria[]` | §15 acceptance criteria + test scenarios |

The skill refuses to finish with an unmappable contract: any blueprint field
not determined by the contract sends the designer back, instead of leaving it
for the model to invent.

## Boundaries preserved

- The model still generates the blueprint; the harness still validates
  (L1–L5); materialization stays harness-controlled. The model never touches
  the filesystem — that guarantee is unchanged.
- **Dataset, splits, training setup, model configuration, and the frozen
  evaluation split are untouched.** No training data was modified; no new
  training was run for this integration.
- `crazy-behavioral` never generates SKILL.md, never creates files, never
  claims validation of a non-existent artifact.

## Preliminary evaluation: existing vs assisted input path

**Method.** 5 representative requests across the dataset's input types
(short_instruction, ambiguous_request, technical_requirement,
business_process, existing_skill_modification). Two paths per request:
(A) the raw instruction as-is; (B) the crazy-behavioral contract for the same
request. Scored two ways:

1. **Blueprint-field answerability (deterministic):** for each of the 6
   REQUIRED content fields the L2 validator checks
   (`name`, `description`, `purpose`, `when_to_use`, `when_not_to_use`,
   `files`), does the input explicitly determine it? (1 / 0.5 / 0)
2. **Rubric 1–5:** behavioral completeness, trigger precision, decision-rule
   completeness, safety & failure handling, acceptance-criteria coverage,
   intent preservation.

**Results.**

| Case | A: fields /6 | B: fields /6 | A: rubric | B: rubric |
|---|---|---|---|---|
| short_instruction (commit messages) | 1.5 | 6 | 1.8 | 4.2 |
| ambiguous_request (idea tracking) | 1.0 | 6 | 1.3 | 3.7 |
| technical_requirement (CSV validator) | 2.5 | 6 | 2.7 | 4.8 |
| business_process (support onboarding) | 2.0 | 6 | 1.8 | 3.8 |
| existing_skill_modification (secret scan) | 2.5 | 6 | 2.2 | 4.7 |
| **Mean** | **1.9** | **6.0** | **2.0** | **4.2** |

The assisted path answers every REQUIRED blueprint field in all 5 cases;
raw instructions average 1.9/6 — they almost never determine the skill name,
negative triggers, or file layout, which are exactly the fields the L2
validator rejects blueprints for. The ambiguous case additionally exercised
the interaction model: one focused question resolved the ambiguity, remaining
uncertainty labeled `[ASSUMED]`/`[OPEN]` instead of invented.

**Limitations (read before citing).** This is a *preliminary, input-side*
comparison: no trained model was involved, so it does not measure
end-to-end generation quality. Contracts were produced by a strong model
following the skill — the skill's usability by a weaker operator is untested.
The rubric was scored by the same party that wrote the contracts (not
independent). n=5 is illustrative, not statistically meaningful. Treat the
improvement as a hypothesis until the trained-model comparison below runs.

## Future work (after the training baseline exists)

Re-run the comparison end-to-end with the fine-tuned model held constant:
same requests, path A vs path B inputs, same decoding config, evaluated with
the frozen harness on the validation split (never the frozen test split):
L1/L2 pass rates, L3 field scores, L4 judge, L5 materialization rate.
That is the experiment that can promote the hypothesis to a claim.
