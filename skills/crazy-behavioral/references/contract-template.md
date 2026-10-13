# Behavioral Contract Template

Fill every section. Every statement carries exactly one provenance label:
`[USER]`, `[DESIGNER]`, `[ASSUMED]`, or `[OPEN]`. Delete the guidance lines
(they start with `>`); keep the section headings.

## 1. Purpose and expected outcomes

> One paragraph: what the skill does, for whom, to what end. Must be
> translatable into a spec-valid skill name and a ≤1024-char description.

## 2. Trigger conditions

> Concrete situations that MUST activate the skill. One per line, each
> starting with the leading trigger word/phrase.

-

## 3. Negative triggers

> Concrete situations that MUST NOT activate it. Be specific about the
> near-miss cases (requests that sound similar but belong elsewhere).

-

## 4. Inputs, context, assumptions, prerequisites

> What the agent receives, what it may assume about the environment,
> what must be true before it starts.

-

## 5. Step-by-step behavior

> Ordered steps. Each step ends with a completion criterion — the condition
> that tells the agent the step is done. No step may depend on implied
> knowledge.

1.
   - Done when:

## 6. Decision rules and branches

> Every choice point as an explicit if/then. Cover the branches the steps
> above can reach.

- IF … THEN … ELSE …

## 7. Steering instructions

> How the agent reasons, what it prioritizes, how it handles judgment calls
> the rules above do not cover. Operational, not aspirational.

## 8. Failure handling and recovery

> What can go wrong at each step, how the agent detects it, what it does next.
> Include: missing inputs, tool failures, ambiguous results.

## 9. Safety and permission boundaries

> Prohibited actions (explicit list). Anything needing authorization, and from
> whom. What the agent must never do even if asked mid-task.

## 10. Evidence and verification

> What the agent must check, observe, or produce to show the work is correct.
> Prefer observed behavior over claims.

## 11. Stopping conditions

> The exact conditions under which the agent stops and reports. Include the
> "cannot proceed" exits.

## 12. Output contract

> The exact shape of the deliverable: format, sections, what each contains.

## 13. Required and optional capabilities

> Tools, integrations, knowledge the skill needs. Mark each REQUIRED or
> OPTIONAL. State tool assumptions outright.

- REQUIRED:
- OPTIONAL:

## 14. Resource requirements

> Scripts, references, templates, data files the skill needs, each with
> proposed path (`scripts/…`, `references/…`, `assets/…`), purpose, and
> content outline. Omit the section entirely if none are needed.

## 15. Acceptance criteria and test scenarios

> Testable criteria plus at least: one positive case, one negative
> (non-trigger) case, one edge case — each with expected behavior.

## 16. Crazy Thing (optional)

> The distinctive behavioral mechanism, if justified. Falsifiable form:
> mechanism (operational), expected advantage (specific), how to test it.
> Omit rather than invent.

## 17. Uncertainties and unresolved questions

> Everything labeled `[OPEN]` collected here, each naming what it blocks.

## Contract → blueprint check

Before delivering, verify each row has a contract source:

| Blueprint field | Covered by |
|---|---|
| `name` | §1 |
| `description` | §1 + §2 |
| `purpose` | §1 |
| `when_to_use[]` | §2 |
| `when_not_to_use[]` | §3 |
| `files[]` (path/purpose/content) | §5 + §12 + §14 |
| `tools[]` | §13 |
| `dependencies[]` | §13 + §14 |
| `evaluation_criteria[]` | §15 |

Any empty cell = incomplete contract. Go back; do not leave it for the model.
