# Error taxonomy E1–E10 (PLAN.md §48)

PLAN.md §47 requires every experiment's `failures.md` to classify failures
with the E1–E10 taxonomy of §48. §48 adopts the taxonomy from the original
implementation questionnaire "as is" and, in the repo, names only its
anchors verbatim:

- **E1 — invalid structure**
- **E10 — format failure**

## Known gap (honest)

The questionnaire itself is not in this repository, so the full original
wording of E2–E9 is **not recoverable from the repo**. The definitions
below for E2–E9 are a documented local operationalisation: each code is
defined by the exact harness signal (`scripts/eval_harness.py` error
strings and L1/L3/L5 outcomes) that maps to it, implemented in
`scripts/failure_analysis.py`. If Roger supplies the original
questionnaire definitions, this table should be reconciled against them.

## Taxonomy and harness-signal mapping

| Code | Definition | Harness signal(s) |
|---|---|---|
| E1 | Invalid structure — parsed JSON is not an object / top-level structure unusable | L2 error `blueprint is not an object` |
| E2 | Missing required field | L2 error `missing field: …` |
| E3 | Wrong field type | L2 errors `field … has wrong type`, `… must be list[str]` |
| E4 | Invalid name (not kebab-case) | L2 error `invalid name: …` |
| E5 | Incomplete file set | L2 errors `files is empty`, `files missing SKILL.md`, `files[i] …` (not an object / field missing or empty) |
| E6 | Unsafe path | L2/L5 errors containing `unsafe` (`…path unsafe: …`, `unsafe path: …`) |
| E7 | Empty or invalid content field | L2 error `empty description` |
| E8 | Materialization failure | L5 fails (`l5_materialize == false`) **while L2 is valid** — frontmatter/materialize errors |
| E9 | Field-fidelity failure | L2 valid but L3 `name_em == 0` (content does not match the reference) |
| E10 | Format failure / other | L1 parse failure (`l1_parse == false`), a missing reference, or **any** harness error string matching no rule above |

Mapping rules worth knowing:

- Classification is **per item, multi-label**: one item can carry several
  codes (e.g. the e0_smoke items are each E2 + E4).
- An L5 failure is **not** counted as E8 when L2 already failed: the
  materialization failure is then a downstream consequence of the schema
  failure, and counting it separately would double-count one root cause.
- `missing field: files` is E2 (missing field), not E5: E5 is reserved
  for a `files` list that is present but incomplete/invalid.
- Unmapped strings fall to **E10/other** and are never dropped; if a new
  harness error appears, its E10 count is the signal to extend this table.

## What the harness cannot see (un-mappable today)

These failure kinds have **no signal** in `scores.json` and therefore
cannot be classified by `scripts/failure_analysis.py`; they need the L4
judge or the expert set (PLAN.md §30/§31):

- Hallucinated capabilities or files that are schema-valid (faithfulness).
- Semantically wrong but well-formed instructions (completeness/consistency).
- Repetition loops / non-termination (only visible in raw generation logs).

Until such a signal exists, items exhibiting only those problems pass
the classifier — a documented blind spot, not a claim that they are good.
