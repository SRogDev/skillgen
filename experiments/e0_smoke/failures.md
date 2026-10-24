# Failure analysis — e0_smoke (SMOKE — Qwen2.5-0.5B-Instruct, zero-shot, val limit 5)

Format: PLAN.md §47 (BEST / WORST / REGRESSION / NEW FAILURE), classified with the §48 taxonomy E1–E10 (see docs/ERROR_TAXONOMY.md).

## Taxonomy counts (items per code; one item can carry several codes)

| Code | Meaning | Items |
|---|---|---|
| E1 | Invalid structure — parsed JSON is not an object / top-level structure unusable | 0 |
| E2 | Missing required field | 5 |
| E3 | Wrong field type | 0 |
| E4 | Invalid name (not kebab-case) | 5 |
| E5 | Incomplete file set (files empty, missing SKILL.md, file entry incomplete) | 0 |
| E6 | Unsafe path | 0 |
| E7 | Empty or invalid content field (e.g. empty description) | 0 |
| E8 | Materialization failure (L5 fails while L2 is valid) | 0 |
| E9 | Field-fidelity failure (L2 valid but L3 name exact-match fails) | 0 |
| E10 | Format failure / other (L1 parse failure, or any unmapped harness signal) | 0 |

Failed items: 5 of 5.

## BEST (passing items)

Count: 0


## WORST (failed items, with taxonomy codes)

Count: 5

- `davila7_claude-code-templates__clinical-reports` — E2, E4
- `anthropics_knowledge-work-plugins__data-visualization` — E2, E4
- `affaan-m_ecc__mysql-patterns` — E2, E4
- `microsoft_azure-skills__azure-resource-visualizer` — E2, E4
- `grafana_skills__send-data` — E2, E4

## REGRESSION

No comparison run provided.

## NEW FAILURE

No comparison run provided.
