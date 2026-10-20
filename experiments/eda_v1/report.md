# SkillGen dataset EDA — v1

Tokenizer/model: `Qwen/Qwen2.5-7B-Instruct` (tokenizer only, CPU). Lengths are for the FULL rendered training text (system prompt from PLAN.md §8 + instruction + blueprint) via the model's chat template.

## Token-length stats (full rendered text)

| Split | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| train | 800 | 12,072.6 | 3,838.5 | 20,743.6 | 34,071.2 | 120,768.0 | 1,947,918 |
| val | 100 | 9,192.4 | 3,522.5 | 18,884.3 | 35,504.2 | 86,666.1 | 99,946 |
| test | 100 | 19,507.1 | 4,466.0 | 24,805.5 | 51,767.3 | 319,161.9 | 571,700 |

## Char-length stats (full rendered text)

| Split | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| train | 800 | 45,901.3 | 14,484.5 | 80,250.6 | 132,935.0 | 426,431.6 | 7,170,763 |
| val | 100 | 36,093.4 | 14,474.5 | 80,835.0 | 122,217.7 | 355,037.7 | 384,703 |
| test | 100 | 63,288.7 | 18,569.0 | 91,712.2 | 215,765.0 | 1,043,639.8 | 1,049,661 |

## Truncation — records OVER each token threshold

| Split | >1024 | >2048 | >4096 | >8192 | >16384 |
|---|---|---|---|---|---|
| train | 696 (87.00%) | 580 (72.50%) | 374 (46.75%) | 218 (27.25%) | 107 (13.38%) |
| val | 86 (86.00%) | 67 (67.00%) | 42 (42.00%) | 29 (29.00%) | 13 (13.00%) |
| test | 87 (87.00%) | 70 (70.00%) | 53 (53.00%) | 29 (29.00%) | 17 (17.00%) |

## Domain distribution

| Domain | train | val | test |
|---|---|---|---|
| analytics | 62 | 7 | 7 |
| business | 50 | 7 | 7 |
| finance | 85 | 10 | 10 |
| marketing | 104 | 13 | 13 |
| operations | 45 | 6 | 6 |
| productivity | 102 | 13 | 13 |
| research | 64 | 8 | 8 |
| sales | 78 | 9 | 9 |
| software | 134 | 17 | 17 |
| support | 12 | 2 | 2 |
| writing | 64 | 8 | 8 |

## Blueprint structure

| Split | records | JSON parse failures | missing SKILL.md | files-per-blueprint distribution |
|---|---|---|---|---|
| train | 800 | 0 | 0 | 1 files × 431, 2 files × 118, 3 files × 65, 4 files × 31, 5 files × 11, 6 files × 23, 7 files × 22, 8 files × 17, 9 files × 15, 10 files × 7, 11 files × 10, 12 files × 1, 13 files × 4, 14 files × 6, 15 files × 6, 16 files × 2, 18 files × 2, 20 files × 1, 21 files × 4, 22 files × 3, 23 files × 1, 24 files × 1, 25 files × 1, 26 files × 2, 27 files × 1, 28 files × 1, 30 files × 1, 31 files × 1, 32 files × 1, 33 files × 2, 35 files × 1, 36 files × 1, 41 files × 1, 48 files × 1, 53 files × 1, 56 files × 1, 74 files × 1, 95 files × 1, 612 files × 1 |
| val | 100 | 0 | 0 | 1 files × 55, 2 files × 16, 3 files × 4, 4 files × 6, 5 files × 1, 6 files × 3, 7 files × 2, 8 files × 5, 9 files × 1, 10 files × 1, 15 files × 2, 17 files × 1, 21 files × 1, 31 files × 1, 34 files × 1 |
| test | 100 | 0 | 0 | 1 files × 49, 2 files × 13, 3 files × 5, 4 files × 4, 5 files × 9, 6 files × 4, 7 files × 2, 8 files × 4, 10 files × 2, 11 files × 1, 12 files × 1, 17 files × 1, 26 files × 1, 34 files × 1, 50 files × 1, 145 files × 1, 235 files × 1 |

## Top-10 longest records overall (tokens)

| # | id | split | domain | tokens | chars |
|---|---|---|---|---|---|
| 1 | forcedotcom_sf-skills__platform-metadata-api-context-get | train | sales | 1,947,918 | 7,170,763 |
| 2 | stanleychanh_tushare-finance-skill-for-claude-code__stanleychanh_tushare-finance-skill-for-claude-code | test | finance | 571,700 | 1,043,579 |
| 3 | larksuite_cli__lark-slides | test | productivity | 316,611 | 1,049,661 |
| 4 | forcedotcom_sf-skills__agentforce-generate | train | sales | 246,482 | 1,054,026 |
| 5 | anthropics_knowledge-work-plugins__video-sdk | test | software | 232,802 | 884,433 |
| 6 | forcedotcom_sf-skills__agentforce-d360-analyze | train | sales | 180,144 | 686,468 |
| 7 | larksuite_cli__lark-sheets | train | productivity | 171,453 | 426,315 |
| 8 | giuseppe-trisciuoglio_developer-kit__aws-sdk-java-v2-bedrock | train | software | 167,825 | 507,080 |
| 9 | vasilyu1983_ai-agents-public__software-ux-research | train | research | 164,829 | 638,843 |
| 10 | vasilyu1983_ai-agents-public__ops-devops-platform | train | operations | 161,436 | 616,181 |

## LENGTH DECISION

- **max_seq_length = 8192 tokens.**
- **Train filter rule:** drop train records whose full rendered token length is > 8192 (filter, never truncate: a truncated blueprint is invalid JSON and would teach the model to emit incomplete blueprints)
- **Excluded from training:** exactly **218 of 800 train records (27.25%)**; 582 records are kept.
- **Val/test are NEVER excluded from evaluation.** Records over the cap are only flagged: val 29/100, test 29/100.

Justification, from the numbers above:

- Train token lengths are extremely right-skewed: p50 = 3,838, p95 = 34,071, p99 = 120,768, max = 1,947,918 — the longest train record even exceeds the model's 131,072-token context window, so the long tail can never be trained whole on any hardware we have.
- At 4096 tokens, 46.75% of train records would be lost — nearly half the dataset. At 16384 the loss falls to 13.38%, but doubling sequence length from 8192 raises activation memory and step time sharply; 8192 is the largest power-of-two length treated as a safe default for 7B QLoRA on a single 16 GB T4 (batch 1, gradient checkpointing).
- 8192 keeps 582 of 800 train records (72.75%) intact — every kept example is a complete, valid blueprint, which matters more for learning the output format than the dropped long tail.
- Fallback (notebooks/00_environment.ipynb): if VRAM < 16 GB or compute capability < 7.0, drop to max_seq_length = 2048, per_device_batch = 1, and raise gradient accumulation instead.
