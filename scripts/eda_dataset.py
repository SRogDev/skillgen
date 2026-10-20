"""Exploratory data analysis of the frozen SkillGen splits (Phase 0).

Renders the FULL training text of every record — system prompt (PLAN.md §8,
verbatim) + user instruction + assistant blueprint — with the base model's
chat template, tokenizes it, and measures what that means for QLoRA training:
token/char length distributions, truncation rates, domain mix, blueprint
structure, and the longest records.

Outputs (in ``experiments/eda_v1/``):
    report.json  — every number, machine-readable
    report.md    — tables + the LENGTH DECISION the numbers justify

Rendering note: ``tokenizer.apply_chat_template(messages, tokenize=False)``
accepts the assistant turn for Qwen2.5-7B-Instruct (verified 2026-10-20), so
the rendered string is exactly the ChatML text the model would be trained on.
If a future template rejects an assistant turn, :func:`render_training_text`
falls back to building the equivalent ChatML string by hand:

    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n{instruction}<|im_end|>\n
    <|im_start|>assistant\n{blueprint}<|im_end|>\n

Run:  uv run python scripts/eda_dataset.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skillgen.data.eda import (  # noqa: E402
    blueprint_stats,
    domain_counts,
    length_stats,
    load_jsonl,
    truncation_report,
)

# PLAN.md §8 — use EXACTLY this system prompt.
SYSTEM_PROMPT = (
    "You are SkillGen, an expert designer of reusable Agent Skills. "
    "Given the user request, output ONLY a valid JSON Skill Blueprint. "
    "No explanations, no markdown fences."
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
THRESHOLDS = [1024, 2048, 4096, 8192, 16384]
OUT_DIR = ROOT / "experiments" / "eda_v1"

# ---------------------------------------------------------------------------
# LENGTH DECISION (the numbers measured by this script justify it — see the
# generated report.md; the counts below are computed from the data, not
# hardcoded):
#
# * max_seq_length = 8192. At 4096 we would lose almost half the train set;
#   at 16384 retention improves, but doubling the sequence length on a
#   16 GB T4 (7B QLoRA, batch 1 + gradient checkpointing) is not a safe
#   default — activation memory and step time grow with sequence length.
#   8192 is the largest power-of-two length that is routinely feasible on
#   that card while still keeping a clear majority of the train records.
# * Train filter: DROP train records whose rendered length is over the cap
#   instead of truncating them. A truncated blueprint is invalid JSON, so
#   training on truncated targets would teach the model to emit incomplete
#   blueprints. Records over the cap are excluded from TRAINING ONLY.
# * Val/test are NEVER excluded from evaluation — records over the cap are
#   only flagged (see `over_cap_flagged` in the report).
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 8192


def render_training_text(tokenizer, record: dict) -> str:
    """Render system+user+assistant as one training string via the chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["instruction"]},
        {"role": "assistant", "content": record["expected_blueprint"]},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        # Fallback: build the equivalent ChatML string by hand (see module
        # docstring). Qwen2.5's template does not reject assistant turns, so
        # this path exists only for future template changes.
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{record['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{record['expected_blueprint']}<|im_end|>\n"
        )


def load_splits() -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": []}
    for shard in sorted(glob.glob(str(ROOT / "data" / "splits" / "train_shard_*.jsonl"))):
        splits["train"].extend(load_jsonl(shard))
    splits["val"] = load_jsonl(ROOT / "data" / "splits" / "val.jsonl")
    splits["test"] = load_jsonl(ROOT / "data" / "splits" / "test.jsonl")
    return splits


def analyse_split(tokenizer, records: list[dict]) -> dict:
    texts = [render_training_text(tokenizer, r) for r in records]
    encodings = tokenizer(texts, add_special_tokens=False)["input_ids"]
    token_lengths = [len(ids) for ids in encodings]
    char_lengths = [len(t) for t in texts]

    ranked = sorted(
        zip(records, token_lengths, char_lengths),
        key=lambda item: item[1],
        reverse=True,
    )
    top10 = [
        {
            "id": record.get("metadata", {}).get("id"),
            "domain": record.get("metadata", {}).get("domain"),
            "token_length": token_length,
            "char_length": char_length,
        }
        for record, token_length, char_length in ranked[:10]
    ]
    return {
        "n_records": len(records),
        "token_stats": length_stats(token_lengths),
        "token_truncation": {
            str(k): v for k, v in truncation_report(token_lengths, THRESHOLDS).items()
        },
        "char_stats": length_stats(char_lengths),
        "domain_counts": domain_counts(records),
        "blueprint_stats": blueprint_stats(records),
        "token_lengths": token_lengths,
        "char_lengths": char_lengths,
        "top10_longest": top10,
    }


def fmt(stats: dict, key: str) -> str:
    value = stats[key]
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def build_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# SkillGen dataset EDA — v1")
    lines.append("")
    lines.append(
        f"Tokenizer/model: `{report['model_id']}` (tokenizer only, CPU). "
        "Lengths are for the FULL rendered training text "
        "(system prompt from PLAN.md §8 + instruction + blueprint) "
        "via the model's chat template."
    )
    lines.append("")

    lines.append("## Token-length stats (full rendered text)")
    lines.append("")
    lines.append("| Split | n | mean | p50 | p90 | p95 | p99 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, split in report["splits"].items():
        s = split["token_stats"]
        lines.append(
            f"| {name} | {s['count']} | {fmt(s, 'mean')} | {fmt(s, 'p50')} | "
            f"{fmt(s, 'p90')} | {fmt(s, 'p95')} | {fmt(s, 'p99')} | {fmt(s, 'max')} |"
        )
    lines.append("")

    lines.append("## Char-length stats (full rendered text)")
    lines.append("")
    lines.append("| Split | n | mean | p50 | p90 | p95 | p99 | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, split in report["splits"].items():
        s = split["char_stats"]
        lines.append(
            f"| {name} | {s['count']} | {fmt(s, 'mean')} | {fmt(s, 'p50')} | "
            f"{fmt(s, 'p90')} | {fmt(s, 'p95')} | {fmt(s, 'p99')} | {fmt(s, 'max')} |"
        )
    lines.append("")

    lines.append("## Truncation — records OVER each token threshold")
    lines.append("")
    header = "| Split | " + " | ".join(f">{t}" for t in THRESHOLDS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(THRESHOLDS) + 1))
    for name, split in report["splits"].items():
        cells = []
        for t in THRESHOLDS:
            entry = split["token_truncation"][str(t)]
            cells.append(f"{entry['n_over']} ({entry['pct_over']:.2f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Domain distribution")
    lines.append("")
    domains = sorted(
        {d for split in report["splits"].values() for d in split["domain_counts"]}
    )
    lines.append("| Domain | " + " | ".join(report["splits"]) + " |")
    lines.append("|" + "---|" * (len(report["splits"]) + 1))
    for domain in domains:
        row = [str(split["domain_counts"].get(domain, 0)) for split in report["splits"].values()]
        lines.append(f"| {domain} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Blueprint structure")
    lines.append("")
    lines.append("| Split | records | JSON parse failures | missing SKILL.md | files-per-blueprint distribution |")
    lines.append("|---|---|---|---|---|")
    for name, split in report["splits"].items():
        b = split["blueprint_stats"]
        dist = ", ".join(
            f"{k} files × {v}" for k, v in sorted(b["files_per_blueprint"].items(), key=lambda kv: int(kv[0]))
        )
        lines.append(
            f"| {name} | {b['total_records']} | {b['parse_failures']} | "
            f"{b['missing_skill_md']} | {dist} |"
        )
    lines.append("")

    lines.append("## Top-10 longest records overall (tokens)")
    lines.append("")
    lines.append("| # | id | split | domain | tokens | chars |")
    lines.append("|---|---|---|---|---|---|")
    for i, item in enumerate(report["top10_longest_overall"], start=1):
        lines.append(
            f"| {i} | {item['id']} | {item['split']} | {item['domain']} | "
            f"{item['token_length']:,} | {item['char_length']:,} |"
        )
    lines.append("")

    d = report["length_decision"]
    train = report["splits"]["train"]["token_stats"]
    lines.append("## LENGTH DECISION")
    lines.append("")
    lines.append(f"- **max_seq_length = {d['max_seq_length']} tokens.**")
    lines.append(
        f"- **Train filter rule:** {d['train_filter_rule']}"
    )
    lines.append(
        f"- **Excluded from training:** exactly **{d['train_records_excluded']} of "
        f"{train['count']} train records ({d['train_pct_excluded']:.2f}%)**; "
        f"{d['train_records_kept']} records are kept."
    )
    lines.append(
        "- **Val/test are NEVER excluded from evaluation.** Records over the cap "
        f"are only flagged: val {d['val_over_cap_flagged']}/"
        f"{report['splits']['val']['n_records']}, test "
        f"{d['test_over_cap_flagged']}/{report['splits']['test']['n_records']}."
    )
    lines.append("")
    lines.append("Justification, from the numbers above:")
    lines.append("")
    for reason in d["rationale"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    splits = load_splits()

    analysed = {name: analyse_split(tokenizer, records) for name, records in splits.items()}

    overall_top10 = sorted(
        (
            {**item, "split": name}
            for name, split in analysed.items()
            for item in split["top10_longest"]
        ),
        key=lambda item: item["token_length"],
        reverse=True,
    )[:10]

    train_trunc = analysed["train"]["token_truncation"][str(MAX_SEQ_LENGTH)]
    excluded = train_trunc["n_over"]
    train_n = analysed["train"]["n_records"]
    train_stats = analysed["train"]["token_stats"]

    decision = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_filter_rule": (
            f"drop train records whose full rendered token length is > {MAX_SEQ_LENGTH} "
            "(filter, never truncate: a truncated blueprint is invalid JSON and would "
            "teach the model to emit incomplete blueprints)"
        ),
        "train_records_excluded": excluded,
        "train_pct_excluded": train_trunc["pct_over"],
        "train_records_kept": train_n - excluded,
        "val_over_cap_flagged": analysed["val"]["token_truncation"][str(MAX_SEQ_LENGTH)]["n_over"],
        "test_over_cap_flagged": analysed["test"]["token_truncation"][str(MAX_SEQ_LENGTH)]["n_over"],
        "rationale": [
            f"Train token lengths are extremely right-skewed: p50 = {train_stats['p50']:,.0f}, "
            f"p95 = {train_stats['p95']:,.0f}, p99 = {train_stats['p99']:,.0f}, "
            f"max = {train_stats['max']:,} — the longest train record even exceeds the "
            "model's 131,072-token context window, so the long tail can never be "
            "trained whole on any hardware we have.",
            f"At 4096 tokens, {analysed['train']['token_truncation']['4096']['pct_over']:.2f}% "
            "of train records would be lost — nearly half the dataset. At 16384 the "
            f"loss falls to {analysed['train']['token_truncation']['16384']['pct_over']:.2f}%, "
            "but doubling sequence length from 8192 raises activation memory and step "
            "time sharply; 8192 is the largest power-of-two length treated as a safe "
            "default for 7B QLoRA on a single 16 GB T4 (batch 1, gradient checkpointing).",
            f"8192 keeps {train_n - excluded} of {train_n} train records "
            f"({100.0 - train_trunc['pct_over']:.2f}%) intact — every kept example is a "
            "complete, valid blueprint, which matters more for learning the output "
            "format than the dropped long tail.",
            "Fallback (notebooks/00_environment.ipynb): if VRAM < 16 GB or compute "
            "capability < 7.0, drop to max_seq_length = 2048, per_device_batch = 1, "
            "and raise gradient accumulation instead.",
        ],
    }

    report = {
        "model_id": MODEL_ID,
        "system_prompt": SYSTEM_PROMPT,
        "thresholds": THRESHOLDS,
        "rendering": "tokenizer.apply_chat_template([system, user, assistant], tokenize=False)",
        "splits": analysed,
        "top10_longest_overall": overall_top10,
        "length_decision": decision,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "report.md").write_text(build_markdown(report), encoding="utf-8")

    print(f"Wrote {OUT_DIR / 'report.json'}")
    print(f"Wrote {OUT_DIR / 'report.md'}")
    for name, split in analysed.items():
        s = split["token_stats"]
        print(
            f"{name}: n={s['count']} token mean={s['mean']:.1f} p50={s['p50']:.1f} "
            f"p95={s['p95']:.1f} p99={s['p99']:.1f} max={s['max']} "
            f"over4096={split['token_truncation']['4096']['pct_over']:.2f}%"
        )
    print(
        f"DECISION: max_seq_length={MAX_SEQ_LENGTH}, excluded train "
        f"{excluded}/{train_n} ({train_trunc['pct_over']:.2f}%)"
    )


if __name__ == "__main__":
    main()
