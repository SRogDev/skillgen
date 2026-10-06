#!/usr/bin/env python3
"""Measure token lengths per split with the real tokenizer (PLAN §23), no GPU needed.

Usage: uv run --extra train python scripts/length_report.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skillgen.data import load_split, to_messages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limits", type=int, nargs="+", default=[2048, 4096, 6144, 8192])
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    for split in ("train", "val"):
        lens = sorted(len(tok.apply_chat_template(to_messages(r), tokenize=True)) for r in load_split(split))
        n = len(lens)
        print(json.dumps({"split": split, "n": n, "p50": lens[n // 2], "p90": lens[int(n * .9)],
                          "p95": lens[int(n * .95)], "max": lens[-1],
                          "fit": {L: sum(x <= L for x in lens) for L in a.limits}}))


if __name__ == "__main__":
    main()
