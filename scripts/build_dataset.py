#!/usr/bin/env python3
"""Build the final dataset JSONL from selection(s) + instructions + blueprints.

Usage:
    python scripts/build_dataset.py --selections data/pilot_selection.jsonl data/selection_full.jsonl \\
        --instructions data/instructions_pilot.json data/instructions_full.json \\
        --out data/dataset_skillgen_1000.jsonl
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", nargs="+", required=True)
    ap.add_argument("--instructions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sel = []
    for s in args.selections:
        sel.extend(json.loads(l) for l in open(s))
    instructions = {}
    for f in args.instructions:
        instructions.update(json.load(open(f)))

    errors, rows, seen = [], [], set()
    for r in sel:
        sid = r["id"]
        bp_file = BASE / "data" / "blueprints" / f"{sid}.json"
        if not bp_file.exists():
            errors.append(f"{sid}: missing blueprint"); continue
        if sid not in instructions:
            errors.append(f"{sid}: missing instruction"); continue
        instruction = instructions[sid].strip()
        if len(instruction) < 40:
            errors.append(f"{sid}: instruction too short")
        if instruction in seen:
            errors.append(f"{sid}: duplicate instruction")
        seen.add(instruction)
        d = json.load(open(bp_file))
        bp, prov = d["blueprint"], d["provenance"]
        if prov.get("validation_errors"):
            errors.append(f"{sid}: blueprint validation errors")
        if not NAME_RE.match(bp["name"]) or "--" in bp["name"]:
            errors.append(f"{sid}: invalid name {bp['name']!r}")
        if not bp["files"]:
            errors.append(f"{sid}: no files")
        desc = bp["description"]
        if desc and len(desc) > 20 and desc in instruction:
            errors.append(f"{sid}: instruction copies description verbatim")
        rows.append({
            "instruction": instruction,
            "expected_blueprint": json.dumps(bp, ensure_ascii=False),
            "metadata": {"id": sid, "domain": r["domain"], "repo": r["repo"],
                         "provenance": prov},
        })

    # near-duplicate instruction check
    instrs = [r["instruction"] for r in rows]
    for i in range(len(instrs)):
        for j in range(i + 1, len(instrs)):
            if difflib.SequenceMatcher(None, instrs[i], instrs[j]).ratio() > 0.8:
                errors.append(f"near-dup: {rows[i]['metadata']['id']} <-> {rows[j]['metadata']['id']}")

    if errors:
        print(f"{len(errors)} VALIDATION ERRORS (showing 30):")
        for e in errors[:30]:
            print(" -", e)
        sys.exit(1)

    out = Path(args.out)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples -> {out}")
    print("domains:", dict(sorted(Counter(r["metadata"]["domain"] for r in rows).items())))


if __name__ == "__main__":
    main()
