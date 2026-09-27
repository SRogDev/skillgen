#!/usr/bin/env python3
"""Build the pilot dataset JSONL: instruction -> expected_blueprint.

Usage:
    python scripts/build_pilot.py

Reads:
    data/pilot_selection.jsonl      (id, domain, repo, relpath)
    data/instructions_pilot.json    (id -> synthetic instruction)
    data/blueprints/<id>.json       (blueprint + provenance)
Writes:
    data/pilot/dataset_pilot.jsonl  (instruction, expected_blueprint, metadata)
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def main():
    sel = [json.loads(l) for l in open(BASE / "data/pilot_selection.jsonl")]
    instructions = json.load(open(BASE / "data/instructions_pilot.json"))
    out_dir = BASE / "data" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    errors, rows = [], []
    seen_instructions = set()
    for r in sel:
        sid = r["id"]
        bp_file = BASE / "data" / "blueprints" / f"{sid}.json"
        if not bp_file.exists():
            errors.append(f"{sid}: missing blueprint file")
            continue
        if sid not in instructions:
            errors.append(f"{sid}: missing instruction")
            continue
        instruction = instructions[sid].strip()
        if len(instruction) < 40:
            errors.append(f"{sid}: instruction too short ({len(instruction)})")
        if instruction in seen_instructions:
            errors.append(f"{sid}: duplicate instruction")
        seen_instructions.add(instruction)

        d = json.load(open(bp_file))
        bp, prov = d["blueprint"], d["provenance"]
        if prov.get("validation_errors"):
            errors.append(f"{sid}: blueprint has validation errors")
        if not NAME_RE.match(bp["name"]) or "--" in bp["name"]:
            errors.append(f"{sid}: invalid blueprint name {bp['name']!r}")
        desc = bp["description"]
        if desc and desc in instruction:
            errors.append(f"{sid}: instruction contains description verbatim")
        if not bp["files"]:
            errors.append(f"{sid}: blueprint has no files")

        rows.append({
            "instruction": instruction,
            "expected_blueprint": json.dumps(bp, ensure_ascii=False),
            "metadata": {
                "id": sid,
                "domain": r["domain"],
                "repo": r["repo"],
                "provenance": prov,
            },
        })

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    out = out_dir / "dataset_pilot.jsonl"
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples -> {out}")


if __name__ == "__main__":
    main()
