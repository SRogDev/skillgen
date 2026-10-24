#!/usr/bin/env python3
"""Validate the §31 expert-set curation scaffold (data/expert/).

Stdlib only. Usage:
    python3 scripts/validate_expert_set.py

Validates every E###.json case file against the curation schema, reports
how many of the 20 cases exist and which IDs are missing. Template files
(`_template*.json`) are ignored. Exit 0 when all present files are valid
(even if fewer than 20 exist — curation is Roger's in-progress work);
exit 1 on any schema error.
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXPERT_DIR = BASE / "data" / "expert"

REQUIRED_FIELDS = {
    "id": str,
    "instruction": str,
    "source": str,
    "source_skill": (str, type(None)),
    "difficulty": str,
    "trap_type": (str, type(None)),
    "curation_notes": str,
    "author": str,
    "created": str,
}

TRAP_TYPES = {
    "nonexistent-field",
    "contradictory",
    "invalid-name",
    "sandbox-escape",
    "scope-creep",
    "ambiguity",
    "language-mix",
}

ID_RE = re.compile(r"^E(\d{3})$")
EXPECTED_IDS = {f"E{i:03d}" for i in range(1, 21)}


def validate_case(path: Path) -> list[str]:
    """Return a list of schema errors (empty = valid)."""
    errors = []
    try:
        case = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path.name}: invalid JSON ({e})"]
    if not isinstance(case, dict):
        return [f"{path.name}: top level must be an object"]
    for field, types in REQUIRED_FIELDS.items():
        if field not in case:
            errors.append(f"{path.name}: missing field '{field}'")
        elif not isinstance(case[field], types):
            errors.append(f"{path.name}: field '{field}' has wrong type")
    extra = set(case) - set(REQUIRED_FIELDS)
    if extra:
        errors.append(f"{path.name}: unexpected fields {sorted(extra)}")
    m = ID_RE.match(str(case.get("id", "")))
    if not m:
        errors.append(f"{path.name}: id must match E### (e.g. E007)")
    elif case.get("id") != path.stem:
        errors.append(f"{path.name}: id '{case.get('id')}' != filename")
    trap = case.get("trap_type")
    if trap is not None and trap not in TRAP_TYPES:
        errors.append(f"{path.name}: unknown trap_type '{trap}'")
    for field in ("instruction", "difficulty", "curation_notes"):
        if isinstance(case.get(field), str) and not case[field].strip():
            errors.append(f"{path.name}: field '{field}' is empty")
    if case.get("author") != "roger":
        errors.append(f"{path.name}: author must be 'roger' (curated by him)")
    return errors


def main() -> int:
    case_files = sorted(
        p for p in EXPERT_DIR.glob("E*.json") if not p.name.startswith("_template")
    )
    errors: list[str] = []
    for path in case_files:
        errors.extend(validate_case(path))
    present = {p.stem for p in case_files}
    missing = sorted(EXPECTED_IDS - present)

    print(f"expert set: {len(present)}/20 cases present")
    if missing:
        print(f"missing: {', '.join(missing)}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print(f"{len(errors)} schema error(s)")
        return 1
    print("all present cases valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
