#!/usr/bin/env python3
"""Convert a real Agent Skill directory -> SkillGen blueprint JSON.

Usage:
    python scripts/convert_skill.py <skill_dir> [--repo-root R] [--out blueprint.json]

Reads SKILL.md (YAML frontmatter + Markdown body), validates against the
Agent Skills spec (agentskills.io), walks supporting files, and emits a
blueprint matching SkillGen's schema (PLAN.md §5).
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")

NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
RESERVED = {"claude", "anthropic"}
MAX_FILE_CHARS = 200_000

WHEN_USE_PATTERNS = [
    r"when to use", r"use cases?", r"triggers?", r"activat",
]
WHEN_NOT_PATTERNS = [
    r"when not to use", r"don'?t use", r"limitations?", r"out of scope",
    r"not suitable", r"avoid when",
]


def split_frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def validate_skill(name, description, dir_name):
    errors = []
    if not name or not NAME_RE.match(name) or "--" in name:
        errors.append(f"invalid name: {name!r}")
    if name in RESERVED:
        errors.append(f"reserved name: {name!r}")
    if name != dir_name:
        errors.append(f"name {name!r} != directory {dir_name!r}")
    if not description or not (1 <= len(description) <= 1024):
        errors.append("description must be 1-1024 chars")
    return errors


def extract_section_items(body, patterns):
    """Return bullet items under the first heading matching any pattern."""
    lines = body.splitlines()
    in_section = False
    items = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            if not in_section and any(re.search(p, heading) for p in patterns):
                in_section = True
                continue
            elif in_section:
                break
        if in_section:
            m = re.match(r"^[-*]\s+(.*)", stripped) or re.match(r"^\d+[.)]\s+(.*)", stripped)
            if m:
                items.append(m.group(1).strip())
            elif stripped:
                items.append(stripped)
    return [i for i in items if i]


def first_paragraph(body):
    for chunk in re.split(r"\n\s*\n", body):
        text = chunk.strip()
        if text and not text.startswith("#") and not text.startswith("---"):
            text = re.sub(r"\s+", " ", text)
            return text[:500]
    return ""


def file_purpose(rel):
    parts = rel.parts
    if rel.name == "SKILL.md":
        return "Main skill definition: YAML frontmatter plus Markdown instructions."
    if parts[0] == "scripts":
        return "Executable helper script used by the skill."
    if parts[0] == "references":
        return "Reference documentation loaded on demand."
    if parts[0] == "assets":
        return "Static asset or template bundled with the skill."
    return "Supporting file bundled with the skill."


def is_binary(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"\0" in chunk
    except OSError:
        return True


def guess_license(skill_dir, repo_root):
    for base in (skill_dir, repo_root):
        if not base:
            continue
        for cand in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING"):
            if (base / cand).exists():
                text = (base / cand).read_text(errors="replace")[:400].lower()
                if "apache license" in text and "version 2.0" in text:
                    return "apache-2.0"
                for lic in ("mit", "bsd", "cc0", "unlicense"):
                    if lic in text:
                        return lic
                return "present (unparsed)"
    return "unknown"


def convert(skill_dir, repo_root=None, repo_slug=""):
    skill_dir = Path(skill_dir)
    dir_name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise ValueError(f"no SKILL.md in {skill_dir}")

    fm, body = split_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
    name = str(fm.get("name", "")).strip()
    description = str(fm.get("description", "")).strip()
    errors = validate_skill(name, description, dir_name)

    when_to_use = extract_section_items(body, WHEN_USE_PATTERNS) or [description]
    when_not_to_use = extract_section_items(body, WHEN_NOT_PATTERNS)

    files = []
    for path in sorted(skill_dir.rglob("*")):
        if ".git" in path.parts or path.is_dir():
            continue
        rel = path.relative_to(skill_dir)
        entry = {"path": rel.as_posix(), "purpose": file_purpose(rel), "content": ""}
        if is_binary(path):
            entry["purpose"] += " (binary — content omitted)"
            entry["content"] = ""
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS] + "\n…[truncated]"
            entry["content"] = text
        files.append(entry)

    blueprint = {
        "blueprint_version": "1.0",
        "name": name,
        "description": description,
        "purpose": first_paragraph(body) or description,
        "when_to_use": when_to_use,
        "when_not_to_use": when_not_to_use,
        "files": files,
    }
    provenance = {
        "source": "skills.sh",
        "repo": repo_slug,
        "skill_url": f"https://skills.sh/{repo_slug}/{name}" if repo_slug else "",
        "license": guess_license(skill_dir, Path(repo_root) if repo_root else None),
        "validation_errors": errors,
    }
    return blueprint, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill_dir")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--repo-slug", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blueprint, provenance = convert(args.skill_dir, args.repo_root, args.repo_slug)
    out = {"blueprint": blueprint, "provenance": provenance}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    if provenance["validation_errors"]:
        print(f"WARNINGS: {provenance['validation_errors']}", file=sys.stderr)


if __name__ == "__main__":
    main()
