#!/usr/bin/env python3
"""Evaluate NL->Skill-Blueprint predictions with the 5-layer ladder.

L1 parse: raw output parses as JSON (code fences stripped)
L2 schema: validates against the Skill Blueprint schema
L3 fields: name exact-match, file-path F1, description token-F1 vs reference
L4 judge: LLM-as-judge rubric 1-5 x4 criteria (needs OPENROUTER_API_KEY)
L5 e2e: blueprint materializes to a temp dir; SKILL.md exists, frontmatter ok

Usage:
    python3 scripts/eval_harness.py --preds preds.jsonl --refs refs.jsonl [--judge]
    preds.jsonl: {"id": ..., "output": "<model raw text>"} per line
    refs.jsonl:  dataset records (instruction + expected_blueprint + metadata.id)

Report printed as JSON to stdout.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
WORD_RE = re.compile(r"[a-z0-9]+")

REQUIRED = {
    "blueprint_version": str,
    "name": str,
    "description": str,
    "purpose": str,
    "when_to_use": list,
    "when_not_to_use": list,
    "files": list,
}


# ---------------- L1 ----------------
def l1_parse(text):
    """Returns (ok, obj). Strips markdown code fences first."""
    t = FENCE_RE.sub("", text or "").strip()
    try:
        return True, json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return False, None


# ---------------- L2 ----------------
def l2_validate_schema(bp):
    """Returns list of error strings (empty = valid)."""
    errs = []
    if not isinstance(bp, dict):
        return ["blueprint is not an object"]
    for k, typ in REQUIRED.items():
        if k not in bp:
            errs.append(f"missing field: {k}")
        elif not isinstance(bp[k], typ):
            errs.append(f"field {k} has wrong type")
    name = bp.get("name", "")
    if isinstance(name, str) and (not NAME_RE.match(name) or "--" in name):
        errs.append(f"invalid name: {name!r}")
    if not bp.get("description"):
        errs.append("empty description")
    files = bp.get("files")
    if isinstance(files, list):
        if not files:
            errs.append("files is empty")
        paths = set()
        for i, f in enumerate(files):
            if not isinstance(f, dict):
                errs.append(f"files[{i}] not an object")
                continue
            for fk in ("path", "purpose", "content"):
                if fk not in f or not isinstance(f[fk], str) or not f[fk]:
                    errs.append(f"files[{i}].{fk} missing/empty")
            p = f.get("path", "")
            if isinstance(p, str):
                if ".." in p or p.startswith("/"):
                    errs.append(f"files[{i}].path unsafe: {p!r}")
                paths.add(p)
        if "SKILL.md" not in paths:
            errs.append("files missing SKILL.md")
    for lk in ("when_to_use", "when_not_to_use"):
        v = bp.get(lk)
        if isinstance(v, list) and not all(isinstance(x, str) for x in v):
            errs.append(f"{lk} must be list[str]")
    return errs


# ---------------- L3 ----------------
def _tok(s):
    return WORD_RE.findall((s or "").lower())


def _f1(pred_set, ref_set):
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    inter = len(pred_set & ref_set)
    p, r = inter / len(pred_set), inter / len(ref_set)
    return 2 * p * r / (p + r) if p + r else 0.0


def l3_field_scores(pred, ref):
    return {
        "name_em": 1.0 if pred.get("name") == ref.get("name") else 0.0,
        "file_path_f1": _f1(
            {f.get("path") for f in pred.get("files", []) if isinstance(f, dict)},
            {f.get("path") for f in ref.get("files", []) if isinstance(f, dict)}),
        "desc_token_f1": _f1(set(_tok(pred.get("description"))),
                             set(_tok(ref.get("description")))),
    }


# ---------------- L5 ----------------
def l5_materialize(bp):
    """Write files to a temp dir; SKILL.md must exist with name:/description: frontmatter."""
    errs = []
    try:
        with tempfile.TemporaryDirectory() as d:
            for f in bp.get("files", []):
                p = Path(d) / f["path"]
                if ".." in f["path"] or f["path"].startswith("/"):
                    return False, [f"unsafe path: {f['path']}"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f.get("content", ""))
            skill = Path(d) / "SKILL.md"
            if not skill.exists():
                return False, ["SKILL.md not materialized"]
            text = skill.read_text()
            m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
            if not m:
                return False, ["SKILL.md frontmatter missing"]
            fm = m.group(1)
            if not re.search(r"^name:\s*\S", fm, re.MULTILINE):
                errs.append("frontmatter missing name:")
            if not re.search(r"^description:\s*\S", fm, re.MULTILINE):
                errs.append("frontmatter missing description:")
    except (OSError, KeyError, TypeError) as e:
        return False, [f"materialize error: {e}"]
    return (len(errs) == 0), errs


# ---------------- L4 (LLM judge) ----------------
JUDGE_RUBRIC = """You are an expert evaluator of AI-generated Agent Skill blueprints.
Score the GENERATED blueprint against the USER REQUEST on four criteria, 1-5 each.
Anchor: 1 = completely wrong/missing, 3 = acceptable with gaps, 5 = excellent.

Criteria:
- format: follows the blueprint schema (name kebab-case, SKILL.md present, fields typed)
- faithfulness: captures the request's intent; no hallucinated files or capabilities
- completeness: all elements of the request are covered
- consistency: name, description, purpose and files agree with each other

USER REQUEST:
{instruction}

GENERATED BLUEPRINT (JSON):
{blueprint}

Reply with ONLY a JSON object: {{"format": n, "faithfulness": n, "completeness": n, "consistency": n}}"""


def run_judge(instruction, blueprint_json, model=None):
    """LLM-as-judge. Requires OPENROUTER_API_KEY. Returns dict of 4 scores."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — judge skipped")
    model = model or os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")
    prompt = JUDGE_RUBRIC.format(instruction=instruction, blueprint_json=blueprint_json)
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    text = body["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {"error": text[:200]}


# ---------------- driver ----------------
def evaluate(preds_path, refs_path, do_judge=False):
    refs = {}
    for line in open(refs_path):
        r = json.loads(line)
        refs[r["metadata"]["id"]] = r
    preds = [json.loads(l) for l in open(preds_path)]

    n = len(preds)
    l1_ok = l2_ok = l5_ok = 0
    name_ems, path_f1s, desc_f1s = [], [], []
    judge_scores, judge_skipped = [], False
    per_item = []

    for p in preds:
        rid = p["id"]
        ref = refs.get(rid)
        item = {"id": rid}
        if ref is None:
            item["error"] = "no reference"
            per_item.append(item)
            continue
        ok1, obj = l1_parse(p.get("output", ""))
        item["l1_parse"] = ok1
        l1_ok += ok1
        ref_bp = json.loads(ref["expected_blueprint"])
        if ok1:
            errs = l2_validate_schema(obj)
            item["l2_errors"] = errs
            l2v = not errs
            item["l2_valid"] = l2v
            l2_ok += l2v
            s3 = l3_field_scores(obj, ref_bp)
            item["l3"] = s3
            name_ems.append(s3["name_em"])
            path_f1s.append(s3["file_path_f1"])
            desc_f1s.append(s3["desc_token_f1"])
            ok5, e5 = l5_materialize(obj)
            item["l5_materialize"] = ok5
            item["l5_errors"] = e5
            l5_ok += ok5
            if do_judge:
                try:
                    js = run_judge(ref["instruction"], json.dumps(obj, ensure_ascii=False)[:6000])
                    item["l4_judge"] = js
                    if "error" not in js:
                        judge_scores.append(js)
                except RuntimeError:
                    judge_skipped = True
        per_item.append(item)

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    report = {
        "n": n,
        "l1_parse_rate": l1_ok / n if n else 0,
        "l2_schema_rate": l2_ok / n if n else 0,
        "l3_name_em": avg(name_ems),
        "l3_file_path_f1": avg(path_f1s),
        "l3_desc_token_f1": avg(desc_f1s),
        "l5_materialize_rate": l5_ok / n if n else 0,
    }
    if do_judge:
        if judge_scores:
            report["l4_judge_mean"] = {
                k: avg([s[k] for s in judge_scores if k in s])
                for k in ("format", "faithfulness", "completeness", "consistency")
            }
            report["l4_judge_n"] = len(judge_scores)
        if judge_skipped:
            report["l4_judge"] = "skipped: OPENROUTER_API_KEY not set"
    report["items"] = per_item
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = evaluate(args.preds, args.refs, do_judge=args.judge)
    items = rep.pop("items")
    print(json.dumps(rep, indent=1))
    if args.out:
        rep["items"] = items
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
