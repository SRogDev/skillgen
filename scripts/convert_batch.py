#!/usr/bin/env python3
"""Convert a selection list to blueprints in parallel.

Usage:
    python scripts/convert_batch.py --selection data/selection_full.jsonl [--workers 8]
Writes data/blueprints/<id>.json. Applies spec name normalization.
"""
import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def norm(d):
    return re.sub(r"-+", "-", d.lower().replace("_", "-").replace(" ", "-")).strip("-")


def convert_one(r):
    repo_dir = BASE / "data" / "raw" / "_repos" / r["repo"]
    out = BASE / "data" / "blueprints" / (r["id"] + ".json")
    if out.exists():
        d = json.load(open(out))
        return ("cached", d["provenance"]["license"])
    owner_repo = r["repo"].replace("_", "/", 1)
    pr = subprocess.run(
        ["python3", str(BASE / "scripts" / "convert_skill.py"),
         str(repo_dir / r["relpath"]),
         "--repo-root", str(repo_dir), "--repo-slug", owner_repo, "--out", str(out)],
        capture_output=True, text=True)
    if pr.returncode != 0:
        return ("error", pr.stderr[-300:])
    d = json.load(open(out))
    bp, prov = d["blueprint"], d["provenance"]
    if prov["validation_errors"]:
        new_name = norm(r["id"].split("__")[-1])
        if not (NAME_RE.match(new_name) and "--" not in new_name):
            return ("badname", new_name)
        prov["original_name"] = bp["name"]
        prov["name_normalized"] = True
        bp["name"] = new_name
        for fe in bp["files"]:
            if fe["path"] == "SKILL.md":
                fe["content"] = re.sub(r"(?m)^name:\s*.*$", f"name: {new_name}",
                                      fe["content"], count=1)
        prov["validation_errors"] = []
        json.dump(d, open(out, "w"), ensure_ascii=False, indent=2)
    return ("ok", prov["license"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.selection)]
    stats, lics, problems = Counter(), Counter(), []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, r): r["id"] for r in rows}
        for i, f in enumerate(as_completed(futs), 1):
            status, info = f.result()
            stats[status] += 1
            if status in ("ok", "cached"):
                lics[info] += 1
            else:
                problems.append((futs[f], status, info))
            if i % 100 == 0:
                print(f"  {i}/{len(rows)} ...")
    print("status:", dict(stats))
    print("licenses:", dict(lics))
    for sid, st, info in problems[:20]:
        print("PROBLEM:", sid, st, info)


if __name__ == "__main__":
    main()
