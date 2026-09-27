#!/usr/bin/env python3
"""Freeze train/val/test splits with leakage control.

- Loads data/dataset_skillgen_1000_shard_*.jsonl
- Clusters near-duplicate instructions (token Jaccard > 0.8); each cluster
  goes to exactly ONE split. Exact-duplicate instructions keep a single copy.
- Stratified by domain, deterministic via --seed (default 7).
- Writes data/splits/{train,val,test}.jsonl + manifest.json (sha256 hashes).

Usage:
    python3 scripts/freeze_splits.py [--seed 7] [--sizes 800 100 100]
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "splits"
NEAR_DUP_JACCARD = 0.8
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_dataset():
    recs = []
    for p in sorted(DATA.glob("dataset_skillgen_1000_shard_*.jsonl")):
        recs += [json.loads(l) for l in open(p)]
    return recs


def _tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def cluster_near_duplicates(recs):
    """Union-find over instruction token-Jaccard > threshold. Returns list of id-lists."""
    toks = [_tokens(r["instruction"]) for r in recs]
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(recs)):
        ti = toks[i]
        if not ti:
            continue
        for j in range(i + 1, len(recs)):
            tj = toks[j]
            if not tj:
                continue
            inter = len(ti & tj)
            if inter / max(len(ti), len(tj)) > NEAR_DUP_JACCARD:
                union(i, j)
    groups = {}
    for i, r in enumerate(recs):
        groups.setdefault(find(i), []).append(r["metadata"]["id"])
    return list(groups.values())


def make_splits(recs, seed, out_dir, sizes=(800, 100, 100), train_shard=100):
    from collections import defaultdict
    import itertools

    by_id = {r["metadata"]["id"]: r for r in recs}
    clusters = cluster_near_duplicates(recs)
    # exact-duplicate instructions: keep a single copy per cluster
    kept_clusters = []
    for cl in clusters:
        seen, uniq = set(), []
        for sid in cl:
            t = by_id[sid]["instruction"]
            if t not in seen:
                seen.add(t)
                uniq.append(sid)
        kept_clusters.append(uniq)

    total_cap = sum(sizes)
    n_unique = sum(len(c) for c in kept_clusters)
    assert total_cap <= n_unique, f"sizes {sizes} exceed {n_unique} unique records"

    names = ["train", "val", "test"]
    rng = random.Random(seed)

    # group clusters by majority domain, shuffle deterministically
    dom_clusters = defaultdict(list)
    for cl in kept_clusters:
        d = Counter(by_id[s]["metadata"]["domain"] for s in cl).most_common(1)[0][0]
        dom_clusters[d].append(cl)
    for d in dom_clusters:
        rng.shuffle(dom_clusters[d])

    # dealing pattern proportional to split sizes, e.g. 80/10/10 -> 8,1,1
    unit = min(sizes)
    pattern = [n for n, s in zip(names, sizes) for _ in range(s // unit)]
    deal = itertools.cycle(pattern)

    splits = {n: [] for n in names}
    remaining = dict(zip(names, sizes))
    for d in sorted(dom_clusters):
        for cl in dom_clusters[d]:
            placed = False
            for _ in range(len(pattern) * 4):
                n = next(deal)
                if remaining[n] >= len(cl):
                    splits[n].extend(cl)
                    remaining[n] -= len(cl)
                    placed = True
                    break
            if not placed:  # cluster bigger than any split's unit share: roomiest split
                n = max(names, key=lambda n: remaining[n])
                splits[n].extend(cl)
                remaining[n] -= len(cl)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "date": str(date.today()),
                "sizes": dict(zip(names, sizes)), "counts": {}, "sha256": {},
                "files": {},
                "near_dup_threshold": NEAR_DUP_JACCARD,
                "clusters": len(kept_clusters)}
    result = []
    for n in names:
        recs_out = [by_id[s] for s in splits[n]]
        if n == "train" and len(recs_out) > train_shard:
            # shard train so no single file nears the GitHub API blob limit;
            # balance by byte size (deal largest-first round-robin), deterministic
            n_shards = (len(recs_out) + train_shard - 1) // train_shard
            blobs = [(len(json.dumps(r, ensure_ascii=False)), r) for r in recs_out]
            blobs.sort(key=lambda t: -t[0])
            buckets = [[] for _ in range(n_shards)]
            for i, (_, r) in enumerate(blobs):
                buckets[i % n_shards].append(r)
            fnames = []
            for i, bucket in enumerate(buckets):
                fn = f"train_shard_{i:02d}.jsonl"
                p = out_dir / fn
                p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in bucket))
                manifest["sha256"][fn] = hashlib.sha256(p.read_bytes()).hexdigest()
                fnames.append(fn)
            manifest["files"][n] = fnames
        else:
            fn = f"{n}.jsonl"
            p = out_dir / fn
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs_out))
            manifest["sha256"][fn] = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest["files"][n] = [fn]
        manifest["counts"][n] = len(recs_out)
        result.append(recs_out)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return result


def verify_manifest(out_dir):
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "manifest.json").read_text())
    for fn, h in man["sha256"].items():
        if hashlib.sha256((out_dir / fn).read_bytes()).hexdigest() != h:
            return False
    # counts must match actual records
    for n, fns in man["files"].items():
        total = sum(1 for fn in fns for _ in open(out_dir / fn))
        if total != man["counts"][n]:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sizes", type=int, nargs=3, default=[800, 100, 100])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    recs = load_dataset()
    print(f"loaded {len(recs)} records")
    train, val, test = make_splits(recs, args.seed, Path(args.out), tuple(args.sizes))
    print(f"train {len(train)} / val {len(val)} / test {len(test)} -> {args.out}")
    print("manifest verified:", verify_manifest(Path(args.out)))
    print("WARNING: test split is now FROZEN. Do not train on it, do not tune against it.")


if __name__ == "__main__":
    main()
