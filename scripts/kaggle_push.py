#!/usr/bin/env python3
"""Push one training config to Kaggle as a private T4 script kernel.

Usage: python scripts/kaggle_push.py --user USER --config configs/smoke.yaml --ref <git sha|branch> [--dry-run]
Then: kaggle kernels status USER/<slug>   and   kaggle kernels output USER/<slug> -p out/
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def build(user: str, config: str, ref: str, out: Path) -> str:
    exp = yaml.safe_load((ROOT / config).read_text())["experiment_id"]
    slug = exp.replace("_", "-")
    code = (ROOT / "kaggle" / "kernel_template.py").read_text()
    (out / "run.py").write_text(code.replace("__REF__", ref).replace("__CONFIG__", config))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=1))
    return f"{user}/{slug}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kaggle", default="kaggle", help="kaggle CLI executable")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = Path(tempfile.mkdtemp())
    kid = build(a.user, a.config, a.ref, d)
    print("kernel", kid, "folder", d)
    if not a.dry_run:
        subprocess.run([a.kaggle, "kernels", "push", "-p", str(d)], check=True)
