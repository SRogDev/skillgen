#!/usr/bin/env python3
"""Train SkillGen with QLoRA. Usage: uv run python scripts/train.py --config configs/qlora.yaml"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skillgen.train import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora.yaml")
    print("adapter saved to", train(ap.parse_args().config))
