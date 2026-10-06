#!/usr/bin/env python3
"""Experiment configs stay consistent with the E1 recipe. Run: uvx pytest tests/test_configs.py"""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parent.parent / "configs"
load = lambda n: yaml.safe_load((CFG / n).read_text())


def test_all_configs_share_model_and_8192():
    base = load("qlora.yaml")
    for name in ("qlora.yaml", "smoke.yaml", "overfit.yaml"):
        c = load(name)
        assert c["model"] == base["model"], name
        assert c["lora"] == base["lora"], name
        assert c["training"]["max_seq_length"] == 8192, name
        assert c["experiment_id"] != base["experiment_id"] or name == "qlora.yaml"


def test_smoke_is_short():
    t = load("smoke.yaml")["training"]
    assert 0 < t["max_steps"] <= 20
    assert t["val_limit"] and t["val_limit"] <= 10


def test_overfit_uses_tiny_train_set_without_early_stopping():
    t = load("overfit.yaml")["training"]
    assert t["train_limit"] == 20
    assert t["early_stopping_patience"] is None


def test_kaggle_push_builds_private_t4_kernel(tmp_path):
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("kp", CFG.parent / "scripts" / "kaggle_push.py")
    kp = importlib.util.module_from_spec(spec); spec.loader.exec_module(kp)
    kid = kp.build("someone", "configs/smoke.yaml", "abc123", tmp_path)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert kid == "someone/skillgen-qlora-v1-smoke" == meta["id"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["is_private"] and meta["enable_gpu"]
    code = (tmp_path / "run.py").read_text()
    assert 'REF = "abc123"' in code and 'CONFIG = "configs/smoke.yaml"' in code
