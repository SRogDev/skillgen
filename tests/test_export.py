"""Phase 5 export-plan tests (PLAN.md §44).

The export plan is pure data: these tests never execute a merge, never
download a model, and never need a GPU. The hard §44 rule under test:
adapter -> merge in 16-bit (``merged_16bit``) -> GGUF ``q4_k_m`` from
the MERGED model. Never merge directly to 4-bit.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    path = REPO_ROOT / "scripts" / "export_model.py"
    spec = importlib.util.spec_from_file_location("export_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_model = _load_export_module()


def _plan(**overrides):
    kwargs = {
        "adapter": "adapters/e1",
        "base": "Qwen/Qwen2.5-7B-Instruct",
        "out_dir": "exports/skillgen-v1",
    }
    kwargs.update(overrides)
    return export_model.build_export_plan(**kwargs)


def _step_names(plan):
    return [step["name"] for step in plan]


def test_merged_16bit_step_before_gguf_step():
    names = _step_names(_plan())
    assert "merged_16bit" in names
    assert "gguf" in names
    assert names.index("merged_16bit") < names.index("gguf")


def test_quantization_value_propagated():
    plan = _plan(quantization="q4_k_m")
    gguf = next(s for s in plan if s["name"] == "gguf")
    assert "q4_k_m" in str(gguf)
    plan2 = _plan(quantization="q8_0")
    gguf2 = next(s for s in plan2 if s["name"] == "gguf")
    assert "q8_0" in str(gguf2)


def test_gguf_step_sources_the_merged_model_not_the_adapter():
    plan = _plan()
    gguf = next(s for s in plan if s["name"] == "gguf")
    # GGUF conversion must start from the merged 16-bit model.
    assert "merged_16bit" in str(gguf)


def test_no_direct_4bit_merge_step():
    plan = _plan()
    merge_steps = [s for s in plan if "merge" in s["name"]]
    for step in merge_steps:
        text = str(step).lower()
        assert "4-bit" not in text and "4bit" not in text and "q4" not in text


def test_plan_mentions_ollama_instructions():
    names = _step_names(_plan())
    assert any("ollama" in n for n in names)


def test_dry_run_cli_exits_0_and_prints_both_steps():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_model.py"),
            "--adapter",
            "adapters/e1",
            "--base",
            "Qwen/Qwen2.5-7B-Instruct",
            "--out-dir",
            "exports/skillgen-v1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "merged_16bit" in proc.stdout
    assert "gguf" in proc.stdout.lower()
    assert "q4_k_m" in proc.stdout
