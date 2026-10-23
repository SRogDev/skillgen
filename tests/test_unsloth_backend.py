"""Unsloth backend tests (Roger's 2026-10-23 direction: Unsloth is the
production training engine).

Unsloth is NOT installed on this CPU-only VM and cannot run here (it
requires CUDA), so these tests inject a FAKE ``unsloth`` module into
``sys.modules`` — a stub ``FastLanguageModel`` that records its calls —
and assert that :mod:`skillgen.training.unsloth_backend` drives it with
exactly the values from the YAML config (no hardcoded duplicates), and
that :func:`require_unsloth` fails with the typed
``UnslothUnavailableError`` (never a raw ImportError) when unsloth is
absent or CUDA is unavailable.

What is proven here: the wiring and the config plumbing. What is NOT
proven: that Unsloth itself trains correctly on a GPU — that is only
verifiable on Kaggle.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from skillgen.training.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Fake unsloth module (records calls)
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self):
        self.peft_applied = False


class _FakeTokenizer:
    pad_token = None
    eos_token = "<|im_end|>"


class _FakeFastLanguageModel:
    calls: dict = {"from_pretrained": [], "get_peft_model": [], "for_inference": []}

    @classmethod
    def reset(cls):
        cls.calls = {"from_pretrained": [], "get_peft_model": [], "for_inference": []}

    @staticmethod
    def from_pretrained(**kwargs):
        _FakeFastLanguageModel.calls["from_pretrained"].append(kwargs)
        return _FakeModel(), _FakeTokenizer()

    @staticmethod
    def get_peft_model(model, **kwargs):
        _FakeFastLanguageModel.calls["get_peft_model"].append((model, kwargs))
        model.peft_applied = True
        return model

    @staticmethod
    def for_inference(model):
        _FakeFastLanguageModel.calls["for_inference"].append(model)
        return model


@pytest.fixture()
def unsloth_backend_with_stub(monkeypatch):
    """Import unsloth_backend with a fake `unsloth` in sys.modules + CUDA on.

    Restores the real (unsloth-less) module state afterwards so other
    tests see the honest local situation.
    """
    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = _FakeFastLanguageModel
    _FakeFastLanguageModel.reset()
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    import skillgen.training.unsloth_backend as backend

    backend = importlib.reload(backend)
    monkeypatch.setattr(backend, "_cuda_available", lambda: True)
    yield backend
    # Restore: drop the stub and reload so the module reflects reality.
    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    importlib.reload(backend)


@pytest.fixture()
def unsloth_backend_real():
    """The module as it honestly is on this VM (unsloth not installed)."""
    import skillgen.training.unsloth_backend as backend

    return importlib.reload(backend)


def _e1_config() -> dict:
    return load_config(CONFIGS_DIR / "qlora_e1.yaml")


# ---------------------------------------------------------------------------
# load_unsloth_model drives FastLanguageModel with the CONFIG's values
# ---------------------------------------------------------------------------


def test_from_pretrained_uses_config_values(unsloth_backend_with_stub):
    backend = unsloth_backend_with_stub
    cfg = _e1_config()
    model, tokenizer = backend.load_unsloth_model(cfg)
    assert model is not None and tokenizer is not None
    (kwargs,) = _FakeFastLanguageModel.calls["from_pretrained"]
    assert kwargs["model_name"] == cfg["model"]["name"]
    assert kwargs["model_name"] == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_seq_length"] == cfg["data"]["max_seq_length"] == 8192
    assert kwargs["dtype"] is None  # auto (fp16 on T4)


def test_get_peft_model_uses_config_values(unsloth_backend_with_stub):
    backend = unsloth_backend_with_stub
    cfg = _e1_config()
    model, _ = backend.load_unsloth_model(cfg)
    assert model.peft_applied is True
    ((_model, kwargs),) = _FakeFastLanguageModel.calls["get_peft_model"]
    assert kwargs["r"] == cfg["lora"]["r"] == 16
    assert kwargs["lora_alpha"] == cfg["lora"]["alpha"] == 32
    assert kwargs["lora_dropout"] == cfg["lora"]["dropout"] == 0.0
    assert kwargs["bias"] == "none"
    assert kwargs["target_modules"] == cfg["lora"]["target_modules"]
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["random_state"] == cfg["training"]["seed"] == 42


def test_enable_inference_wraps_for_inference(unsloth_backend_with_stub):
    backend = unsloth_backend_with_stub
    model = _FakeModel()
    out = backend.enable_inference(model)
    assert out is model
    assert _FakeFastLanguageModel.calls["for_inference"] == [model]


# ---------------------------------------------------------------------------
# require_unsloth: typed error, never a raw ImportError traceback
# ---------------------------------------------------------------------------


def test_require_unsloth_raises_typed_error_when_absent(unsloth_backend_real):
    backend = unsloth_backend_real
    # On this VM unsloth is genuinely not installed.
    assert backend.unsloth_importable() is False
    with pytest.raises(backend.UnslothUnavailableError) as excinfo:
        backend.require_unsloth()
    message = str(excinfo.value)
    assert "unsloth" in message.lower()
    assert "kaggle" in message.lower()  # actionable: where/how to get it


def test_require_unsloth_raises_typed_error_when_no_cuda(monkeypatch):
    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = _FakeFastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    import skillgen.training.unsloth_backend as backend

    backend = importlib.reload(backend)
    try:
        assert backend.unsloth_importable() is True
        assert backend._cuda_available() is False  # this VM has no GPU
        with pytest.raises(backend.UnslothUnavailableError) as excinfo:
            backend.require_unsloth()
        assert "cuda" in str(excinfo.value).lower()
    finally:
        monkeypatch.delitem(sys.modules, "unsloth", raising=False)
        importlib.reload(backend)


# ---------------------------------------------------------------------------
# Config contract: production configs are unsloth; ONLY the smoke is peft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "qlora_e1.yaml",
        "qlora_e2_no_masking.yaml",
        "qlora_e3_epochs1.yaml",
        "qlora_e3_epochs3.yaml",
        "qlora_e4_rank8.yaml",
        "qlora_e4_rank32.yaml",
        "qlora_e5_attention_only.yaml",
        "qlora_e6_data250.yaml",
        "qlora_e6_data500.yaml",
    ],
)
def test_production_configs_use_unsloth_backend(filename):
    cfg = load_config(CONFIGS_DIR / filename)
    assert cfg["training"]["backend"] == "unsloth"


def test_smoke_config_uses_peft_backend():
    cfg = load_config(CONFIGS_DIR / "smoke_cpu.yaml")
    assert cfg["training"]["backend"] == "peft"
