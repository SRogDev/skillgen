"""SkillGen serving API (PLAN.md §41-§43).

``POST /generate-skill`` with ``{"input": "<user request>"}`` returns
``200 {"blueprint": {...}, "validation_report": {...}}``.

Post-inference validation pipeline (§42): generate -> JSON parse (L1)
-> schema validation (L2) -> on failure, repair: re-generate with the
validation errors fed back, MAX 2 retries -> still invalid: a typed
reject error (HTTP 422, ``error.code == "blueprint_invalid"``).
Validation reuses the EXISTING harness validators from
``scripts/eval_harness.py`` — imported the same way
``src/skillgen/evaluation/baseline.py`` already does — never
reimplemented here.

Generation defaults for the real backend (§41): temperature 0.3,
top_p 0.9, max_new_tokens 4096, repetition_penalty 1.1, the §8 system
prompt verbatim, and the model's native chat template.

v1 is local-only: no auth. Backends are pluggable behind the
:class:`ModelBackend` protocol so tests (and the demo default) run
with a deterministic fake — no model download. Backend selection
(``_default_backend``): ``SKILLGEN_BACKEND=unsloth`` selects the
GPU-only :class:`UnslothBackend` (Roger's 2026-10-23 "Unsloth para
todo" direction: Unsloth for all GPU work — training, export, real
serving inference, Kaggle baselines); ``SKILLGEN_BACKEND=transformers``
or a bare ``SKILLGEN_MODEL_ID`` selects :class:`TransformersBackend`;
otherwise the demo :class:`FakeBackend` is the default, and
``GET /health`` always names the active backend so nobody mistakes
the demo for the real (not yet trained at 7B) model. Selecting the
Unsloth backend on a machine without CUDA/unsloth fails at first
generation with the typed ``UnslothUnavailableError`` from
``skillgen.training.unsloth_backend``, surfaced as the endpoint's
typed 502 — never a crash traceback.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Reuse the exact §8 system prompt and the existing harness validators.
from skillgen.evaluation.baseline import SYSTEM_PROMPT, eval_harness

__all__ = [
    "SYSTEM_PROMPT",
    "ModelBackend",
    "FakeBackend",
    "TransformersBackend",
    "UnslothBackend",
    "build_repair_text",
    "create_app",
    "app",
    "MAX_INPUT_CHARS",
    "MAX_ATTEMPTS",
]

#: Input size guard (§43: input max ~8k tokens; enforced in characters).
MAX_INPUT_CHARS = 32000

#: 1 initial generation + MAX 2 repair retries (§42).
MAX_ATTEMPTS = 3

#: §41 generation defaults for the real backend.
DEFAULT_GENERATION = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
}

# A real, short train-split blueprint (id ``larksuite_cli__good-skill``,
# ``data/splits/train_shard_*.jsonl``) — it passes ``l2_validate_schema``.
# The demo FakeBackend returns it verbatim; it is demo data, NOT model
# output.
DEMO_BLUEPRINT = (
    '{"blueprint_version": "1.0", "name": "good-skill", '
    '"description": "This is a properly formatted skill.", '
    '"purpose": "This skill follows all the formatting rules.", '
    '"when_to_use": ["This is a properly formatted skill."], '
    '"when_not_to_use": [], '
    '"files": [{"path": "SKILL.md", '
    '"purpose": "Main skill definition: YAML frontmatter plus Markdown instructions.", '
    '"content": "---\\nname: good-skill\\nversion: 1.0.0\\n'
    'description: \\"This is a properly formatted skill.\\"\\nmetadata:\\n'
    '  requires:\\n    bins: [\\"lark-cli\\"]\\n---\\n\\n# Good Skill\\n\\n'
    'This skill follows all the formatting rules.\\n"}]}'
)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelBackend(Protocol):
    """Anything that can turn a user request text into raw model output."""

    name: str

    def generate(self, user_text: str) -> str:
        """Return the raw (unparsed) model output for ``user_text``."""
        ...


class FakeBackend:
    """Deterministic backend for tests and the local demo.

    Constructed with a list of canned outputs, returned in order;
    once the list is exhausted the LAST output repeats forever.
    """

    def __init__(self, outputs: list[str], name: str = "fake"):
        if not outputs:
            raise ValueError("FakeBackend needs at least one canned output")
        self._outputs = list(outputs)
        self._index = 0
        self.name = name

    def generate(self, user_text: str) -> str:
        idx = min(self._index, len(self._outputs) - 1)
        self._index += 1
        return self._outputs[idx]


class TransformersBackend:
    """Real backend: a Hugging Face causal LM (+ optional PEFT adapter).

    Lazy: the model/tokenizer load on the FIRST ``generate()`` call,
    never at import time or app creation, so importing this module and
    starting the server stay cheap and download-free until used.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        **generation_overrides,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.generation = {**DEFAULT_GENERATION, **generation_overrides}
        self.name = f"transformers:{model_id}"
        if adapter_path:
            self.name += f"+adapter:{adapter_path}"
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        # Imported here (not at module level) to keep loading lazy.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def generate(self, user_text: str) -> str:
        import torch

        self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **self.generation)
        # Decode ONLY the newly generated tokens.
        new_ids = output_ids[0][input_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)


class UnslothBackend:
    """Real backend on GPU: Unsloth ``FastLanguageModel`` + fast inference.

    The production serving path on a GPU host (Roger's 2026-10-23
    direction: Unsloth for all GPU work). Lazy like
    :class:`TransformersBackend`: nothing loads at import or app
    creation; the first ``generate()`` loads the 4-bit model through
    ``skillgen.training.unsloth_backend`` and switches it to inference
    mode with ``FastLanguageModel.for_inference``. If ``adapter_path``
    is set it is loaded directly (an Unsloth-saved adapter directory
    loads through ``from_pretrained`` and brings its base with it).

    On a machine without unsloth or CUDA, the first ``generate()``
    raises the typed ``UnslothUnavailableError`` — the endpoint turns
    it into a clean 502, never a traceback.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        max_seq_length: int = 8192,
        **generation_overrides,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.generation = {**DEFAULT_GENERATION, **generation_overrides}
        self.name = f"unsloth:{model_id}"
        if adapter_path:
            self.name += f"+adapter:{adapter_path}"
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        # Imported here (not at module level) to keep loading lazy —
        # and because `import unsloth` must precede transformers
        # imports in the process (unsloth_backend handles the guard).
        from skillgen.training.unsloth_backend import (
            enable_inference,
            require_unsloth,
        )

        require_unsloth()
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.adapter_path or self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,  # auto: fp16 on T4-class GPUs
            load_in_4bit=True,
        )
        self._model = enable_inference(model)
        self._tokenizer = tokenizer

    def generate(self, user_text: str) -> str:
        import torch

        self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **self.generation)
        # Decode ONLY the newly generated tokens.
        new_ids = output_ids[0][input_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Repair prompt (§42)
# ---------------------------------------------------------------------------


def build_repair_text(
    original_input: str, previous_output: str, errors: list[str]
) -> str:
    """Build the repair request: original input + previous output + errors.

    Pure function (unit-testable). The validation errors are included
    verbatim so the model sees exactly what the harness rejected.
    """
    error_block = "\n".join(f"- {err}" for err in errors)
    return (
        f"{original_input}\n\n"
        "Your previous output was rejected by the Skill Blueprint "
        "validator. Previous output:\n"
        f"{previous_output}\n\n"
        "Validation errors:\n"
        f"{error_block}\n\n"
        "Fix ALL of these errors and output ONLY a valid JSON Skill "
        "Blueprint. No explanations, no markdown fences."
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


def _default_backend() -> ModelBackend:
    backend_kind = os.environ.get("SKILLGEN_BACKEND", "").strip().lower()
    model_id = os.environ.get("SKILLGEN_MODEL_ID")
    adapter_path = os.environ.get("SKILLGEN_ADAPTER_PATH")
    if backend_kind == "unsloth":
        # GPU-only production path; construction is lazy — a machine
        # without CUDA fails typed at first generate(), not here.
        return UnslothBackend(
            model_id or "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            adapter_path=adapter_path,
        )
    if backend_kind == "transformers" or model_id:
        return TransformersBackend(
            model_id or "Qwen/Qwen2.5-7B-Instruct", adapter_path=adapter_path
        )
    # Demo default — clearly named so /health never lets it pass as real.
    return FakeBackend([DEMO_BLUEPRINT], name="fake-demo (NOT a trained model)")


def create_app(backend: ModelBackend | None = None) -> FastAPI:
    if backend is None:
        backend = _default_backend()
    app = FastAPI(title="SkillGen", version="0.1.0")
    app.state.backend = backend

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": getattr(backend, "name", backend.__class__.__name__),
        }

    @app.post("/generate-skill")
    def generate_skill(request: GenerateRequest):
        original_input = request.input
        current_text = original_input
        last_errors: list[str] = []
        last_output = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw = backend.generate(current_text)
            except Exception as exc:  # backend failure -> typed 502
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "code": "backend_error",
                            "message": f"{type(exc).__name__}: {exc}",
                            "attempts": attempt,
                        }
                    },
                )
            last_output = raw
            ok, obj = eval_harness.l1_parse(raw)
            if not ok:
                errors = ["L1 parse failure: output is not valid JSON"]
            else:
                errors = list(eval_harness.l2_validate_schema(obj))
            if ok and not errors:
                return {
                    "blueprint": obj,
                    "validation_report": {
                        "valid": True,
                        "attempts": attempt,
                        "errors": [],
                    },
                }
            last_errors = errors
            if attempt < MAX_ATTEMPTS:
                current_text = build_repair_text(original_input, raw, errors)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "blueprint_invalid",
                    "message": (
                        "Generated blueprint failed validation after "
                        f"{MAX_ATTEMPTS} attempts."
                    ),
                    "attempts": MAX_ATTEMPTS,
                    "last_errors": last_errors,
                    "last_output": last_output[:2000],
                }
            },
        )

    return app


#: Module-level app for ``uvicorn skillgen.serving.app:app``.
app = create_app()
