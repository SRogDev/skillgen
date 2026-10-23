"""Phase 5 serving tests (PLAN.md §41-§43).

Uses FastAPI's TestClient with FakeBackend only — no model downloads,
no GPU, no network. The valid blueprint below is a real (short) train
record's ``expected_blueprint`` (id ``larksuite_cli__good-skill`` from
``data/splits/train_shard_*.jsonl``); it passes the harness's
``l2_validate_schema``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from skillgen.serving.app import FakeBackend, build_repair_text, create_app

VALID_BLUEPRINT = (
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

# Parses as JSON (L1 passes) but fails L2 schema validation.
INVALID_BLUEPRINT = '{"name": "Bad Name"}'


def _client(outputs):
    return TestClient(create_app(FakeBackend(outputs)), raise_server_exceptions=False)


def test_health_reports_backend_name():
    client = _client([VALID_BLUEPRINT])
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "fake" in body["backend"].lower()


def test_valid_path_first_attempt():
    client = _client([VALID_BLUEPRINT])
    resp = client.post("/generate-skill", json={"input": "make me a good skill"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["blueprint"], dict)
    assert body["blueprint"]["name"] == "good-skill"
    assert body["validation_report"]["valid"] is True
    assert body["validation_report"]["attempts"] == 1
    assert body["validation_report"]["errors"] == []


def test_repair_path_second_attempt():
    client = _client([INVALID_BLUEPRINT, VALID_BLUEPRINT])
    resp = client.post("/generate-skill", json={"input": "make me a good skill"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["blueprint"]["name"] == "good-skill"
    assert body["validation_report"]["valid"] is True
    assert body["validation_report"]["attempts"] == 2


def test_reject_path_after_max_retries():
    client = _client([INVALID_BLUEPRINT])
    resp = client.post("/generate-skill", json={"input": "make me a good skill"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "blueprint_invalid"
    assert body["error"]["attempts"] == 3
    assert body["error"]["last_errors"]


def test_empty_input_is_422():
    client = _client([VALID_BLUEPRINT])
    resp = client.post("/generate-skill", json={"input": ""})
    assert resp.status_code == 422


def test_backend_error_is_502():
    class BoomBackend:
        name = "boom"

        def generate(self, user_text: str) -> str:
            raise RuntimeError("backend exploded")

    client = TestClient(create_app(BoomBackend()), raise_server_exceptions=False)
    resp = client.post("/generate-skill", json={"input": "hello"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "backend_error"


def test_build_repair_text_includes_errors_verbatim():
    errors = ["missing field: files", "invalid name: 'Bad Name'"]
    text = build_repair_text("original request", INVALID_BLUEPRINT, errors)
    for err in errors:
        assert err in text
    assert "original request" in text
    # The previous (invalid) output is fed back so the model can fix it.
    assert INVALID_BLUEPRINT in text


def test_valid_blueprint_fixture_actually_validates():
    # Guard the fixture itself against the real harness validator.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_harness.py"
    spec = importlib.util.spec_from_file_location("eval_harness_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ok, obj = module.l1_parse(VALID_BLUEPRINT)
    assert ok
    assert module.l2_validate_schema(obj) == []
    assert json.loads(VALID_BLUEPRINT)["name"] == "good-skill"


# ---------------------------------------------------------------------------
# UnslothBackend (Roger's 2026-10-23 "Unsloth para todo" direction)
#
# GPU-only and lazy. On this CPU VM unsloth is not installed, so selecting
# it must surface the typed UnslothUnavailableError as a clean 502 — never
# a crash traceback — and /health must name it honestly.
# ---------------------------------------------------------------------------


def test_default_backend_selection_unsloth_via_env(monkeypatch):
    from skillgen.serving.app import UnslothBackend, _default_backend

    monkeypatch.setenv("SKILLGEN_BACKEND", "unsloth")
    monkeypatch.setenv("SKILLGEN_MODEL_ID", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    backend = _default_backend()  # construction must NOT load anything
    assert isinstance(backend, UnslothBackend)
    assert backend.name.startswith("unsloth:")


def test_default_backend_selection_transformers_via_env(monkeypatch):
    from skillgen.serving.app import TransformersBackend, _default_backend

    monkeypatch.setenv("SKILLGEN_BACKEND", "transformers")
    monkeypatch.setenv("SKILLGEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    backend = _default_backend()
    assert isinstance(backend, TransformersBackend)


def test_unsloth_backend_health_names_it():
    from skillgen.serving.app import UnslothBackend

    backend = UnslothBackend("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    client = TestClient(create_app(backend), raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["backend"].startswith("unsloth:")


def test_unsloth_backend_unavailable_is_typed_502():
    from skillgen.serving.app import UnslothBackend

    backend = UnslothBackend("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    client = TestClient(create_app(backend), raise_server_exceptions=False)
    resp = client.post("/generate-skill", json={"input": "make me a skill"})
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["code"] == "backend_error"
    # The typed error from the unsloth backend module, not a raw traceback.
    assert "UnslothUnavailableError" in error["message"]
