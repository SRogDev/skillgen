"""Training-config loading for SkillGen (PLAN.md §53).

Configs are plain YAML files under ``configs/`` with the sections
``model`` / ``quantization`` / ``lora`` / ``training`` / ``data`` plus the
top-level fields ``experiment_id``, ``output_dir`` and
``train_on_assistant_only``. One file per experiment; per PLAN.md §19 each
E-variant differs from E1 in exactly one intended key (enforced by
``tests/test_prepare.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a training config YAML into a plain dict."""
    with Path(path).open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return config


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dicts into dotted keys (lists stay whole).

    Used by the config-diff test to assert that an E-variant changes
    exactly its intended key(s) versus E1.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat
