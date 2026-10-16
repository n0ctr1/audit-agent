"""Pipeline presets for benchmark runs.

A preset is a set of environment overrides applied for the duration of a run
and recorded in the results metadata. Settings are re-read on enter and exit.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

# Every preset sets all of these explicitly, so a value left in .env cannot
# silently change what a preset means.
ABLATION_FLAGS = (
    "THREAT_MODEL_ENABLED",
    "HUNTER_EXPERTS_ENABLED",
    "HUNTER_STANDALONE_PASSES_ENABLED",
    "EXPLOIT_ENABLED",
    "SKEPTIC_ENABLED",
    "REFUTER_ENABLED",
)

# Flags switched off by each preset.
DISABLED: dict[str, tuple[str, ...]] = {
    # Everything on (the defaults).
    "full": (),
    # Skeptic verdicts are final.
    "no_refuter": ("REFUTER_ENABLED",),
    # Hunter without mixture-of-experts and standalone passes (temporal, crypto,
    # cross-contract, aggregation, boundary); per-function analysis, navigator
    # and triage stay on.
    "no_experts": ("HUNTER_EXPERTS_ENABLED", "HUNTER_STANDALONE_PASSES_ENABLED"),
    "no_threat_model": ("THREAT_MODEL_ENABLED",),
    # No Exploit / Skeptic / Refuter: every hypothesis is reported as unverified.
    "hunter_only": ("EXPLOIT_ENABLED", "SKEPTIC_ENABLED", "REFUTER_ENABLED"),
    # Not the pipeline at all: one LLM agent with bash + file_edit
    # (agents/baseline.py, budgets BASELINE_MAX_*). Findings are unverified.
    "baseline": ABLATION_FLAGS,
}

PRESETS: dict[str, dict[str, str]] = {
    name: {flag: "false" if flag in off else "true" for flag in ABLATION_FLAGS}
    for name, off in DISABLED.items()
}


def preset_env(name: str) -> dict[str, str]:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return dict(PRESETS[name])


@contextmanager
def applied_preset(name: str) -> Iterator[dict[str, str]]:
    """Set the preset's env vars, yield them, then restore the previous environment."""
    from audit_agent.config import reset_settings

    env = preset_env(name)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    reset_settings()
    try:
        yield env
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        reset_settings()
