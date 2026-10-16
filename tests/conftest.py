"""Session-wide test isolation.

The test suite must behave the same on a fresh clone (no `.env`) and on a
developer machine whose `.env` points at a real LLM endpoint, real Docker images
or `AUDIT_AGENT_TOOL_MODE=strict`. Tests therefore never read `.env` and default
to `AUDIT_AGENT_TOOL_MODE=mock`; tests that need another mode set it explicitly
with `monkeypatch.setenv`.
"""

from __future__ import annotations

import pytest

from audit_agent.config import Settings, reset_settings

Settings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "mock")
    # Reports written by full-audit tests must not land in ./audit-reports.
    monkeypatch.setenv("AUDIT_REPORTS_DIR", str(tmp_path_factory.mktemp("audit-reports")))
    reset_settings()
    yield
    reset_settings()
