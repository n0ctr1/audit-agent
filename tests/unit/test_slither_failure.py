"""run_slither must not hide a Slither crash behind an empty finding list."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from audit_agent.tools.errors import ToolExecutionError


async def test_missing_report_raises(tmp_path, monkeypatch):
    from audit_agent.tools import slither

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")
    crashed = SimpleNamespace(exit_code=1, output="", error="FileNotFoundError: 'forge'")
    with patch.object(slither.DockerRunner, "run", new=AsyncMock(return_value=crashed)):
        with pytest.raises(ToolExecutionError, match="no report"):
            await slither.run_slither(str(tmp_path), [], "a")


async def test_report_present_is_parsed(tmp_path, monkeypatch):
    from audit_agent.tools import slither

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")

    async def fake_run(self, **kw):
        (tmp_path / "slither_output.json").write_text(
            '{"results": {"detectors": [{"check": "reentrancy-eth", "impact": "High", '
            '"description": "d", "elements": []}]}}')
        return SimpleNamespace(exit_code=255, output="", error="")

    with patch.object(slither.DockerRunner, "run", new=fake_run):
        out = await slither.run_slither(str(tmp_path), [], "a")
    assert len(out) == 1


async def test_scout_lenient_degrades_on_slither_crash(tmp_path, monkeypatch):
    """lenient: the crash marks slither degraded instead of silently returning []."""
    from audit_agent.agents.scout import ScoutAgent

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "lenient")
    agent = ScoutAgent()
    with patch("audit_agent.agents.scout.run_slither",
               new=AsyncMock(side_effect=ToolExecutionError("slither", "no report"))):
        findings, degraded = await agent._run_slither(str(tmp_path), [], "a", "lenient", [])
    assert findings == [] and "slither" in degraded
