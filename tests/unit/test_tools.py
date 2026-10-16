"""Unit tests for tool error policy (no Docker required)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from audit_agent.tools.errors import (
    MockRunResult,
    ToolExecutionError,
    ToolTimeoutError,
    ToolUnavailableError,
)


# ── DockerRunner ──────────────────────────────────────────────────────────────

class TestDockerRunnerStrict:
    def test_raises_tool_unavailable_when_docker_absent(self, monkeypatch):
        """In strict mode, run() raises ToolUnavailableError if Docker is down."""
        monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")
        import audit_agent.config as cfg
        cfg.reset_settings()

        with patch("docker.from_env", side_effect=Exception("socket not found")):
            from audit_agent.sandbox.docker_runner import DockerRunner

            runner = DockerRunner(tool_mode="strict")
            assert runner._client is None
            assert runner._unavailable_error is not None

            with pytest.raises(ToolUnavailableError) as exc_info:
                import asyncio
                asyncio.run(runner.run(image="test", command=["echo", "hi"]))

            assert exc_info.value.tool == "docker"
            assert "Docker" in str(exc_info.value) or "docker" in str(exc_info.value)

    def test_raises_tool_unavailable_in_lenient_mode_too(self, monkeypatch):
        """lenient mode still raises ToolUnavailableError; callers decide to degrade."""
        with patch("docker.from_env", side_effect=Exception("no daemon")):
            from audit_agent.sandbox.docker_runner import DockerRunner

            runner = DockerRunner(tool_mode="lenient")
            with pytest.raises(ToolUnavailableError):
                import asyncio
                asyncio.run(runner.run(image="test", command=["echo", "hi"]))


class TestDockerRunnerMock:
    @pytest.mark.asyncio
    async def test_returns_mock_result_in_mock_mode(self):
        """mock mode skips Docker and returns MockRunResult — never exit_code=0 fake success."""
        from audit_agent.sandbox.docker_runner import DockerRunner

        runner = DockerRunner(tool_mode="mock")
        # Should NOT attempt Docker at all
        result = await runner.run(image="audit-agent/slither:latest", command=["slither", "."])

        assert isinstance(result, MockRunResult), (
            f"Expected MockRunResult in mock mode, got {type(result).__name__}"
        )
        assert result.status == "mocked"
        assert result.data is None

    @pytest.mark.asyncio
    async def test_mock_mode_never_contacts_docker(self):
        """Verify Docker SDK is never called in mock mode."""
        with patch("docker.from_env") as mock_from_env:
            from audit_agent.sandbox.docker_runner import DockerRunner

            runner = DockerRunner(tool_mode="mock")
            await runner.run(image="any-image", command=["any-cmd"])
            mock_from_env.assert_not_called()


# ── Slither wrapper ───────────────────────────────────────────────────────────

class TestSlitherWrapper:
    @pytest.mark.asyncio
    async def test_returns_mock_result_in_mock_mode(self, tmp_path, monkeypatch):
        """slither wrapper in mock mode returns MockRunResult, not []."""
        monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "mock")
        import audit_agent.config as cfg
        cfg.reset_settings()

        from audit_agent.tools.slither import run_slither

        result = await run_slither(str(tmp_path), [], audit_id="test")

        assert isinstance(result, MockRunResult), (
            f"Expected MockRunResult in mock mode, got {type(result).__name__}. "
            "Returning [] would hide that the tool did not run."
        )

    @pytest.mark.asyncio
    async def test_crash_without_report_raises(self, tmp_path):
        """Slither crashed before writing its report → ToolExecutionError (not an empty list).

        Scout turns this into a halt (strict) or a degraded tool (lenient).
        """
        from audit_agent.sandbox.docker_runner import RunResult
        from audit_agent.tools.errors import ToolExecutionError

        mock_result = RunResult(exit_code=1, output="", error="some slither error", duration_ms=0)

        with patch(
            "audit_agent.sandbox.docker_runner.DockerRunner.run",
            new=AsyncMock(return_value=mock_result),
        ):
            from audit_agent.tools.slither import run_slither

            with pytest.raises(ToolExecutionError):
                await run_slither(str(tmp_path), [], audit_id="test")

    @pytest.mark.asyncio
    async def test_invalid_json_in_report_file_returns_empty(self, tmp_path):
        """Invalid JSON in the report file is logged as error, returns empty findings."""
        from audit_agent.sandbox.docker_runner import RunResult

        report_path = tmp_path / "slither_output.json"

        async def run_writes_broken_report(*args, **kwargs):
            # The wrapper deletes any stale report first; write it as the container would.
            report_path.write_text("not json at all !!!")
            return RunResult(exit_code=255, output="", error="", duration_ms=0)

        with patch(
            "audit_agent.sandbox.docker_runner.DockerRunner.run",
            new=run_writes_broken_report,
        ):
            from audit_agent.tools.slither import run_slither

            result = await run_slither(str(tmp_path), [], audit_id="test")
            assert isinstance(result, list)
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_propagates_tool_unavailable(self, tmp_path):
        """ToolUnavailableError from DockerRunner propagates through slither wrapper."""
        with patch(
            "audit_agent.sandbox.docker_runner.DockerRunner.run",
            new=AsyncMock(
                side_effect=ToolUnavailableError("docker", "Docker daemon unreachable")
            ),
        ):
            from audit_agent.tools.slither import run_slither

            with pytest.raises(ToolUnavailableError):
                await run_slither(str(tmp_path), [], audit_id="test")


# ── Scout agent — strict vs lenient mode ─────────────────────────────────────

class TestScoutAgentToolMode:
    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_tool_unavailable(self, tmp_path, monkeypatch):
        """In strict mode, ToolUnavailableError from Slither propagates out of Scout."""
        monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")
        import audit_agent.config as cfg
        cfg.reset_settings()

        (tmp_path / "Token.sol").write_text("pragma solidity ^0.8.20;\n")

        with patch(
            "audit_agent.agents.scout.run_slither",
            new=AsyncMock(
                side_effect=ToolUnavailableError("docker", "Docker daemon unreachable")
            ),
        ):
            from audit_agent.agents.scout import ScoutAgent

            agent = ScoutAgent()
            state = {
                "audit_id": "test-strict",
                "mode": "review",
                "target_dir": str(tmp_path),
                "platform": "generic",
                "budget_tokens": 100_000,
                "budget_usd": 10.0,
                "invariants": [],
                "static_findings": [],
                "hypotheses": [],
                "exploits": [],
                "validated_findings": [],
                "tokens_used": 0,
                "cost_usd": 0.0,
                "errors": [],
                "run_checkpoint_id": "",
                "degraded_tools": [],
            }

            with pytest.raises(ToolUnavailableError):
                await agent.run(state)

    @pytest.mark.asyncio
    async def test_mock_mode_marks_tool_degraded(self, tmp_path, monkeypatch):
        """In mock mode, Slither result is MockRunResult → tool added to degraded_tools."""
        monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "mock")
        import audit_agent.config as cfg
        cfg.reset_settings()

        (tmp_path / "Token.sol").write_text("pragma solidity ^0.8.20;\n")

        with patch(
            "audit_agent.agents.scout.run_slither",
            new=AsyncMock(return_value=MockRunResult()),
        ):
            with patch(
                "audit_agent.agents.scout.ScoutAgent._summarize_protocol",
                new=AsyncMock(
                    return_value=__import__(
                        "audit_agent.models.schemas", fromlist=["ProtocolSummary"]
                    ).ProtocolSummary(
                        name="Mock",
                        category="other",
                        key_mechanisms=[],
                        user_roles=[],
                        sol_files=[str(tmp_path / "Token.sol")],
                    )
                ),
            ):
                from audit_agent.agents.scout import ScoutAgent

                agent = ScoutAgent()
                result = await agent.run({
                    "audit_id": "test-mock",
                    "mode": "review",
                    "target_dir": str(tmp_path),
                    "platform": "generic",
                    "budget_tokens": 100_000,
                    "budget_usd": 10.0,
                    "invariants": [],
                    "static_findings": [],
                    "hypotheses": [],
                    "exploits": [],
                    "validated_findings": [],
                    "tokens_used": 0,
                    "cost_usd": 0.0,
                    "errors": [],
                    "run_checkpoint_id": "",
                    "degraded_tools": [],
                })

        assert "slither" in result["degraded_tools"], (
            "slither must appear in degraded_tools when running in mock mode"
        )
        assert result["static_findings"] == [], (
            "static_findings must be empty when Slither ran in mock mode"
        )

    @pytest.mark.asyncio
    async def test_lenient_mode_marks_tool_degraded_on_unavailable(
        self, tmp_path, monkeypatch
    ):
        """lenient mode: ToolUnavailableError from Slither → degraded_tools, no exception."""
        monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "lenient")
        # Only Slither is under test; keep the other analyzers off so the result
        # does not depend on which Docker images exist on the machine.
        monkeypatch.setenv("ADERYN_ENABLED", "false")
        monkeypatch.setenv("SEMGREP_ENABLED", "false")
        import audit_agent.config as cfg
        cfg.reset_settings()

        (tmp_path / "Token.sol").write_text("pragma solidity ^0.8.20;\n")

        with patch(
            "audit_agent.agents.scout.run_slither",
            new=AsyncMock(
                side_effect=ToolUnavailableError("docker", "no docker")
            ),
        ):
            with patch(
                "audit_agent.agents.scout.ScoutAgent._summarize_protocol",
                new=AsyncMock(
                    return_value=__import__(
                        "audit_agent.models.schemas", fromlist=["ProtocolSummary"]
                    ).ProtocolSummary(
                        name="Mock",
                        category="other",
                        key_mechanisms=[],
                        user_roles=[],
                        sol_files=[str(tmp_path / "Token.sol")],
                    )
                ),
            ):
                from audit_agent.agents.scout import ScoutAgent

                agent = ScoutAgent()
                result = await agent.run({
                    "audit_id": "test-lenient",
                    "mode": "review",
                    "target_dir": str(tmp_path),
                    "platform": "generic",
                    "budget_tokens": 100_000,
                    "budget_usd": 10.0,
                    "invariants": [],
                    "static_findings": [],
                    "hypotheses": [],
                    "exploits": [],
                    "validated_findings": [],
                    "tokens_used": 0,
                    "cost_usd": 0.0,
                    "errors": [],
                    "run_checkpoint_id": "",
                    "degraded_tools": [],
                })

        assert "docker" in result["degraded_tools"]
        assert result["static_findings"] == []
