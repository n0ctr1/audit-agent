"""Tool execution error hierarchy and mock result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """Base for all tool execution errors."""

    def __init__(self, tool: str, message: str, output: str = "", stderr: str = "") -> None:
        self.tool = tool
        self.output = output
        self.stderr = stderr
        super().__init__(f"[{tool}] {message}")


class ToolUnavailableError(ToolError):
    """Tool cannot run — infrastructure issue (Docker unreachable, image missing, etc.)."""


class ToolExecutionError(ToolError):
    """Tool ran but failed: non-zero exit code, parse error, etc."""


class ToolTimeoutError(ToolError):
    """Tool exceeded its wall-clock time limit."""


@dataclass
class MockRunResult:
    """Returned by DockerRunner when AUDIT_AGENT_TOOL_MODE=mock.

    Signals that the tool was intentionally skipped. Callers MUST NOT treat
    this as a successful run — it is a distinct type precisely to prevent
    silent fake-success behaviour.
    """

    status: str = "mocked"
    data: Any = None
