"""BASELINE — a single tool-using LLM agent, for comparison with the pipeline.

One model with two tools, ``bash`` (inside the Foundry Docker image, network
disabled, on a copy of the project) and ``file_edit``, investigates the in-scope
code and submits findings through the same ``report_hypotheses`` schema Hunter
uses. It runs under explicit token / wall-clock / step budgets so that "complex
pipeline vs simple agent at equal budget" can be measured with the benchmark
(``--preset baseline``).

Transport mirrors the navigator: tool observations are fed back as plain user
messages (the multi-turn shape every provider supports through ``complete()``).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from audit_agent.agents.hunter import _HYPOTHESIS_TOOL
from audit_agent.config import get_settings
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.models.schemas import Hypothesis
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.baseline")

_SEVERITIES = ("critical", "high", "medium", "low")
_OBS_LIMIT = 12_000      # chars of tool output fed back per call
_HISTORY_KEEP = 40       # messages kept after the task statement

BASH_TOOL = ToolDefinition(
    name="bash",
    description=(
        "Run a bash command in /project (a copy of the target repository) inside a "
        "Docker container with Foundry (forge, cast, anvil) and no network access. "
        "Use it to list and read files, grep, and compile/run Foundry tests."
    ),
    parameters={
        "type": "object",
        "required": ["command"],
        "properties": {"command": {"type": "string"}},
    },
)

FILE_EDIT_TOOL = ToolDefinition(
    name="file_edit",
    description=(
        "Create or edit a file under /project. With `old_string`, replace its single "
        "occurrence by `new_string`; without it, write `new_string` as the whole file."
    ),
    parameters={
        "type": "object",
        "required": ["path", "new_string"],
        "properties": {
            "path": {"type": "string", "description": "path relative to /project"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
    },
)

REPORT_TOOL = ToolDefinition(
    name="report_hypotheses",
    description=(
        "Submit ALL vulnerabilities you found in the in-scope code, once, at the end. "
        "This ends the audit."
    ),
    parameters=_HYPOTHESIS_TOOL.parameters,
)

BASELINE_SYSTEM = """You are a smart contract security auditor working alone on a Solidity code base.

Goal: find High and Medium severity vulnerabilities in the in-scope files and
report them with the report_hypotheses tool. You have a shell (bash) in a
container with Foundry and no network, and a file_edit tool. You may write and
run Foundry tests to check your ideas. Work efficiently: your token, time and
step budget is limited, and you must call report_hypotheses before it runs out.

For each vulnerability give a specific title, the vulnerability class, the
affected functions as Contract.function, a step-by-step attack scenario,
preconditions, severity and a 0-100 priority. Only report issues in in-scope
files. If you find nothing credible, report an empty list."""


class BaselineAgent:
    """Single-LLM agent with bash + file_edit; returns hypotheses."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        shell_factory=None,  # (project_copy_dir: str) -> session with start/run/close
        max_tokens: Optional[int] = None,
        max_seconds: Optional[int] = None,
        max_steps: Optional[int] = None,
        clock=time.monotonic,
    ) -> None:
        s = get_settings()
        self._provider = provider or get_provider()
        self._model = model or s.baseline_model or s.hunter_model
        self._max_tokens = max_tokens if max_tokens is not None else s.baseline_max_tokens
        self._max_seconds = max_seconds if max_seconds is not None else s.baseline_max_seconds
        self._max_steps = max_steps if max_steps is not None else s.baseline_max_steps
        self._command_timeout = s.baseline_command_timeout
        self._shell_factory = shell_factory
        self._clock = clock
        self.tokens_used = 0
        self.steps = 0
        self.stop_reason = ""

    def _make_shell(self, project_copy: str):
        if self._shell_factory is not None:
            return self._shell_factory(project_copy)
        from audit_agent.tools.shell_session import DockerShellSession, MockShellSession

        if get_settings().tool_mode == "mock":
            return MockShellSession(project_copy)
        return DockerShellSession(project_copy, command_timeout=self._command_timeout)

    async def run(
        self,
        target_dir: str,
        in_scope: list[str],
        audit_id: str,
        documentation: str = "",
    ) -> list[Hypothesis]:
        tmp = Path(tempfile.mkdtemp(prefix="baseline-"))
        project = tmp / "project"
        shutil.copytree(target_dir, project, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "out", "cache", "node_modules",
                                                      "audit_logs", ".slither_analysis.json"))
        shell = self._make_shell(str(project))
        try:
            await shell.start()
            raw = await self._loop(shell, project, in_scope, documentation)
        finally:
            await shell.close()
            shutil.rmtree(tmp, ignore_errors=True)
        return _to_hypotheses(raw, audit_id)

    # ── agent loop ────────────────────────────────────────────────────────
    async def _loop(self, shell, project: Path, in_scope: list[str], documentation: str) -> list[dict]:
        task = (
            "In-scope files (paths relative to /project):\n"
            + "\n".join(f"- {p}" for p in in_scope)
            + (f"\n\nProject documentation (truncated):\n{documentation[:6000]}" if documentation else "")
            + f"\n\nBudget: {self._max_tokens} tokens, {self._max_seconds} s, {self._max_steps} tool steps."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        t0 = self._clock()
        al = get_audit_logger()

        while True:
            out_of_budget = (
                self.tokens_used >= self._max_tokens
                or self._clock() - t0 >= self._max_seconds
                or self.steps >= self._max_steps
            )
            resp = await self._complete(messages, force_report=out_of_budget)
            if resp is None:
                self.stop_reason = "llm_error"
                return []
            self.tokens_used += int(resp.input_tokens or 0) + int(resp.output_tokens or 0)

            report = next((tc for tc in resp.tool_calls if tc.name == "report_hypotheses"), None)
            if report is not None:
                self.stop_reason = "budget" if out_of_budget else "reported"
                hyps = report.arguments.get("hypotheses") or []
                if al:
                    al.log_decision("baseline", "report",
                                    f"{len(hyps)} hypotheses after {self.steps} steps",
                                    {"steps": self.steps, "tokens": self.tokens_used,
                                     "stop_reason": self.stop_reason})
                return hyps if isinstance(hyps, list) else []
            if out_of_budget:
                self.stop_reason = "budget_no_report"
                return []

            if not resp.tool_calls:
                messages.append({"role": "assistant", "content": (resp.text or "")[:2000]})
                messages.append({"role": "user", "content":
                                 "Use a tool: bash, file_edit, or report_hypotheses to finish."})
                self.steps += 1
                continue

            observations = []
            for tc in resp.tool_calls:
                self.steps += 1
                obs = await self._run_tool(shell, project, tc.name, tc.arguments or {})
                observations.append(f"OBSERVATION ({tc.name}):\n{obs[:_OBS_LIMIT]}")
                if al:
                    al.log_decision("baseline", "tool", f"{tc.name} -> {len(obs)} chars",
                                    {"tool": tc.name,
                                     "args": json.dumps(tc.arguments)[:300]})
            messages.append({"role": "assistant", "content": json.dumps(
                [{"tool": tc.name, "args": tc.arguments} for tc in resp.tool_calls])[:4000]})
            messages.append({"role": "user", "content": "\n\n".join(observations)})
            if len(messages) > _HISTORY_KEEP + 1:
                messages = [messages[0]] + messages[-_HISTORY_KEEP:]
                if messages[1]["role"] != "assistant":
                    messages.pop(1)

    async def _complete(self, messages: list[dict], force_report: bool):
        system = BASELINE_SYSTEM
        tools = [BASH_TOOL, FILE_EDIT_TOOL, REPORT_TOOL]
        choice: Any = "auto"
        if force_report:
            system += ("\n\nYOUR BUDGET IS EXHAUSTED. Call report_hypotheses now with every "
                       "vulnerability you found (empty list if none).")
            tools, choice = [REPORT_TOOL], {"name": "report_hypotheses"}
        try:
            return await asyncio.wait_for(
                self._provider.complete(model=self._model, system=system, messages=messages,
                                        tools=tools, tool_choice=choice, max_tokens=8192),
                timeout=900,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("baseline.llm_failed", error=str(exc))
            return None

    async def _run_tool(self, shell, project: Path, name: str, args: dict[str, Any]) -> str:
        if name == "bash":
            code, out = await shell.run(str(args.get("command", "")))
            return f"exit={code}\n{out}"
        if name == "file_edit":
            return _file_edit(project, args)
        return f"unknown tool {name!r}"


def _file_edit(project: Path, args: dict[str, Any]) -> str:
    rel = str(args.get("path", "")).lstrip("/")
    if rel.startswith("project/"):
        rel = rel[len("project/"):]
    target = (project / rel).resolve()
    if not str(target).startswith(str(project.resolve()) + "/"):
        return "error: path must stay inside /project"
    new = str(args.get("new_string", ""))
    old = args.get("old_string")
    if old:
        if not target.exists():
            return f"error: {rel} does not exist"
        text = target.read_text(encoding="utf-8", errors="replace")
        n = text.count(old)
        if n != 1:
            return f"error: old_string found {n} times in {rel} (must be exactly once)"
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {rel}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new, encoding="utf-8")
    return f"wrote {rel} ({len(new)} chars)"


def _to_hypotheses(raw: list[dict], audit_id: str) -> list[Hypothesis]:
    out: list[Hypothesis] = []
    for item in raw or []:
        if not isinstance(item, dict) or not str(item.get("title", "")).strip():
            continue
        sev = str(item.get("estimated_severity", "medium")).lower()
        try:
            prio = float(item.get("priority_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            prio = 0.0
        out.append(Hypothesis(
            id=str(uuid.uuid4()),
            audit_id=audit_id,
            title=str(item["title"]).strip(),
            vulnerability_class=item.get("vulnerability_class"),
            affected_functions=[str(f) for f in (item.get("affected_functions") or []) if str(f).strip()],
            attack_scenario=str(item.get("attack_scenario", "")),
            required_preconditions=[str(p) for p in (item.get("required_preconditions") or [])],
            estimated_severity=sev if sev in _SEVERITIES else "medium",  # type: ignore[arg-type]
            priority_score=prio,
        ))
    return out
