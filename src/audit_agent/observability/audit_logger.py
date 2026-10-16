"""
Detailed audit logging for post-run analysis.

Captures every LLM call (full prompt + response), every tool execution
(command + stdout + stderr + exit code), and every agent decision with
its inputs/outputs. All data written to structured files under
{target_dir}/audit_logs/{audit_id}/.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from audit_agent.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMCallRecord:
    """Single LLM API call with full request and response."""
    call_id: str
    agent: str
    purpose: str
    model: str
    timestamp: str = ""
    duration_seconds: float = 0.0

    # Request
    system_prompt: str = ""
    user_message: str = ""
    tools: list = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 0

    # Response
    response_text: str = ""
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    cost_usd: float = 0.0
    error: str = ""


@dataclass
class ToolRunRecord:
    """Single tool/Docker execution."""
    run_id: str
    agent: str
    tool_name: str
    purpose: str
    timestamp: str = ""
    duration_seconds: float = 0.0

    command: str = ""
    working_dir: str = ""
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""

    poc_code: str = ""
    hypothesis_id: str = ""
    attempt_number: int = 0

    status: str = ""
    parsed_result: dict = field(default_factory=dict)


@dataclass
class AgentRecord:
    """Full agent execution record."""
    agent: str
    phase_number: int
    timestamp_start: str = ""
    timestamp_end: str = ""
    duration_seconds: float = 0.0

    input_summary: dict = field(default_factory=dict)
    output_summary: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    llm_call_ids: list = field(default_factory=list)
    tool_run_ids: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class AuditLogger:
    """
    Per-audit logger that writes structured logs to disk.

    All methods are safe to call even if initialization fails — errors
    are caught and logged via structlog but never propagate to callers.
    """

    def __init__(self, target_dir: Path, audit_id: str) -> None:
        self.audit_id = audit_id
        self.log_dir = Path(target_dir) / "audit_logs" / audit_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "agents").mkdir(exist_ok=True)
        (self.log_dir / "llm_calls").mkdir(exist_ok=True)
        (self.log_dir / "tools").mkdir(exist_ok=True)
        (self.log_dir / "poc").mkdir(exist_ok=True)

        self._llm_call_counter = 0
        self._tool_run_counter = 0
        self._start_time = time.time()
        self._total_cost_usd = 0.0
        self._active_llm_call_ids: list[str] = []
        self._active_tool_run_ids: list[str] = []
        self._active_decisions: list[dict] = []

        self._audit_log = open(self.log_dir / "audit.log", "a", encoding="utf-8")
        self._jsonl = open(self.log_dir / "audit.jsonl", "a", encoding="utf-8")

        self._log_event("audit_start", {
            "audit_id": audit_id,
            "target_dir": str(target_dir),
            "timestamp": _now(),
        })

    def log_llm_call_start(
        self,
        agent: str,
        purpose: str,
        model: str,
        system_prompt: str,
        user_message: str,
        tools: Optional[list] = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
    ) -> LLMCallRecord:
        """Log the start of an LLM call. Returns record to update with response later."""
        self._llm_call_counter += 1
        call_id = f"{self._llm_call_counter:03d}"

        record = LLMCallRecord(
            call_id=call_id,
            agent=agent,
            purpose=purpose,
            model=model,
            timestamp=_now(),
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools or [],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._active_llm_call_ids.append(call_id)
        self._log_event("llm_call_start", {
            "call_id": call_id,
            "agent": agent,
            "purpose": purpose,
            "model": model,
            "system_prompt_length": len(system_prompt),
            "user_message_length": len(user_message),
            "tools_count": len(tools or []),
        })

        self._write_text(
            f"\n{'='*80}\n"
            f"LLM CALL #{call_id} | {agent} | {purpose} | {model}\n"
            f"{'='*80}\n"
            f"--- SYSTEM PROMPT ({len(system_prompt)} chars) ---\n"
            f"{_truncate(system_prompt, 5000)}\n"
            f"--- USER MESSAGE ({len(user_message)} chars) ---\n"
            f"{_truncate(user_message, 10000)}\n"
            f"--- TOOLS: {len(tools or [])} defined ---\n"
        )

        return record

    def log_llm_call_end(
        self,
        record: LLMCallRecord,
        response_text: str = "",
        tool_calls: Optional[list] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str = "",
        cost_usd: float = 0.0,
        error: str = "",
        duration_seconds: float = 0.0,
    ) -> None:
        """Update LLM call record with response and save to disk."""
        if cost_usd == 0.0 and (input_tokens > 0 or output_tokens > 0):
            cost_usd = _estimate_cost(record.model, input_tokens, output_tokens)
        self._total_cost_usd += cost_usd

        record.response_text = response_text
        record.tool_calls = tool_calls or []
        record.input_tokens = input_tokens
        record.output_tokens = output_tokens
        record.stop_reason = stop_reason
        record.cost_usd = cost_usd
        record.error = error
        record.duration_seconds = duration_seconds

        filename = f"{record.call_id}_{record.agent}_{_safe_filename(record.purpose)}.json"
        self._write_json(self.log_dir / "llm_calls" / filename, asdict(record))

        self._log_event("llm_call_end", {
            "call_id": record.call_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "stop_reason": stop_reason,
            "duration_seconds": duration_seconds,
            "error": error[:500] if error else "",
            "response_length": len(response_text),
            "tool_calls_count": len(tool_calls or []),
        })

        self._write_text(
            f"--- RESPONSE ({len(response_text)} chars, "
            f"{input_tokens}+{output_tokens} tokens, "
            f"${cost_usd:.4f}, {duration_seconds:.1f}s, "
            f"stop={stop_reason}) ---\n"
            f"{_truncate(response_text, 10000)}\n"
        )
        if tool_calls:
            self._write_text(f"--- TOOL CALLS: {len(tool_calls)} ---\n")
            for tc in tool_calls:
                self._write_text(f"  {json.dumps(tc, indent=2, default=str)[:3000]}\n")
        if error:
            self._write_text(f"--- ERROR ---\n{error}\n")

    def log_tool_run(
        self,
        agent: str,
        tool_name: str,
        purpose: str,
        command: str = "",
        working_dir: str = "",
        exit_code: int = -1,
        stdout: str = "",
        stderr: str = "",
        poc_code: str = "",
        hypothesis_id: str = "",
        attempt_number: int = 0,
        status: str = "",
        parsed_result: Optional[dict] = None,
        duration_seconds: float = 0.0,
    ) -> ToolRunRecord:
        """Log a complete tool/Docker execution."""
        self._tool_run_counter += 1
        run_id = f"tool_{self._tool_run_counter:03d}"
        self._active_tool_run_ids.append(run_id)

        record = ToolRunRecord(
            run_id=run_id,
            agent=agent,
            tool_name=tool_name,
            purpose=purpose,
            timestamp=_now(),
            duration_seconds=duration_seconds,
            command=command,
            working_dir=working_dir,
            exit_code=exit_code,
            stdout=stdout[:50000],
            stderr=stderr[:50000],
            poc_code=poc_code,
            hypothesis_id=hypothesis_id,
            attempt_number=attempt_number,
            status=status,
            parsed_result=parsed_result or {},
        )

        filename = f"{run_id}_{tool_name}_{_safe_filename(purpose)}.json"
        self._write_json(self.log_dir / "tools" / filename, asdict(record))

        if poc_code:
            poc_filename = f"{hypothesis_id or 'unknown'}"
            if attempt_number > 0:
                poc_filename += f"_attempt{attempt_number}"
            poc_filename += ".sol"
            (self.log_dir / "poc" / poc_filename).write_text(poc_code, encoding="utf-8")

        self._log_event("tool_run", {
            "run_id": run_id,
            "agent": agent,
            "tool_name": tool_name,
            "purpose": purpose,
            "exit_code": exit_code,
            "status": status,
            "duration_seconds": duration_seconds,
            "stdout_length": len(stdout),
            "stderr_length": len(stderr),
            "hypothesis_id": hypothesis_id,
            "attempt_number": attempt_number,
        })

        self._write_text(
            f"\n{'─'*60}\n"
            f"TOOL RUN #{run_id} | {tool_name} | {purpose}\n"
            f"{'─'*60}\n"
            f"Command: {command}\n"
            f"Exit code: {exit_code} | Status: {status} | Duration: {duration_seconds:.1f}s\n"
        )
        if poc_code:
            self._write_text(f"--- PoC CODE ({len(poc_code)} chars) ---\n{_truncate(poc_code, 5000)}\n")
        if stdout:
            self._write_text(f"--- STDOUT ({len(stdout)} chars) ---\n{_truncate(stdout, 5000)}\n")
        if stderr:
            self._write_text(f"--- STDERR ({len(stderr)} chars) ---\n{_truncate(stderr, 3000)}\n")

        return record

    def log_agent_start(
        self,
        agent: str,
        phase_number: int,
        input_summary: dict,
    ) -> AgentRecord:
        """Log agent start. Returns record to finalize at end."""
        self._active_llm_call_ids = []
        self._active_tool_run_ids = []
        self._active_decisions = []

        record = AgentRecord(
            agent=agent,
            phase_number=phase_number,
            timestamp_start=_now(),
            input_summary=input_summary,
        )

        self._log_event("agent_start", {
            "agent": agent,
            "phase": phase_number,
            "input_keys": list(input_summary.keys()),
        })

        self._write_text(
            f"\n{'#'*80}\n"
            f"AGENT: {agent} (Phase {phase_number})\n"
            f"{'#'*80}\n"
            f"Input summary:\n{json.dumps(input_summary, indent=2, default=str)[:5000]}\n"
        )

        return record

    def log_agent_end(
        self,
        record: AgentRecord,
        output_summary: dict,
        decisions: Optional[list] = None,
        llm_call_ids: Optional[list] = None,
        tool_run_ids: Optional[list] = None,
        warnings: Optional[list] = None,
        errors: Optional[list] = None,
    ) -> None:
        """Log agent completion and save full record."""
        record.timestamp_end = _now()
        record.duration_seconds = _elapsed(record.timestamp_start)
        record.output_summary = output_summary
        record.decisions = decisions or list(self._active_decisions)
        record.llm_call_ids = llm_call_ids or list(self._active_llm_call_ids)
        record.tool_run_ids = tool_run_ids or list(self._active_tool_run_ids)
        record.warnings = warnings or []
        record.errors = errors or []

        filename = f"{record.phase_number:02d}_{record.agent}.json"
        self._write_json(self.log_dir / "agents" / filename, asdict(record))

        self._log_event("agent_end", {
            "agent": record.agent,
            "phase": record.phase_number,
            "duration_seconds": record.duration_seconds,
            "llm_calls": len(record.llm_call_ids),
            "tool_runs": len(record.tool_run_ids),
            "output_keys": list(output_summary.keys()),
            "warnings_count": len(record.warnings),
            "errors_count": len(record.errors),
        })

        self._write_text(
            f"\nAgent {record.agent} completed in {record.duration_seconds:.1f}s\n"
            f"LLM calls: {len(record.llm_call_ids)}, Tool runs: {len(record.tool_run_ids)}\n"
            f"Decisions: {json.dumps(decisions or [], indent=2, default=str)[:3000]}\n"
            f"Output summary:\n{json.dumps(output_summary, indent=2, default=str)[:5000]}\n"
        )
        if warnings:
            self._write_text(f"Warnings: {warnings}\n")
        if errors:
            self._write_text(f"ERRORS: {errors}\n")

    def log_decision(
        self,
        agent: str,
        decision: str,
        reason: str,
        data: Optional[dict] = None,
    ) -> None:
        """Log a specific decision point."""
        self._active_decisions.append({
            "decision": decision,
            "reason": reason,
            "data": data or {},
        })
        self._log_event("decision", {
            "agent": agent,
            "decision": decision,
            "reason": reason,
            "data": data or {},
        })
        self._write_text(f"  DECISION [{agent}]: {decision} — {reason}\n")

    def write_summary(self, final_state: dict) -> None:
        """Write final audit summary and close log files."""
        total_duration = time.time() - self._start_time

        summary = {
            "audit_id": self.audit_id,
            "total_duration_seconds": total_duration,
            "total_llm_calls": self._llm_call_counter,
            "total_tool_runs": self._tool_run_counter,
            "timestamp_start": datetime.fromtimestamp(
                self._start_time, tz=timezone.utc
            ).isoformat(),
            "timestamp_end": _now(),
            "scope_files_count": len(final_state.get("scope_in_scope") or []),
            "hypotheses_count": len(final_state.get("hypotheses") or []),
            "exploits_count": len(final_state.get("exploits") or []),
            "validated_findings_count": len(final_state.get("validated_findings") or []),
            "findings_by_verdict": _count_by_field(
                final_state.get("validated_findings") or [], "status"
            ),
            "reports_count": len(final_state.get("reports") or {}),
            "tokens_used": final_state.get("tokens_used", 0),
            "cost_usd": self._total_cost_usd or final_state.get("cost_usd", 0.0),
            "degraded_tools": final_state.get("degraded_tools") or [],
            "errors": [str(e) for e in (final_state.get("errors") or [])],
        }

        self._write_json(self.log_dir / "summary.json", summary)

        self._write_text(
            f"\n{'='*80}\n"
            f"AUDIT COMPLETE\n"
            f"{'='*80}\n"
            f"{json.dumps(summary, indent=2, default=str)}\n"
        )

        self._audit_log.close()
        self._jsonl.close()

        logger.info("Audit logs written", log_dir=str(self.log_dir))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: dict) -> None:
        event = {"ts": _now(), "event": event_type, **data}
        self._jsonl.write(json.dumps(event, default=str) + "\n")
        self._jsonl.flush()

    def _write_text(self, text: str) -> None:
        self._audit_log.write(text)
        self._audit_log.flush()

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_current_logger: Optional[AuditLogger] = None


def set_audit_logger(al: Optional[AuditLogger]) -> None:
    global _current_logger
    _current_logger = al


def get_audit_logger() -> Optional[AuditLogger]:
    return _current_logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(start_iso: str) -> float:
    start = datetime.fromisoformat(start_iso)
    return (datetime.now(timezone.utc) - start).total_seconds()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 50
    return (
        text[:half]
        + f"\n\n... [{len(text) - max_len} chars truncated] ...\n\n"
        + text[-half:]
    )


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:60]


_MODEL_COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts and model pricing."""
    rates = _MODEL_COST_PER_MILLION.get(model)
    if not rates:
        for key, val in _MODEL_COST_PER_MILLION.items():
            if key in model:
                rates = val
                break
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def _count_by_field(items: list, field_name: str) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            val = item.get(field_name, "unknown")
        else:
            val = getattr(item, field_name, "unknown")
        counts[str(val)] = counts.get(str(val), 0) + 1
    return counts
