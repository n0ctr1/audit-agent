"""Agentic codebase navigator for Hunter (roadmap piece #3).

Instead of feeding the LLM a fixed, truncated code slice and asking once, the
navigator lets the model *investigate a lead* the way a senior auditor does:
pull exactly the functions/files it wants, follow callers/callees and state
writers across the whole codebase, then report hypotheses when it has enough
evidence.

Transport: a provider-agnostic ReAct loop. Every step the model is FORCED to
call the single `investigate` tool (structured output — robust for thinking
models). Tool observations are fed back as plain `user` messages, which is the
only multi-turn shape both the OpenAI and Anthropic providers support through
the shared `complete()` interface (neither round-trips native tool_result here).

The navigator's tools are READ-ONLY (no code execution) and operate over the
in-scope source files plus the rich SlitherAnalysis (call graph, taint,
external-call targets).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

from audit_agent.knowledge.graph.builders import SlitherAnalysis, SlitherFunction
from audit_agent.knowledge.graph.context import InMemoryGraph
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.navigator")


# ─────────────────────────────────────────────────────────────────────────────
# The single structured tool the model uses every step.
# ─────────────────────────────────────────────────────────────────────────────

_HYPOTHESIS_ITEM_SCHEMA = {
    "type": "object",
    "required": ["title", "vulnerability_class", "affected_functions",
                 "attack_scenario", "estimated_severity", "priority_score"],
    "properties": {
        "title": {"type": "string"},
        "vulnerability_class": {"type": "string", "description": "SWC-ID or class name"},
        "affected_functions": {"type": "array", "items": {"type": "string"}},
        "attack_scenario": {"type": "string", "description": "Step-by-step, concrete"},
        "required_preconditions": {"type": "array", "items": {"type": "string"}},
        "estimated_severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "priority_score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
    },
}

INVESTIGATE_TOOL = ToolDefinition(
    name="investigate",
    description=(
        "Take ONE step of a security investigation. Either call a read-only "
        "navigation tool to gather more evidence, or finish with action='report' "
        "and your vulnerability hypotheses. Call this exactly once per turn."
    ),
    parameters={
        "type": "object",
        "required": ["thought", "action"],
        "properties": {
            "thought": {
                "type": "string",
                "description": "Brief reasoning: what you suspect and why this step.",
            },
            "action": {
                "type": "string",
                "enum": [
                    "read_function", "read_file", "grep", "get_callers",
                    "get_callees", "get_state_writers", "list_functions",
                    "get_function_facts", "report",
                ],
            },
            "target": {
                "type": "string",
                "description": (
                    "Argument for the action: a function name (read_function, "
                    "get_callers, get_callees, get_function_facts), a file path "
                    "(read_file), a regex (grep), a state-variable name "
                    "(get_state_writers), or a contract name (list_functions). "
                    "Empty for report / list_functions(all)."
                ),
            },
            "start_line": {"type": "integer", "description": "read_file: optional start line"},
            "end_line": {"type": "integer", "description": "read_file: optional end line"},
            "hypotheses": {
                "type": "array",
                "items": _HYPOTHESIS_ITEM_SCHEMA,
                "description": "Only when action='report': the vulnerabilities found.",
            },
        },
    },
)

NAVIGATOR_SYSTEM = """You are a world-class smart-contract security auditor investigating ONE lead.

You do NOT see the whole codebase at once. Instead you pull exactly what you need
using read-only tools, the way a senior auditor navigates a repo:

- read_function <name>      — full source of a function
- get_function_facts <name> — visibility, state reads/writes, TAINTED writes,
                              external-call targets (which state var holds the callee),
                              attack-surface score
- get_callers <name>        — who can reach this function
- get_callees <name>        — what this function calls
- get_state_writers <var>   — every function (any contract) that writes a state variable
- grep <regex>              — search all in-scope source
- read_file <path>          — a file (optionally start_line/end_line)
- list_functions <contract?> — enumerate functions (with scores/flags)

METHOD:
1. Start from the lead. Understand what the code is SUPPOSED to do and its invariants.
2. FOLLOW THE DATA. If a value is user-tainted and feeds accounting/transfers/pricing,
   trace where it goes (get_callees, get_state_writers, grep). If a function reads a
   state var, find who writes it and whether that write can be attacker-influenced.
3. THINK IN MULTI-STEP / CROSS-FUNCTION terms: a bug is often a CHAIN — function A
   sets state that function B trusts. Pull both before concluding.
4. Challenge every trust assumption and every arithmetic sign/rounding/boundary.
5. Stop gathering once you can state a concrete, exploitable scenario — then report.

RULES:
- Every turn: call `investigate` exactly once. Output ONLY the tool call.
- Prefer FEWER, DEEPER findings. Do not report generic/style/gas issues.
- Do not report bugs that require trusted-admin cooperation.
- When you report, each hypothesis MUST have a concrete step-by-step attack scenario
  and name the exact functions/variables involved. priority_score 0-100.
- If after investigating you find nothing credible, action='report' with an empty
  hypotheses list. Do not invent issues.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Read-only tools over the codebase + SlitherAnalysis
# ─────────────────────────────────────────────────────────────────────────────

class CodebaseNavigator:
    """Read-only navigation tools. Pure (no LLM, no code execution)."""

    def __init__(
        self,
        analysis: SlitherAnalysis,
        in_scope_files: list[str],
        target_dir: str = "",
        per_obs_chars: int = 6000,
    ) -> None:
        self._analysis = analysis
        self._graph = InMemoryGraph(analysis)
        self._target_dir = target_dir
        self._per_obs = per_obs_chars
        # Resolve in-scope files to readable paths once.
        self._files: list[Path] = []
        for f in in_scope_files:
            p = Path(f)
            if not p.exists() and target_dir:
                p = Path(target_dir) / f.lstrip("./")
            if p.exists():
                self._files.append(p)
        # Index functions by (lower) name for quick lookup.
        self._by_name: dict[str, list[SlitherFunction]] = {}
        self._contract_of: dict[str, str] = {}
        for c in analysis.contracts:
            for fn in c.functions:
                self._by_name.setdefault(fn.name.lower(), []).append(fn)
                self._contract_of[fn.id] = c.name

    # ── dispatch ──────────────────────────────────────────────────────────
    def run_tool(self, action: str, target: str = "", start_line: int | None = None,
                 end_line: int | None = None) -> str:
        try:
            if action == "read_function":
                return self.read_function(target)
            if action == "get_function_facts":
                return self.get_function_facts(target)
            if action == "get_callers":
                return self.get_callers(target)
            if action == "get_callees":
                return self.get_callees(target)
            if action == "get_state_writers":
                return self.get_state_writers(target)
            if action == "grep":
                return self.grep(target)
            if action == "read_file":
                return self.read_file(target, start_line, end_line)
            if action == "list_functions":
                return self.list_functions(target or None)
            return f"(unknown action '{action}')"
        except Exception as exc:  # never let a tool crash the loop
            return f"(tool error in {action}: {exc})"

    # ── helpers ───────────────────────────────────────────────────────────
    def _find_fn(self, name: str) -> Optional[SlitherFunction]:
        if not name:
            return None
        key = name.strip().split("(")[0].lower()
        cands = self._by_name.get(key)
        if cands:
            # prefer the highest attack-surface one if overloaded
            return max(cands, key=lambda f: f.attack_surface_score)
        # maybe an id was passed
        for c in self._analysis.contracts:
            for fn in c.functions:
                if fn.id == name:
                    return fn
        return None

    def _read_fn_source(self, fn: SlitherFunction) -> str:
        if fn.source_code:
            return fn.source_code
        loc = fn.source_location
        if loc and loc.file and loc.start_line and loc.end_line:
            for cand in self._candidate_paths(loc.file):
                try:
                    lines = cand.read_text(errors="ignore").splitlines()
                    return "\n".join(lines[loc.start_line - 1: loc.end_line])
                except OSError:
                    continue
        return ""

    def _candidate_paths(self, file_str: str) -> list[Path]:
        cands = [Path(file_str)]
        if self._target_dir:
            cands.append(Path(self._target_dir) / file_str)
            if file_str.startswith("/target/"):
                cands.append(Path(self._target_dir) / file_str[len("/target/"):])
        return cands

    # ── tools ─────────────────────────────────────────────────────────────
    def read_function(self, name: str) -> str:
        fn = self._find_fn(name)
        if fn is None:
            return f"(function '{name}' not found — try list_functions or grep)"
        src = self._read_fn_source(fn)
        contract = self._contract_of.get(fn.id, "?")
        header = f"// {contract}.{fn.signature} [{fn.visibility} {fn.state_mutability}]"
        if not src:
            return header + "\n(source not available)"
        return (header + "\n" + src)[: self._per_obs]

    def get_function_facts(self, name: str) -> str:
        fn = self._find_fn(name)
        if fn is None:
            return f"(function '{name}' not found)"
        contract = self._contract_of.get(fn.id, "?")
        targets = ", ".join(
            f"{t.get('dest')}.{t.get('fn')}" for t in (fn.external_targets or [])
        ) or "none"
        return (
            f"{contract}.{fn.signature}\n"
            f"  visibility: {fn.visibility} {fn.state_mutability}\n"
            f"  attack_surface_score: {fn.attack_surface_score:.1f}\n"
            f"  reads: {[r.split('-')[-1] for r in fn.reads]}\n"
            f"  writes: {[w.split('-')[-1] for w in fn.writes]}\n"
            f"  TAINTED writes (user-influenced): {fn.tainted_state_writes or 'none'}\n"
            f"  external call targets: {targets}\n"
            f"  flags: external_calls={fn.has_external_calls} handles_value={fn.handles_value} "
            f"reads_oracle={fn.reads_oracle} privileged={fn.is_privileged}"
        )

    def get_callers(self, name: str) -> str:
        fn = self._find_fn(name)
        if fn is None:
            return f"(function '{name}' not found)"
        callers = self._graph._callers_of.get(fn.id, set())
        if not callers:
            return f"({fn.name} has no internal callers in scope — likely an entrypoint)"
        rows = []
        for cid in sorted(callers):
            cf = self._graph._fns.get(cid)
            if cf:
                rows.append(f"  {self._contract_of.get(cid,'?')}.{cf.name} [{cf.visibility}]")
        return f"Callers of {fn.name}:\n" + "\n".join(rows)

    def get_callees(self, name: str) -> str:
        fn = self._find_fn(name)
        if fn is None:
            return f"(function '{name}' not found)"
        rows = []
        for cid in fn.calls:
            cf = self._graph._fns.get(cid)
            label = f"{cf.name} [{cf.visibility}]" if cf else cid
            rows.append(f"  {label}")
        ext = [f"  (external) {t.get('dest')}.{t.get('fn')}" for t in (fn.external_targets or [])]
        body = "\n".join(rows + ext) or "  (no calls)"
        return f"{fn.name} calls:\n" + body

    def get_state_writers(self, var: str) -> str:
        if not var:
            return "(provide a state variable name)"
        key = var.strip().lower()
        rows = []
        for c in self._analysis.contracts:
            for fn in c.functions:
                for w in fn.writes:
                    if w.split("-")[-1].lower() == key:
                        taint = " [TAINTED]" if var in (fn.tainted_state_writes or []) else ""
                        rows.append(f"  {c.name}.{fn.name}{taint}")
        if not rows:
            return f"(no in-scope writers found for '{var}')"
        return f"Writers of state var '{var}':\n" + "\n".join(sorted(set(rows)))

    def grep(self, pattern: str, max_results: int = 40) -> str:
        if not pattern:
            return "(provide a regex)"
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"(invalid regex: {exc})"
        hits: list[str] = []
        for p in self._files:
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"  {p.name}:{i}: {line.strip()[:160]}")
                        if len(hits) >= max_results:
                            return f"grep '{pattern}' (truncated at {max_results}):\n" + "\n".join(hits)
            except OSError:
                continue
        return f"grep '{pattern}':\n" + ("\n".join(hits) if hits else "  (no matches)")

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        if not path:
            return "(provide a file path)"
        target: Optional[Path] = None
        base = path.split("/")[-1]
        for p in self._files:
            if str(p) == path or p.name == base or str(p).endswith(path):
                target = p
                break
        if target is None:
            for c in self._candidate_paths(path):
                if c.exists():
                    target = c
                    break
        if target is None:
            return f"(file '{path}' not in scope — use list_functions/grep)"
        try:
            lines = target.read_text(errors="ignore").splitlines()
        except OSError as exc:
            return f"(cannot read {path}: {exc})"
        if start:
            lines = lines[max(0, start - 1): (end or start + 120)]
        body = "\n".join(lines)
        return f"// {target.name}\n{body[: self._per_obs]}"

    def list_functions(self, contract: str | None = None) -> str:
        rows = []
        for c in self._analysis.contracts:
            if contract and c.name.lower() != contract.lower():
                continue
            for fn in c.functions:
                if fn.visibility in ("external", "public") or fn.attack_surface_score > 0:
                    rows.append((fn.attack_surface_score, f"  {c.name}.{fn.name} "
                                 f"[{fn.visibility}] score={fn.attack_surface_score:.1f}"
                                 f"{' $' if fn.handles_value else ''}"
                                 f"{' oracle' if fn.reads_oracle else ''}"))
        rows.sort(key=lambda t: t[0], reverse=True)
        listing = "\n".join(r for _, r in rows[:60])
        return (f"Functions{' in ' + contract if contract else ''} "
                f"(by attack surface):\n{listing}") or "(no functions)"


# ─────────────────────────────────────────────────────────────────────────────
# The agentic ReAct session
# ─────────────────────────────────────────────────────────────────────────────

class NavigatorSession:
    """Runs one bounded investigation of a single lead, returning raw hypotheses."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        navigator: CodebaseNavigator,
        max_steps: int = 6,
        max_tokens: int = 4096,
        step_timeout: float = 180.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._nav = navigator
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._step_timeout = step_timeout

    async def investigate(self, lead_prompt: str, audit_id: str = "") -> list[dict]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": lead_prompt}]
        al = get_audit_logger()

        for step in range(self._max_steps):
            force_report = step == self._max_steps - 1
            args = await self._step(messages, force_report)
            if args is None:
                break

            action = (args.get("action") or "").strip()
            if action == "report" or force_report:
                hyps = args.get("hypotheses") or []
                if al:
                    al.log_decision("navigator", "report",
                        f"lead reported {len(hyps)} hypotheses after {step + 1} steps",
                        {"steps": step + 1, "count": len(hyps)})
                return hyps if isinstance(hyps, list) else []

            target = args.get("target", "") or ""
            obs = self._nav.run_tool(
                action, target,
                args.get("start_line"), args.get("end_line"),
            )
            if al:
                al.log_decision("navigator", "tool",
                    f"{action}({target[:40]}) -> {len(obs)} chars",
                    {"action": action, "target": target[:80]})

            # Record a compact transcript so the model keeps its chain of reasoning.
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": (args.get("thought") or "")[:500],
                    "action": action, "target": target,
                }),
            })
            messages.append({
                "role": "user",
                "content": f"OBSERVATION ({action}):\n{obs}\n\n"
                           "Continue investigating, or set action='report' when you can "
                           "state a concrete exploit (or confirm none).",
            })

        return []

    async def _step(self, messages: list[dict], force_report: bool) -> Optional[dict]:
        system = NAVIGATOR_SYSTEM
        if force_report:
            system += (
                "\n\nYOU HAVE REACHED YOUR INVESTIGATION BUDGET. You MUST now call "
                "investigate with action='report' and your final hypotheses "
                "(empty list if nothing credible)."
            )
        try:
            resp = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    tools=[INVESTIGATE_TOOL],
                    tool_choice={"name": "investigate"},
                    messages=messages,
                ),
                timeout=self._step_timeout,
            )
        except Exception as exc:
            log.warning("navigator.step_failed", error=str(exc))
            return None
        for tc in resp.tool_calls:
            if tc.name == "investigate":
                return tc.arguments
        return None
