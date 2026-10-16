"""SCOUT — Intake Agent: first-pass understanding of the codebase."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from audit_agent.agents.base import BaseAgent
from audit_agent.config import get_settings
from audit_agent.llm import get_provider
from audit_agent.graph.state import AuditState
from audit_agent.knowledge.fork_detector.differ import DifferentialAnalyzer
from audit_agent.knowledge.fork_detector.fingerprints import compute_file_hash
from audit_agent.models.schemas import ForkAnalysis, ProtocolSummary, ScopeMap
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.scope import resolve_scope
from audit_agent.tools.aderyn import run_aderyn
from audit_agent.tools.errors import MockRunResult, ToolError, ToolUnavailableError
from audit_agent.tools.semgrep import run_semgrep
from audit_agent.tools.slither import run_slither
from audit_agent.tools.slither_rich import run_slither_rich

_DEPENDENCY_DIRS = frozenset(["lib", "node_modules", "dependencies", "vendor", "test", "tests", "mock", "mocks", "script", "scripts"])
# Directories whose contents are always out-of-scope (dependencies)
_OOS_DIR_PREFIXES = ("@openzeppelin", "@chainlink", "@uniswap", "solady", "solmate")
_OOS_FILE_ENDINGS = (".t.sol", ".s.sol")
_DOC_PATTERNS = ("*.md", "*.txt", "*.rst")
_DOC_IGNORE_DIRS = frozenset({"lib", "node_modules", ".git", "out", "cache", "artifacts", "build"})
_DOC_MAX_CHARS = 50_000


def _enumerate_sol_files(target_dir: str) -> tuple[list[str], list[str]]:
    root = Path(target_dir).resolve()
    in_scope: list[str] = []
    out_of_scope: list[str] = []

    # Load explicit out-of-scope list if it exists
    oos_file_path = root / "out_of_scope.txt"
    explicit_oos_files = set()

    if oos_file_path.exists():
        try:
            lines = oos_file_path.read_text().splitlines()
            for line in lines:
                clean_path = line.strip()
                if clean_path.startswith("./"):
                    clean_path = clean_path[2:]
                if clean_path:
                    explicit_oos_files.add(clean_path)
        except Exception:
            pass

    for path in sorted(root.rglob("*.sol")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_str = str(rel)
        parts = set(rel.parts[:-1])

        if rel_str in explicit_oos_files:
            out_of_scope.append(str(path))
            continue

        if (
            (parts & _DEPENDENCY_DIRS)
            or any(rel_str.startswith(p) for p in _OOS_DIR_PREFIXES)
            or any(rel_str.endswith(ext) for ext in _OOS_FILE_ENDINGS)
        ):
            out_of_scope.append(str(path))
        else:
            in_scope.append(str(path))

    return in_scope, out_of_scope


def _build_scope_map(target_dir: str, in_scope: list[str], out_of_scope: list[str]) -> ScopeMap:
    return ScopeMap(
        in_scope=in_scope,
        out_of_scope=out_of_scope,
        total_sol_files=len(in_scope) + len(out_of_scope),
        total_loc=_count_loc(in_scope),
    )


def _run_fork_detection(in_scope: list[str]) -> ForkAnalysis:
    """Compute file hashes and run DifferentialAnalyzer against known protocol fingerprints."""
    analyzer = DifferentialAnalyzer()
    hashes: dict[str, str] = {}
    for path_str in in_scope:
        path = Path(path_str)
        if path.exists():
            try:
                hashes[path.name] = compute_file_hash(path)
            except OSError:
                pass

    result = analyzer.analyze(hashes)
    if result.is_fork and result.best_match:
        m = result.best_match
        return ForkAnalysis(
            is_fork=True,
            base_protocol=m.protocol.name,
            similarity_score=m.similarity_score,
            modified_files=[mf.project_path for mf in m.modified_files],
            critical_modifications=[
                f"{mf.project_path}: {mf.modification_type}"
                for mf in m.modified_files
                if mf.modification_type in ("changed_formula", "major_rewrite")
            ],
        )
    return ForkAnalysis(is_fork=False)


def _collect_documentation(target_dir: Path) -> str:
    """Collect README, whitepapers, and other docs from the project root."""
    doc_texts: list[str] = []
    for pattern in _DOC_PATTERNS:
        for f in sorted(target_dir.rglob(pattern)):
            if any(part in _DOC_IGNORE_DIRS for part in f.parts):
                continue
            try:
                text = f.read_text(errors="ignore")
                doc_texts.append(f"--- {f.relative_to(target_dir)} ---\n{text}")
            except Exception:
                continue
    combined = "\n\n".join(doc_texts)
    return combined[:_DOC_MAX_CHARS]


def _count_loc(files: list[str]) -> int:
    total = 0
    for f in files:
        try:
            with open(f) as fh:
                total += sum(1 for line in fh if line.strip())
        except OSError:
            pass
    return total


_SCOUT_SYSTEM = """\
You are a senior smart contract security auditor performing first-pass triage.
Given a list of Solidity files, produce a concise protocol summary.

Respond ONLY with a JSON object matching this schema (no markdown fences):
{
  "name": "<protocol name or 'Unknown'>",
  "category": "<DEX|lending|bridge|staking|NFT|governance|utility|other>",
  "key_mechanisms": ["<mechanism 1>", "..."],
  "user_roles": ["<role 1>", "..."],
  "raw_notes": "<any important observations>"
}
"""


class ScoutAgent(BaseAgent):
    async def run(self, state: AuditState) -> AuditState:
        self.log.info("Scout: starting intake", target_dir=state.get("target_dir"))

        target_dir = state.get("target_dir", ".")
        audit_id = state.get("audit_id", "unknown")
        tool_mode = get_settings().tool_mode
        target_path = Path(target_dir)

        al = get_audit_logger()
        ar = al.log_agent_start("scout", 1, {
            "target_dir": target_dir,
            "audit_id": audit_id,
            "mode": state.get("mode"),
            "scope_file": state.get("scope_file"),
        }) if al else None

        # Resolve scope: explicit file from state, auto-detect, or all non-library
        scope_file_str = state.get("scope_file")
        scope_file_path = Path(scope_file_str) if scope_file_str else None
        scope_result = resolve_scope(target_path, scope_file=scope_file_path)

        in_scope = [str(p) for p in scope_result.in_scope]
        out_of_scope = [str(p) for p in scope_result.out_of_scope]
        context_files = [str(p) for p in scope_result.context_files]

        total_loc = _count_loc(in_scope)
        self.log.info(
            "Scout: found Solidity files",
            in_scope=len(in_scope),
            context=len(context_files),
            out_of_scope=len(out_of_scope),
            loc=total_loc,
            scope_source=scope_result.scope_source,
        )
        if al:
            al.log_decision("scout", "scope_resolved",
                f"{len(in_scope)} in-scope, {len(out_of_scope)} out-of-scope, {total_loc} LOC",
                {"in_scope_count": len(in_scope), "out_of_scope_count": len(out_of_scope),
                 "loc": total_loc, "scope_source": scope_result.scope_source})

        degraded_tools: list[str] = list(state.get("degraded_tools", []))
        static_findings = []

        if in_scope:
            static_findings, degraded_tools = await self._run_slither(
                target_dir, in_scope, audit_id, tool_mode, degraded_tools, al
            )
            # Best-effort full-codebase model via the Slither Python API. Writes
            # slither_rich.json to target_dir; Mapper prefers it over detector JSON.
            # Any failure is non-fatal — the detector path remains the fallback.
            await self._run_slither_rich(target_dir, audit_id)

            # Additional static analyzers (additive findings). Each degrades
            # independently and never blocks the others.
            settings = get_settings()
            if settings.aderyn_enabled:
                extra, degraded_tools = await self._run_static_tool(
                    "aderyn",
                    lambda: run_aderyn(target_dir, in_scope, audit_id),
                    target_dir, audit_id, tool_mode, degraded_tools, al,
                )
                static_findings = [*static_findings, *extra]
            if settings.semgrep_enabled:
                extra, degraded_tools = await self._run_static_tool(
                    "semgrep",
                    lambda: run_semgrep(
                        target_dir, in_scope, audit_id, settings.semgrep_config
                    ),
                    target_dir, audit_id, tool_mode, degraded_tools, al,
                )
                static_findings = [*static_findings, *extra]

        protocol_summary = ProtocolSummary(
            name="Unknown",
            category="other",
            key_mechanisms=[],
            user_roles=[],
            sol_files=in_scope,
        )
        if in_scope:
            protocol_summary = await self._summarize_protocol(in_scope, audit_id, al)

        fork_analysis = _run_fork_detection(in_scope)
        scope_map = _build_scope_map(target_dir, in_scope, out_of_scope)

        documentation = _collect_documentation(target_path)
        self.log.info(
            "Scout: documentation collected",
            chars=len(documentation),
        )

        if degraded_tools:
            self.log.warning(
                "Scout: audit running in DEGRADED mode — some tools were unavailable",
                degraded_tools=degraded_tools,
            )

        result = {
            **state,
            "protocol_summary": protocol_summary,
            "fork_analysis": fork_analysis,
            "scope_map": scope_map,
            "scope_in_scope": in_scope,
            "scope_context_files": context_files,
            "scope_source": scope_result.scope_source,
            "static_findings": static_findings,
            "degraded_tools": degraded_tools,
            "documentation": documentation,
        }

        if al and ar:
            al.log_agent_end(ar, {
                "files_in_scope": len(in_scope),
                "static_findings": len(static_findings),
                "protocol_name": protocol_summary.name,
                "protocol_category": protocol_summary.category,
                "degraded_tools": degraded_tools,
                "documentation_chars": len(documentation),
            })

        return result  # type: ignore[return-value]

    async def _run_slither(
        self,
        target_dir: str,
        in_scope: list[str],
        audit_id: str,
        tool_mode: str,
        degraded_tools: list[str],
        al=None,
    ) -> tuple[list, list[str]]:
        """Run Slither, handling mock/degraded modes at the agent level."""
        t0 = time.time()
        try:
            result = await run_slither(target_dir, in_scope, audit_id)
            duration = time.time() - t0

            if isinstance(result, MockRunResult):
                self.log.warning(
                    "Slither: running in mock mode — no real findings collected",
                    audit_id=audit_id,
                )
                if al:
                    al.log_tool_run(
                        agent="scout", tool_name="slither", purpose="static_analysis",
                        command="slither . --json slither_output.json",
                        working_dir=target_dir, exit_code=0,
                        status="mock", duration_seconds=duration,
                    )
                return [], [*degraded_tools, "slither"]

            self.log.info("Scout: Slither complete", findings=len(result))
            if al:
                al.log_tool_run(
                    agent="scout", tool_name="slither", purpose="static_analysis",
                    command="slither . --json slither_output.json",
                    working_dir=target_dir, exit_code=0,
                    status="success",
                    parsed_result={"findings_count": len(result)},
                    duration_seconds=duration,
                )
            return result, degraded_tools

        except ToolUnavailableError as exc:
            duration = time.time() - t0
            if al:
                al.log_tool_run(
                    agent="scout", tool_name="slither", purpose="static_analysis",
                    command="slither . --json slither_output.json",
                    working_dir=target_dir, exit_code=-1,
                    status="unavailable", stderr=str(exc),
                    duration_seconds=duration,
                )
            if tool_mode == "strict":
                self.log.error(
                    "Scout: Slither unavailable in strict mode — halting audit",
                    tool=exc.tool, error=str(exc),
                )
                raise
            self.log.warning(
                "Scout: Slither unavailable (degraded mode) — continuing without it",
                tool=exc.tool, error=str(exc),
            )
            return [], [*degraded_tools, exc.tool]

        except ToolError as exc:
            duration = time.time() - t0
            if al:
                al.log_tool_run(
                    agent="scout", tool_name="slither", purpose="static_analysis",
                    command="slither . --json slither_output.json",
                    working_dir=target_dir, exit_code=-1,
                    status="error", stderr=str(exc),
                    duration_seconds=duration,
                )
            if tool_mode == "strict":
                self.log.error(
                    "Scout: Slither failed in strict mode — halting audit",
                    tool=exc.tool, error=str(exc),
                )
                raise
            self.log.warning(
                "Scout: Slither failed (degraded mode) — continuing without it",
                tool=exc.tool, error=str(exc),
            )
            return [], [*degraded_tools, exc.tool]

    async def _run_static_tool(
        self,
        tool_name: str,
        runner,
        target_dir: str,
        audit_id: str,
        tool_mode: str,
        degraded_tools: list[str],
        al=None,
    ) -> tuple[list, list[str]]:
        """Run an auxiliary static analyzer (aderyn/semgrep) with the same
        mock/degraded handling as Slither. Returns (findings, degraded_tools).

        Unlike Slither, these are *additive* — a strict-mode failure still raises
        (infrastructure is broken), but mock/unavailable simply marks the tool as
        degraded so the audit keeps its Slither findings.
        """
        t0 = time.time()
        try:
            result = await runner()
            duration = time.time() - t0

            if isinstance(result, MockRunResult):
                self.log.warning(f"{tool_name}: mock mode — no real findings", audit_id=audit_id)
                if al:
                    al.log_tool_run(
                        agent="scout", tool_name=tool_name, purpose="static_analysis",
                        command=tool_name, working_dir=target_dir, exit_code=0,
                        status="mock", duration_seconds=duration,
                    )
                return [], [*degraded_tools, tool_name]

            self.log.info(f"Scout: {tool_name} complete", findings=len(result))
            if al:
                al.log_tool_run(
                    agent="scout", tool_name=tool_name, purpose="static_analysis",
                    command=tool_name, working_dir=target_dir, exit_code=0,
                    status="success", parsed_result={"findings_count": len(result)},
                    duration_seconds=duration,
                )
            return result, degraded_tools

        except (ToolUnavailableError, ToolError) as exc:
            duration = time.time() - t0
            if al:
                al.log_tool_run(
                    agent="scout", tool_name=tool_name, purpose="static_analysis",
                    command=tool_name, working_dir=target_dir, exit_code=-1,
                    status="error", stderr=str(exc), duration_seconds=duration,
                )
            if tool_mode == "strict":
                self.log.error(
                    f"Scout: {tool_name} unavailable in strict mode — halting audit",
                    tool=getattr(exc, "tool", tool_name), error=str(exc),
                )
                raise
            self.log.warning(
                f"Scout: {tool_name} unavailable (degraded mode) — continuing without it",
                tool=getattr(exc, "tool", tool_name), error=str(exc),
            )
            return [], [*degraded_tools, tool_name]

    async def _run_slither_rich(self, target_dir: str, audit_id: str) -> None:
        """Best-effort full-codebase Slither-API analysis. Never raises."""
        t0 = time.time()
        try:
            result = await run_slither_rich(target_dir, audit_id)
        except Exception as exc:
            self.log.warning("Scout: rich Slither analysis errored", error=str(exc))
            return
        if isinstance(result, MockRunResult):
            self.log.info("Scout: rich Slither analysis skipped (mock mode)")
            return
        if not result:
            self.log.info(
                "Scout: rich Slither analysis unavailable — Mapper will use detector JSON"
            )
            return
        n_fns = sum(len(c.get("functions", [])) for c in result.get("contracts", []))
        self.log.info(
            "Scout: rich Slither analysis complete",
            contracts=len(result.get("contracts", [])),
            functions=n_fns,
            duration_s=round(time.time() - t0, 1),
        )

    @staticmethod
    def _heuristic_summary(sol_files: list[Path]) -> ProtocolSummary:
        """Determine protocol type from contract/function names — no LLM required."""
        all_names = " ".join(f.stem.lower() for f in sol_files)

        category = "utility"
        if any(k in all_names for k in ["swap", "pool", "amm", "pair"]):
            category = "DEX"
        elif any(k in all_names for k in ["lend", "borrow", "vault", "aave"]):
            category = "lending"
        elif any(k in all_names for k in ["bridge", "l1", "l2", "cross"]):
            category = "bridge"
        elif any(k in all_names for k in ["stake", "staking", "validator"]):
            category = "staking"
        elif any(k in all_names for k in ["govern", "vote", "proposal"]):
            category = "governance"
        elif any(k in all_names for k in ["nft", "erc721", "erc1155"]):
            category = "NFT"

        return ProtocolSummary(
            name="Unknown",
            category=category,
            key_mechanisms=[],
            user_roles=["user", "admin"],
            sol_files=[str(f) for f in sol_files],
        )

    async def _summarize_protocol(
        self, sol_files: list[str], audit_id: str, al=None
    ) -> ProtocolSummary:
        import json

        sol_paths = [Path(f) for f in sol_files]
        settings = get_settings()
        provider = get_provider()

        file_sections: list[str] = []
        for f in sol_files[:10]:
            try:
                lines = Path(f).read_text(errors="ignore").splitlines()
                snippet = "\n".join(lines[:200])
                file_sections.append(f"// === {Path(f).name} ===\n{snippet}")
            except OSError:
                pass

        user_content = "Solidity source files in scope:\n\n" + "\n\n".join(file_sections)
        user_content = user_content[:15000]

        llm_record = None
        if al:
            llm_record = al.log_llm_call_start(
                agent="scout", purpose="summarize_protocol",
                model=settings.scout_model,
                system_prompt=_SCOUT_SYSTEM, user_message=user_content,
                max_tokens=4096,
            )

        t0 = time.time()
        try:
            resp = await asyncio.wait_for(
                provider.complete(
                    model=settings.scout_model,
                    max_tokens=4096,
                    system=_SCOUT_SYSTEM,
                    messages=[{"role": "user", "content": user_content}],
                ),
                timeout=300.0,
            )
            duration = resp.duration_seconds or (time.time() - t0)
            raw = resp.text

            if al and llm_record:
                al.log_llm_call_end(
                    llm_record,
                    response_text=raw,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    stop_reason=resp.stop_reason,
                    duration_seconds=duration,
                )

            if not raw.strip():
                self.log.warning("Scout: LLM returned empty response, using heuristic fallback")
                return self._heuristic_summary(sol_paths)
            # Strip markdown code fences if LLM wrapped JSON in ```json ... ```
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean[clean.find("\n") + 1:]
                end = clean.rfind("```")
                if end != -1:
                    clean = clean[:end]
                clean = clean.strip()
            data = json.loads(clean)
            return ProtocolSummary(
                name=data.get("name", "Unknown"),
                category=data.get("category", "other"),
                key_mechanisms=data.get("key_mechanisms", []),
                user_roles=data.get("user_roles", []),
                sol_files=sol_files,
                raw_notes=data.get("raw_notes", ""),
            )
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            duration = time.time() - t0
            if al and llm_record:
                al.log_llm_call_end(llm_record, error=str(exc), duration_seconds=duration)
            self.log.warning("Scout: LLM summarization failed", error=str(exc))
            return self._heuristic_summary(sol_paths)
        except Exception as exc:
            duration = time.time() - t0
            if al and llm_record:
                al.log_llm_call_end(llm_record, error=str(exc), duration_seconds=duration)
            self.log.warning("Scout: LLM summarization failed", error=str(exc))
            return self._heuristic_summary(sol_paths)
