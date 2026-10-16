"""MAPPER agent — builds Neo4j code graph and mines invariants (§3.2).

Orchestrates:
  1. Slither analysis (real Docker or injected fixture JSON)
  2. Graph construction via GraphBuilder → GraphStore
  3. Attack surface scoring (embedded in builders.py)
  4. Invariant mining (NatSpec + LLM docs + LLM inferred)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.agents.invariant_miner import InvariantMiner, extract_natspec_invariants
from audit_agent.config import get_settings
from audit_agent.graph.state import AuditState
from audit_agent.knowledge.graph.builders import (
    BuildStats,
    GraphBuilder,
    GraphStore,
    InMemoryGraphStore,
    SlitherAnalysis,
    parse_slither_json,
)
from audit_agent.models.schemas import Invariant
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.mapper")


@dataclass
class MapperResult:
    """Summary returned by the mapper for CLI display and state updates."""

    contracts: int = 0
    functions: int = 0
    relationships: int = 0
    state_vars: int = 0
    invariants_natspec: int = 0
    invariants_docs: int = 0
    invariants_inferred: int = 0
    top_surface_functions: list[dict[str, Any]] = field(default_factory=list)
    graph_id: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def total_invariants(self) -> int:
        return self.invariants_natspec + self.invariants_docs + self.invariants_inferred


class MapperAgent(BaseAgent):
    """Parses contracts, builds code graph, extracts invariants."""

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        invariant_miner: Optional[InvariantMiner] = None,
        slither_json: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # If no store provided, use InMemoryGraphStore (runs without Neo4j)
        self._graph_store: GraphStore = graph_store or InMemoryGraphStore()
        self._miner = invariant_miner or InvariantMiner()
        # Optional injected Slither output (for tests / offline mode)
        self._slither_json = slither_json

    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        target_dir = state.get("target_dir", ".")
        docs_dir = state.get("docs_dir")

        self.log.info("Mapper: starting", audit_id=audit_id, target_dir=target_dir)

        al = get_audit_logger()
        ar = al.log_agent_start("mapper", 2, {
            "audit_id": audit_id,
            "target_dir": target_dir,
            "docs_dir": docs_dir,
            "scope_files": len(state.get("scope_in_scope") or []),
        }) if al else None

        # Prefer the new scope_in_scope field (set by Scout via scope.py);
        # fall back to scope_map.in_scope for backward compatibility.
        scope_map = state.get("scope_map")
        in_scope_files: list[str] | None = state.get("scope_in_scope") or (
            scope_map.in_scope if scope_map is not None else None
        )
        result = await self._run_mapper(audit_id, target_dir, docs_dir, in_scope_files=in_scope_files)

        stats = result["stats"]
        if al:
            al.log_decision("mapper", "graph_built",
                f"{stats.contracts} contracts, {stats.functions} functions, "
                f"{stats.relationships} relationships",
                {"contracts": stats.contracts, "functions": stats.functions,
                 "relationships": stats.relationships, "state_vars": stats.state_vars})

        # Merge new invariants with any already in state
        existing = list(state.get("invariants") or [])
        state_docs = state.get("documentation") or ""
        new_invariants = await self._mine_invariants(
            result.get("analysis"), audit_id, docs_dir, state_docs, target_dir=target_dir
        )
        all_invariants = existing + new_invariants

        self.log.info(
            "Mapper: complete",
            contracts=stats.contracts,
            functions=stats.functions,
            relationships=stats.relationships,
            invariants=len(new_invariants),
        )
        if al:
            al.log_decision("mapper", "invariants_mined",
                f"{len(new_invariants)} invariants extracted",
                {"total_new": len(new_invariants), "total_all": len(all_invariants)})

        output = {
            **state,
            "graph_id": result["graph_id"],
            "invariants": all_invariants,
            "slither_analysis_raw": result.get("slither_raw"),
        }

        if al and ar:
            al.log_agent_end(ar, {
                "graph_id": result["graph_id"],
                "contracts": stats.contracts,
                "functions": stats.functions,
                "relationships": stats.relationships,
                "new_invariants": len(new_invariants),
                "total_invariants": len(all_invariants),
            })

        return output  # type: ignore[return-value]

    async def run_standalone(
        self,
        target_dir: str,
        audit_id: str = "standalone",
        docs_text: str = "",
    ) -> MapperResult:
        """Entry point for the `ingest` CLI command — returns rich MapperResult."""
        analysis = self._get_analysis(target_dir)
        if analysis is None:
            return MapperResult(errors=["Failed to load Slither analysis"])

        builder = GraphBuilder(self._graph_store)
        stats = await builder.build(analysis)

        # Mine invariants
        natspec_invs = extract_natspec_invariants(analysis, audit_id)
        docs_invs: list[Invariant] = []
        inferred_invs: list[Invariant] = []
        if docs_text:
            docs_invs = await self._miner.extract_from_docs(analysis, docs_text, audit_id)
        inferred_invs = await self._miner.extract_inferred(analysis, audit_id)
        
        ignored_funcs = {"_beforeTokenTransfer", "_afterTokenTransfer", "_update", "_mint", "_burn", "_transfer", "decimals", "name", "symbol"}

        top_fns = sorted(
            [
                {"id": fn.id, "name": fn.name, "score": fn.attack_surface_score,
                 "contract": c.name, "visibility": fn.visibility}
                for c in analysis.contracts
                if "src" in c.source_file or "contracts" in c.source_file
                for fn in c.functions
                if fn.name not in ignored_funcs and not fn.name.startswith("echidna_")
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:10]

        return MapperResult(
            contracts=stats.contracts,
            functions=stats.functions,
            relationships=stats.relationships,
            state_vars=stats.state_vars,
            invariants_natspec=len(natspec_invs),
            invariants_docs=len(docs_invs),
            invariants_inferred=len(inferred_invs),
            top_surface_functions=top_fns,
            graph_id=f"graph-{audit_id}",
        )

    async def _run_mapper(
        self, audit_id: str, target_dir: str, docs_dir: Optional[str],
        in_scope_files: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        slither_raw, analysis = self._get_analysis_with_raw(target_dir)
        graph_id = f"graph-{audit_id}"
        if analysis is None:
            return {"stats": BuildStats(), "graph_id": graph_id, "analysis": None, "slither_raw": None}

        builder = GraphBuilder(self._graph_store)
        stats = await builder.build(analysis, in_scope_files=in_scope_files)
        return {"stats": stats, "graph_id": graph_id, "analysis": analysis, "slither_raw": slither_raw}

    async def _mine_invariants(
        self,
        analysis: Optional[SlitherAnalysis],
        audit_id: str,
        docs_dir: Optional[str],
        state_docs: str = "",
        target_dir: str = "",
    ) -> list[Invariant]:
        if analysis is None:
            return []
        docs_text = _read_docs(docs_dir) if docs_dir else state_docs
        return await self._miner.mine_all(analysis, audit_id, docs_text, target_dir=target_dir)

    def _get_analysis(self, target_dir: str) -> Optional[SlitherAnalysis]:
        _raw, analysis = self._get_analysis_with_raw(target_dir)
        return analysis

    def _get_analysis_with_raw(
        self, target_dir: str
    ) -> tuple[Optional[dict[str, Any]], Optional[SlitherAnalysis]]:
        """Return (raw_json, analysis). raw_json is None when Slither ran via Docker."""
        if self._slither_json is not None:
            analysis = parse_slither_json(self._slither_json)
            _supplement_analysis_from_source(analysis, target_dir)
            return self._slither_json, analysis

        # Prefer the rich Slither-API model (full call graph, taint, real
        # reads/writes) produced by Scout. It is already in normalized format,
        # so parse_slither_json consumes it directly. The detector-derived
        # _supplement is unnecessary here (rich already has every function).
        for rich_name in ("slither_rich.json", ".slither_rich.json"):
            rich_path = Path(target_dir) / rich_name
            if rich_path.exists():
                try:
                    raw = json.loads(rich_path.read_text())
                    if raw.get("contracts"):
                        analysis = parse_slither_json(raw)
                        self.log.info(
                            "Mapper: using rich Slither model",
                            contracts=len(analysis.contracts),
                            functions=len(analysis.all_functions()),
                        )
                        return raw, analysis
                except Exception as exc:
                    self.log.warning("Mapper: rich Slither JSON parse failed", error=str(exc))

        # Look for cached detector slither output in target dir
        cached = Path(target_dir) / ".slither_analysis.json"
        if cached.exists():
            try:
                raw = json.loads(cached.read_text())
                analysis = parse_slither_json(raw)
                _supplement_analysis_from_source(analysis, target_dir)
                return raw, analysis
            except Exception as exc:
                self.log.warning("Mapper: cached slither JSON parse failed", error=str(exc))

        # No Slither available offline — log and return empty analysis
        self.log.warning(
            "Mapper: Slither not available (no Docker or cached output). "
            "Graph will be empty. Run with Docker or provide slither_json fixture."
        )
        return None, SlitherAnalysis()


def _read_docs(docs_dir: str) -> str:
    """Read all .md / .txt files from docs directory."""
    parts: list[str] = []
    for path in sorted(Path(docs_dir).rglob("*.md")):
        try:
            parts.append(path.read_text(errors="replace")[:3000])
        except OSError:
            pass
    for path in sorted(Path(docs_dir).rglob("*.txt")):
        try:
            parts.append(path.read_text(errors="replace")[:1000])
        except OSError:
            pass
    return "\n\n".join(parts)[:8000]


def _supplement_analysis_from_source(analysis: SlitherAnalysis, target_dir: str) -> None:
    """Add public/external functions not found by Slither (it only reports functions with findings).

    Slither's native --json output only includes functions that appear in detector findings.
    Clean functions like deposit() are invisible to the graph without this supplement.
    """
    if not target_dir:
        return
    tp = Path(target_dir)
    sd = tp / "src"
    source_dir = sd if sd.is_dir() else tp

    _SKIP = {"lib", "node_modules", "test", "tests", "script", "scripts", "out"}

    from audit_agent.knowledge.graph.builders import (
        SlitherContract, SlitherFunction, SlitherLocation,
    )

    existing_contracts = {c.name: c for c in analysis.contracts}

    for sol_file in source_dir.glob("*.sol"):
        rel_parts = {p.lower() for p in sol_file.relative_to(tp).parts[:-1]}
        if rel_parts & _SKIP:
            continue
        try:
            content = sol_file.read_text(errors="ignore")
        except OSError:
            continue

        cm = re.search(r"\bcontract\s+(\w+)", content)
        if not cm:
            continue
        cname = cm.group(1)

        if cname not in existing_contracts:
            contract = SlitherContract(
                id=f"contract-{cname}",
                name=cname,
                source_file=str(sol_file),
            )
            analysis.contracts.append(contract)
            existing_contracts[cname] = contract

        contract = existing_contracts[cname]
        existing_fn_names = {fn.name for fn in contract.functions}

        for fn_m in re.finditer(
            r"function\s+(\w+)\s*\(([^)]*)\)\s*(external|public)[^{;]*",
            content,
        ):
            fname = fn_m.group(1)
            if fname in existing_fn_names:
                continue
            fn = SlitherFunction(
                id=f"fn-{cname}-{fname}",
                name=fname,
                signature=f"{fname}({fn_m.group(2).strip()})",
                contract_id=f"contract-{cname}",
                visibility=fn_m.group(3),
                source_location=SlitherLocation(file=str(sol_file), start_line=0, end_line=0),
            )
            # Heuristic flags from source text
            fn_body_start = content.find(fn_m.group(0))
            fn_snippet = content[fn_body_start:fn_body_start + 300]
            if ".call{value:" in fn_snippet or ".transfer(" in fn_snippet or ".send(" in fn_snippet:
                fn.has_external_calls = True
                fn.handles_value = True
            if "payable" in fn_m.group(0):
                fn.handles_value = True
            fn.attack_surface_score = float(fn.handles_value) * 30 + (
                20 if fn.visibility == "external" else 10
            )
            contract.functions.append(fn)
            existing_fn_names.add(fname)
