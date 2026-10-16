"""SubgraphContextBuilder — minimal code slice engine for Hunter (§8.4).

Two implementations of GraphAdapter:
  - InMemoryGraph: built from SlitherAnalysis, used in tests and offline mode.
  - NullGraphAdapter: fallback when no graph is available.

Production Neo4j adapter lives in neo4j_client.py (queries already in queries.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, Protocol, runtime_checkable

from audit_agent.agents.scoring import ExternalCallInfo
from audit_agent.knowledge.graph.builders import (
    SlitherAnalysis,
    SlitherFunction,
    SlitherLocation,
    SlitherNatSpec,
)
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.graph.context")


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubgraphContext:
    entry_function_id: str
    code_slice: str
    state_interactions: list[dict[str, Any]] = field(default_factory=list)
    external_calls: list[dict[str, Any]] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    reachable_function_ids: list[str] = field(default_factory=list)
    similar_historical_cases: list[dict[str, Any]] = field(default_factory=list)
    estimated_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# GraphAdapter protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class GraphAdapter(Protocol):
    async def get_function_meta(self, fn_id: str) -> Optional[dict[str, Any]]: ...
    async def bfs_callees(self, fn_id: str, max_depth: int) -> list[dict[str, Any]]: ...
    async def get_external_calls_for_fns(self, fn_ids: list[str]) -> list[dict[str, Any]]: ...
    async def get_state_interactions_for_fns(self, fn_ids: list[str]) -> list[dict[str, Any]]: ...
    async def get_modifiers_for_fns(self, fn_ids: list[str]) -> list[str]: ...
    async def get_callers(self, fn_id: str) -> list[dict[str, Any]]: ...
    async def get_shared_state_writers(self, fn_id: str) -> list[dict[str, Any]]: ...


# ─────────────────────────────────────────────────────────────────────────────
# NullGraphAdapter — graceful no-op when nothing is available
# ─────────────────────────────────────────────────────────────────────────────

class NullGraphAdapter:
    async def get_function_meta(self, fn_id: str) -> Optional[dict[str, Any]]:
        return None

    async def bfs_callees(self, fn_id: str, max_depth: int) -> list[dict[str, Any]]:
        return []

    async def get_external_calls_for_fns(self, fn_ids: list[str]) -> list[dict[str, Any]]:
        return []

    async def get_state_interactions_for_fns(self, fn_ids: list[str]) -> list[dict[str, Any]]:
        return []

    async def get_modifiers_for_fns(self, fn_ids: list[str]) -> list[str]:
        return []

    async def get_callers(self, fn_id: str) -> list[dict[str, Any]]:
        return []

    async def get_shared_state_writers(self, fn_id: str) -> list[dict[str, Any]]:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# InMemoryGraph — test-only GraphAdapter with real BFS traversal
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryGraph:
    """GraphAdapter built from SlitherAnalysis. Supports real BFS traversal.

    Two construction paths:
      1. InMemoryGraph(analysis) — from parsed SlitherAnalysis
      2. InMemoryGraph.empty() + graph.add_function(...) — for hand-crafted test graphs
    """

    def __init__(self, analysis: Optional[SlitherAnalysis] = None) -> None:
        self._fns: dict[str, SlitherFunction] = {}
        self._contract_of: dict[str, str] = {}
        # Reverse edges: callee_id → set of caller fn_ids
        self._callers_of: dict[str, set[str]] = {}
        # State var → set of fn_ids that write it
        self._writers_of: dict[str, set[str]] = {}
        if analysis is not None:
            for contract in analysis.contracts:
                for fn in contract.functions:
                    self._fns[fn.id] = fn
                    self._contract_of[fn.id] = contract.name
            self._build_reverse_indices()

    @classmethod
    def empty(cls) -> "InMemoryGraph":
        return cls(analysis=None)

    def _build_reverse_indices(self) -> None:
        """Build _callers_of and _writers_of from the forward edges."""
        for fn_id, fn in self._fns.items():
            for callee_id in fn.calls:
                self._callers_of.setdefault(callee_id, set()).add(fn_id)
            for sv_id in fn.writes:
                self._writers_of.setdefault(sv_id, set()).add(fn_id)

    # ── GraphAdapter implementation ────────────────────────────────────────

    async def get_function_meta(self, fn_id: str) -> Optional[dict[str, Any]]:
        fn = self._fns.get(fn_id)
        if fn is None:
            return None
        loc = fn.source_location
        return {
            "id": fn.id,
            "name": fn.name,
            "signature": fn.signature,
            "visibility": fn.visibility,
            "state_mutability": fn.state_mutability,
            "modifiers": fn.modifiers,
            "attack_surface_score": fn.attack_surface_score,
            "contract": self._contract_of.get(fn_id, ""),
            "has_external_calls": fn.has_external_calls,
            "handles_value": fn.handles_value,
            "reads_oracle": fn.reads_oracle,
            "is_privileged": fn.is_privileged,
            "source_location_file": loc.file if loc else "",
            "source_location_start": loc.start_line if loc else 0,
            "source_location_end": loc.end_line if loc else 0,
        }

    async def bfs_callees(self, fn_id: str, max_depth: int) -> list[dict[str, Any]]:
        """BFS through CALLS edges from fn_id, not including the entry itself."""
        visited: set[str] = {fn_id}
        queue: deque[tuple[str, int]] = deque()
        # seed with direct callees of the entry
        entry_fn = self._fns.get(fn_id)
        if entry_fn:
            for callee_id in entry_fn.calls:
                if callee_id not in visited:
                    queue.append((callee_id, 1))

        result: list[dict[str, Any]] = []
        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            meta = await self.get_function_meta(current_id)
            if meta:
                result.append(meta)

            if depth < max_depth:
                fn = self._fns.get(current_id)
                if fn:
                    for callee_id in fn.calls:
                        if callee_id not in visited:
                            queue.append((callee_id, depth + 1))

        return result

    async def get_external_calls_for_fns(self, fn_ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for fn_id in fn_ids:
            fn = self._fns.get(fn_id)
            if fn:
                for ec in fn.external_calls:
                    result.append({
                        "fn_id": fn_id,
                        "fn_name": fn.name,
                        "call_type": ec.call_type,
                        "target_type": ec.target_type,
                        "line": ec.line,
                    })
        return result

    async def get_state_interactions_for_fns(self, fn_ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for fn_id in fn_ids:
            fn = self._fns.get(fn_id)
            if fn:
                for sv_id in fn.reads:
                    result.append({
                        "fn_id": fn_id,
                        "rel_type": "READS",
                        "var_name": sv_id.split(".")[-1],
                        "var_full_id": sv_id,
                    })
                for sv_id in fn.writes:
                    result.append({
                        "fn_id": fn_id,
                        "rel_type": "WRITES",
                        "var_name": sv_id.split(".")[-1],
                        "var_full_id": sv_id,
                    })
        return result

    async def get_modifiers_for_fns(self, fn_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        for fn_id in fn_ids:
            fn = self._fns.get(fn_id)
            if fn:
                for m in fn.modifiers:
                    seen.add(m)
        return sorted(seen)

    async def get_callers(self, fn_id: str) -> list[dict[str, Any]]:
        """Return metadata for all functions that directly call fn_id (backward edge)."""
        caller_ids = self._callers_of.get(fn_id, set())
        result: list[dict[str, Any]] = []
        for caller_id in sorted(caller_ids):
            meta = await self.get_function_meta(caller_id)
            if meta:
                result.append(meta)
        return result

    async def get_shared_state_writers(self, fn_id: str) -> list[dict[str, Any]]:
        """Return functions from OTHER contracts that write state variables read by fn_id.

        Useful for detecting cross-contract read/write conflicts (e.g., price oracle skew,
        shared registry manipulation).
        """
        fn = self._fns.get(fn_id)
        if fn is None:
            return []
        my_contract = self._contract_of.get(fn_id, "")
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for sv_id in fn.reads:
            for writer_id in self._writers_of.get(sv_id, set()):
                if writer_id == fn_id or writer_id in seen:
                    continue
                writer_contract = self._contract_of.get(writer_id, "")
                if writer_contract != my_contract:
                    meta = await self.get_function_meta(writer_id)
                    if meta:
                        meta = dict(meta)
                        meta["shared_var"] = sv_id.split(".")[-1]
                        result.append(meta)
                        seen.add(writer_id)
        return result

    # ── Test helper: manual graph construction ─────────────────────────────

    def add_function(
        self,
        fn_id: str,
        name: str,
        contract: str = "TestContract",
        visibility: str = "external",
        state_mutability: str = "nonpayable",
        calls: Optional[list[str]] = None,
        external_calls: Optional[list[ExternalCallInfo]] = None,
        reads: Optional[list[str]] = None,
        writes: Optional[list[str]] = None,
        modifiers: Optional[list[str]] = None,
        attack_surface_score: float = 0.0,
    ) -> None:
        fn = SlitherFunction(
            id=fn_id,
            name=name,
            signature=name,
            contract_id=contract,
            visibility=visibility,
            state_mutability=state_mutability,
            modifiers=modifiers or [],
            calls=calls or [],
            external_calls=external_calls or [],
            reads=reads or [],
            writes=writes or [],
            source_location=SlitherLocation(),
            natspec=SlitherNatSpec(),
        )
        fn.attack_surface_score = attack_surface_score
        self._fns[fn_id] = fn
        self._contract_of[fn_id] = contract
        # Update reverse indices incrementally
        for callee_id in (calls or []):
            self._callers_of.setdefault(callee_id, set()).add(fn_id)
        for sv_id in (writes or []):
            self._writers_of.setdefault(sv_id, set()).add(fn_id)


# ─────────────────────────────────────────────────────────────────────────────
# Token estimation and code-slice formatting
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars)."""
    return max(0, len(text) // 4)


def _format_fn_slice(fn_meta: dict[str, Any], base_dir: str = "") -> str:
    contract = fn_meta.get("contract", "?")
    name = fn_meta.get("name", "?")
    sig = fn_meta.get("signature", name)
    vis = fn_meta.get("visibility", "")
    mut = fn_meta.get("state_mutability", "")
    score = fn_meta.get("attack_surface_score", 0.0)
    mods = fn_meta.get("modifiers", [])

    header = f"[{contract}] fn {sig} {vis} {mut}  // score={score:.1f}"
    if mods:
        header += f"\n  modifiers: {mods}"

    # Try to read the actual function body from source file
    src_file = fn_meta.get("source_location_file", "")
    start_line = fn_meta.get("source_location_start", 0)
    end_line = fn_meta.get("source_location_end", 0)
    if src_file and start_line and end_line and start_line <= end_line:
        candidates: list[Path] = [Path(src_file)]
        if base_dir:
            # Slither stores filename_short as a relative path (e.g. lib/panoptic-v1.1/...)
            candidates.append(Path(base_dir) / src_file)
            # Slither inside Docker may also report /target/... absolute paths
            if src_file.startswith("/target/"):
                candidates.append(Path(base_dir) / src_file[len("/target/"):])
        for candidate in candidates:
            try:
                lines = candidate.read_text(errors="ignore").splitlines()
                body_lines = lines[start_line - 1 : end_line]
                if body_lines:
                    return header + "\n" + "\n".join(body_lines)
            except OSError:
                continue

    return header


def _format_state_summary(interactions: list[dict[str, Any]]) -> str:
    reads = sorted({i["var_name"] for i in interactions if i["rel_type"] == "READS"})
    writes = sorted({i["var_name"] for i in interactions if i["rel_type"] == "WRITES"})
    parts: list[str] = []
    if reads:
        parts.append(f"reads={reads}")
    if writes:
        parts.append(f"writes={writes}")
    return "// state vars: " + ", ".join(parts) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# SubgraphContextBuilder
# ─────────────────────────────────────────────────────────────────────────────

# Type alias for the optional async RAG search callable
RagSearch = Callable[[str, int], Coroutine[Any, Any, list[dict[str, Any]]]]


class SubgraphContextBuilder:
    """Builds minimal code slices for Hunter analysis (§8.4).

    Guarantees the returned code_slice fits within max_tokens.
    """

    def __init__(
        self,
        graph: GraphAdapter,
        qdrant_search: Optional[RagSearch] = None,
        target_dir: str = "",
    ) -> None:
        self._graph = graph
        self._qdrant_search = qdrant_search
        self._target_dir = target_dir

    async def build_context(
        self,
        entry_function_id: str,
        max_depth: int = 8,
        max_tokens: int = 6000,
    ) -> SubgraphContext:
        # 1. Entry function metadata
        entry_meta = await self._graph.get_function_meta(entry_function_id)

        # 2. Forward reachability through CALLS
        callees = await self._graph.bfs_callees(entry_function_id, max_depth)
        reachable_ids = [entry_function_id] + [fn["id"] for fn in callees]

        # 3. External calls on all reachable functions
        external_calls = await self._graph.get_external_calls_for_fns(reachable_ids)

        # 4. State variable interactions
        state_interactions = await self._graph.get_state_interactions_for_fns(reachable_ids)

        # 5. Modifiers
        modifiers = await self._graph.get_modifiers_for_fns(reachable_ids)

        # 6. Assemble code slice, respecting max_tokens budget
        all_metas = ([entry_meta] if entry_meta else []) + callees
        slice_parts: list[str] = []
        tokens_used = 0

        for fn_meta in all_metas:
            part = _format_fn_slice(fn_meta, base_dir=self._target_dir)
            cost = _estimate_tokens(part)
            if tokens_used + cost > max_tokens:
                slice_parts.append("// ... [truncated — max_tokens reached]")
                break
            slice_parts.append(part)
            tokens_used += cost

        # Append state summary if budget allows
        if state_interactions:
            sv_line = _format_state_summary(state_interactions)
            if sv_line:
                cost = _estimate_tokens(sv_line)
                if tokens_used + cost <= max_tokens:
                    slice_parts.append(sv_line)
                    tokens_used += cost

        code_slice = "\n\n".join(slice_parts)

        # 7. RAG: similar historical vulnerabilities from Qdrant
        similar_cases: list[dict[str, Any]] = []
        if self._qdrant_search and code_slice:
            try:
                similar_cases = await self._qdrant_search(code_slice[:2000], 5)
            except Exception as exc:
                log.warning("context.rag_failed", error=str(exc))

        log.debug(
            "context.built",
            entry=entry_function_id,
            reachable=len(reachable_ids),
            tokens=tokens_used,
        )

        return SubgraphContext(
            entry_function_id=entry_function_id,
            code_slice=code_slice,
            state_interactions=state_interactions,
            external_calls=external_calls,
            modifiers=modifiers,
            reachable_function_ids=reachable_ids,
            similar_historical_cases=similar_cases,
            estimated_tokens=tokens_used,
        )

    async def build_cross_contract_context(
        self,
        entry_function_id: str,
        max_depth: int = 8,
        max_tokens: int = 6000,
    ) -> SubgraphContext:
        """Build context enriched with cross-contract backward reachability.

        Adds to the standard forward context:
        - Callers of entry_function_id (who can trigger it from other contracts)
        - Functions in other contracts that write state variables read by entry_function_id
          (potential cross-contract state manipulation vectors)
        """
        base = await self.build_context(entry_function_id, max_depth, max_tokens)

        callers = await self._graph.get_callers(entry_function_id)
        shared_writers = await self._graph.get_shared_state_writers(entry_function_id)

        extra_parts: list[str] = []
        tokens_remaining = max_tokens - base.estimated_tokens

        if callers and tokens_remaining > 0:
            caller_lines = ["// === Callers (backward reachability) ==="]
            for c in callers:
                line = _format_fn_slice(c)
                cost = _estimate_tokens(line)
                if tokens_remaining - cost < 0:
                    break
                caller_lines.append(line)
                tokens_remaining -= cost
            extra_parts.append("\n".join(caller_lines))

        if shared_writers and tokens_remaining > 0:
            writer_lines = ["// === Cross-contract state writers (shared var conflicts) ==="]
            for w in shared_writers:
                shared_var = w.get("shared_var", "?")
                line = _format_fn_slice(w) + f"  // writes shared var: {shared_var}"
                cost = _estimate_tokens(line)
                if tokens_remaining - cost < 0:
                    break
                writer_lines.append(line)
                tokens_remaining -= cost
            extra_parts.append("\n".join(writer_lines))

        if extra_parts:
            enriched_slice = base.code_slice + "\n\n" + "\n\n".join(extra_parts)
        else:
            enriched_slice = base.code_slice

        log.debug(
            "context.cross_contract_built",
            entry=entry_function_id,
            callers=len(callers),
            shared_writers=len(shared_writers),
        )

        return SubgraphContext(
            entry_function_id=base.entry_function_id,
            code_slice=enriched_slice,
            state_interactions=base.state_interactions,
            external_calls=base.external_calls,
            modifiers=base.modifiers,
            reachable_function_ids=base.reachable_function_ids,
            similar_historical_cases=base.similar_historical_cases,
            estimated_tokens=max_tokens - tokens_remaining,
        )
