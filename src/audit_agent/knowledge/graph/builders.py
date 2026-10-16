"""Parse Slither analysis JSON → Python node/edge dataclasses.

Two-phase design:
  1. parse_slither_json()  → SlitherAnalysis  (pure parsing, no I/O)
  2. GraphBuilder.build()  → upserts nodes/edges into a GraphStore

GraphStore is a protocol implemented by both Neo4jClient and InMemoryGraphStore.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from audit_agent.agents.scoring import ExternalCallInfo, ScoredFunction, attack_surface_score
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.graph.builders")


# ─────────────────────────────────────────────────────────────────────────────
# Slither intermediate representation (Python dataclasses)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SlitherLocation:
    file: str = ""
    start_line: int = 0
    end_line: int = 0


@dataclass
class SlitherNatSpec:
    notice: str = ""
    dev: str = ""


@dataclass
class SlitherStateVar:
    id: str
    name: str
    type: str
    contract_id: str
    visibility: str = "internal"
    is_constant: bool = False
    is_immutable: bool = False
    initial_value: str = ""


@dataclass
class SlitherModifier:
    id: str
    name: str
    contract_id: str
    source_code: str = ""


@dataclass
class SlitherEvent:
    id: str
    name: str
    contract_id: str
    parameters: list[str] = field(default_factory=list)


@dataclass
class SlitherFunction:
    id: str
    name: str
    signature: str
    contract_id: str
    visibility: str = "internal"
    state_mutability: str = "nonpayable"
    modifiers: list[str] = field(default_factory=list)
    parameters: list[dict[str, str]] = field(default_factory=list)
    returns: list[dict[str, str]] = field(default_factory=list)
    source_location: SlitherLocation = field(default_factory=SlitherLocation)
    natspec: SlitherNatSpec = field(default_factory=SlitherNatSpec)
    # relationship targets (IDs of other functions called internally)
    calls: list[str] = field(default_factory=list)
    external_calls: list[ExternalCallInfo] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)    # state var IDs
    writes: list[str] = field(default_factory=list)   # state var IDs
    emits: list[str] = field(default_factory=list)    # event names
    # raw source text (optional; populated when fixture/parser provides it)
    source_code: str = ""
    # analysis flags (populated by Slither or heuristics)
    has_external_calls: bool = False
    handles_value: bool = False
    reads_oracle: bool = False
    is_privileged: bool = False
    is_constructor: bool = False
    # Rich-analysis extras (populated only by the Slither-API path; empty otherwise)
    # External-call targets: [{"dest": "<state var holding callee>", "fn": "<callee fn>",
    #                          "kind": "high_level|delegatecall|...", "line": int}]
    external_targets: list[dict[str, Any]] = field(default_factory=list)
    # Names of written state variables reachable from user-controlled input (taint).
    tainted_state_writes: list[str] = field(default_factory=list)
    # attack surface score (computed post-parse)
    attack_surface_score: float = 0.0


@dataclass
class SlitherContract:
    id: str
    name: str
    source_file: str
    kind: str = "contract"
    compiler_version: str = ""
    is_abstract: bool = False
    is_interface: bool = False
    inherits: list[str] = field(default_factory=list)
    natspec: SlitherNatSpec = field(default_factory=SlitherNatSpec)
    state_variables: list[SlitherStateVar] = field(default_factory=list)
    functions: list[SlitherFunction] = field(default_factory=list)
    modifiers: list[SlitherModifier] = field(default_factory=list)
    events: list[SlitherEvent] = field(default_factory=list)


@dataclass
class SlitherAnalysis:
    contracts: list[SlitherContract] = field(default_factory=list)
    slither_version: str = ""
    solidity_version: str = ""

    # Convenience accessors
    def all_functions(self) -> list[SlitherFunction]:
        return [fn for c in self.contracts for fn in c.functions]

    def all_state_vars(self) -> list[SlitherStateVar]:
        return [sv for c in self.contracts for sv in c.state_variables]

    def contract_by_id(self, cid: str) -> SlitherContract | None:
        return next((c for c in self.contracts if c.id == cid), None)

    def function_by_id(self, fid: str) -> SlitherFunction | None:
        return next((f for c in self.contracts for f in c.functions if f.id == fid), None)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_slither_json(data: dict[str, Any]) -> SlitherAnalysis:
    """Parse Slither output into SlitherAnalysis.

    Handles two formats:
    - Normalized format: top-level ``contracts`` list (used in tests/fixtures)
    - Native Slither ``--json`` format: ``results.detectors`` with ``elements``

    Also computes attack_surface_score for every function.
    """
    analysis = SlitherAnalysis(
        slither_version=data.get("slither_version", ""),
        solidity_version=data.get("solidity_version", ""),
    )

    if data.get("contracts"):
        # Normalized format
        for c_data in data["contracts"]:
            contract = _parse_contract(c_data)
            for fn in contract.functions:
                fn.attack_surface_score = _score_function(fn)
            analysis.contracts.append(contract)
    elif (data.get("results") or {}).get("detectors"):
        # Native Slither --json format: extract structure from detector elements
        analysis = _parse_slither_native(data)

    log.debug(
        "builders.parsed",
        contracts=len(analysis.contracts),
        functions=len(analysis.all_functions()),
    )
    return analysis


def _parse_slither_native(data: dict[str, Any]) -> SlitherAnalysis:
    """Extract contract/function/state-var structure from native Slither --json output.

    Slither's ``--json`` flag emits ``results.detectors`` where each detector has
    ``elements``: a chain of function → node(external_call | variable_written |
    variable_read) records.  We reconstruct a full code graph by collecting:

      • contracts and functions from ``type == "function"`` elements
      • state-variable writes/reads from ``type == "node"`` elements whose
        ``additional_fields.underlying_type`` is ``"variables_written"`` or
        ``"variables_read"``
      • external-call flags from ``underlying_type == "external_calls"`` nodes
      • function signatures from ``type_specific_fields.signature``
    """
    analysis = SlitherAnalysis()
    contracts_by_name: dict[str, SlitherContract] = {}
    functions_by_key: dict[str, SlitherFunction] = {}  # "ContractName.fnName"
    state_vars_by_key: dict[str, SlitherStateVar] = {}  # "ContractName.varName"

    def _ensure_contract(cname: str, src_file: str = "") -> SlitherContract:
        if cname not in contracts_by_name:
            contracts_by_name[cname] = SlitherContract(
                id=f"contract-{cname}",
                name=cname,
                source_file=src_file,
            )
        return contracts_by_name[cname]

    def _ensure_function(cname: str, fname: str, signature: str, src: dict) -> SlitherFunction:
        key = f"{cname}.{fname}"
        if key not in functions_by_key:
            lines = src.get("lines", [])
            fn = SlitherFunction(
                id=f"fn-{cname}-{fname}",
                name=fname,
                signature=signature or f"{fname}()",
                contract_id=f"contract-{cname}",
                visibility="external",
                source_location=SlitherLocation(
                    file=src.get("filename_short", ""),
                    start_line=lines[0] if lines else 0,
                    end_line=lines[-1] if lines else 0,
                ),
            )
            functions_by_key[key] = fn
            _ensure_contract(cname, src.get("filename_short", "")).functions.append(fn)
        return functions_by_key[key]

    def _ensure_state_var(cname: str, vname: str) -> SlitherStateVar:
        key = f"{cname}.{vname}"
        if key not in state_vars_by_key:
            sv = SlitherStateVar(
                id=f"sv-{cname}-{vname}",
                name=vname,
                type="unknown",
                contract_id=f"contract-{cname}",
                visibility="internal",
            )
            state_vars_by_key[key] = sv
            _ensure_contract(cname).state_variables.append(sv)
        return state_vars_by_key[key]

    def _contract_of_elem(elem: dict) -> str | None:
        """Walk type_specific_fields to find the parent contract name."""
        tsf = elem.get("type_specific_fields") or {}
        parent = tsf.get("parent") or {}
        if isinstance(parent, dict):
            if parent.get("type") == "contract":
                return parent.get("name")
            if parent.get("type") == "function":
                gp = (parent.get("type_specific_fields") or {}).get("parent") or {}
                if isinstance(gp, dict) and gp.get("type") == "contract":
                    return gp.get("name")
        return None

    def _function_of_elem(elem: dict) -> tuple[str, str, str] | None:
        """Return (contract_name, fn_name, signature) or None."""
        tsf = elem.get("type_specific_fields") or {}
        parent = tsf.get("parent") or {}
        if isinstance(parent, dict) and parent.get("type") == "function":
            fn_name = parent.get("name", "")
            sig = (parent.get("type_specific_fields") or {}).get("signature") or fn_name + "()"
            gp = (parent.get("type_specific_fields") or {}).get("parent") or {}
            if isinstance(gp, dict) and gp.get("type") == "contract":
                return gp.get("name"), fn_name, sig
        return None

    for detector in (data.get("results") or {}).get("detectors") or []:
        check = detector.get("check", "")
        for elem in detector.get("elements") or []:
            elem_type = elem.get("type", "")
            tsf = elem.get("type_specific_fields") or {}
            src = elem.get("source_mapping") or {}

            if elem_type == "function":
                cname = (tsf.get("parent") or {}).get("name")
                fname = elem.get("name") or ""
                sig = tsf.get("signature") or f"{fname}()"
                if cname and fname:
                    fn = _ensure_function(cname, fname, sig, src)
                    # Slither flags: infer from detector type
                    if "reentrancy" in check:
                        fn.has_external_calls = True
                    if "suicidal" in check or "selfdestruct" in check:
                        fn.is_privileged = True

            elif elem_type in ("variable", "node"):
                af = elem.get("additional_fields") or {}
                underlying = af.get("underlying_type", "")
                var_name = af.get("variable_name") or ""

                fn_tuple = _function_of_elem(elem)
                cname = fn_tuple[0] if fn_tuple else _contract_of_elem(elem)

                if fn_tuple and cname:
                    fn_cname, fn_fname, fn_sig = fn_tuple
                    fn_src_elem = next(
                        (e for e in detector.get("elements") or []
                         if e.get("type") == "function" and e.get("name") == fn_fname),
                        {},
                    )
                    fn = _ensure_function(fn_cname, fn_fname, fn_sig, fn_src_elem.get("source_mapping") or {})

                    if underlying == "external_calls":
                        fn.has_external_calls = True
                        # ETH-value transfer: look for {value: in the call name
                        if "{value:" in (elem.get("name") or ""):
                            fn.handles_value = True

                    elif underlying == "variables_written" and var_name:
                        sv = _ensure_state_var(fn_cname, var_name)
                        if sv.id not in fn.writes:
                            fn.writes.append(sv.id)

                    elif underlying == "variables_read" and var_name:
                        sv = _ensure_state_var(fn_cname, var_name)
                        if sv.id not in fn.reads:
                            fn.reads.append(sv.id)

    # Score functions after all flags are set
    for fn in (f for c in contracts_by_name.values() for f in c.functions):
        fn.attack_surface_score = _score_function(fn)

    analysis.contracts = list(contracts_by_name.values())
    return analysis


def parse_slither_json_file(path: Path) -> SlitherAnalysis:
    data = json.loads(path.read_text())
    return parse_slither_json(data)


def _parse_contract(data: dict[str, Any]) -> SlitherContract:
    c = SlitherContract(
        id=data["id"],
        name=data["name"],
        source_file=data.get("source_file", ""),
        kind=data.get("kind", "contract"),
        compiler_version=data.get("compiler_version", ""),
        is_abstract=data.get("is_abstract", False),
        is_interface=data.get("is_interface", False),
        inherits=data.get("inherits", []),
        natspec=_parse_natspec(data.get("natspec", {})),
    )
    for sv in data.get("state_variables", []):
        c.state_variables.append(_parse_state_var(sv, c.id))
    for fn in data.get("functions", []):
        c.functions.append(_parse_function(fn, c.id))
    for mod in data.get("modifiers", []):
        c.modifiers.append(SlitherModifier(
            id=mod["id"], name=mod["name"],
            contract_id=c.id, source_code=mod.get("source_code", ""),
        ))
    for ev in data.get("events", []):
        c.events.append(SlitherEvent(
            id=ev["id"], name=ev["name"],
            contract_id=c.id, parameters=ev.get("parameters", []),
        ))
    return c


def _parse_state_var(data: dict[str, Any], contract_id: str) -> SlitherStateVar:
    return SlitherStateVar(
        id=data["id"], name=data["name"], type=data.get("type", ""),
        contract_id=contract_id, visibility=data.get("visibility", "internal"),
        is_constant=data.get("is_constant", False),
        is_immutable=data.get("is_immutable", False),
        initial_value=data.get("initial_value", ""),
    )


def _parse_function(data: dict[str, Any], contract_id: str) -> SlitherFunction:
    loc_data = data.get("source_location", {})
    ext_calls = [
        ExternalCallInfo(
            id=ec.get("id", ""),
            call_type=ec.get("call_type", "call"),
            target_type=ec.get("target_type", "user_controlled"),
            line=ec.get("line", 0),
        )
        for ec in data.get("external_calls", [])
    ]
    return SlitherFunction(
        id=data["id"], name=data["name"], signature=data.get("signature", data["name"]),
        contract_id=contract_id,
        visibility=data.get("visibility", "internal"),
        state_mutability=data.get("state_mutability", "nonpayable"),
        modifiers=data.get("modifiers", []),
        parameters=data.get("parameters", []),
        returns=data.get("returns", []),
        source_location=SlitherLocation(
            file=loc_data.get("file", ""),
            start_line=loc_data.get("start_line", 0),
            end_line=loc_data.get("end_line", 0),
        ),
        natspec=_parse_natspec(data.get("natspec", {})),
        calls=data.get("calls", []),
        external_calls=ext_calls,
        reads=data.get("reads", []),
        writes=data.get("writes", []),
        emits=data.get("emits", []),
        has_external_calls=data.get("has_external_calls", bool(ext_calls)),
        handles_value=data.get("handles_value", False),
        reads_oracle=data.get("reads_oracle", False),
        is_privileged=data.get("is_privileged", False),
        is_constructor=data.get("is_constructor", False),
        external_targets=data.get("external_targets", []),
        tainted_state_writes=data.get("tainted_state_writes", []),
        source_code=data.get("source_code", ""),
    )


def _parse_natspec(data: dict[str, Any]) -> SlitherNatSpec:
    return SlitherNatSpec(notice=data.get("notice", ""), dev=data.get("dev", ""))


def _score_function(fn: SlitherFunction) -> float:
    # Detect division/multiplication from source code heuristic
    src = getattr(fn, "source_code", "") or ""
    has_div = "/" in src or "div" in src.lower()
    has_mul = "*" in src or "mul" in src.lower()

    sf = ScoredFunction(
        id=fn.id, name=fn.name,
        visibility=fn.visibility,  # type: ignore[arg-type]
        state_mutability=fn.state_mutability,  # type: ignore[arg-type]
        modifiers=fn.modifiers,
        external_calls=fn.external_calls,
        writes_state_vars=fn.writes,
        handles_value=fn.handles_value,
        reads_oracle=fn.reads_oracle,
        is_privileged=fn.is_privileged,
        has_division=has_div,
        has_multiplication=has_mul,
    )
    return attack_surface_score(sf)


# ─────────────────────────────────────────────────────────────────────────────
# GraphStore protocol + InMemoryGraphStore (for tests)
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class GraphStore(Protocol):
    async def run_write(self, query: str, **params: Any) -> None: ...
    async def run(self, query: str, **params: Any) -> list[dict[str, Any]]: ...


@dataclass
class CapturedQuery:
    query: str
    params: dict[str, Any]


class InMemoryGraphStore:
    """Drop-in GraphStore for offline tests. Records all queries."""

    def __init__(self) -> None:
        self.queries: list[CapturedQuery] = []
        # In-memory node/edge storage keyed by label+id
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

    async def run_write(self, query: str, **params: Any) -> None:
        self.queries.append(CapturedQuery(query=query, params=params))
        # Very minimal Cypher interpretation for MERGE on id
        if "MERGE" in query and "id:" in query:
            label = _extract_label(query)
            node_id = params.get("id") or params.get("props", {}).get("id", "")
            if label and node_id:
                key = f"{label}:{node_id}"
                existing = self.nodes.get(key, {})
                existing.update(params.get("props", {}))
                self.nodes[key] = existing

    async def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(CapturedQuery(query=query, params=params))
        return []

    def node_count(self, label: str) -> int:
        return sum(1 for k in self.nodes if k.startswith(f"{label}:"))

    def queries_for(self, keyword: str) -> list[CapturedQuery]:
        return [q for q in self.queries if keyword.upper() in q.query.upper()]


def _extract_label(query: str) -> str:
    """Extract node label from a MERGE (:Label ...) statement."""
    import re
    m = re.search(r"MERGE\s*\([\w]*:(\w+)", query)
    return m.group(1) if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder — writes nodes+edges to any GraphStore
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuildStats:
    contracts: int = 0
    functions: int = 0
    state_vars: int = 0
    modifiers: int = 0
    events: int = 0
    relationships: int = 0


IGNORE_PATHS = frozenset([
    "lib/", "node_modules/", "@openzeppelin/", "@solady/",
    "forge-std/", "test/", "script/",
])


def _should_include_contract(file_path: str) -> bool:
    """Return True if the contract's source file is in-scope (not a dependency)."""
    return not any(ignore in file_path for ignore in IGNORE_PATHS)


class GraphBuilder:
    """Translate a SlitherAnalysis into Neo4j nodes and edges via a GraphStore."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    async def build(
        self,
        analysis: SlitherAnalysis,
        in_scope_files: list[str] | None = None,
    ) -> BuildStats:
        stats = BuildStats()
        total = len(analysis.contracts)
        included = 0
        excluded = 0
        for contract in analysis.contracts:
            # Filter out dependency contracts (OZ, lib/, etc.)
            if in_scope_files is not None:
                # Explicit in-scope list provided by Scout
                if not any(contract.source_file in f or f.endswith(contract.source_file)
                           for f in in_scope_files):
                    excluded += 1
                    continue
            elif not _should_include_contract(contract.source_file):
                excluded += 1
                continue
            included += 1
            await self._upsert_contract(contract)
            stats.contracts += 1
            for sv in contract.state_variables:
                await self._upsert_state_var(sv)
                stats.state_vars += 1
            for mod in contract.modifiers:
                await self._upsert_modifier(mod)
                stats.modifiers += 1
            for ev in contract.events:
                await self._upsert_event(ev)
                stats.events += 1
            for fn in contract.functions:
                await self._upsert_function(fn)
                stats.functions += 1
                rel_count = await self._upsert_function_relationships(fn, contract)
                stats.relationships += rel_count
            # inheritance
            for parent_id in contract.inherits:
                await self._store.run_write(
                    "MATCH (c:Contract {id: $cid}), (p:Contract {id: $pid}) "
                    "MERGE (c)-[:INHERITS_FROM]->(p)",
                    cid=contract.id, pid=parent_id,
                )
                stats.relationships += 1
        log.info(
            "Mapper: scope filter",
            total_contracts=total,
            included=included,
            excluded=excluded,
        )
        log.info("graph.build_complete", **vars(stats))
        return stats

    async def _upsert_contract(self, c: SlitherContract) -> None:
        await self._store.run_write(
            "MERGE (n:Contract {id: $id}) SET n += $props",
            id=c.id,
            props={
                "id": c.id, "name": c.name, "file_path": c.source_file,
                "kind": c.kind, "compiler_version": c.compiler_version,
                "is_abstract": c.is_abstract,
                "inherits_from": c.inherits,
                "is_in_scope": True,
                "is_fork_of": None,
            },
        )

    async def _upsert_state_var(self, sv: SlitherStateVar) -> None:
        await self._store.run_write(
            "MERGE (n:StateVar {id: $id}) SET n += $props "
            "WITH n MATCH (c:Contract {id: $cid}) MERGE (n)-[:DEFINED_IN]->(c)",
            id=sv.id,
            props={
                "id": sv.id, "name": sv.name, "type": sv.type,
                "contract_id": sv.contract_id, "visibility": sv.visibility,
                "is_constant": sv.is_constant, "is_immutable": sv.is_immutable,
                "initial_value": sv.initial_value,
            },
            cid=sv.contract_id,
        )

    async def _upsert_modifier(self, mod: SlitherModifier) -> None:
        await self._store.run_write(
            "MERGE (n:Modifier {id: $id}) SET n += $props",
            id=mod.id,
            props={
                "id": mod.id, "name": mod.name,
                "contract_id": mod.contract_id, "source_code": mod.source_code,
            },
        )

    async def _upsert_event(self, ev: SlitherEvent) -> None:
        await self._store.run_write(
            "MERGE (n:Event {id: $id}) SET n += $props",
            id=ev.id,
            props={
                "id": ev.id, "name": ev.name,
                "contract_id": ev.contract_id, "parameters": ev.parameters,
            },
        )

    async def _upsert_function(self, fn: SlitherFunction) -> None:
        await self._store.run_write(
            "MERGE (n:Function {id: $id}) SET n += $props "
            "SET n.attack_surface_score = $attack_surface_score "
            "WITH n MATCH (c:Contract {id: $cid}) MERGE (n)-[:DEFINED_IN]->(c)",
            id=fn.id,
            attack_surface_score=fn.attack_surface_score,
            props={
                "id": fn.id, "name": fn.name, "signature": fn.signature,
                "contract_id": fn.contract_id,
                "visibility": fn.visibility, "state_mutability": fn.state_mutability,
                "modifiers": fn.modifiers,
                "parameters": [p.get("type", "") for p in fn.parameters],
                "returns": [r.get("type", "") for r in fn.returns],
                "source_location_file": fn.source_location.file,
                "source_location_start": fn.source_location.start_line,
                "source_location_end": fn.source_location.end_line,
                "natspec": fn.natspec.notice,
                "handles_value": fn.handles_value,
                "has_external_calls": fn.has_external_calls,
                "reads_oracle": fn.reads_oracle,
                "is_privileged": fn.is_privileged,
                "attack_surface_score": fn.attack_surface_score,
            },
            cid=fn.contract_id,
        )

    async def _upsert_function_relationships(
        self, fn: SlitherFunction, contract: SlitherContract
    ) -> int:
        count = 0
        # CALLS (internal)
        for callee_id in fn.calls:
            await self._store.run_write(
                "MATCH (caller:Function {id: $fid}), (callee:Function {id: $cid}) "
                "MERGE (caller)-[:CALLS]->(callee)",
                fid=fn.id, cid=callee_id,
            )
            count += 1
        # CALLS_EXTERNAL
        for ec in fn.external_calls:
            await self._store.run_write(
                "MERGE (ec:ExternalCall {id: $ecid}) SET ec += $props "
                "WITH ec MATCH (f:Function {id: $fid}) "
                "MERGE (f)-[:CALLS_EXTERNAL]->(ec)",
                ecid=ec.id,
                props={
                    "id": ec.id, "call_type": ec.call_type,
                    "target_type": ec.target_type, "line": ec.line,
                },
                fid=fn.id,
            )
            count += 1
        # READS
        for sv_id in fn.reads:
            await self._store.run_write(
                "MATCH (f:Function {id: $fid}), (sv:StateVar {id: $svid}) "
                "MERGE (f)-[:READS]->(sv)",
                fid=fn.id, svid=sv_id,
            )
            count += 1
        # WRITES
        for sv_id in fn.writes:
            await self._store.run_write(
                "MATCH (f:Function {id: $fid}), (sv:StateVar {id: $svid}) "
                "MERGE (f)-[:WRITES]->(sv)",
                fid=fn.id, svid=sv_id,
            )
            count += 1
        # USES_MODIFIER
        for mod_name in fn.modifiers:
            mod_id = f"{fn.contract_id}.{mod_name}"
            await self._store.run_write(
                "MATCH (f:Function {id: $fid}) "
                "MERGE (m:Modifier {id: $mid}) ON CREATE SET m.name = $mname "
                "MERGE (f)-[:USES_MODIFIER]->(m)",
                fid=fn.id, mid=mod_id, mname=mod_name,
            )
            count += 1
        # EMITS
        for event_name in fn.emits:
            event_id = f"{fn.contract_id}.{event_name}"
            await self._store.run_write(
                "MATCH (f:Function {id: $fid}) "
                "MERGE (e:Event {id: $eid}) ON CREATE SET e.name = $ename "
                "MERGE (f)-[:EMITS]->(e)",
                fid=fn.id, eid=event_id, ename=event_name,
            )
            count += 1
        return count
