"""THREAT MODELER — builds a value-centric threat model (roadmap #6).

A senior auditor doesn't grep for `onlyOwner`; they first ask "where is the
money, and who can move it?" then aim everything at that. This stage produces a
structured ThreatModel after Mapper:

  • deterministic extraction from the rich Slither model — which functions MOVE
    VALUE (payable / token transfers), which are PRIVILEGED, which read ORACLES,
    and what EXTERNAL CONTRACTS the protocol trusts;
  • LLM enrichment (grounded on those facts + the protocol summary + docs) into
    actors, assets, trust boundaries, value flows, and a ranked list of
    prioritized targets;
  • a deterministic fallback model when the LLM is unavailable.

Hunter consumes `prioritized_targets` to bias analysis (and navigator leads)
toward value-concentrated code instead of raw attack-surface score.

Runs as a graph node: mapper → threat_modeler → hunter.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.config import get_settings
from audit_agent.graph.state import AuditState
from audit_agent.knowledge.graph.builders import SlitherAnalysis, parse_slither_json
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.models.schemas import (
    PrioritizedTarget,
    ThreatActor,
    ThreatModel,
)
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.threat_modeler")

_VALUE_CALLEES = frozenset({
    "transfer", "transferfrom", "send", "safetransfer", "safetransferfrom",
    "mint", "burn", "deposit", "withdraw",
})
_VALUE_NAMES = frozenset({
    "withdraw", "deposit", "borrow", "repay", "redeem", "claim", "swap",
    "liquidate", "mint", "burn", "stake", "unstake", "flashloan", "collect",
})
_ASSET_HINTS = (
    "balance", "reserve", "supply", "collateral", "debt", "deposit",
    "liquidity", "vault", "pool", "total", "shares", "stake",
)


_THREAT_TOOL = ToolDefinition(
    name="build_threat_model",
    description=(
        "Produce a value-centric threat model for the protocol. Focus on WHERE "
        "VALUE CONCENTRATES and WHO can attack it. Call exactly once."
    ),
    parameters={
        "type": "object",
        "required": ["actors", "prioritized_targets"],
        "properties": {
            "actors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "capability": {"type": "string"},
                        "relevance": {"type": "string"},
                    },
                },
            },
            "assets": {"type": "array", "items": {"type": "string"}},
            "trust_boundaries": {"type": "array", "items": {"type": "string"}},
            "external_dependencies": {"type": "array", "items": {"type": "string"}},
            "value_flows": {"type": "array", "items": {"type": "string"}},
            "prioritized_targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["function", "value_at_risk"],
                    "properties": {
                        "function": {"type": "string"},
                        "contract": {"type": "string"},
                        "reason": {"type": "string"},
                        "value_at_risk": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                        },
                    },
                },
            },
            "notes": {"type": "string"},
        },
    },
)

_THREAT_SYSTEM = (
    "You are a senior smart-contract auditor building a threat model BEFORE diving "
    "into code. Think like an attacker deciding where to spend effort.\n\n"
    "Given the protocol summary, documentation, and concrete code facts (which "
    "functions move value, which are privileged, which read oracles, which external "
    "contracts are trusted), produce:\n"
    "- ACTORS: realistic adversaries for THIS protocol (e.g. flash-loan attacker, "
    "malicious token/oracle, sandwich/MEV searcher, first depositor, whale, a "
    "privileged role acting within its trust assumptions ONLY if relevant).\n"
    "- ASSETS: what holds or represents value (token balances, reserves, shares, "
    "collateral, fees).\n"
    "- TRUST BOUNDARIES & EXTERNAL DEPENDENCIES: where the protocol trusts outside "
    "input or contracts.\n"
    "- VALUE FLOWS: how value enters and leaves.\n"
    "- PRIORITIZED_TARGETS: the functions an attacker would hit FIRST, each with a "
    "value_at_risk rating and a one-line reason. Rank by where the most value is "
    "reachable, not by superficial attack surface.\n\n"
    "Be specific to the actual functions named in the facts. Output ONLY the tool call."
)


class ThreatModelerAgent(BaseAgent):
    """Produces a value-centric ThreatModel to steer Hunter prioritization."""

    _LLM_TIMEOUT_SECS = 180.0

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self._provider = provider or get_provider()
        self._model = model or settings.scout_model
        self._enabled = getattr(settings, "threat_model_enabled", True)

    # ── LangGraph node ────────────────────────────────────────────────────
    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        if not self._enabled:
            self.log.info("ThreatModeler: disabled via settings")
            return state

        analysis = self._analysis_from_state(state)
        in_scope = list(state.get("scope_in_scope") or [])
        facts = self._extract_facts(analysis, in_scope) if analysis else {}

        self.log.info("ThreatModeler: starting", audit_id=audit_id,
                      value_movers=len(facts.get("value_movers", [])),
                      privileged=len(facts.get("privileged", [])))
        al = get_audit_logger()
        ar = al.log_agent_start("threat_modeler", 3, {
            "audit_id": audit_id,
            "value_movers": len(facts.get("value_movers", [])),
            "external_deps": len(facts.get("external_deps", [])),
        }) if al else None

        tm: Optional[ThreatModel] = None
        try:
            tm = await self._llm_threat_model(state, facts, audit_id)
        except Exception as exc:
            self.log.warning("ThreatModeler: LLM failed", error=str(exc))
        if tm is None:
            tm = self._fallback_model(facts, audit_id)

        self.log.info("ThreatModeler: complete",
                      actors=len(tm.actors), targets=len(tm.prioritized_targets))
        if al and ar:
            al.log_agent_end(ar, {
                "actors": [a.name for a in tm.actors][:8],
                "prioritized_targets": [t.function for t in tm.prioritized_targets][:12],
            })

        return {**state, "threat_model": tm}

    # ── deterministic extraction ──────────────────────────────────────────
    def _extract_facts(self, analysis: SlitherAnalysis, in_scope: list[str]) -> dict[str, Any]:
        scope_base = {self._base(p) for p in in_scope} if in_scope else set()

        value_movers: list[tuple[str, str, float]] = []  # (contract, fn, score)
        privileged: list[tuple[str, str, float]] = []
        oracle_readers: list[tuple[str, str]] = []
        external_deps: set[str] = set()
        assets: set[str] = set()

        for c in analysis.contracts:
            if scope_base and self._base(c.source_file) not in scope_base:
                continue
            for sv in c.state_variables:
                if any(h in sv.name.lower() for h in _ASSET_HINTS):
                    assets.add(sv.name)
            for fn in c.functions:
                callees = {(t.get("fn") or "").lower() for t in (fn.external_targets or [])}
                for t in (fn.external_targets or []):
                    dest = t.get("dest") or ""
                    if dest and not dest.startswith("TMP_") and dest not in ("this",):
                        external_deps.add(dest)
                is_value = (
                    fn.handles_value
                    or bool(callees & _VALUE_CALLEES)
                    or fn.name.lower() in _VALUE_NAMES
                )
                if is_value and fn.visibility in ("external", "public"):
                    value_movers.append((c.name, fn.name, fn.attack_surface_score))
                if fn.is_privileged:
                    privileged.append((c.name, fn.name, fn.attack_surface_score))
                if fn.reads_oracle:
                    oracle_readers.append((c.name, fn.name))

        value_movers.sort(key=lambda t: t[2], reverse=True)
        privileged.sort(key=lambda t: t[2], reverse=True)
        return {
            "value_movers": value_movers,
            "privileged": privileged,
            "oracle_readers": oracle_readers,
            "external_deps": sorted(external_deps),
            "assets": sorted(assets),
        }

    # ── LLM enrichment ────────────────────────────────────────────────────
    async def _llm_threat_model(
        self, state: AuditState, facts: dict[str, Any], audit_id: str
    ) -> Optional[ThreatModel]:
        ps = state.get("protocol_summary")
        documentation = (state.get("documentation") or "")[:2500]

        def _fns(rows, n=15):
            return ", ".join(f"{c}.{f}" for c, f, *_ in rows[:n]) or "none"

        summary_block = ""
        if ps is not None:
            summary_block = (
                f"name={getattr(ps, 'name', '?')} category={getattr(ps, 'category', '?')}\n"
                f"mechanisms={getattr(ps, 'key_mechanisms', [])}\n"
                f"roles={getattr(ps, 'user_roles', [])}"
            )

        prompt = (
            f"## Protocol Summary\n{summary_block or '(none)'}\n\n"
            f"## Documentation (excerpt)\n{documentation or '(none)'}\n\n"
            f"## Code facts (from static analysis)\n"
            f"- Value-moving functions: {_fns(facts.get('value_movers', []))}\n"
            f"- Privileged functions: {_fns(facts.get('privileged', []))}\n"
            f"- Oracle-reading functions: "
            f"{', '.join(f'{c}.{f}' for c, f in facts.get('oracle_readers', [])[:12]) or 'none'}\n"
            f"- External dependencies (state vars holding external contracts): "
            f"{', '.join(facts.get('external_deps', [])[:15]) or 'none'}\n"
            f"- Value-bearing state variables: {', '.join(facts.get('assets', [])[:15]) or 'none'}\n\n"
            "Build the threat model. Prioritize targets by reachable value."
        )

        al = get_audit_logger()
        rec = al.log_llm_call_start(
            agent="threat_modeler", purpose="build_threat_model", model=self._model,
            system_prompt=_THREAT_SYSTEM, user_message=prompt,
            tools=[_THREAT_TOOL], max_tokens=4096,
        ) if al else None

        t0 = time.time()
        try:
            resp = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model, max_tokens=4096, system=_THREAT_SYSTEM,
                    tools=[_THREAT_TOOL], tool_choice={"name": "build_threat_model"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._LLM_TIMEOUT_SECS,
            )
        except Exception as exc:
            if al and rec:
                al.log_llm_call_end(rec, error=str(exc), duration_seconds=time.time() - t0)
            log.warning("threat_modeler.llm_unavailable", error=str(exc))
            return None

        args: Optional[dict[str, Any]] = None
        for tc in resp.tool_calls:
            if tc.name == "build_threat_model":
                args = tc.arguments
        if al and rec:
            al.log_llm_call_end(
                rec, tool_calls=[{"name": "build_threat_model"}] if args else [],
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason, duration_seconds=time.time() - t0,
            )
        if not args:
            return None
        return self._parse_model(args, audit_id)

    @staticmethod
    def _parse_model(args: dict[str, Any], audit_id: str) -> ThreatModel:
        actors = []
        for a in (args.get("actors") or []):
            if isinstance(a, dict) and a.get("name"):
                actors.append(ThreatActor(
                    name=str(a["name"]), capability=str(a.get("capability", "")),
                    relevance=str(a.get("relevance", "")),
                ))
        targets = []
        for t in (args.get("prioritized_targets") or []):
            if not isinstance(t, dict) or not t.get("function"):
                continue
            var = str(t.get("value_at_risk", "medium")).lower()
            if var not in ("critical", "high", "medium", "low"):
                var = "medium"
            targets.append(PrioritizedTarget(
                function=str(t["function"]), contract=str(t.get("contract", "")),
                reason=str(t.get("reason", "")), value_at_risk=var,  # type: ignore[arg-type]
            ))
        return ThreatModel(
            id=str(uuid.uuid4()), audit_id=audit_id, actors=actors,
            assets=[str(x) for x in (args.get("assets") or [])][:30],
            trust_boundaries=[str(x) for x in (args.get("trust_boundaries") or [])][:30],
            external_dependencies=[str(x) for x in (args.get("external_dependencies") or [])][:30],
            value_flows=[str(x) for x in (args.get("value_flows") or [])][:30],
            prioritized_targets=targets,
            notes=str(args.get("notes", "")),
        )

    # ── deterministic fallback ────────────────────────────────────────────
    @staticmethod
    def _fallback_model(facts: dict[str, Any], audit_id: str) -> ThreatModel:
        targets: list[PrioritizedTarget] = []
        for c, f, *_ in facts.get("value_movers", [])[:10]:
            targets.append(PrioritizedTarget(
                function=f, contract=c, value_at_risk="high",
                reason="moves token/ETH value",
            ))
        for c, f, *_ in facts.get("privileged", [])[:6]:
            if not any(t.function == f for t in targets):
                targets.append(PrioritizedTarget(
                    function=f, contract=c, value_at_risk="medium",
                    reason="privileged state-changing function",
                ))
        actors = [
            ThreatActor(name="External user", capability="calls public/external functions"),
        ]
        if facts.get("oracle_readers"):
            actors.append(ThreatActor(
                name="Price manipulator / flash-loan attacker",
                capability="manipulates oracle/AMM price within a transaction"))
        if facts.get("value_movers"):
            actors.append(ThreatActor(
                name="Value extractor",
                capability="seeks to withdraw more value than entitled"))
        return ThreatModel(
            id=str(uuid.uuid4()), audit_id=audit_id, actors=actors,
            assets=facts.get("assets", []),
            external_dependencies=facts.get("external_deps", []),
            prioritized_targets=targets,
            notes="Deterministic fallback threat model (LLM unavailable).",
        )

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _base(path: str) -> str:
        return (path or "").split("/")[-1]

    @staticmethod
    def _analysis_from_state(state: AuditState) -> Optional[SlitherAnalysis]:
        raw = state.get("slither_analysis_raw")
        if not raw:
            return None
        try:
            return parse_slither_json(raw)
        except Exception:
            return None
