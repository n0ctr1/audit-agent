"""CHAINER — composes individual hypotheses into multi-step attack chains (roadmap #5).

A senior auditor's highest-value output is rarely a single finding — it is the
realization that finding A *enables* finding B which *unlocks* C. This stage takes
the full hypothesis pool (Hunter + InvariantProver) and the rich code structure
(state-variable write→read flows, call edges from the Slither-API model) and asks
the model to compose grounded attack chains.

Grounding matters: we precompute the concrete structural connections between
hypotheses (which one writes a state var another reads, who calls whom) and feed
them in, so the model chains things that are actually connected rather than
hallucinating plausible-sounding sequences.

Each discovered chain is emitted BOTH as a structured `AttackChain` (kept in
state for reporting) AND as a composite, high-priority `Hypothesis` so it flows
through Exploit → Skeptic → Scribe like any other finding.

Runs as a graph node: invariant_prover → chainer → exploit.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.config import get_settings
from audit_agent.graph.state import AuditState
from audit_agent.knowledge.graph.builders import SlitherAnalysis, parse_slither_json
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.models.schemas import AttackChain, AttackChainStep, Hypothesis
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.chainer")


_CHAIN_TOOL = ToolDefinition(
    name="report_chains",
    description=(
        "Report multi-step attack chains composed from the provided hypotheses. "
        "Only chain hypotheses that have a REAL data-flow or call connection. "
        "Call this exactly once."
    ),
    parameters={
        "type": "object",
        "required": ["chains"],
        "properties": {
            "chains": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title", "combined_severity", "combined_scenario",
                                 "component_indices", "steps"],
                    "properties": {
                        "title": {"type": "string"},
                        "combined_severity": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                        },
                        "root_cause": {"type": "string"},
                        "combined_scenario": {
                            "type": "string",
                            "description": "Full narrative of the chained exploit, end to end.",
                        },
                        "component_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices (from the numbered list) of the chained hypotheses. >=2.",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "order": {"type": "integer"},
                                    "function": {"type": "string"},
                                    "action": {"type": "string"},
                                    "effect": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
)

_CHAINER_SYSTEM = (
    "You are a senior smart-contract security auditor specializing in COMPOSING "
    "individual issues into end-to-end exploit chains.\n\n"
    "You are given numbered vulnerability hypotheses and the concrete structural "
    "connections between them (which hypothesis writes a state variable another "
    "reads, and which functions call which).\n\n"
    "Your job: find chains where one hypothesis ENABLES another — e.g. step 1 sets "
    "or corrupts state X, step 2 trusts X to misprice/misaccount, step 3 extracts "
    "value. A chain must:\n"
    "- contain >=2 hypotheses linked by a REAL connection from the list (data flow "
    "or call edge), not just thematic similarity;\n"
    "- have a concrete, ordered attacker timeline;\n"
    "- escalate severity only when the composition genuinely unlocks higher impact "
    "than any single link.\n\n"
    "Do NOT fabricate chains with no structural connection. If nothing composes, "
    "return an empty chains list. Output ONLY the tool call."
)


class ChainerAgent(BaseAgent):
    """Composes hypotheses into AttackChains + composite hypotheses."""

    _LLM_TIMEOUT_SECS = 200.0

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self._provider = provider or get_provider()
        self._model = model or settings.hunter_model
        self._enabled = getattr(settings, "chainer_enabled", True)
        self._max_chains = getattr(settings, "chainer_max_chains", 6)

    # ── LangGraph node ────────────────────────────────────────────────────
    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        hyps: list[Hypothesis] = list(state.get("hypotheses") or [])

        if not self._enabled:
            self.log.info("Chainer: disabled via settings")
            return state
        if len(hyps) < 2:
            self.log.info("Chainer: <2 hypotheses — nothing to chain")
            return {**state, "chains": list(state.get("chains") or [])}

        self.log.info("Chainer: starting", audit_id=audit_id, hypotheses=len(hyps))
        al = get_audit_logger()
        ar = al.log_agent_start("chainer", 4, {
            "audit_id": audit_id, "hypotheses": len(hyps),
        }) if al else None

        analysis = self._analysis_from_state(state)
        connections = self._build_connections(hyps, analysis)

        if al:
            al.log_decision("chainer", "connections",
                f"{len(connections)} structural links among {len(hyps)} hypotheses",
                {"connections": connections[:10]})

        chains: list[AttackChain] = []
        composite: list[Hypothesis] = []
        try:
            raw_chains = await self._llm_chains(hyps, connections, audit_id)
            chains, composite = self._build_chains(raw_chains, hyps, audit_id)
        except Exception as exc:
            self.log.warning("Chainer: composition failed", error=str(exc))

        self.log.info("Chainer: complete", chains=len(chains))
        if al and ar:
            al.log_agent_end(ar, {
                "chains": len(chains),
                "titles": [c.title for c in chains],
            })

        return {
            **state,
            "chains": list(state.get("chains") or []) + chains,
            "hypotheses": hyps + composite,
        }

    # ── Structural connection mining ──────────────────────────────────────
    def _build_connections(
        self, hyps: list[Hypothesis], analysis: Optional[SlitherAnalysis]
    ) -> list[str]:
        """Return human-readable structural links between hypotheses.

        A link exists when one hypothesis's functions write a state var that
        another's functions read, or when one calls into the other.
        """
        if analysis is None:
            return []

        # Index functions by id and base name.
        by_id: dict[str, Any] = {}
        by_name: dict[str, list[Any]] = defaultdict(list)
        for c in analysis.contracts:
            for fn in c.functions:
                by_id[fn.id] = fn
                by_name[fn.name.lower()].append(fn)

        def _resolve(aff: str) -> list[Any]:
            if aff in by_id:
                return [by_id[aff]]
            base = aff.split("(")[0].split(".")[-1].split("-")[-1].lower()
            return by_name.get(base, [])

        # Aggregate reads/writes/callee-names per hypothesis.
        info: list[dict[str, set]] = []
        for h in hyps:
            reads: set[str] = set()
            writes: set[str] = set()
            callees: set[str] = set()
            fn_names: set[str] = set()
            for aff in (h.affected_functions or []):
                for fn in _resolve(aff):
                    fn_names.add(fn.name.lower())
                    reads.update(r.split("-")[-1].lower() for r in fn.reads)
                    writes.update(w.split("-")[-1].lower() for w in fn.writes)
                    for cid in fn.calls:
                        callee = by_id.get(cid)
                        if callee:
                            callees.add(callee.name.lower())
            info.append({"reads": reads, "writes": writes,
                         "callees": callees, "fns": fn_names})

        connections: list[str] = []
        for i in range(len(hyps)):
            for j in range(len(hyps)):
                if i == j:
                    continue
                shared = info[i]["writes"] & info[j]["reads"]
                if shared:
                    connections.append(
                        f"H{i} ({hyps[i].title[:40]}) writes state [{', '.join(sorted(shared)[:4])}] "
                        f"that H{j} ({hyps[j].title[:40]}) reads"
                    )
                if info[i]["callees"] & info[j]["fns"]:
                    connections.append(
                        f"H{i} ({hyps[i].title[:40]}) calls into a function of "
                        f"H{j} ({hyps[j].title[:40]})"
                    )
        return connections

    # ── LLM composition ───────────────────────────────────────────────────
    async def _llm_chains(
        self, hyps: list[Hypothesis], connections: list[str], audit_id: str
    ) -> list[dict[str, Any]]:
        listing = "\n".join(
            f"[{i}] {h.title} (severity={h.estimated_severity}, class={h.vulnerability_class})\n"
            f"     functions: {h.affected_functions}\n"
            f"     scenario: {(h.attack_scenario or '')[:240]}"
            for i, h in enumerate(hyps)
        )
        conn_block = (
            "\n".join(f"- {c}" for c in connections[:40])
            if connections else "(no structural connections detected — be very conservative)"
        )
        prompt = (
            f"## Hypotheses\n{listing}\n\n"
            f"## Structural connections between them\n{conn_block}\n\n"
            "Compose grounded multi-step attack chains (>=2 hypotheses each), "
            "preferring chains that follow the structural connections above. "
            "Call report_chains with the chains (empty list if none compose)."
        )
        al = get_audit_logger()
        rec = al.log_llm_call_start(
            agent="chainer", purpose="compose_chains", model=self._model,
            system_prompt=_CHAINER_SYSTEM, user_message=prompt,
            tools=[_CHAIN_TOOL], max_tokens=4096,
        ) if al else None

        t0 = time.time()
        try:
            resp = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model, max_tokens=4096, system=_CHAINER_SYSTEM,
                    tools=[_CHAIN_TOOL], tool_choice={"name": "report_chains"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._LLM_TIMEOUT_SECS,
            )
        except Exception as exc:
            if al and rec:
                al.log_llm_call_end(rec, error=str(exc), duration_seconds=time.time() - t0)
            log.warning("chainer.llm_unavailable", error=str(exc))
            return []

        chains_raw: list[dict[str, Any]] = []
        for tc in resp.tool_calls:
            if tc.name == "report_chains":
                chains_raw = tc.arguments.get("chains", []) or []
        if al and rec:
            al.log_llm_call_end(
                rec, tool_calls=[{"name": "report_chains", "chains": len(chains_raw)}],
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason, duration_seconds=time.time() - t0,
            )
        return chains_raw

    # ── Build AttackChain + composite Hypothesis ──────────────────────────
    def _build_chains(
        self, raw_chains: list[dict[str, Any]], hyps: list[Hypothesis], audit_id: str
    ) -> tuple[list[AttackChain], list[Hypothesis]]:
        chains: list[AttackChain] = []
        composite: list[Hypothesis] = []

        for item in raw_chains[: self._max_chains]:
            indices = [int(i) for i in (item.get("component_indices") or [])
                       if isinstance(i, (int, float)) and 0 <= int(i) < len(hyps)]
            indices = list(dict.fromkeys(indices))  # dedupe, preserve order
            if len(indices) < 2:
                continue  # a chain needs >=2 real links

            components = [hyps[i] for i in indices]
            comp_ids = [h.id for h in components]
            affected = list(dict.fromkeys(
                f for h in components for f in (h.affected_functions or [])
            ))
            severity = str(item.get("combined_severity", "high")).lower()
            if severity not in ("critical", "high", "medium", "low"):
                severity = "high"
            scenario = (item.get("combined_scenario") or "").strip()
            root_cause = (item.get("root_cause") or "").strip()
            title = (item.get("title") or "Composed attack chain").strip()

            steps = []
            for s in (item.get("steps") or []):
                steps.append(AttackChainStep(
                    order=int(s.get("order", len(steps) + 1)),
                    function=str(s.get("function", "")),
                    action=str(s.get("action", "")),
                    effect=str(s.get("effect", "")),
                ))
            # Attach component hypothesis ids to steps positionally where possible.
            for k, st in enumerate(steps):
                if k < len(comp_ids):
                    st.hypothesis_id = comp_ids[k]

            priority = min(99.0, max((h.priority_score for h in components), default=70.0) + 10.0)

            chain = AttackChain(
                id=str(uuid.uuid4()), audit_id=audit_id, title=title, steps=steps,
                combined_severity=severity,  # type: ignore[arg-type]
                combined_scenario=scenario or title, root_cause=root_cause,
                component_hypothesis_ids=comp_ids, affected_functions=affected,
                priority_score=priority,
            )
            chains.append(chain)

            step_lines = "\n".join(
                f"  {st.order}. {st.function}: {st.action}"
                + (f" → {st.effect}" if st.effect else "")
                for st in steps
            )
            full_scenario = scenario
            if step_lines:
                full_scenario += "\n\nChain steps:\n" + step_lines
            if root_cause:
                full_scenario += f"\n\nRoot cause: {root_cause}"
            full_scenario += (
                "\n\nComposed from: "
                + "; ".join(f"[{i}] {hyps[i].title[:60]}" for i in indices)
            )

            composite.append(Hypothesis(
                id=str(uuid.uuid4()), audit_id=audit_id,
                title=f"Attack chain: {title}"[:160],
                vulnerability_class="Attack Chain",
                affected_functions=affected,
                attack_scenario=full_scenario,
                required_preconditions=[],
                estimated_severity=severity,  # type: ignore[arg-type]
                similar_historical_cases=[],
                priority_score=priority,
                confidence=0.7,
            ))

        return chains, composite

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _analysis_from_state(state: AuditState) -> Optional[SlitherAnalysis]:
        raw = state.get("slither_analysis_raw")
        if not raw:
            return None
        try:
            return parse_slither_json(raw)
        except Exception:
            return None
