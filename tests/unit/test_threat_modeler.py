"""Tests for the value-centric threat modeler (roadmap #6)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from audit_agent.agents.hunter import HunterAgent, _format_threat_summary, _threat_targets_and_summary
from audit_agent.agents.threat_modeler import ThreatModelerAgent
from audit_agent.knowledge.graph.builders import parse_slither_json
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.llm.mock_provider import MockProvider
from audit_agent.models.schemas import PrioritizedTarget, ThreatActor, ThreatModel

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "slither_rich.json"


def _rich() -> dict:
    return json.loads(FIXTURE.read_text())


def _state():
    return {
        "audit_id": "audit-1",
        "slither_analysis_raw": _rich(),
        "scope_in_scope": [],
        "documentation": "A collateralized lending pool with an oracle.",
        "protocol_summary": None,
    }


# ── deterministic extraction ──────────────────────────────────────────────────

def test_extract_facts_finds_value_oracle_external():
    analysis = parse_slither_json(_rich())
    agent = ThreatModelerAgent(provider=MockProvider())
    facts = agent._extract_facts(analysis, [])
    value_fn_names = {f for _, f, *_ in facts["value_movers"]}
    assert {"borrow", "liquidate"} & value_fn_names  # token movers detected
    oracle_fn_names = {f for _, f in facts["oracle_readers"]}
    assert {"borrow", "liquidate"} & oracle_fn_names
    assert any("oracle" in d.lower() for d in facts["external_deps"])  # priceOracle dep


# ── LLM path ───────────────────────────────────────────────────────────────────

async def test_run_builds_threat_model_from_llm():
    args = {
        "actors": [{"name": "Flash-loan attacker", "capability": "manipulates price"}],
        "assets": ["collateralBalance", "borrowedAmount"],
        "external_dependencies": ["priceOracle"],
        "prioritized_targets": [
            {"function": "liquidate", "contract": "CollateralLendingPool",
             "reason": "drains collateral", "value_at_risk": "critical"},
            {"function": "borrow", "value_at_risk": "high"},
        ],
        "notes": "oracle is the crux",
    }

    async def cb(*, model, **kw):
        return LLMResponse(
            text="", tool_calls=[ToolCall(id="t", name="build_threat_model", arguments=args)],
            input_tokens=0, output_tokens=0, stop_reason="tool_use", model=model,
        )

    agent = ThreatModelerAgent(provider=MockProvider(callback=cb))
    out = await agent.run(_state())
    tm = out["threat_model"]
    assert isinstance(tm, ThreatModel)
    assert tm.actors[0].name == "Flash-loan attacker"
    assert any(t.function == "liquidate" and t.value_at_risk == "critical"
               for t in tm.prioritized_targets)


async def test_run_falls_back_when_llm_fails():
    agent = ThreatModelerAgent(provider=MockProvider(side_effect=RuntimeError("down")))
    out = await agent.run(_state())
    tm = out["threat_model"]
    assert isinstance(tm, ThreatModel)
    # Fallback still ranks the value-moving functions.
    target_fns = {t.function for t in tm.prioritized_targets}
    assert {"borrow", "liquidate"} & target_fns
    assert "Deterministic fallback" in tm.notes


async def test_disabled_is_noop():
    agent = ThreatModelerAgent(provider=MockProvider())
    agent._enabled = False
    out = await agent.run(_state())
    assert out.get("threat_model") is None


# ── Hunter consumption ──────────────────────────────────────────────────────────

def test_threat_targets_and_summary_helpers():
    tm = ThreatModel(
        audit_id="a",
        actors=[ThreatActor(name="Whale", capability="large deposits")],
        assets=["reserves"],
        prioritized_targets=[
            PrioritizedTarget(function="CollateralLendingPool.liquidate(address)",
                              value_at_risk="critical", reason="drain"),
        ],
    )
    targets, summary = _threat_targets_and_summary(tm)
    assert "liquidate" in targets            # base name extracted
    assert "liquidate" in summary
    assert "Whale" in summary
    assert _threat_targets_and_summary(None) == (set(), "")


async def test_hunter_boosts_threat_prioritized_function():
    """A lower-attack-surface function flagged by the threat model should still be
    pulled into Hunter's top-N analysis via the boost."""
    analysis = parse_slither_json(_rich())
    from audit_agent.knowledge.graph.context import SubgraphContextBuilder, InMemoryGraph

    hunter = HunterAgent(provider=MockProvider(), top_functions=2)
    context_builder = SubgraphContextBuilder(graph=InMemoryGraph(analysis))

    processed: list[str] = []

    async def tracking(fn, contract_name, *args, **kwargs):
        processed.append(fn.name)
        from audit_agent.knowledge.graph.context import SubgraphContext
        return [], (fn, contract_name, SubgraphContext(entry_function_id=fn.id, code_slice=""))

    hunter._analyze_one_function = tracking  # type: ignore[assignment]

    await hunter._generate(
        analysis=analysis, audit_id="a", context_builder=context_builder,
        static_findings=[],
        in_scope_files=[],
        priority_targets={"depositcollateral"},  # boost this specific fn
    )
    assert "depositCollateral" in processed
