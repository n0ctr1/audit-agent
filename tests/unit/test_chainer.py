"""Tests for the attack-chain composer (roadmap #5)."""

from __future__ import annotations

import json
from pathlib import Path

from audit_agent.agents.chainer import ChainerAgent
from audit_agent.knowledge.graph.builders import parse_slither_json
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.llm.mock_provider import MockProvider
from audit_agent.models.schemas import Hypothesis

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "slither_rich.json"


def _rich() -> dict:
    return json.loads(FIXTURE.read_text())


def _hyp(title: str, fns: list[str], sev: str = "medium", score: float = 60.0) -> Hypothesis:
    return Hypothesis(
        id=title.replace(" ", "-"), audit_id="a", title=title,
        vulnerability_class="logic", affected_functions=fns,
        attack_scenario=f"scenario for {title}", estimated_severity=sev,  # type: ignore[arg-type]
        priority_score=score,
    )


def _state(hyps, with_analysis=True):
    return {
        "audit_id": "audit-1",
        "hypotheses": hyps,
        "chains": [],
        "slither_analysis_raw": _rich() if with_analysis else None,
    }


def _chain_response(model, chains):
    return LLMResponse(
        text="", tool_calls=[ToolCall(id="c", name="report_chains", arguments={"chains": chains})],
        input_tokens=0, output_tokens=0, stop_reason="tool_use", model=model,
    )


# ── structural connections ────────────────────────────────────────────────────

def test_connections_detect_write_read_link():
    analysis = parse_slither_json(_rich())
    hyps = [_hyp("borrow bug", ["borrow"]), _hyp("liquidate bug", ["liquidate"])]
    agent = ChainerAgent(provider=MockProvider())
    conns = agent._build_connections(hyps, analysis)
    # borrow writes borrowedAmount, liquidate reads it → a link must exist.
    assert any("borrowedamount" in c.lower() for c in conns)


def test_connections_empty_without_analysis():
    hyps = [_hyp("a", ["borrow"]), _hyp("b", ["liquidate"])]
    agent = ChainerAgent(provider=MockProvider())
    assert agent._build_connections(hyps, None) == []


# ── composition ────────────────────────────────────────────────────────────────

async def test_fewer_than_two_hypotheses_is_noop():
    agent = ChainerAgent(provider=MockProvider())
    out = await agent.run(_state([_hyp("only one", ["borrow"])]))
    assert out["chains"] == []
    assert len(out["hypotheses"]) == 1


async def test_chain_becomes_attackchain_and_composite_hypothesis():
    hyps = [_hyp("borrow bug", ["borrow"], score=70.0),
            _hyp("liquidate bug", ["liquidate"], sev="high", score=75.0)]

    async def cb(*, model, **kw):
        return _chain_response(model, [{
            "title": "Inflate debt then liquidate",
            "combined_severity": "critical",
            "root_cause": "borrowedAmount trusted across functions",
            "combined_scenario": "Attacker borrows to set borrowedAmount, then liquidates.",
            "component_indices": [0, 1],
            "steps": [
                {"order": 1, "function": "borrow", "action": "inflate borrowedAmount", "effect": "state X set"},
                {"order": 2, "function": "liquidate", "action": "liquidate on stale value", "effect": "drain"},
            ],
        }])

    agent = ChainerAgent(provider=MockProvider(callback=cb))
    out = await agent.run(_state(hyps))

    assert len(out["chains"]) == 1
    chain = out["chains"][0]
    assert chain.combined_severity == "critical"
    assert set(chain.component_hypothesis_ids) == {"borrow-bug", "liquidate-bug"}
    assert chain.priority_score >= 85.0  # escalated above components
    assert len(chain.steps) == 2 and chain.steps[0].hypothesis_id == "borrow-bug"

    # A composite hypothesis was appended (2 original + 1 composite).
    assert len(out["hypotheses"]) == 3
    composite = [h for h in out["hypotheses"] if h.vulnerability_class == "Attack Chain"]
    assert len(composite) == 1
    assert composite[0].estimated_severity == "critical"
    assert "Chain steps:" in composite[0].attack_scenario


async def test_single_component_chain_is_rejected():
    hyps = [_hyp("a", ["borrow"]), _hyp("b", ["liquidate"])]

    async def cb(*, model, **kw):
        return _chain_response(model, [{
            "title": "not really a chain", "combined_severity": "high",
            "combined_scenario": "x", "component_indices": [0], "steps": [],
        }])

    agent = ChainerAgent(provider=MockProvider(callback=cb))
    out = await agent.run(_state(hyps))
    assert out["chains"] == []
    assert len(out["hypotheses"]) == 2  # no composite added


async def test_llm_failure_is_graceful():
    hyps = [_hyp("a", ["borrow"]), _hyp("b", ["liquidate"])]
    agent = ChainerAgent(provider=MockProvider(side_effect=RuntimeError("down")))
    out = await agent.run(_state(hyps))
    assert out["chains"] == []
    assert len(out["hypotheses"]) == 2


async def test_out_of_range_indices_ignored():
    hyps = [_hyp("a", ["borrow"]), _hyp("b", ["liquidate"])]

    async def cb(*, model, **kw):
        return _chain_response(model, [{
            "title": "bad indices", "combined_severity": "high", "combined_scenario": "x",
            "component_indices": [0, 99], "steps": [],  # 99 invalid → only 1 valid → rejected
        }])

    agent = ChainerAgent(provider=MockProvider(callback=cb))
    out = await agent.run(_state(hyps))
    assert out["chains"] == []
