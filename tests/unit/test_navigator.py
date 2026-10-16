"""Tests for the agentic codebase navigator (roadmap piece #3).

Tool tests run against the rich Slither fixture + the real test-lending source
(FlatPool.sol). The ReAct-loop tests drive a scripted MockProvider that emits
`investigate` tool calls, verifying tools execute, observations feed back, and
the report/force-report termination paths work — all offline, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from audit_agent.agents.navigator import CodebaseNavigator, NavigatorSession
from audit_agent.knowledge.graph.builders import parse_slither_json
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.llm.mock_provider import MockProvider

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "slither_rich.json"
LENDING = Path(__file__).resolve().parents[2] / "test-contracts" / "test-lending"


def _navigator() -> CodebaseNavigator:
    analysis = parse_slither_json(json.loads(FIXTURE.read_text()))
    in_scope = [str(LENDING / "src" / "FlatPool.sol")]
    return CodebaseNavigator(analysis, in_scope_files=in_scope, target_dir=str(LENDING))


# ── Tool correctness (pure, no LLM) ───────────────────────────────────────────

def test_facts_surface_taint_and_external_targets():
    nav = _navigator()
    facts = nav.get_function_facts("borrow")
    assert "borrowedAmount" in facts
    assert "TAINTED" in facts
    assert "priceOracle" in facts  # external call target named by its state var


def test_state_writers_span_functions():
    nav = _navigator()
    writers = nav.get_state_writers("borrowedAmount")
    assert "borrow" in writers
    assert "liquidate" in writers


def test_callers_reverse_edges():
    nav = _navigator()
    callers = nav.get_callers("_getCollateralValue")
    assert "borrow" in callers
    assert "liquidate" in callers


def test_grep_searches_real_source():
    nav = _navigator()
    out = nav.grep("getPrice")
    assert "FlatPool.sol" in out
    assert "getPrice" in out


def test_read_function_returns_source():
    nav = _navigator()
    out = nav.read_function("borrow")
    assert "borrow" in out
    # body or header present; fixture carries source_code so body should appear
    assert "collateral" in out.lower() or "borrowedAmount" in out


def test_unknown_function_is_graceful():
    nav = _navigator()
    assert "not found" in nav.read_function("doesNotExist")
    assert "(tool error" not in nav.run_tool("read_function", "doesNotExist")


# ── ReAct loop ────────────────────────────────────────────────────────────────

def _investigate_response(model: str, args: dict) -> LLMResponse:
    return LLMResponse(
        text="", tool_calls=[ToolCall(id="t", name="investigate", arguments=args)],
        input_tokens=0, output_tokens=0, stop_reason="tool_use", model=model,
    )


async def test_session_runs_tools_then_reports():
    seen_messages: list[list] = []
    script = [
        {"thought": "look at borrow", "action": "read_function", "target": "borrow"},
        {"thought": "check taint", "action": "get_function_facts", "target": "borrow"},
        {"thought": "report", "action": "report", "hypotheses": [
            {"title": "Oracle price trust in borrow", "vulnerability_class": "oracle",
             "affected_functions": ["borrow"], "attack_scenario": "manipulate getPrice",
             "estimated_severity": "high", "priority_score": 80.0},
        ]},
    ]
    calls = {"i": 0}

    async def cb(*, model, system, messages, tools=None, tool_choice=None,
                 temperature=0.0, max_tokens=4096):
        seen_messages.append(messages)
        args = script[calls["i"]]
        calls["i"] += 1
        return _investigate_response(model, args)

    session = NavigatorSession(MockProvider(callback=cb), "m", _navigator(), max_steps=6)
    hyps = await session.investigate("LEAD: investigate borrow", audit_id="a1")

    assert calls["i"] == 3
    assert len(hyps) == 1 and hyps[0]["title"].startswith("Oracle price")
    # The 3rd call must have seen observations from the prior two tool steps.
    final_msgs = "\n".join(m["content"] for m in seen_messages[-1])
    assert "OBSERVATION (read_function)" in final_msgs
    assert "OBSERVATION (get_function_facts)" in final_msgs
    assert "TAINTED" in final_msgs  # facts observation was fed back


async def test_session_force_reports_on_budget_exhaustion():
    """A model that never reports must still terminate at the step budget."""
    calls = {"i": 0}

    async def cb(*, model, system, messages, tools=None, tool_choice=None,
                 temperature=0.0, max_tokens=4096):
        calls["i"] += 1
        # Always ask to read again — never reports voluntarily.
        return _investigate_response(model, {"thought": "more", "action": "read_function",
                                             "target": "borrow"})

    session = NavigatorSession(MockProvider(callback=cb), "m", _navigator(), max_steps=3)
    hyps = await session.investigate("LEAD: investigate borrow")

    # Loop stopped at the budget (did not run forever) and returned no hypotheses.
    assert calls["i"] == 3
    assert hyps == []
