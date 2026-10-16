"""Tests for the breadth-triage model-cascade stage (roadmap #7)."""

from __future__ import annotations

import json
from pathlib import Path

from audit_agent.agents.hunter import HunterAgent
from audit_agent.agents.triage import FunctionTriager
from audit_agent.knowledge.graph.builders import parse_slither_json
from audit_agent.knowledge.graph.context import (
    InMemoryGraph, SubgraphContext, SubgraphContextBuilder,
)
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.llm.mock_provider import MockProvider

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "slither_rich.json"


def _analysis():
    return parse_slither_json(json.loads(FIXTURE.read_text()))


def _all_fns(analysis):
    return [(c.name, fn) for c in analysis.contracts for fn in c.functions]


# ── FunctionTriager ────────────────────────────────────────────────────────────

async def test_triage_scores_every_function_across_batches():
    analysis = _analysis()
    fns = _all_fns(analysis)
    assert len(fns) > 5  # enough to span multiple small batches

    seen_batches = {"n": 0}

    async def cb(*, model, system, messages, tools=None, tool_choice=None, **kw):
        seen_batches["n"] += 1
        # Parse the indices we were given and flag them all with suspicion 70.
        content = messages[0]["content"]
        flags = []
        import re
        for m in re.finditer(r"\[(\d+)\]", content):
            flags.append({"index": int(m.group(1)), "suspicion": 70})
        return LLMResponse(
            text="", tool_calls=[ToolCall(id="t", name="report_triage", arguments={"flags": flags})],
            input_tokens=0, output_tokens=0, stop_reason="tool_use", model=model,
        )

    triager = FunctionTriager(provider=MockProvider(callback=cb), batch_size=5)
    scores = triager.triage  # alias
    result = await triager.triage(fns, audit_id="a")

    # Every function got a score, and batching actually happened (>1 batch).
    assert len(result) == len(fns)
    assert all(v == 70 for v in result.values())
    assert seen_batches["n"] >= 2


async def test_triage_graceful_on_failure():
    analysis = _analysis()
    triager = FunctionTriager(provider=MockProvider(side_effect=RuntimeError("down")))
    result = await triager.triage(_all_fns(analysis), audit_id="a")
    assert result == {}  # no crash, no scores


async def test_triage_empty_input():
    triager = FunctionTriager(provider=MockProvider())
    assert await triager.triage([], audit_id="a") == {}


# ── Hunter cascade integration ──────────────────────────────────────────────────

class _FakeTriager:
    """Returns a fixed suspicion map; never calls an LLM."""
    def __init__(self, scores):
        self._scores = scores

    async def triage(self, functions, audit_id=""):
        return dict(self._scores)


async def test_triage_boost_promotes_low_surface_function_to_depth():
    """A function with low static surface but high triage suspicion must reach the
    expensive depth pass (top-N), proving the cascade promotes breadth findings."""
    analysis = _analysis()
    lending = [c for c in analysis.contracts if c.name == "CollateralLendingPool"][0]
    # A real, low-surface function that would NOT win top-1 on static score alone.
    low_fn = next(f for f in lending.functions if f.name == "depositCollateral")
    top_by_score = max(
        (f for c in analysis.contracts for f in c.functions if f.attack_surface_score > 0),
        key=lambda f: f.attack_surface_score,
    )
    assert low_fn.id != top_by_score.id  # it is not the natural winner

    hunter = HunterAgent(
        provider=MockProvider(), top_functions=1,
        triager=_FakeTriager({low_fn.id: 100.0}),
    )
    # Make the suspicion boost decisive so the assertion is deterministic and does
    # not depend on the fixture's exact score spread — we are testing the WIRING.
    hunter._triage_weight = 10_000.0
    context_builder = SubgraphContextBuilder(graph=InMemoryGraph(analysis))

    processed: list[str] = []

    async def tracking(fn, contract_name, *args, **kwargs):
        processed.append(fn.id)
        return [], (fn, contract_name, SubgraphContext(entry_function_id=fn.id, code_slice=""))

    hunter._analyze_one_function = tracking  # type: ignore[assignment]

    await hunter._generate(
        analysis=analysis, audit_id="a", context_builder=context_builder,
        static_findings=[], in_scope_files=[],
    )
    # With top_functions=1 and a decisive triage boost, the suspicious low-surface
    # function is the one promoted into the expensive depth pass.
    assert low_fn.id in processed
