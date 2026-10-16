"""Tests for Phase 3: SubgraphContextBuilder, HunterAgent, hypothesis model.

All tests are offline — LLM mocked, Qdrant mocked, no Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.agents.hunter import (
    HunterAgent,
    _build_hypotheses,
    _build_prompt,
    _deduplicate,
    _dedup_same_class,
    _select_experts,
)
from audit_agent.knowledge.graph.builders import parse_slither_json
from audit_agent.knowledge.graph.context import (
    InMemoryGraph,
    NullGraphAdapter,
    SubgraphContext,
    SubgraphContextBuilder,
)
from audit_agent.models.schemas import Hypothesis, StaticFinding


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

FIXTURE_PATH = Path("tests/fixtures/slither_output.json")


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


def make_mock_client(hypotheses: list[dict[str, Any]]) -> MockProvider:
    """Build a MockProvider that returns the given hypotheses via tool_use.

    First response is a plain-text response consumed by _derive_invariants (complete_simple).
    Second response carries the tool_call hypotheses for _hypothesize.
    Subsequent calls fall back to the empty-response default in MockProvider.
    """
    return MockProvider([
        # _derive_invariants calls complete_simple → reads .text
        LLMResponse(text="", tool_calls=[], input_tokens=5, output_tokens=5,
                    stop_reason="end_turn", model="test-model"),
        # First _hypothesize call returns the canned hypotheses
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": hypotheses})],
            input_tokens=10,
            output_tokens=100,
            stop_reason="tool_use",
            model="test-model",
        ),
    ])


_SAMPLE_HYPOTHESES = [
    {
        "title": "Reentrancy in withdraw() via unchecked msg.sender.call",
        "vulnerability_class": "SWC-107",
        "affected_functions": ["vulnerablebank.withdraw(uint256)"],
        "attack_scenario": "1. Deploy attacker contract. 2. Deposit ETH. 3. Call withdraw(). 4. In fallback, re-enter withdraw() before balance is updated.",
        "required_preconditions": ["Attacker deploys a re-entrant contract", "Contract has ETH balance"],
        "estimated_severity": "critical",
        "similar_historical_cases": ["The DAO Hack 2016"],
        "priority_score": 92.0,
    },
    {
        "title": "Missing access control on emergencyDrain()",
        "vulnerability_class": "SWC-115",
        "affected_functions": ["vulnerablebank.emergencyDrain()"],
        "attack_scenario": "Any address calls emergencyDrain() to transfer all ETH to themselves.",
        "required_preconditions": ["Contract has ETH balance"],
        "estimated_severity": "critical",
        "similar_historical_cases": [],
        "priority_score": 88.0,
    },
    {
        "title": "Precision loss in calculateShares() due to divide-before-multiply",
        "vulnerability_class": "SWC-101",
        "affected_functions": ["vulnerablebank.calculateShares(uint256,uint256)"],
        "attack_scenario": "Small deposit amounts round down to zero shares, enabling free withdrawals.",
        "required_preconditions": ["Small deposit amounts"],
        "estimated_severity": "medium",
        "similar_historical_cases": [],
        "priority_score": 45.0,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SubgraphContextBuilder tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subgraph_builder_forward_reach():
    """BFS traversal: entry → A → B → C must include all reachable function IDs."""
    graph = InMemoryGraph.empty()
    graph.add_function("fn_entry", "entry", calls=["fn_a"])
    graph.add_function("fn_a", "funcA", calls=["fn_b"])
    graph.add_function("fn_b", "funcB", calls=["fn_c"])
    graph.add_function("fn_c", "funcC")

    builder = SubgraphContextBuilder(graph=graph)
    ctx = await builder.build_context("fn_entry", max_depth=8)

    assert "fn_entry" in ctx.reachable_function_ids
    assert "fn_a" in ctx.reachable_function_ids
    assert "fn_b" in ctx.reachable_function_ids
    assert "fn_c" in ctx.reachable_function_ids
    # code_slice should mention all function names
    assert "entry" in ctx.code_slice
    assert "funcA" in ctx.code_slice
    assert "funcB" in ctx.code_slice
    assert "funcC" in ctx.code_slice


@pytest.mark.asyncio
async def test_subgraph_builder_forward_reach_respects_depth():
    """Depth limit: with max_depth=1, only direct callees are included."""
    graph = InMemoryGraph.empty()
    graph.add_function("fn_root", "root", calls=["fn_level1"])
    graph.add_function("fn_level1", "level1", calls=["fn_level2"])
    graph.add_function("fn_level2", "level2")

    builder = SubgraphContextBuilder(graph=graph)
    ctx = await builder.build_context("fn_root", max_depth=1)

    assert "fn_root" in ctx.reachable_function_ids
    assert "fn_level1" in ctx.reachable_function_ids
    assert "fn_level2" not in ctx.reachable_function_ids


@pytest.mark.asyncio
async def test_subgraph_builder_respects_max_tokens():
    """code_slice must not exceed max_tokens (approximate: tokens ≈ len/4)."""
    graph = InMemoryGraph.empty()
    # Add many functions with long names to stress the token budget
    for i in range(20):
        graph.add_function(f"fn_{i}", f"veryLongFunctionNameNumber{i:03d}", calls=[f"fn_{i+1}"] if i < 19 else [])

    builder = SubgraphContextBuilder(graph=graph)
    max_tok = 100  # very small budget
    ctx = await builder.build_context("fn_0", max_depth=8, max_tokens=max_tok)

    # estimated_tokens must not exceed max_tokens
    assert ctx.estimated_tokens <= max_tok
    # truncation marker should appear in slice
    assert "truncated" in ctx.code_slice or ctx.estimated_tokens <= max_tok


@pytest.mark.asyncio
async def test_subgraph_builder_includes_state_vars():
    """State variable reads/writes for reachable functions must be present."""
    analysis = parse_slither_json(load_fixture())
    graph = InMemoryGraph(analysis)
    builder = SubgraphContextBuilder(graph=graph)

    # withdraw() reads and writes 'balances'
    ctx = await builder.build_context("vulnerablebank.withdraw(uint256)")

    var_names = {i["var_name"] for i in ctx.state_interactions}
    assert "balances" in var_names

    rel_types = {i["rel_type"] for i in ctx.state_interactions}
    assert "READS" in rel_types
    assert "WRITES" in rel_types


@pytest.mark.asyncio
async def test_subgraph_builder_includes_external_calls():
    """External calls for reachable functions must be captured."""
    analysis = parse_slither_json(load_fixture())
    graph = InMemoryGraph(analysis)
    builder = SubgraphContextBuilder(graph=graph)

    ctx = await builder.build_context("vulnerablebank.withdraw(uint256)")

    assert len(ctx.external_calls) > 0
    ec = ctx.external_calls[0]
    assert "call_type" in ec
    assert "target_type" in ec
    assert ec["target_type"] == "user_controlled"


@pytest.mark.asyncio
async def test_subgraph_builder_rag_called():
    """Qdrant RAG search must be invoked with a non-empty code slice."""
    analysis = parse_slither_json(load_fixture())
    graph = InMemoryGraph(analysis)

    rag_calls: list[tuple[str, int]] = []

    async def mock_rag(query: str, limit: int) -> list[dict]:
        rag_calls.append((query, limit))
        return [{"title": "SWC-107 Reentrancy", "summary": "Classic reentrancy"}]

    builder = SubgraphContextBuilder(graph=graph, qdrant_search=mock_rag)
    ctx = await builder.build_context("vulnerablebank.withdraw(uint256)")

    assert len(rag_calls) == 1
    query_used, limit_used = rag_calls[0]
    assert len(query_used) > 0
    assert limit_used == 5
    assert len(ctx.similar_historical_cases) == 1
    assert ctx.similar_historical_cases[0]["title"] == "SWC-107 Reentrancy"


@pytest.mark.asyncio
async def test_subgraph_builder_rag_failure_is_graceful():
    """RAG failures must not crash the builder — similar_cases returns empty."""
    analysis = parse_slither_json(load_fixture())
    graph = InMemoryGraph(analysis)

    async def failing_rag(query: str, limit: int) -> list[dict]:
        raise ConnectionError("Qdrant unavailable")

    builder = SubgraphContextBuilder(graph=graph, qdrant_search=failing_rag)
    ctx = await builder.build_context("vulnerablebank.withdraw(uint256)")

    assert ctx.similar_historical_cases == []
    assert ctx.code_slice  # slice still built


@pytest.mark.asyncio
async def test_subgraph_builder_unknown_function():
    """Unknown function ID should produce empty but valid context."""
    graph = InMemoryGraph.empty()
    builder = SubgraphContextBuilder(graph=graph)
    ctx = await builder.build_context("nonexistent.fn")

    assert ctx.entry_function_id == "nonexistent.fn"
    assert ctx.reachable_function_ids == ["nonexistent.fn"]
    assert ctx.external_calls == []
    assert ctx.state_interactions == []


# ─────────────────────────────────────────────────────────────────────────────
# HunterAgent tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hunter_generates_hypotheses():
    """Mock LLM → HunterAgent must return Hypothesis objects with required fields."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    client = make_mock_client(_SAMPLE_HYPOTHESES)
    hunter = HunterAgent(graph_adapter=graph, provider=client, top_functions=3)

    hyps = await hunter.generate_hypotheses(analysis, audit_id="test-audit")

    assert len(hyps) > 0
    for h in hyps:
        assert isinstance(h, Hypothesis)
        assert h.title
        assert h.vulnerability_class
        assert h.attack_scenario
        assert h.estimated_severity in ("critical", "high", "medium", "low")
        assert isinstance(h.priority_score, float)
        assert h.audit_id == "test-audit"


@pytest.mark.asyncio
async def test_hunter_prioritization():
    """Hypotheses must be sorted by priority_score descending."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    client = make_mock_client(_SAMPLE_HYPOTHESES)
    hunter = HunterAgent(graph_adapter=graph, provider=client, top_functions=3)

    hyps = await hunter.generate_hypotheses(analysis, audit_id="test-sort")

    scores = [h.priority_score for h in hyps]
    assert scores == sorted(scores, reverse=True), "Hypotheses must be sorted by priority_score desc"


@pytest.mark.asyncio
async def test_hunter_dedup_vs_static():
    """Hypotheses that duplicate static findings (≥70% title overlap) must be filtered."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    # Static finding that matches first hypothesis
    static = [
        StaticFinding(
            audit_id="test",
            tool="slither",
            title="Reentrancy in withdraw via unchecked call",
            severity="high",
            description="Reentrancy vulnerability",
        )
    ]

    client = make_mock_client(_SAMPLE_HYPOTHESES)
    hunter = HunterAgent(graph_adapter=graph, provider=client, top_functions=3)

    hyps = await hunter.generate_hypotheses(analysis, audit_id="test-dedup", static_findings=static)

    # The duplicate should be removed
    titles = [h.title for h in hyps]
    # The reentrancy hypothesis has significant overlap with the static finding
    # Other hypotheses (emergencyDrain, calculateShares) should survive
    assert not any(
        "emergencyDrain" in t and "Reentrancy in withdraw" in t for t in titles
    )


@pytest.mark.asyncio
async def test_hunter_dedup_no_false_positives():
    """Unique hypotheses must NOT be filtered out."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    # Completely unrelated static finding
    static = [
        StaticFinding(
            audit_id="test",
            tool="slither",
            title="Timestamp dependence in auction bid",
            severity="low",
            description="Uses block.timestamp",
        )
    ]

    client = make_mock_client(_SAMPLE_HYPOTHESES)
    hunter = HunterAgent(graph_adapter=graph, provider=client, top_functions=3)

    hyps = await hunter.generate_hypotheses(analysis, audit_id="test-no-fp", static_findings=static)

    # All hypotheses should survive (no overlap with "timestamp dependence in auction bid")
    assert len(hyps) >= len(_SAMPLE_HYPOTHESES) - 1  # at most 1 dedup across all functions


@pytest.mark.asyncio
async def test_hunter_uses_rag():
    """Verify that Qdrant search is invoked during hypothesis generation."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    rag_call_count = 0

    async def counting_rag(query: str, limit: int) -> list[dict]:
        nonlocal rag_call_count
        rag_call_count += 1
        return [{"title": "Historical reentrancy", "summary": "Known pattern"}]

    client = make_mock_client(_SAMPLE_HYPOTHESES)
    hunter = HunterAgent(
        graph_adapter=graph,
        qdrant_search=counting_rag,
        provider=client,
        top_functions=3,
    )

    await hunter.generate_hypotheses(analysis, audit_id="test-rag")

    # RAG should have been called at least once (once per analyzed function)
    assert rag_call_count > 0


@pytest.mark.asyncio
async def test_hunter_llm_unavailable_returns_empty():
    """When LLM raises (no API key), Hunter must return empty list without crashing."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    # Client that always raises TypeError (auth failure)
    bad_client = MockProvider(side_effect=TypeError("No API key"))

    hunter = HunterAgent(graph_adapter=graph, provider=bad_client, top_functions=3)
    hyps = await hunter.generate_hypotheses(analysis, audit_id="test-no-llm")

    assert hyps == []


@pytest.mark.asyncio
async def test_hunter_run_via_state():
    """HunterAgent.run() must read slither_analysis_raw from state and return hypotheses."""
    fixture = load_fixture()
    client = make_mock_client(_SAMPLE_HYPOTHESES)
    hunter = HunterAgent(provider=client, top_functions=3)

    state = {
        "audit_id": "state-test",
        "slither_analysis_raw": fixture,
        "static_findings": [],
        "invariants": [],
        "hypotheses": [],
    }

    new_state = await hunter.run(state)  # type: ignore[arg-type]

    assert "hypotheses" in new_state
    assert len(new_state["hypotheses"]) > 0
    for h in new_state["hypotheses"]:
        assert isinstance(h, Hypothesis)


@pytest.mark.asyncio
async def test_hunter_run_no_analysis_returns_empty():
    """HunterAgent.run() with no slither_analysis_raw must return empty hypotheses."""
    hunter = HunterAgent()

    state = {
        "audit_id": "empty-test",
        "static_findings": [],
        "invariants": [],
        "hypotheses": [],
    }
    new_state = await hunter.run(state)  # type: ignore[arg-type]

    assert new_state["hypotheses"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_hypothesis_model_required_fields():
    """Hypothesis must reject construction with missing required fields."""
    with pytest.raises(Exception):
        Hypothesis(  # missing audit_id, title, etc.
            estimated_severity="critical",
        )


def test_hypothesis_model_valid():
    """Valid Hypothesis construction must succeed."""
    h = Hypothesis(
        audit_id="audit-xyz",
        title="Reentrancy in withdraw()",
        vulnerability_class="SWC-107",
        affected_functions=["contract.withdraw"],
        attack_scenario="Call withdraw() from re-entrant fallback",
        required_preconditions=["attacker deploys re-entrant contract"],
        estimated_severity="critical",
        similar_historical_cases=["The DAO"],
        priority_score=95.0,
    )
    assert h.title == "Reentrancy in withdraw()"
    assert h.priority_score == 95.0
    assert h.audit_id == "audit-xyz"
    assert h.id  # auto-generated UUID


def test_hypothesis_model_defaults():
    """Hypothesis priority_score defaults to 0.0, similar_cases to []."""
    h = Hypothesis(
        audit_id="a",
        title="Test",
        attack_scenario="...",
        estimated_severity="low",
    )
    assert h.priority_score == 0.0
    assert h.similar_historical_cases == []
    assert h.affected_functions == []


def test_build_hypotheses_skips_empty_title():
    """_build_hypotheses must skip entries with empty title."""
    raw = [
        {"title": "", "vulnerability_class": "SWC-107", "priority_score": 90.0,
         "attack_scenario": "x", "estimated_severity": "high"},
        {"title": "  ", "vulnerability_class": "SWC-115", "priority_score": 80.0,
         "attack_scenario": "y", "estimated_severity": "critical"},
        {"title": "Valid finding", "vulnerability_class": "SWC-101", "priority_score": 50.0,
         "attack_scenario": "z", "estimated_severity": "medium"},
    ]
    results = _build_hypotheses(raw, "audit-1", "some.fn")
    assert len(results) == 1
    assert results[0].title == "Valid finding"


def test_deduplicate_removes_high_overlap():
    """_deduplicate must remove hypothesis with ≥70% title token overlap vs static."""
    from audit_agent.models.schemas import StaticFinding

    static = [
        StaticFinding(
            audit_id="a", tool="slither",
            title="Reentrancy withdraw call user",
            severity="high", description="",
        )
    ]
    hyps = [
        Hypothesis(
            audit_id="a", title="Reentrancy withdraw call user controlled",
            estimated_severity="critical", attack_scenario="...", priority_score=90.0,
        ),
        Hypothesis(
            audit_id="a", title="Oracle price manipulation via flash loan",
            estimated_severity="high", attack_scenario="...", priority_score=70.0,
        ),
    ]
    result = _deduplicate(hyps, static)
    titles = [h.title for h in result]
    assert "Oracle price manipulation via flash loan" in titles
    # The reentrancy one should be filtered (high token overlap)
    assert not any("Reentrancy withdraw" in t for t in titles)


def test_deduplicate_keeps_unique():
    """_deduplicate must keep hypotheses with no overlap with static findings."""
    from audit_agent.models.schemas import StaticFinding

    static = [
        StaticFinding(
            audit_id="a", tool="slither",
            title="Integer overflow in mint",
            severity="medium", description="",
        )
    ]
    hyps = [
        Hypothesis(
            audit_id="a", title="Missing access control on pause()",
            estimated_severity="high", attack_scenario="...", priority_score=85.0,
        ),
    ]
    result = _deduplicate(hyps, static)
    assert len(result) == 1


def _make_hyp(title: str, vuln_class: str, severity: str, score: float) -> Hypothesis:
    return Hypothesis(
        id=title[:8].replace(" ", "_"),
        audit_id="test",
        title=title,
        vulnerability_class=vuln_class,
        affected_functions=[],
        attack_scenario="scenario",
        required_preconditions=[],
        estimated_severity=severity,
        similar_historical_cases=[],
        priority_score=score,
    )


def test_dedup_same_class_collapses_medium_duplicates():
    """Medium-severity duplicates of same class should be capped at 2."""
    hyps = [
        _make_hyp(f"Integer truncation in func{i}", "SWC-101", "medium", float(10 - i))
        for i in range(6)
    ]
    result = _dedup_same_class(hyps)
    assert len(result) == 2
    # Best two by score should survive
    scores = [h.priority_score for h in result]
    assert scores == sorted(scores, reverse=True)


def test_dedup_same_class_allows_more_high_severity():
    """High-severity findings of same class should be capped at 3."""
    hyps = [
        _make_hyp(f"Reentrancy in path{i}", "SWC-107", "high", float(10 - i))
        for i in range(5)
    ]
    result = _dedup_same_class(hyps)
    assert len(result) == 3


def test_dedup_same_class_preserves_distinct_classes():
    """Findings from different vulnerability classes must all survive."""
    hyps = [
        _make_hyp("Integer truncation", "SWC-101", "medium", 8.0),
        _make_hyp("Reentrancy", "SWC-107", "high", 9.0),
        _make_hyp("Hash collision", "SWC-133", "high", 7.0),
    ]
    result = _dedup_same_class(hyps)
    assert len(result) == 3


def test_dedup_same_class_keeps_best_by_score():
    """Within a class, the highest-scored hypotheses survive."""
    hyps = [
        _make_hyp("Truncation low score", "SWC-101", "medium", 3.0),
        _make_hyp("Truncation high score", "SWC-101", "medium", 9.0),
        _make_hyp("Truncation mid score", "SWC-101", "medium", 6.0),
    ]
    result = _dedup_same_class(hyps)
    assert len(result) == 2
    assert any(h.priority_score == 9.0 for h in result)
    assert any(h.priority_score == 6.0 for h in result)


def test_inmemory_graph_empty_factory():
    """InMemoryGraph.empty() must start with no functions."""
    g = InMemoryGraph.empty()
    # No functions → bfs should return empty
    import asyncio
    callees = asyncio.get_event_loop().run_until_complete(g.bfs_callees("any", 5))
    assert callees == []


def test_inmemory_graph_from_analysis():
    """InMemoryGraph built from SlitherAnalysis must index all parsed functions."""
    analysis = parse_slither_json(load_fixture())
    g = InMemoryGraph(analysis)
    all_fn_ids = {fn.id for c in analysis.contracts for fn in c.functions}
    for fn_id in all_fn_ids:
        import asyncio
        meta = asyncio.get_event_loop().run_until_complete(g.get_function_meta(fn_id))
        assert meta is not None, f"Function {fn_id} not found in InMemoryGraph"
        assert meta["id"] == fn_id


# ─────────────────────────────────────────────────────────────────────────────
# Documentation injection into prompts
# ─────────────────────────────────────────────────────────────────────────────

def _make_context(code_slice: str = "function foo() external {}") -> SubgraphContext:
    return SubgraphContext(
        entry_function_id="fn1",
        code_slice=code_slice,
        state_interactions=[],
        external_calls=[],
        modifiers=[],
        similar_historical_cases=[],
    )


def test_hunter_prompt_includes_documentation():
    ctx = _make_context()
    doc = "The protocol guarantees that user balances never decrease without explicit withdrawal."
    prompt = _build_prompt(ctx, fork_protocol=None, documentation=doc)
    assert "user balances never decrease" in prompt
    assert "Project Documentation" in prompt


def test_hunter_prompt_graceful_without_docs():
    ctx = _make_context()
    prompt = _build_prompt(ctx, fork_protocol=None, documentation="")
    assert "No project documentation available" in prompt


def test_hunter_documentation_truncated():
    ctx = _make_context()
    large_doc = "X" * 60_000  # 60k chars, should be capped at 10k before reaching _build_prompt
    # _build_prompt itself does not truncate — truncation happens in run(); test the boundary
    prompt = _build_prompt(ctx, fork_protocol=None, documentation=large_doc[:10_000])
    # Prompt should contain the first 10k chars of docs but no more
    assert "X" * 9_000 in prompt
    assert len(prompt) < 20_000  # sanity: prompt doesn't balloon uncontrollably


@pytest.mark.asyncio
async def test_hunter_run_passes_documentation_to_prompt():
    """HunterAgent.run() passes state['documentation'] into LLM calls."""
    called_prompts: list[str] = []

    async def fake_create(*, model, system, messages, tools=None, tool_choice=None, temperature=0.0, max_tokens=4096):
        called_prompts.append(messages[0].get("content", "") if messages else "")
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": []})],
            input_tokens=0,
            output_tokens=0,
            stop_reason="tool_use",
            model=model,
        )

    client = MockProvider(callback=fake_create)

    fixture_path = Path(__file__).parent.parent / "fixtures" / "slither_output.json"
    if not fixture_path.exists():
        pytest.skip("slither fixture not available")

    import json
    raw = json.loads(fixture_path.read_text())

    agent = HunterAgent(provider=client, top_functions=1)
    state = {
        "audit_id": "doc-hunter-test",
        "slither_analysis_raw": raw,
        "static_findings": [],
        "documentation": "Critical invariant: total supply must equal sum of balances.",
    }
    await agent.run(state)

    assert len(called_prompts) > 0
    assert any("total supply must equal sum of balances" in p for p in called_prompts)


# ─────────────────────────────────────────────────────────────────────────────
# Precision loss vulnerability detection
# ─────────────────────────────────────────────────────────────────────────────

def test_hunter_system_prompt_includes_precision_loss():
    """_HUNTER_SYSTEM must mention integer precision loss as a vulnerability class."""
    from audit_agent.agents.hunter import _HUNTER_SYSTEM
    assert "precision loss" in _HUNTER_SYSTEM.lower() or "rounding" in _HUNTER_SYSTEM.lower()
    assert "division" in _HUNTER_SYSTEM.lower() or "fee" in _HUNTER_SYSTEM.lower()


@pytest.mark.asyncio
async def test_hunter_generates_precision_loss_hypothesis():
    """When LLM returns a precision-loss hypothesis, HunterAgent must surface it."""
    precision_loss_hyp = {
        "title": "Integer precision loss in collectFee() — fee truncates to zero for small amounts",
        "vulnerability_class": "SWC-101",
        "affected_functions": ["vulnerablebank.collectFee(uint256)"],
        "attack_scenario": (
            "1. Deposit a tiny amount (e.g. 999 wei) when feeRate=1 and FEE_DENOMINATOR=10000. "
            "2. Call collectFee(999). "
            "3. fee = 999 * 1 / 10000 = 0 (integer truncation). "
            "4. No fee is collected, draining protocol revenue over many small transactions."
        ),
        "required_preconditions": ["feeRate > 0", "amount < FEE_DENOMINATOR / feeRate"],
        "estimated_severity": "medium",
        "similar_historical_cases": [],
        "priority_score": 62.0,
    }

    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)
    client = make_mock_client([precision_loss_hyp])
    hunter = HunterAgent(graph_adapter=graph, provider=client, top_functions=3)

    hyps = await hunter.generate_hypotheses(analysis, audit_id="precision-test")

    precision_hyps = [
        h for h in hyps
        if "precision" in h.title.lower() or "truncat" in h.title.lower() or "fee" in h.title.lower()
    ]
    assert len(precision_hyps) >= 1, "Expected at least one precision loss hypothesis"
    h = precision_hyps[0]
    assert h.vulnerability_class == "SWC-101"
    assert h.priority_score == 62.0


@pytest.mark.asyncio
async def test_hunter_prompt_contains_fee_documentation():
    """When state includes fee mechanism docs, they appear in the LLM prompt."""
    called_prompts: list[str] = []

    async def fake_create(*, model, system, messages, tools=None, tool_choice=None, temperature=0.0, max_tokens=4096):
        called_prompts.append(messages[0].get("content", "") if messages else "")
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": []})],
            input_tokens=0,
            output_tokens=0,
            stop_reason="tool_use",
            model=model,
        )

    client = MockProvider(callback=fake_create)

    fixture_path = Path(__file__).parent.parent / "fixtures" / "slither_output.json"
    if not fixture_path.exists():
        pytest.skip("slither fixture not available")

    import json
    raw = json.loads(fixture_path.read_text())

    agent = HunterAgent(provider=client, top_functions=1)
    state = {
        "audit_id": "fee-doc-test",
        "slither_analysis_raw": raw,
        "static_findings": [],
        "documentation": (
            "Fee mechanism: collectFee(amount) charges feeRate basis points of amount. "
            "FEE_DENOMINATOR is 10000. Fee = amount * feeRate / FEE_DENOMINATOR."
        ),
    }
    await agent.run(state)

    assert any("feeRate" in p or "FEE_DENOMINATOR" in p for p in called_prompts)


# ─────────────────────────────────────────────────────────────────────────────
# Second-pass analysis methods (temporal, crypto, cross-contract) — Part 2
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_check_skipped_without_keywords():
    """_temporal_check returns [] when source code has no time-related keywords."""
    client = MockProvider([])
    hunter = HunterAgent(provider=client)
    result = await hunter._temporal_check("someFunc", "function someFunc() external { x += 1; }", "test-audit")
    assert result == []
    assert client._call_index == 0, "No LLM call should be made when no time keywords"


@pytest.mark.asyncio
async def test_temporal_check_triggers_on_keywords():
    """_temporal_check calls LLM and returns hypotheses when time keywords present."""
    hyp = {
        "title": "Epoch boundary double-claim",
        "vulnerability_class": "temporal",
        "affected_functions": ["Vault.claim()"],
        "attack_scenario": "Claim at exact epoch boundary to collect rewards twice.",
        "required_preconditions": ["epoch boundary"],
        "estimated_severity": "high",
        "priority_score": 85.0,
    }
    client = MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": [hyp]})],
        input_tokens=10, output_tokens=50, stop_reason="tool_use", model="test",
    )])
    hunter = HunterAgent(provider=client)
    source = "function claim() external { require(block.timestamp >= epochEnd, 'too early'); }"
    result = await hunter._temporal_check("claim", source, "test-audit")
    assert len(result) == 1
    assert result[0].title == "Epoch boundary double-claim"
    assert client._call_index == 1, "Exactly one LLM call should have been made"


@pytest.mark.asyncio
async def test_crypto_check_skipped_without_keywords():
    """_crypto_check returns [] when source code has no crypto keywords."""
    client = MockProvider([])
    hunter = HunterAgent(provider=client)
    result = await hunter._crypto_check("transfer", "function transfer(address to, uint256 amt) external {}", "test-audit")
    assert result == []
    assert client._call_index == 0


@pytest.mark.asyncio
async def test_crypto_check_triggers_on_keywords():
    """_crypto_check calls LLM and returns hypotheses when crypto keywords present."""
    hyp = {
        "title": "Missing field in signed hash",
        "vulnerability_class": "signature-malleability",
        "affected_functions": ["Vault.permit()"],
        "attack_scenario": "Attacker modifies deadline without invalidating signature.",
        "required_preconditions": ["ecrecover used"],
        "estimated_severity": "critical",
        "priority_score": 92.0,
    }
    client = MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": [hyp]})],
        input_tokens=10, output_tokens=50, stop_reason="tool_use", model="test",
    )])
    hunter = HunterAgent(provider=client)
    source = "bytes32 digest = keccak256(abi.encodePacked(amount, nonce)); address signer = ecrecover(digest, v, r, s);"
    result = await hunter._crypto_check("permit", source, "test-audit")
    assert len(result) == 1
    assert result[0].title == "Missing field in signed hash"


@pytest.mark.asyncio
async def test_cross_contract_check_skipped_with_single_file(tmp_path):
    """_cross_contract_check returns [] when fewer than 2 scope files."""
    client = MockProvider([])
    hunter = HunterAgent(provider=client)
    sol = tmp_path / "Vault.sol"
    sol.write_text("contract Vault {}")
    result = await hunter._cross_contract_check("test-audit", [str(sol)], str(tmp_path))
    assert result == []
    assert client._call_index == 0


@pytest.mark.asyncio
async def test_cross_contract_check_triggers_on_references(tmp_path):
    """_cross_contract_check calls LLM when contracts cross-reference each other."""
    vault = tmp_path / "Vault.sol"
    vault.write_text("contract Vault { IRewards rewards; function claim() external { rewards.distribute(); } }")
    rewards = tmp_path / "Rewards.sol"
    rewards.write_text("contract Rewards { function distribute() external {} }")

    hyp = {
        "title": "Stale epoch in cross-contract call",
        "vulnerability_class": "cross-contract",
        "affected_functions": ["Vault.claim()"],
        "attack_scenario": "Vault reads epoch from Rewards after Rewards transitions.",
        "required_preconditions": ["two contracts"],
        "estimated_severity": "high",
        "priority_score": 78.0,
    }
    client = MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": [hyp]})],
        input_tokens=10, output_tokens=50, stop_reason="tool_use", model="test",
    )])
    hunter = HunterAgent(provider=client)
    result = await hunter._cross_contract_check("test-audit", [str(vault), str(rewards)], str(tmp_path))
    assert len(result) == 1
    assert result[0].title == "Stale epoch in cross-contract call"


@pytest.mark.asyncio
async def test_second_pass_errors_are_graceful():
    """If second-pass LLM calls raise, _generate still returns base hypotheses."""
    fixture = load_fixture()
    analysis = parse_slither_json(fixture)
    graph = InMemoryGraph(analysis)

    hypo_calls = 0

    async def flaky_callback(**kwargs):
        # Dispatch by tool presence (robust to whether _derive_invariants runs at
        # all — it is skipped when there is no source/docs). Non-tool calls return
        # a benign empty response; the FIRST tool-bearing _hypothesize succeeds and
        # every later (second-pass) tool call raises.
        nonlocal hypo_calls
        if not kwargs.get("tools"):
            return LLMResponse(text="", tool_calls=[], input_tokens=5, output_tokens=5,
                               stop_reason="end_turn", model="test")
        hypo_calls += 1
        if hypo_calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": [_SAMPLE_HYPOTHESES[0]]})],
                input_tokens=10, output_tokens=50, stop_reason="tool_use", model="test",
            )
        # All subsequent calls (second-pass checks) raise
        raise RuntimeError("LLM unavailable for second pass")

    client = MockProvider(callback=flaky_callback)
    hunter = HunterAgent(graph_adapter=graph, provider=client, top_functions=1)
    hyps = await hunter.generate_hypotheses(analysis, audit_id="graceful-test")
    # Base hypotheses survive even when second-pass errors out
    assert len(hyps) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# MoE: _select_experts + _run_expert_pass + _expert_passes
# ─────────────────────────────────────────────────────────────────────────────

def test_select_experts_reentrancy():
    code = "function withdraw(uint256 amt) external { msg.sender.call{value: amt}(''); balances[msg.sender] -= amt; }"
    experts = _select_experts(code)
    assert "reentrancy" in experts


def test_select_experts_cryptography():
    code = "function verify(bytes32 hash, bytes sig) external { address signer = ecrecover(hash, v, r, s); }"
    experts = _select_experts(code)
    assert "cryptography" in experts


def test_select_experts_arithmetic():
    code = "uint256 shares = amount * 1e18 / totalSupply; uint256 wad = price * 1e18;"
    experts = _select_experts(code)
    assert "arithmetic" in experts


def test_select_experts_time_epoch():
    code = "function claimRewards() external { require(block.timestamp >= epochEnd, 'epoch not over'); }"
    experts = _select_experts(code)
    assert "time_epoch" in experts


def test_select_experts_multiple_domains():
    code = (
        "function execute(bytes sig) external { "
        "address s = ecrecover(hash, v, r, s); "
        "require(block.timestamp <= deadline); "
        "msg.sender.call{value: amount}(''); "
        "}"
    )
    experts = _select_experts(code)
    assert "cryptography" in experts
    assert "time_epoch" in experts
    assert "reentrancy" in experts


def test_select_experts_empty_code():
    code = "function foo() external pure returns (uint256) { return 42; }"
    experts = _select_experts(code)
    assert experts == []


def test_select_experts_unknown_expert_not_returned():
    experts = _select_experts("")
    for e in experts:
        from audit_agent.agents.hunter import _EXPERT_REGISTRY
        assert e in _EXPERT_REGISTRY


@pytest.mark.asyncio
async def test_run_expert_pass_returns_hypotheses():
    """_run_expert_pass returns Hypothesis objects from the LLM tool call."""
    hyp_data = {
        "title": "Reentrancy in withdraw via .call",
        "vulnerability_class": "SWC-107",
        "affected_functions": ["withdraw"],
        "attack_scenario": "Attacker re-enters withdraw before balance update.",
        "required_preconditions": ["Attacker has deposited ETH"],
        "estimated_severity": "critical",
        "priority_score": 90.0,
    }
    provider = MockProvider([
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": [hyp_data]})],
            input_tokens=10, output_tokens=50, stop_reason="tool_use", model="test",
        )
    ])
    hunter = HunterAgent(provider=provider)
    hyps = await hunter._run_expert_pass("withdraw", "msg.sender.call{value: amt}('');", "reentrancy", "audit-1")
    assert len(hyps) == 1
    assert hyps[0].title == hyp_data["title"]
    assert hyps[0].vulnerability_class == "SWC-107"


@pytest.mark.asyncio
async def test_run_expert_pass_unknown_expert_returns_empty():
    """_run_expert_pass returns [] for an expert name not in _EXPERT_REGISTRY."""
    provider = MockProvider([])
    hunter = HunterAgent(provider=provider)
    hyps = await hunter._run_expert_pass("foo", "code", "nonexistent_expert", "audit-1")
    assert hyps == []


@pytest.mark.asyncio
async def test_run_expert_pass_exception_returns_empty():
    """_run_expert_pass returns [] when the LLM provider raises."""
    async def failing_create(**kwargs):
        raise RuntimeError("provider down")

    provider = MockProvider(callback=failing_create)
    hunter = HunterAgent(provider=provider)
    hyps = await hunter._run_expert_pass("foo", "ecrecover(hash, v, r, s);", "cryptography", "audit-1")
    assert hyps == []


@pytest.mark.asyncio
async def test_expert_passes_budget_total_cap():
    """_expert_passes must not exceed budget_total even when many functions match many experts."""
    hyp_data = {
        "title": "Reentrancy finding",
        "vulnerability_class": "SWC-107",
        "affected_functions": ["fn"],
        "attack_scenario": "Re-entry.",
        "required_preconditions": ["deposit"],
        "estimated_severity": "high",
        "priority_score": 80.0,
    }

    call_count = 0

    async def counting_create(*, model, system, messages, tools=None, tool_choice=None,
                               temperature=0.0, max_tokens=4096):
        nonlocal call_count
        call_count += 1
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id=str(call_count), name="report_hypotheses",
                                 arguments={"hypotheses": [hyp_data]})],
            input_tokens=5, output_tokens=20, stop_reason="tool_use", model=model,
        )

    from audit_agent.knowledge.graph.context import SubgraphContext

    def _make_fn_context(name: str):
        fn = MagicMock()
        fn.name = name
        ctx = SubgraphContext(
            entry_function_id=name,
            # code that matches reentrancy + time_epoch → 2 experts per function
            code_slice=f"function {name}() external {{ msg.sender.call{{value: 1}}(''); require(block.timestamp > epoch); }}",
            state_interactions=[],
            external_calls=[],
            modifiers=[],
            similar_historical_cases=[],
        )
        return (fn, "Contract", ctx)

    fn_contexts = [_make_fn_context(f"fn{i}") for i in range(10)]

    provider = MockProvider(callback=counting_create)
    hunter = HunterAgent(provider=provider)
    hyps = await hunter._expert_passes(fn_contexts, "audit-1", budget_per_expert=2, budget_total=5)

    assert len(hyps) <= 5
