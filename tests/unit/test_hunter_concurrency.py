"""Tests for Hunter pass parallelism (roadmap piece #1).

Verifies the shared concurrency semaphore actually bounds simultaneous in-flight
LLM calls, and that passes run concurrently rather than strictly sequentially.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from audit_agent.agents.hunter import HunterAgent, _gather_limited
from audit_agent.knowledge.graph.context import SubgraphContext
from audit_agent.llm.base import LLMResponse
from audit_agent.llm.mock_provider import MockProvider


def _reentrancy_ctx(name: str):
    fn = MagicMock()
    fn.name = name
    fn.id = name
    ctx = SubgraphContext(
        entry_function_id=name,
        # `.call{` triggers exactly the reentrancy expert in _select_experts.
        code_slice="function f() external { msg.sender.call{value: 1}(''); }",
    )
    return (fn, "C", ctx)


async def test_expert_passes_respect_concurrency_limit():
    """With max_concurrency=3 and 8 expert tasks, peak in-flight must stay ≤ 3."""
    tracker = {"inflight": 0, "peak": 0}

    async def cb(**kwargs):
        tracker["inflight"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["inflight"])
        await asyncio.sleep(0.02)
        tracker["inflight"] -= 1
        return LLMResponse(
            text="", tool_calls=[], input_tokens=0, output_tokens=0,
            stop_reason="end_turn", model="m",
        )

    fn_contexts = [_reentrancy_ctx(f"fn{i}") for i in range(8)]
    hunter = HunterAgent(provider=MockProvider(callback=cb), max_concurrency=3)

    await hunter._expert_passes(fn_contexts, "audit-1", budget_per_expert=2, budget_total=100)

    assert tracker["peak"] <= 3, f"semaphore breached: peak={tracker['peak']}"
    assert tracker["peak"] >= 2, "passes did not actually run concurrently"


async def test_gather_limited_preserves_order_and_caps():
    """_gather_limited keeps input order and never exceeds the semaphore size."""
    tracker = {"inflight": 0, "peak": 0}
    sem = asyncio.Semaphore(2)

    async def task(i: int):
        tracker["inflight"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["inflight"])
        await asyncio.sleep(0.01)
        tracker["inflight"] -= 1
        return i

    results = await _gather_limited([task(i) for i in range(6)], sem)
    assert results == [0, 1, 2, 3, 4, 5]
    assert tracker["peak"] <= 2


async def test_gather_limited_isolates_exceptions():
    """One failing coroutine must not abort the others (return_exceptions)."""
    sem = asyncio.Semaphore(4)

    async def ok(i):
        return i

    async def boom():
        raise RuntimeError("nope")

    results = await _gather_limited([ok(1), boom(), ok(3)], sem)
    assert results[0] == 1
    assert isinstance(results[1], Exception)
    assert results[2] == 3
