"""Unit tests for the LangGraph workflow skeleton."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from audit_agent.graph.workflow import build_graph
from audit_agent.graph.state import AuditState


@pytest.mark.asyncio
async def test_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_scout_node_called(tmp_path) -> None:
    """Smoke test: graph runs Scout and returns state."""
    from audit_agent.models.schemas import ForkAnalysis, ProtocolSummary

    mock_state_update = {
        "protocol_summary": ProtocolSummary(
            name="Mock", category="other", key_mechanisms=[], user_roles=[], sol_files=[]
        ),
        "fork_analysis": ForkAnalysis(is_fork=False),
        "static_findings": [],
    }

    with patch(
        "audit_agent.graph.nodes.ScoutAgent.run",
        new=AsyncMock(return_value=mock_state_update),
    ):
        graph = build_graph()
        initial: AuditState = {
            "audit_id": "test-graph-001",
            "mode": "review",
            "target_dir": str(tmp_path),
            "platform": "generic",
            "budget_tokens": 100_000,
            "budget_usd": 10.0,
            "invariants": [],
            "static_findings": [],
            "hypotheses": [],
            "exploits": [],
            "validated_findings": [],
            "tokens_used": 0,
            "cost_usd": 0.0,
            "errors": [],
            "run_checkpoint_id": "",
        }

        result = await graph.ainvoke(initial)

    assert result["protocol_summary"].name == "Mock"
