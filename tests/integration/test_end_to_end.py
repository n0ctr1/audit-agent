"""Integration tests — require Docker (testcontainers).

Run with:
    pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "contracts"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_initial_state(
    audit_id: str,
    target_dir: str,
    mode: str = "review",
) -> dict:  # type: ignore[type-arg]
    return {
        "audit_id": audit_id,
        "mode": mode,
        "target_dir": target_dir,
        "docs_dir": None,
        "known_issues": None,
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
        "degraded_tools": [],
    }


# ── test 1: audit persists to Postgres ────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_audit_persists_to_postgres(
    apply_migrations, reset_db_engine, monkeypatch, tmp_path
):
    """Full audit in mock mode writes audit_id to Postgres with status='completed'."""
    import shutil

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "mock")

    import audit_agent.config as cfg_mod
    cfg_mod.reset_settings()

    # Copy fixture contracts to tmp path so we don't mutate the source
    contract_dir = tmp_path / "contracts"
    shutil.copytree(FIXTURES_DIR, contract_dir)

    from audit_agent.cli import _run_audit
    from audit_agent.observability.logging import get_logger

    audit_id = str(uuid.uuid4())
    await _run_audit(
        target_dir=str(contract_dir),
        mode="review",
        docs_dir=None,
        platform="generic",
        budget_usd=10.0,
        start_metrics=False,
        log=get_logger("test"),
        audit_id=audit_id,
    )

    # Verify row exists in Postgres
    from sqlalchemy import select

    from audit_agent.storage.postgres import AuditRun, get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(AuditRun).where(AuditRun.id == uuid.UUID(audit_id))
        )
        run = result.scalar_one_or_none()

    assert run is not None, "AuditRun row was not written to Postgres"
    assert run.status == "completed", f"Expected status='completed', got '{run.status}'"
    assert str(run.id) == audit_id


# ── test 2: audit is resumable via MemorySaver ───────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_audit_resumable_from_checkpoint(monkeypatch, tmp_path):
    """Graph with MemorySaver persists state; re-invoking with same thread_id
    returns the cached final state without re-running Scout."""
    import shutil

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "mock")

    import audit_agent.config as cfg_mod
    cfg_mod.reset_settings()

    contract_dir = tmp_path / "contracts"
    shutil.copytree(FIXTURES_DIR, contract_dir)

    from langgraph.checkpoint.memory import MemorySaver

    from audit_agent.graph.workflow import build_graph

    saver = MemorySaver()
    graph = build_graph(checkpointer=saver)

    audit_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": audit_id}}

    initial = _make_initial_state(audit_id, str(contract_dir))

    # First run — Scout executes
    run_count = {"n": 0}
    original_scout = None

    import audit_agent.agents.scout as scout_mod

    original_run = scout_mod.ScoutAgent.run

    async def counting_run(self, state):  # type: ignore[no-untyped-def]
        run_count["n"] += 1
        return await original_run(self, state)

    monkeypatch.setattr(scout_mod.ScoutAgent, "run", counting_run)

    state_after_first = await graph.ainvoke(initial, config=config)
    assert run_count["n"] == 1, "Scout should have run once"

    # Checkpoint is now stored in MemorySaver
    snapshot = await graph.aget_state(config)
    assert snapshot is not None, "Checkpoint should have been saved"
    saved_audit_id = snapshot.values.get("audit_id")
    assert saved_audit_id == audit_id

    # "Resume" — re-invoke with same thread_id and None (use checkpointed state)
    # In this graph (single node, already completed), LangGraph returns the
    # cached final state immediately without re-running Scout.
    state_after_resume = await graph.ainvoke(None, config=config)  # type: ignore[arg-type]
    assert run_count["n"] == 1, "Scout must NOT run again — state was checkpointed"
    assert state_after_resume["audit_id"] == audit_id


# ── test 3: strict mode fails loudly when tool unavailable ───────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_unavailable_fails_loudly_in_strict_mode(monkeypatch, tmp_path):
    """In strict mode, ToolUnavailableError propagates all the way up (no silent success)."""
    import shutil
    from unittest.mock import patch

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")

    import audit_agent.config as cfg_mod
    cfg_mod.reset_settings()

    contract_dir = tmp_path / "contracts"
    shutil.copytree(FIXTURES_DIR, contract_dir)

    # Force Docker to appear unavailable regardless of environment
    with patch("docker.from_env", side_effect=Exception("No Docker daemon")):
        from audit_agent.tools.errors import ToolUnavailableError
        from audit_agent.graph.workflow import build_graph

        graph = build_graph(checkpointer=None)
        initial = _make_initial_state(str(uuid.uuid4()), str(contract_dir))

        with pytest.raises(ToolUnavailableError) as exc_info:
            await graph.ainvoke(initial)

    assert "docker" in str(exc_info.value).lower() or exc_info.value.tool == "docker"
