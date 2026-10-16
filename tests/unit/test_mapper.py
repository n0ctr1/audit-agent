"""Unit tests for Mapper: Slither JSON parsing + graph building.

All tests offline — no Neo4j, no Docker, no API.
Uses tests/fixtures/slither_output.json as the Slither output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_agent.knowledge.graph.builders import (
    GraphBuilder,
    InMemoryGraphStore,
    SlitherAnalysis,
    parse_slither_json,
    parse_slither_json_file,
)
from audit_agent.agents.mapper import MapperAgent

FIXTURE = Path(__file__).parent.parent / "fixtures" / "slither_output.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


# ── Slither JSON parsing ───────────────────────────────────────────────────────

def test_parse_fixture_loads():
    data = load_fixture()
    analysis = parse_slither_json(data)
    assert isinstance(analysis, SlitherAnalysis)


def test_fixture_contract_count():
    analysis = parse_slither_json(load_fixture())
    assert len(analysis.contracts) == 2


def test_fixture_contract_names():
    analysis = parse_slither_json(load_fixture())
    names = {c.name for c in analysis.contracts}
    assert names == {"VulnerableBank", "SimpleToken"}


def test_vulnerable_bank_function_count():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    assert len(bank.functions) == 6  # constructor + deposit + withdraw + emergencyDrain + calculateShares + collectFee


def test_simple_token_function_count():
    analysis = parse_slither_json(load_fixture())
    token = next(c for c in analysis.contracts if c.name == "SimpleToken")
    assert len(token.functions) == 4  # constructor + transfer + approve + transferFrom


def test_withdraw_has_external_call():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    withdraw = next(f for f in bank.functions if f.name == "withdraw")
    assert withdraw.has_external_calls is True
    assert len(withdraw.external_calls) == 1
    assert withdraw.external_calls[0].call_type == "call"
    assert withdraw.external_calls[0].target_type == "user_controlled"


def test_withdraw_reads_and_writes_balances():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    withdraw = next(f for f in bank.functions if f.name == "withdraw")
    assert "vulnerablebank.balances" in withdraw.reads
    assert "vulnerablebank.balances" in withdraw.writes


def test_withdraw_natspec_captured():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    withdraw = next(f for f in bank.functions if f.name == "withdraw")
    assert "VULNERABLE" in withdraw.natspec.notice
    assert "reentrancy" in withdraw.natspec.notice.lower()


def test_deposit_is_payable_and_handles_value():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    deposit = next(f for f in bank.functions if f.name == "deposit")
    assert deposit.state_mutability == "payable"
    assert deposit.handles_value is True


def test_emergency_drain_is_privileged():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    drain = next(f for f in bank.functions if f.name == "emergencyDrain")
    assert drain.is_privileged is True


def test_state_variables_parsed():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    sv_names = {sv.name for sv in bank.state_variables}
    assert sv_names == {"balances", "owner", "totalShares", "feeRate", "feesCollected"}


def test_events_parsed():
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    event_names = {e.name for e in bank.events}
    assert event_names == {"Deposited", "Withdrawn"}


def test_attack_surface_score_computed_post_parse():
    """After parsing, every function should have a non-negative score."""
    analysis = parse_slither_json(load_fixture())
    for fn in analysis.all_functions():
        assert fn.attack_surface_score >= 0.0


def test_withdraw_highest_score_in_bank():
    """withdraw() and emergencyDrain() should have the highest scores in VulnerableBank."""
    analysis = parse_slither_json(load_fixture())
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    scores = {fn.name: fn.attack_surface_score for fn in bank.functions}
    high_score_fns = {"withdraw", "emergencyDrain"}
    # Both should score higher than deposit and calculateShares
    for fn_name in high_score_fns:
        assert scores[fn_name] > scores.get("calculateShares", 0)


def test_all_functions_accessor():
    analysis = parse_slither_json(load_fixture())
    all_fns = analysis.all_functions()
    # VulnerableBank(6) + SimpleToken(4)
    assert len(all_fns) == 10


def test_contract_by_id():
    analysis = parse_slither_json(load_fixture())
    c = analysis.contract_by_id("vulnerablebank")
    assert c is not None
    assert c.name == "VulnerableBank"


def test_function_by_id():
    analysis = parse_slither_json(load_fixture())
    fn = analysis.function_by_id("vulnerablebank.withdraw(uint256)")
    assert fn is not None
    assert fn.name == "withdraw"


def test_parse_slither_json_file():
    analysis = parse_slither_json_file(FIXTURE)
    assert len(analysis.contracts) == 2


# ── Graph builder ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_builder_creates_contract_nodes():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    stats = await builder.build(analysis)

    assert stats.contracts == 2
    # Verify MERGE queries were issued for Contract nodes
    contract_merges = store.queries_for("Contract")
    assert len(contract_merges) > 0


@pytest.mark.asyncio
async def test_builder_creates_function_nodes():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    stats = await builder.build(analysis)

    assert stats.functions == 10  # 6 + 4


@pytest.mark.asyncio
async def test_builder_creates_state_var_nodes():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    stats = await builder.build(analysis)

    # VulnerableBank: 5 vars, SimpleToken: 6 vars
    assert stats.state_vars == 11


@pytest.mark.asyncio
async def test_builder_creates_relationships():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    stats = await builder.build(analysis)

    assert stats.relationships > 0


@pytest.mark.asyncio
async def test_builder_external_call_relationship():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    await builder.build(analysis)

    # Should have CALLS_EXTERNAL relationship for withdraw()
    ext_queries = store.queries_for("CALLS_EXTERNAL")
    assert len(ext_queries) > 0


@pytest.mark.asyncio
async def test_builder_writes_relationship():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    await builder.build(analysis)

    writes_queries = store.queries_for("WRITES")
    assert len(writes_queries) > 0


@pytest.mark.asyncio
async def test_builder_defined_in_relationship():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    await builder.build(analysis)

    defined_in = store.queries_for("DEFINED_IN")
    assert len(defined_in) > 0


@pytest.mark.asyncio
async def test_in_memory_store_captures_queries():
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    await builder.build(analysis)

    assert len(store.queries) > 0


@pytest.mark.asyncio
async def test_builder_attack_surface_score_in_payload():
    """Function node payload must include attack_surface_score."""
    analysis = parse_slither_json(load_fixture())
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    await builder.build(analysis)

    # Find function queries with props
    fn_queries = [q for q in store.queries if "Function" in q.query and "attack_surface_score" in q.query]
    assert len(fn_queries) > 0


# ── MapperAgent standalone ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mapper_agent_with_fixture():
    fixture_data = load_fixture()
    mapper = MapperAgent(slither_json=fixture_data)
    result = await mapper.run_standalone(
        target_dir="tests/fixtures/contracts",
        audit_id="test-phase2",
    )
    assert result.contracts == 2
    assert result.functions == 10
    assert result.relationships > 0


@pytest.mark.asyncio
async def test_mapper_agent_top_surface_functions():
    fixture_data = load_fixture()
    mapper = MapperAgent(slither_json=fixture_data)
    result = await mapper.run_standalone("tests/fixtures/contracts", audit_id="test")
    assert len(result.top_surface_functions) > 0
    # Top function should have the highest score
    scores = [fn["score"] for fn in result.top_surface_functions]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_mapper_agent_no_errors_with_fixture():
    fixture_data = load_fixture()
    mapper = MapperAgent(slither_json=fixture_data)
    result = await mapper.run_standalone("tests/fixtures/contracts", audit_id="test")
    assert result.errors == []


# ── Problem A: Mapper scope filter ───────────────────────────────────────────

def _make_oz_slither_data() -> dict:
    """Slither fixture with 1 in-scope contract + 1 OZ dependency contract."""
    return {
        "contracts": [
            {
                "id": "mytoken",
                "name": "MyToken",
                "source_file": "src/MyToken.sol",
                "kind": "contract",
                "compiler_version": "^0.8.20",
                "is_abstract": False,
                "inherits": [],
                "natspec": {},
                "state_variables": [],
                "functions": [],
                "modifiers": [],
                "events": [],
            },
            {
                "id": "erc20",
                "name": "ERC20",
                "source_file": "lib/@openzeppelin/contracts/token/ERC20/ERC20.sol",
                "kind": "contract",
                "compiler_version": "^0.8.20",
                "is_abstract": False,
                "inherits": [],
                "natspec": {},
                "state_variables": [],
                "functions": [],
                "modifiers": [],
                "events": [],
            },
        ]
    }


@pytest.mark.asyncio
async def test_mapper_excludes_openzeppelin():
    """GraphBuilder must skip contracts from lib/@openzeppelin/ paths."""
    from audit_agent.knowledge.graph.builders import GraphBuilder, InMemoryGraphStore, parse_slither_json

    data = _make_oz_slither_data()
    analysis = parse_slither_json(data)
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    stats = await builder.build(analysis)

    assert stats.contracts == 1, f"Expected 1 in-scope contract, got {stats.contracts}"


@pytest.mark.asyncio
async def test_mapper_scope_filter_log(capsys):
    """Builder must emit scope filter log (total/included/excluded)."""
    from audit_agent.knowledge.graph.builders import GraphBuilder, InMemoryGraphStore, parse_slither_json

    data = _make_oz_slither_data()
    analysis = parse_slither_json(data)
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    await builder.build(analysis)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "excluded" in combined or "scope" in combined.lower()


@pytest.mark.asyncio
async def test_mapper_in_scope_files_param():
    """Explicit in_scope_files list takes precedence over path-based filter."""
    from audit_agent.knowledge.graph.builders import GraphBuilder, InMemoryGraphStore, parse_slither_json

    data = _make_oz_slither_data()
    analysis = parse_slither_json(data)
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    # Only pass the MyToken file as in-scope
    stats = await builder.build(analysis, in_scope_files=["src/MyToken.sol"])
    assert stats.contracts == 1
