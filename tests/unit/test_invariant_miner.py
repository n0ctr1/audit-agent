"""Unit tests for InvariantMiner — all offline, LLM mocked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.agents.invariant_miner import (
    InvariantMiner,
    extract_natspec_invariants,
    mine_arithmetic_invariants,
    _classify_invariant_type,
    _generate_echidna_stub,
)
from audit_agent.knowledge.graph.builders import (
    SlitherAnalysis,
    parse_slither_json,
)
from audit_agent.models.schemas import Invariant

FIXTURE = Path(__file__).parent.parent / "fixtures" / "slither_output.json"


def load_analysis() -> SlitherAnalysis:
    return parse_slither_json(json.loads(FIXTURE.read_text()))


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: NatSpec extraction (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def test_natspec_extracts_from_contract_notice():
    """Contract-level NatSpec with invariant keywords should produce an invariant."""
    analysis = load_analysis()
    # Inject a clear invariant into the contract NatSpec
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    bank.natspec.notice = "totalShares must always equal sum of all user share balances"

    results = extract_natspec_invariants(analysis, audit_id="test")
    found = [i for i in results if "totalShares" in i.natural_language]
    assert len(found) >= 1
    assert found[0].source == "natspec"


def test_natspec_skips_vulnerability_notices():
    """Lines marked VULNERABLE: should not be treated as invariants."""
    analysis = load_analysis()
    results = extract_natspec_invariants(analysis, audit_id="test")
    for inv in results:
        assert "VULNERABLE" not in inv.natural_language
        assert "BUG:" not in inv.natural_language


def test_natspec_skips_short_text():
    """Text shorter than 15 chars should not produce invariants."""
    analysis = load_analysis()
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    bank.natspec.notice = "always true"  # too short AND no invariant keyword
    results = extract_natspec_invariants(analysis, audit_id="test")
    # Should not crash, just return without adding junk
    for inv in results:
        assert len(inv.natural_language) >= 15


def test_natspec_invariant_has_echidna_property():
    analysis = load_analysis()
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    bank.natspec.notice = "totalSupply must always equal sum of all balances in the system"

    results = extract_natspec_invariants(analysis, audit_id="test")
    matching = [i for i in results if "totalSupply" in i.natural_language]
    if matching:
        inv = matching[0]
        assert inv.echidna_property.startswith("function echidna_")
        assert "public returns (bool)" in inv.echidna_property


def test_natspec_global_type_classification():
    text = "totalSupply == sum of all balances"
    assert _classify_invariant_type(text) == "global"


def test_natspec_transition_type_classification():
    text = "after deposit, user balance increases"
    assert _classify_invariant_type(text) == "transition"


def test_natspec_economic_type_classification():
    text = "user cannot withdraw more value than deposited"
    assert _classify_invariant_type(text) == "economic"


def test_echidna_stub_generated():
    stub = _generate_echidna_stub("total supply equals sum balances")
    assert stub.startswith("function echidna_")
    assert "returns (bool)" in stub


def test_natspec_invariant_has_audit_id():
    analysis = load_analysis()
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    bank.natspec.notice = "totalShares must always equal sum of all deposit shares"

    results = extract_natspec_invariants(analysis, audit_id="audit-abc")
    if results:
        assert results[0].audit_id == "audit-abc"


def test_natspec_invariant_has_affected_functions():
    analysis = load_analysis()
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    bank.natspec.notice = "totalShares must always equal sum of all deposit shares"

    results = extract_natspec_invariants(analysis, audit_id="test")
    if results:
        # Contract-level invariant should reference all functions
        assert len(results[0].affected_functions) > 0


def test_empty_analysis_returns_no_invariants():
    empty = SlitherAnalysis()
    results = extract_natspec_invariants(empty, audit_id="test")
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# LLM mock helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_client(invariants: list[dict[str, Any]]) -> MockProvider:
    return MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="extract_invariants", arguments={"invariants": invariants})],
        input_tokens=0,
        output_tokens=0,
        stop_reason="tool_use",
        model="test-model",
    )])


_SAMPLE_INVARIANTS = [
    {
        "natural_language": "totalSupply must always equal the sum of all user balances",
        "invariant_type": "global",
        "affected_functions": ["transfer", "transferFrom"],
        "solidity_assert": "assertEq(token.totalSupply(), sumBalances());",
        "echidna_property": "function echidna_total_supply() public returns (bool) { return totalSupply == sumBalances(); }",
        "halmos_spec": "",
        "confidence": 0.9,
    },
    {
        "natural_language": "after transfer(to, x), balanceOf[to] increases by x",
        "invariant_type": "transition",
        "affected_functions": ["transfer"],
        "solidity_assert": "assertEq(balanceOf[to], prevBalance + x);",
        "echidna_property": "function echidna_transfer_balance() public returns (bool) { return true; }",
        "halmos_spec": "",
        "confidence": 0.85,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: Docs extraction (LLM mocked)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_docs_extraction_returns_invariants():
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_from_docs(
        analysis, "## Invariants\n- totalSupply == sum of balances", audit_id="test"
    )
    assert len(results) == 2
    assert all(isinstance(i, Invariant) for i in results)


@pytest.mark.asyncio
async def test_docs_invariants_source_is_docs():
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_from_docs(analysis, "docs text", audit_id="test")
    for inv in results:
        assert inv.source == "docs"


@pytest.mark.asyncio
async def test_docs_invariants_have_natural_language():
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_from_docs(analysis, "docs text", audit_id="test")
    for inv in results:
        assert inv.natural_language
        assert len(inv.natural_language) > 10


@pytest.mark.asyncio
async def test_docs_invariants_have_correct_types():
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_from_docs(analysis, "docs text", audit_id="test")
    types = {inv.invariant_type for inv in results}
    assert "global" in types
    assert "transition" in types


@pytest.mark.asyncio
async def test_docs_extraction_empty_llm_response():
    """LLM returns empty list → no invariants, no crash."""
    client = _make_llm_client([])
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_from_docs(analysis, "docs text", audit_id="test")
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: Inferred extraction (LLM mocked)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_inferred_extraction_returns_invariants():
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_inferred(analysis, audit_id="test")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_inferred_invariants_source_is_inferred():
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_inferred(analysis, audit_id="test")
    for inv in results:
        assert inv.source == "inferred"


@pytest.mark.asyncio
async def test_inferred_skips_empty_natural_language():
    bad_data = [{"natural_language": "", "invariant_type": "global",
                 "affected_functions": [], "solidity_assert": "", "echidna_property": "",
                 "confidence": 0.5}]
    client = _make_llm_client(bad_data)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.extract_inferred(analysis, audit_id="test")
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Full mine_all (all three sources)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mine_all_combines_sources():
    analysis = load_analysis()
    # Inject a real natspec invariant so source 1 fires
    bank = next(c for c in analysis.contracts if c.name == "VulnerableBank")
    bank.natspec.notice = "totalShares must always equal sum of all deposit shares"

    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)

    results = await miner.mine_all(analysis, audit_id="test", docs_text="protocol docs here")
    sources = {inv.source for inv in results}
    # Should have results from at least two sources (natspec + docs or natspec + inferred)
    assert len(sources) >= 2


@pytest.mark.asyncio
async def test_mine_all_no_docs_skips_docs_source():
    """If docs_text is empty, docs extraction is skipped."""
    client = _make_llm_client(_SAMPLE_INVARIANTS)
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.mine_all(analysis, audit_id="test", docs_text="")
    # docs extraction not called, so only natspec + inferred sources
    sources = {inv.source for inv in results}
    assert "docs" not in sources


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic invariant mining (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fn(name: str, fn_id: str, source_code: str):
    """Build a minimal SlitherFunction-like object for arithmetic mining tests."""
    from dataclasses import dataclass

    @dataclass
    class FakeFn:
        name: str
        id: str
        source_code: str

    return FakeFn(name=name, id=fn_id, source_code=source_code)


def test_arithmetic_miner_detects_division_by_10000():
    """Function with / 10000 must generate a truncation-warning invariant."""
    fn = _make_fn(
        name="doSomething",
        fn_id="contract.doSomething(uint256)",
        source_code="uint256 result = amount * rate / 10000;",
    )
    invs = mine_arithmetic_invariants([fn], audit_id="test")
    assert len(invs) >= 1
    combined = " ".join(inv.natural_language for inv in invs)
    assert "10000" in combined
    assert any("truncat" in inv.natural_language.lower() or "division" in inv.natural_language.lower() for inv in invs)


def test_arithmetic_miner_fee_function():
    """collectFee with division generates both fee-nonzero and truncation invariants."""
    fn = _make_fn(
        name="collectFee",
        fn_id="bank.collectFee(uint256)",
        source_code="uint256 fee = amount * feeRate / 10000;\nfeesCollected += fee;",
    )
    invs = mine_arithmetic_invariants([fn], audit_id="test")
    # Must have at least: fee-nonzero invariant + truncation invariant
    assert len(invs) >= 2
    types = {inv.invariant_type for inv in invs}
    assert "economic" in types
    # One invariant must mention non-zero / precision
    has_nonzero = any("non-zero" in inv.natural_language or "precision" in inv.natural_language.lower() for inv in invs)
    assert has_nonzero
    # Affected functions must reference collectFee
    for inv in invs:
        assert "bank.collectFee(uint256)" in inv.affected_functions


def test_arithmetic_miner_ignores_no_division():
    """Function without division produces no arithmetic invariants."""
    fn = _make_fn(
        name="deposit",
        fn_id="bank.deposit()",
        source_code="balances[msg.sender] += msg.value;",
    )
    invs = mine_arithmetic_invariants([fn], audit_id="test")
    assert invs == []


def test_arithmetic_miner_ignores_small_divisor():
    """Division by small constants (< 100) is not flagged."""
    fn = _make_fn(
        name="halfOf",
        fn_id="contract.halfOf(uint256)",
        source_code="return amount / 2;",
    )
    invs = mine_arithmetic_invariants([fn], audit_id="test")
    # No divisor >= 100, so no truncation invariant
    # name doesn't match fee keywords either
    assert invs == []


def test_arithmetic_miner_fee_keywords():
    """Fee-keyword functions (rate, reward, yield, share) are always flagged."""
    for kw_name in ("setFeeRate", "calcReward", "getYield", "sharePrice", "interestAccrued"):
        fn = _make_fn(
            name=kw_name,
            fn_id=f"c.{kw_name}()",
            source_code=f"return x / 1000;",
        )
        invs = mine_arithmetic_invariants([fn], audit_id="test")
        assert len(invs) >= 1, f"Expected invariant for fee-keyword function {kw_name}"
        assert any("non-zero" in inv.natural_language for inv in invs), f"Missing nonzero invariant for {kw_name}"


def test_arithmetic_miner_fixture_collectfee():
    """collectFee from the real fixture must generate precision invariants."""
    analysis = load_analysis()
    all_fns = [fn for c in analysis.contracts for fn in c.functions]
    invs = mine_arithmetic_invariants(all_fns, audit_id="fixture-test")

    collect_fee_invs = [
        inv for inv in invs
        if "collectFee" in " ".join(inv.affected_functions)
    ]
    assert len(collect_fee_invs) >= 1, "Expected precision invariant for collectFee from fixture"
    combined = " ".join(inv.natural_language for inv in collect_fee_invs)
    assert "10000" in combined or "non-zero" in combined or "precision" in combined.lower()


@pytest.mark.asyncio
async def test_mine_all_includes_arithmetic_invariants():
    """mine_all() must include arithmetic invariants in its output."""
    client = _make_llm_client([])  # LLM returns nothing — arithmetic still fires
    miner = InvariantMiner(provider=client)
    analysis = load_analysis()

    results = await miner.mine_all(analysis, audit_id="test", docs_text="")
    arithmetic = [inv for inv in results if inv.source == "inferred" and "10000" in inv.natural_language]
    assert len(arithmetic) >= 1, "Expected at least one arithmetic invariant about division by 10000"


def test_hunter_receives_precision_invariants_in_prompt():
    """When fn_invariants contain a truncation warning, _build_prompt includes 'truncate'."""
    from audit_agent.agents.hunter import _build_prompt
    from audit_agent.knowledge.graph.context import SubgraphContext
    from audit_agent.models.schemas import Invariant

    ctx = SubgraphContext(
        entry_function_id="bank.collectFee(uint256)",
        code_slice="function collectFee(uint256 amount) external { uint256 fee = amount * feeRate / 10000; }",
        state_interactions=[],
        external_calls=[],
        modifiers=[],
        similar_historical_cases=[],
    )
    inv = Invariant(
        audit_id="t",
        invariant_type="economic",
        natural_language=(
            "collectFee() divides by 10000. For inputs smaller than 10000, "
            "integer division truncates to 0 — fee is silently lost."
        ),
        affected_functions=["bank.collectFee(uint256)"],
        solidity_assert="// TRUNCATION RISK",
        echidna_property="",
        source="inferred",
        confidence=0.90,
    )

    prompt = _build_prompt(ctx, fork_protocol=None, documentation="", fn_invariants=[inv])

    assert "truncate" in prompt.lower() or "truncat" in prompt.lower()
    assert "Invariants For This Function" in prompt
    assert "MUST generate a hypothesis" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Source fallback: read from real .sol file when source_code is empty
# ─────────────────────────────────────────────────────────────────────────────

def test_arithmetic_miner_reads_sol_file(tmp_path):
    """Function with empty source_code + target_dir → reads .sol, finds division."""
    from dataclasses import dataclass

    @dataclass
    class FakeLoc:
        file: str = ""
        start_line: int = 0
        end_line: int = 0

    @dataclass
    class FakeFnNoSource:
        name: str
        id: str
        source_code: str = ""
        source_location: FakeLoc = None  # type: ignore[assignment]

        def __post_init__(self):
            if self.source_location is None:
                self.source_location = FakeLoc()

    # Write a real .sol file to tmp_path
    sol_content = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract FeeVault {
    uint256 public feeRate = 100;
    function collectFee(uint256 amount) external returns (uint256) {
        uint256 fee = amount * feeRate / 10000;
        return fee;
    }
}
"""
    (tmp_path / "FeeVault.sol").write_text(sol_content)

    fn = FakeFnNoSource(name="collectFee", id="feevault.collectFee(uint256)")
    invs = mine_arithmetic_invariants([fn], audit_id="test", target_dir=str(tmp_path))

    assert len(invs) >= 2, f"Expected ≥2 invariants, got {len(invs)}: {[i.natural_language for i in invs]}"
    combined = " ".join(inv.natural_language for inv in invs)
    assert "10000" in combined
    assert "non-zero" in combined or "truncat" in combined.lower()


def test_arithmetic_miner_no_source_no_file():
    """Function with no source_code and no target_dir → graceful skip, 0 invariants."""
    from dataclasses import dataclass

    @dataclass
    class FakeFnEmpty:
        name: str = "collectFee"
        id: str = "c.collectFee()"
        source_code: str = ""

    fn = FakeFnEmpty()
    invs = mine_arithmetic_invariants([fn], audit_id="test", target_dir="")
    assert invs == []


def test_arithmetic_miner_reads_fixture_sol():
    """Fallback 3 finds collectFee in the real VulnerableBank.sol fixture."""
    from dataclasses import dataclass

    @dataclass
    class FakeFnNoSource:
        name: str
        id: str
        source_code: str = ""

    fn = FakeFnNoSource(name="collectFee", id="vulnerablebank.collectFee(uint256)")
    contracts_dir = str(Path(__file__).parent.parent / "fixtures" / "contracts")
    invs = mine_arithmetic_invariants([fn], audit_id="test", target_dir=contracts_dir)

    assert len(invs) >= 1, "Expected at least one invariant from sol file scan"
    combined = " ".join(inv.natural_language for inv in invs)
    assert "10000" in combined or "non-zero" in combined
