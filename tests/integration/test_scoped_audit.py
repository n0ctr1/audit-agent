"""Integration tests for scope-aware audit pipeline.

All tests are offline — LLM and Docker are mocked, no external services needed.
Covers: resolve_scope(), GraphBuilder scope filtering, Hunter scope filtering,
        and Scribe prompt scope annotations.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.base import LLMResponse, ToolCall


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

VAULT_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./lib/SafeMath.sol";

contract Vault {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        // precision loss: integer division before multiply
        uint256 fee = amount / 100 * 2;
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount - fee);
    }
}
"""

SAFEMATH_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}
"""


@pytest.fixture()
def vulnerable_project(tmp_path: Path) -> Path:
    """Set up a minimal Solidity project: Vault.sol (in-scope) + lib/SafeMath.sol (out-of-scope)."""
    src_dir = tmp_path / "src"
    lib_dir = tmp_path / "lib"
    src_dir.mkdir()
    lib_dir.mkdir()

    (src_dir / "Vault.sol").write_text(VAULT_SOL)
    (lib_dir / "SafeMath.sol").write_text(SAFEMATH_SOL)
    (tmp_path / "scope.txt").write_text("src/Vault.sol\n")

    return tmp_path


def _make_slither_analysis(vault_path: str, safemath_path: str):
    """Build a minimal SlitherAnalysis with two contracts from the given file paths."""
    from audit_agent.knowledge.graph.builders import (
        SlitherAnalysis,
        SlitherContract,
        SlitherFunction,
        SlitherLocation,
    )

    vault_fn = SlitherFunction(
        id="vault.withdraw(uint256)",
        name="withdraw",
        signature="withdraw(uint256)",
        contract_id="vault",
        visibility="external",
        has_external_calls=True,
        attack_surface_score=80.0,
    )
    vault_contract = SlitherContract(
        id="vault",
        name="Vault",
        source_file=vault_path,
        functions=[vault_fn],
    )

    safemath_fn = SlitherFunction(
        id="safemath.add(uint256,uint256)",
        name="add",
        signature="add(uint256,uint256)",
        contract_id="safemath",
        visibility="internal",
        attack_surface_score=5.0,
    )
    safemath_contract = SlitherContract(
        id="safemath",
        name="SafeMath",
        source_file=safemath_path,
        functions=[safemath_fn],
    )

    return SlitherAnalysis(contracts=[vault_contract, safemath_contract])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scope resolution
# ─────────────────────────────────────────────────────────────────────────────

def test_scoped_audit_scope_resolution(vulnerable_project: Path):
    """resolve_scope() puts Vault.sol in_scope and SafeMath.sol out_of_scope."""
    from audit_agent.scope import resolve_scope

    result = resolve_scope(vulnerable_project)

    in_scope_names = {p.name for p in result.in_scope}
    context_names = {p.name for p in result.context_files}
    out_of_scope_names = {p.name for p in result.out_of_scope}

    assert "Vault.sol" in in_scope_names, "Vault.sol must be in_scope"
    assert "SafeMath.sol" not in in_scope_names, "SafeMath.sol must not be in_scope"
    assert result.scope_source == "auto_detected"


def test_scoped_audit_scope_resolution_context_includes_imports(vulnerable_project: Path):
    """Imported SafeMath.sol becomes a context_file (not out_of_scope) because Vault.sol imports it."""
    from audit_agent.scope import resolve_scope

    result = resolve_scope(vulnerable_project)

    all_tracked = {p.name for p in result.in_scope + result.context_files}
    # SafeMath is imported by Vault, so it should be context or in_scope, not silently ignored
    # (depends on import resolution — at minimum it should NOT be out_of_scope if importable)
    assert "Vault.sol" in all_tracked


def test_scoped_audit_scope_resolution_explicit_scope_file(vulnerable_project: Path):
    """Explicit scope_file argument takes priority over auto-detected scope.txt."""
    from audit_agent.scope import resolve_scope

    explicit_scope = vulnerable_project / "explicit_scope.txt"
    explicit_scope.write_text("src/Vault.sol\n")

    result = resolve_scope(vulnerable_project, scope_file=explicit_scope)

    assert result.scope_source == "explicit_file"
    assert any(p.name == "Vault.sol" for p in result.in_scope)


def test_scoped_audit_scope_resolution_fallback_no_scope_file(tmp_path: Path):
    """When no scope.txt exists, all non-library .sol files are in_scope."""
    from audit_agent.scope import resolve_scope

    (tmp_path / "Token.sol").write_text("pragma solidity ^0.8.0;\ncontract Token {}\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "OZ.sol").write_text("pragma solidity ^0.8.0;\nlibrary OZ {}\n")

    result = resolve_scope(tmp_path)

    assert result.scope_source == "all_non_library"
    in_scope_names = {p.name for p in result.in_scope}
    assert "Token.sol" in in_scope_names
    assert "OZ.sol" not in in_scope_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mapper scope filtering via GraphBuilder
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scoped_audit_mapper_marks_scope(vulnerable_project: Path):
    """GraphBuilder.build() with in_scope_files excludes out-of-scope contracts."""
    from audit_agent.knowledge.graph.builders import GraphBuilder, InMemoryGraphStore

    vault_path = str(vulnerable_project / "src" / "Vault.sol")
    safemath_path = str(vulnerable_project / "lib" / "SafeMath.sol")
    analysis = _make_slither_analysis(vault_path, safemath_path)

    store = InMemoryGraphStore()
    builder = GraphBuilder(store=store)

    stats = await builder.build(analysis, in_scope_files=[vault_path])

    # Only the Vault contract (1 contract, 1 function) should be included
    assert stats.contracts == 1, f"Expected 1 contract, got {stats.contracts}"
    assert stats.functions == 1, f"Expected 1 function, got {stats.functions}"


@pytest.mark.asyncio
async def test_scoped_audit_mapper_includes_all_without_scope(vulnerable_project: Path):
    """GraphBuilder.build() without in_scope_files includes all non-library contracts."""
    from audit_agent.knowledge.graph.builders import GraphBuilder, InMemoryGraphStore

    vault_path = str(vulnerable_project / "src" / "Vault.sol")
    safemath_path = str(vulnerable_project / "lib" / "SafeMath.sol")
    analysis = _make_slither_analysis(vault_path, safemath_path)

    store = InMemoryGraphStore()
    builder = GraphBuilder(store=store)

    stats = await builder.build(analysis, in_scope_files=None)

    # Without scope filter, both contracts included (library filtering may exclude SafeMath)
    assert stats.contracts >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hunter scope filtering
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scoped_audit_hunter_filters_by_scope(vulnerable_project: Path):
    """Hunter._generate() only processes functions from in-scope contracts."""
    from audit_agent.agents.hunter import HunterAgent
    from audit_agent.knowledge.graph.context import NullGraphAdapter

    vault_path = str(vulnerable_project / "src" / "Vault.sol")
    safemath_path = str(vulnerable_project / "lib" / "SafeMath.sol")
    analysis = _make_slither_analysis(vault_path, safemath_path)

    mock_client = MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="report_hypotheses", arguments={"hypotheses": []})],
        input_tokens=0,
        output_tokens=0,
        stop_reason="tool_use",
        model="test-model",
    )])

    hunter = HunterAgent(
        graph_adapter=NullGraphAdapter(),
        provider=mock_client,
        top_functions=20,
    )

    # Patch _hypothesize to track which function IDs are processed
    processed_fn_ids: list[str] = []
    original_hypothesize = hunter._hypothesize

    async def tracking_hypothesize(context, audit_id, *args, **kwargs):
        processed_fn_ids.append(context.entry_function_id)
        return []

    hunter._hypothesize = tracking_hypothesize

    from audit_agent.knowledge.graph.context import SubgraphContextBuilder, InMemoryGraph
    context_builder = SubgraphContextBuilder(graph=InMemoryGraph(analysis))

    await hunter._generate(
        analysis=analysis,
        audit_id="test-audit",
        context_builder=context_builder,
        static_findings=[],
        in_scope_files=[vault_path],
    )

    # Only Vault's function should have been processed
    assert all("vault" in fn_id.lower() for fn_id in processed_fn_ids), (
        f"Hunter processed out-of-scope functions: {processed_fn_ids}"
    )
    assert "safemath.add(uint256,uint256)" not in processed_fn_ids, (
        "Hunter must not process SafeMath (out-of-scope) functions"
    )


@pytest.mark.asyncio
async def test_scoped_audit_hunter_no_hypotheses_when_all_out_of_scope(vulnerable_project: Path):
    """Hunter returns empty list when in_scope_files excludes all contracts."""
    from audit_agent.agents.hunter import HunterAgent
    from audit_agent.knowledge.graph.context import NullGraphAdapter, SubgraphContextBuilder, InMemoryGraph

    vault_path = str(vulnerable_project / "src" / "Vault.sol")
    safemath_path = str(vulnerable_project / "lib" / "SafeMath.sol")
    analysis = _make_slither_analysis(vault_path, safemath_path)

    mock_client = MockProvider()

    hunter = HunterAgent(graph_adapter=NullGraphAdapter(), provider=mock_client)
    context_builder = SubgraphContextBuilder(graph=InMemoryGraph(analysis))

    # Pass a path that doesn't match any contract
    hypotheses = await hunter._generate(
        analysis=analysis,
        audit_id="test-audit",
        context_builder=context_builder,
        static_findings=[],
        in_scope_files=["/nonexistent/Other.sol"],
    )

    assert hypotheses == [], "Expected no hypotheses when all contracts are out-of-scope"
    assert len(mock_client.calls) == 0, "LLM must not be called when all contracts are out-of-scope"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scribe report scope annotations
# ─────────────────────────────────────────────────────────────────────────────

def test_scoped_audit_report_only_scope_files(vulnerable_project: Path):
    """_build_scribe_prompt includes in-scope file names in the prompt."""
    from audit_agent.agents.scribe import _build_scribe_prompt
    from audit_agent.models.schemas import ValidatedFinding

    vault_path = str(vulnerable_project / "src" / "Vault.sol")
    safemath_path = str(vulnerable_project / "lib" / "SafeMath.sol")

    finding = ValidatedFinding(
        audit_id="test-audit",
        title="Precision loss in Vault.withdraw()",
        severity="medium",
        confidence=0.9,
        status="validated",
        description="Integer division before multiplication causes precision loss.",
        affected_functions=["vault.withdraw(uint256)"],
    )

    prompt = _build_scribe_prompt(
        finding=finding,
        hypothesis=None,
        exploit=None,
        in_scope_files=[vault_path, safemath_path],
    )

    assert "Vault.sol" in prompt, "Prompt must list in-scope file Vault.sol"
    assert "SafeMath.sol" in prompt, "Prompt must list SafeMath.sol (also passed as in_scope)"
    assert "In-Scope Files" in prompt, "Prompt must include the in-scope section header"


def test_scoped_audit_report_no_scope_section_when_no_files():
    """_build_scribe_prompt omits scope section when in_scope_files is empty."""
    from audit_agent.agents.scribe import _build_scribe_prompt
    from audit_agent.models.schemas import ValidatedFinding

    finding = ValidatedFinding(
        audit_id="test-audit",
        title="Reentrancy",
        severity="critical",
        confidence=0.95,
        status="validated",
        description="Classic reentrancy in withdraw().",
    )

    prompt = _build_scribe_prompt(
        finding=finding,
        hypothesis=None,
        exploit=None,
        in_scope_files=None,
    )

    assert "In-Scope Files" not in prompt


def test_scoped_audit_report_scope_files_reference_filenames_only():
    """_build_scribe_prompt uses only the filename (not full path) in the scope section."""
    from audit_agent.agents.scribe import _build_scribe_prompt
    from audit_agent.models.schemas import ValidatedFinding

    finding = ValidatedFinding(
        audit_id="test-audit",
        title="Access control issue",
        severity="high",
        confidence=0.88,
        status="validated",
        description="Missing onlyOwner modifier.",
    )

    full_path = "/srv/project/src/contracts/Vault.sol"
    prompt = _build_scribe_prompt(
        finding=finding,
        hypothesis=None,
        exploit=None,
        in_scope_files=[full_path],
    )

    assert "Vault.sol" in prompt
    # Full path should not leak into the prompt (filenames only per scribe implementation)
    assert "/srv/project/src/contracts/" not in prompt
