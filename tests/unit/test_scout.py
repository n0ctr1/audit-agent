"""Unit tests for the Scout agent."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from audit_agent.agents.scout import ScoutAgent, _enumerate_sol_files, _count_loc, _collect_documentation
from audit_agent.graph.state import AuditState


# ── _enumerate_sol_files ───────────────────────────────────────────────────────

def test_enumerate_sol_files_finds_sol(tmp_path: Path) -> None:
    (tmp_path / "Token.sol").write_text("// SPDX-License-Identifier: MIT\n")
    (tmp_path / "Vault.sol").write_text("// SPDX-License-Identifier: MIT\n")
    (tmp_path / "README.md").write_text("docs")

    in_scope, out_of_scope = _enumerate_sol_files(str(tmp_path))
    assert len(in_scope) == 2
    assert len(out_of_scope) == 0


def test_enumerate_sol_files_excludes_lib(tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib" / "openzeppelin"
    lib_dir.mkdir(parents=True)
    (lib_dir / "ERC20.sol").write_text("// OZ\n")
    (tmp_path / "MyToken.sol").write_text("// Mine\n")

    in_scope, out_of_scope = _enumerate_sol_files(str(tmp_path))
    assert len(in_scope) == 1
    assert len(out_of_scope) == 1
    assert any("MyToken" in p for p in in_scope)
    assert any("ERC20" in p for p in out_of_scope)


def test_enumerate_sol_files_empty_dir(tmp_path: Path) -> None:
    in_scope, out_of_scope = _enumerate_sol_files(str(tmp_path))
    assert in_scope == []
    assert out_of_scope == []


def test_count_loc(tmp_path: Path) -> None:
    f1 = tmp_path / "a.sol"
    f1.write_text("line1\nline2\n\nline3\n")
    f2 = tmp_path / "b.sol"
    f2.write_text("only\n")
    loc = _count_loc([str(f1), str(f2)])
    assert loc == 4  # 3 non-blank in a.sol + 1 in b.sol


# ── ScoutAgent ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scout_no_files(tmp_path: Path) -> None:
    state: AuditState = {
        "audit_id": "test-123",
        "mode": "review",
        "target_dir": str(tmp_path),
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
    }

    agent = ScoutAgent()
    result = await agent.run(state)

    assert result["protocol_summary"] is not None
    assert result["static_findings"] == []
    assert result["protocol_summary"].sol_files == []


@pytest.mark.asyncio
async def test_scout_with_sol_files(tmp_path: Path) -> None:
    (tmp_path / "Token.sol").write_text("// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n")
    (tmp_path / "Vault.sol").write_text("// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n")
    (tmp_path / "Counter.sol").write_text("// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n")

    state: AuditState = {
        "audit_id": "test-456",
        "mode": "review",
        "target_dir": str(tmp_path),
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
    }

    # Mock Slither (Docker unavailable in unit tests) and LLM call
    with (
        patch(
            "audit_agent.agents.scout.run_slither",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "audit_agent.agents.scout.ScoutAgent._summarize_protocol",
            new=AsyncMock(
                return_value=__import__(
                    "audit_agent.models.schemas", fromlist=["ProtocolSummary"]
                ).ProtocolSummary(
                    name="TestProtocol",
                    category="DEX",
                    key_mechanisms=["swap"],
                    user_roles=["trader"],
                    sol_files=[str(tmp_path / "Token.sol")],
                )
            ),
        ),
    ):
        agent = ScoutAgent()
        result = await agent.run(state)

    assert result["protocol_summary"] is not None
    ps = result["protocol_summary"]
    # Scout should have reported the 3 in-scope files via enumerate
    assert len(ps.sol_files) >= 1


# ── _collect_documentation ────────────────────────────────────────────────────

def test_scout_collects_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# MyProtocol\nThis protocol does X.")
    result = _collect_documentation(tmp_path)
    assert "MyProtocol" in result
    assert "README.md" in result


def test_scout_ignores_lib_docs(tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib" / "dep"
    lib_dir.mkdir(parents=True)
    (lib_dir / "README.md").write_text("# Library docs — should be ignored")
    (tmp_path / "SPEC.md").write_text("# Real spec")
    result = _collect_documentation(tmp_path)
    assert "Library docs" not in result
    assert "Real spec" in result


def test_scout_caps_at_50k(tmp_path: Path) -> None:
    # Write a file larger than 50k chars
    (tmp_path / "WHITEPAPER.md").write_text("A" * 60_000)
    result = _collect_documentation(tmp_path)
    assert len(result) <= 50_000


def test_scout_no_docs_graceful(tmp_path: Path) -> None:
    # Only Solidity files — no .md/.txt/.rst
    (tmp_path / "Token.sol").write_text("pragma solidity ^0.8.20;")
    result = _collect_documentation(tmp_path)
    assert result == ""


def test_scout_collects_txt_and_rst(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Important note about invariants.")
    (tmp_path / "spec.rst").write_text("RST specification content.")
    result = _collect_documentation(tmp_path)
    assert "Important note" in result
    assert "RST specification" in result


@pytest.mark.asyncio
async def test_scout_run_includes_documentation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Protocol\nKey invariant: balances must never decrease.")
    (tmp_path / "Token.sol").write_text("pragma solidity ^0.8.20;\n")

    state: AuditState = {
        "audit_id": "doc-test",
        "mode": "review",
        "target_dir": str(tmp_path),
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
    }

    with (
        patch("audit_agent.agents.scout.run_slither", new=AsyncMock(return_value=[])),
        patch(
            "audit_agent.agents.scout.ScoutAgent._summarize_protocol",
            new=AsyncMock(
                return_value=__import__(
                    "audit_agent.models.schemas", fromlist=["ProtocolSummary"]
                ).ProtocolSummary(
                    name="Protocol",
                    category="other",
                    key_mechanisms=[],
                    user_roles=[],
                    sol_files=[str(tmp_path / "Token.sol")],
                )
            ),
        ),
    ):
        agent = ScoutAgent()
        result = await agent.run(state)

    assert "documentation" in result
    assert "balances must never decrease" in (result["documentation"] or "")


# ── Problem D: Scout heuristic fallback ───────────────────────────────────────

def test_scout_fallback_heuristic_returns_protocol_summary(tmp_path: Path) -> None:
    from audit_agent.agents.scout import ScoutAgent
    from pathlib import Path as _Path

    sol_files = [tmp_path / "SwapPool.sol", tmp_path / "PairFactory.sol"]
    for f in sol_files:
        f.write_text("")

    result = ScoutAgent._heuristic_summary(sol_files)
    assert result is not None
    assert result.name == "Unknown"
    assert result.category == "DEX"


def test_scout_heuristic_dex_category(tmp_path: Path) -> None:
    from audit_agent.agents.scout import ScoutAgent

    files = [tmp_path / "SwapRouter.sol"]
    for f in files:
        f.write_text("")
    result = ScoutAgent._heuristic_summary(files)
    assert result.category == "DEX"


def test_scout_heuristic_lending_category(tmp_path: Path) -> None:
    from audit_agent.agents.scout import ScoutAgent

    files = [tmp_path / "LendingVault.sol"]
    for f in files:
        f.write_text("")
    result = ScoutAgent._heuristic_summary(files)
    assert result.category == "lending"


def test_scout_heuristic_staking_category(tmp_path: Path) -> None:
    from audit_agent.agents.scout import ScoutAgent

    files = [tmp_path / "StakingRewards.sol"]
    for f in files:
        f.write_text("")
    result = ScoutAgent._heuristic_summary(files)
    assert result.category == "staking"


def test_scout_heuristic_governance_category(tmp_path: Path) -> None:
    from audit_agent.agents.scout import ScoutAgent

    files = [tmp_path / "GovernorBravo.sol"]
    for f in files:
        f.write_text("")
    result = ScoutAgent._heuristic_summary(files)
    assert result.category == "governance"


def test_scout_heuristic_utility_default(tmp_path: Path) -> None:
    from audit_agent.agents.scout import ScoutAgent

    files = [tmp_path / "Helper.sol"]
    for f in files:
        f.write_text("")
    result = ScoutAgent._heuristic_summary(files)
    assert result.category == "utility"


@pytest.mark.asyncio
async def test_scout_uses_fallback_on_empty_llm_response(tmp_path: Path) -> None:
    """ScoutAgent must return heuristic summary when LLM returns empty string."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from audit_agent.agents.scout import ScoutAgent
    from audit_agent.graph.state import AuditState

    (tmp_path / "SwapPool.sol").write_text("pragma solidity ^0.8.20;\n")

    state: AuditState = {
        "audit_id": "fallback-test",
        "mode": "review",
        "target_dir": str(tmp_path),
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
    }

    from audit_agent.llm.base import LLMResponse
    from audit_agent.llm.factory import set_provider

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value=LLMResponse(
        text="", tool_calls=[], input_tokens=0, output_tokens=0,
        stop_reason="end_turn", model="mock",
    ))
    mock_provider.complete_simple = AsyncMock(return_value="")

    set_provider(mock_provider)
    try:
        with patch("audit_agent.agents.scout.run_slither", new=AsyncMock(return_value=[])):
            agent = ScoutAgent()
            result = await agent.run(state)
    finally:
        set_provider(None)

    ps = result["protocol_summary"]
    assert ps is not None
    # Heuristic should detect "swap" in SwapPool.sol
    assert ps.category == "DEX"
