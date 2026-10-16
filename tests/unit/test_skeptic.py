"""Tests for Phase 5: SkepticAgent adversarial validation.

All tests are offline — LLM mocked, no Docker, no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.agents.skeptic import (
    SkepticAgent,
    _build_validation_prompt,
    _heuristic_verdict,
    _heuristic_impact,
)
from audit_agent.models.schemas import ExploitResult, Hypothesis, ValidatedFinding


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_hypothesis(
    title: str = "Reentrancy in withdraw()",
    vuln_class: str = "SWC-107",
    severity: str = "critical",
) -> Hypothesis:
    return Hypothesis(
        audit_id="test-audit",
        title=title,
        vulnerability_class=vuln_class,
        affected_functions=["vulnerablebank.withdraw(uint256)"],
        attack_scenario="Attacker calls withdraw recursively before balance update",
        required_preconditions=["attacker has deposited ETH"],
        estimated_severity=severity,
        priority_score=90.0,
    )


def _make_exploit(
    status: str = "exploited",
    hypothesis_id: str = "",
) -> ExploitResult:
    return ExploitResult(
        hypothesis_id=hypothesis_id or "test-hyp-id",
        test_type="unit",
        status=status,  # type: ignore[arg-type]
        poc_code="// SPDX-License-Identifier: UNLICENSED\npragma solidity ^0.8.20;",
        execution_trace="[PASS] test_exploit_withdraw (gas: 54321)",
        gas_used=54321,
        reproduction_steps=["[PASS] test_exploit_withdraw (gas: 54321)"],
    )


def _make_llm_client(
    verdict: str = "true_positive",
    confidence: float = 0.9,
    reasoning: str = "PoC passes with real logic",
    real_impact: str = "Full drain of contract balance",
    suggested_severity: str = "critical",
) -> MockProvider:
    return MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="validate_exploit", arguments={
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "real_impact": real_impact,
            "suggested_severity": suggested_severity,
        })],
        input_tokens=0,
        output_tokens=0,
        stop_reason="tool_use",
        model="test-model",
    )])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Heuristic verdict helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicVerdict:
    def test_exploited_gives_true_positive(self):
        exploit = _make_exploit("exploited")
        verdict, confidence = _heuristic_verdict(exploit)
        assert verdict == "true_positive"
        assert confidence >= 0.6

    def test_blocked_gives_false_positive(self):
        exploit = _make_exploit("blocked")
        verdict, confidence = _heuristic_verdict(exploit)
        assert verdict == "false_positive"
        assert confidence >= 0.7

    def test_partial_gives_needs_review(self):
        exploit = _make_exploit("partial")
        verdict, confidence = _heuristic_verdict(exploit)
        assert verdict == "needs_review"

    def test_error_gives_needs_review(self):
        exploit = _make_exploit("error")
        verdict, confidence = _heuristic_verdict(exploit)
        assert verdict == "needs_review"
        assert confidence <= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildValidationPrompt:
    def test_includes_hypothesis_title(self):
        hyp = _make_hypothesis("Reentrancy in withdraw()")
        exploit = _make_exploit()
        prompt = _build_validation_prompt(exploit, hyp)
        assert "Reentrancy in withdraw()" in prompt

    def test_includes_exploit_status(self):
        exploit = _make_exploit("blocked")
        prompt = _build_validation_prompt(exploit, None)
        assert "blocked" in prompt

    def test_handles_no_hypothesis(self):
        exploit = _make_exploit()
        prompt = _build_validation_prompt(exploit, None)
        assert "No matching hypothesis" in prompt

    def test_truncates_poc_code(self):
        exploit = _make_exploit()
        exploit = exploit.model_copy(update={"poc_code": "x" * 5000})
        prompt = _build_validation_prompt(exploit, None)
        # prompt should not contain full 5000-char string
        assert len(prompt) < 5000


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM-based validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSkepticLLMValidation:
    @pytest.mark.asyncio
    async def test_true_positive_from_llm(self):
        client = _make_llm_client(verdict="true_positive", confidence=0.95)
        agent = SkepticAgent(provider=client)
        hyp = _make_hypothesis()
        exploit = _make_exploit("exploited", hypothesis_id=hyp.id)

        finding = await agent.validate(exploit, hyp, "test-audit")

        assert finding.status == "validated"
        assert finding.confidence == 0.95
        assert finding.severity == "critical"
        assert "PoC passes" in (finding.skeptic_notes or "")

    @pytest.mark.asyncio
    async def test_false_positive_from_llm(self):
        client = _make_llm_client(
            verdict="false_positive",
            confidence=0.85,
            reasoning="Straw-man contract — real implementation has onlyOwner guard",
            real_impact="No real impact",
            suggested_severity="info",
        )
        agent = SkepticAgent(provider=client)
        hyp = _make_hypothesis()
        exploit = _make_exploit("exploited", hypothesis_id=hyp.id)

        finding = await agent.validate(exploit, hyp, "test-audit")

        assert finding.status == "false_positive"
        assert finding.confidence == 0.85

    @pytest.mark.asyncio
    async def test_needs_review_from_llm(self):
        client = _make_llm_client(
            verdict="needs_review",
            confidence=0.55,
            reasoning="Partial exploit — economic impact unclear",
            real_impact="Potential partial drain",
            suggested_severity="high",
        )
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("partial")

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.status == "unverified"
        assert 0.4 <= finding.confidence <= 0.7

    @pytest.mark.asyncio
    async def test_llm_verdict_overrides_heuristic(self):
        # exploit.status="blocked" would give false_positive heuristically,
        # but LLM says true_positive
        client = _make_llm_client(verdict="true_positive", confidence=0.80)
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("blocked")

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.status == "validated"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heuristic fallback when LLM fails
# ─────────────────────────────────────────────────────────────────────────────

class TestSkepticHeuristicFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_auth_error(self):
        client = MockProvider(side_effect=Exception("authentication error"))
        agent = SkepticAgent(provider=client)
        hyp = _make_hypothesis()
        exploit = _make_exploit("exploited", hypothesis_id=hyp.id)

        finding = await agent.validate(exploit, hyp, "test-audit")

        # Heuristic: exploited → true_positive → validated
        assert finding.status == "validated"
        assert "Heuristic" in (finding.skeptic_notes or "")

    @pytest.mark.asyncio
    async def test_fallback_blocked_gives_false_positive(self):
        client = MockProvider(side_effect=Exception("timeout"))
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("blocked")

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.status == "false_positive"

    @pytest.mark.asyncio
    async def test_fallback_uses_hypothesis_severity(self):
        client = MockProvider(side_effect=Exception("err"))
        agent = SkepticAgent(provider=client)
        hyp = _make_hypothesis(severity="high")
        exploit = _make_exploit("exploited", hypothesis_id=hyp.id)

        finding = await agent.validate(exploit, hyp, "test-audit")

        assert finding.severity == "high"


# ─────────────────────────────────────────────────────────────────────────────
# 5. ValidatedFinding fields
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatedFindingFields:
    @pytest.mark.asyncio
    async def test_finding_has_hypothesis_title(self):
        client = _make_llm_client()
        agent = SkepticAgent(provider=client)
        hyp = _make_hypothesis("Critical reentrancy")
        exploit = _make_exploit(hypothesis_id=hyp.id)

        finding = await agent.validate(exploit, hyp, "test-audit")

        assert finding.title == "Critical reentrancy"
        assert finding.audit_id == "test-audit"

    @pytest.mark.asyncio
    async def test_finding_has_poc_code(self):
        client = _make_llm_client()
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit()

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.poc_code is not None
        assert "pragma solidity" in (finding.poc_code or "")

    @pytest.mark.asyncio
    async def test_finding_no_hypothesis_uses_fallback_title(self):
        client = MockProvider(side_effect=Exception("err"))
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("error")
        exploit = exploit.model_copy(update={"hypothesis_id": "some-id-123"})

        finding = await agent.validate(exploit, None, "test-audit")

        assert "some-id-123" in finding.title


# ─────────────────────────────────────────────────────────────────────────────
# 6. SkepticAgent.run() — LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

class TestSkepticAgentRun:
    @pytest.mark.asyncio
    async def test_run_populates_validated_findings(self):
        client = _make_llm_client(verdict="true_positive", confidence=0.9)
        agent = SkepticAgent(provider=client)

        hyp = _make_hypothesis()
        exploit = _make_exploit("exploited", hypothesis_id=hyp.id)

        state = {
            "audit_id": "run-test",
            "hypotheses": [hyp],
            "exploits": [exploit],
        }
        new_state = await agent.run(state)

        assert "validated_findings" in new_state
        assert len(new_state["validated_findings"]) == 1
        assert new_state["validated_findings"][0].status == "validated"

    @pytest.mark.asyncio
    async def test_run_handles_empty_exploits(self):
        client = _make_llm_client()
        agent = SkepticAgent(provider=client)

        state = {"audit_id": "empty-test", "hypotheses": [], "exploits": []}
        new_state = await agent.run(state)

        assert new_state["validated_findings"] == []

    @pytest.mark.asyncio
    async def test_run_processes_multiple_exploits(self):
        call_count = 0

        async def mock_callback(*, model, system, messages, tools=None, tool_choice=None, temperature=0.0, max_tokens=4096):
            nonlocal call_count
            call_count += 1
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id=str(call_count), name="validate_exploit", arguments={
                    "verdict": "true_positive",
                    "confidence": 0.8,
                    "reasoning": "ok",
                    "real_impact": "drain",
                    "suggested_severity": "high",
                })],
                input_tokens=0,
                output_tokens=0,
                stop_reason="tool_use",
                model=model,
            )

        client = MockProvider(callback=mock_callback)
        agent = SkepticAgent(provider=client)

        hyp1 = _make_hypothesis("Bug A")
        hyp2 = _make_hypothesis("Bug B")
        e1 = _make_exploit("exploited", hypothesis_id=hyp1.id)
        e2 = _make_exploit("partial", hypothesis_id=hyp2.id)

        state = {
            "audit_id": "multi-test",
            "hypotheses": [hyp1, hyp2],
            "exploits": [e1, e2],
        }
        new_state = await agent.run(state)

        assert len(new_state["validated_findings"]) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_run_preserves_existing_state_fields(self):
        client = MockProvider(side_effect=Exception("err"))
        agent = SkepticAgent(provider=client)

        state = {
            "audit_id": "preserve-test",
            "hypotheses": [],
            "exploits": [],
            "some_other_field": "keep-me",
        }
        new_state = await agent.run(state)

        assert new_state.get("some_other_field") == "keep-me"


# ─────────────────────────────────────────────────────────────────────────────
# 7. needs_review routing and logging
# ─────────────────────────────────────────────────────────────────────────────

class TestSkepticNeedsReview:
    @pytest.mark.asyncio
    async def test_skeptic_partial_high_confidence_needs_review(self):
        """exploit.status=partial + LLM confidence=0.85 → needs_review (unverified)."""
        client = _make_llm_client(
            verdict="needs_review",
            confidence=0.85,
            reasoning="PoC failed but code clearly vulnerable",
            real_impact="Potential fund drain",
            suggested_severity="high",
        )
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("partial")

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.status == "unverified"
        assert finding.confidence == 0.85

    @pytest.mark.asyncio
    async def test_skeptic_partial_low_confidence_fp(self):
        """exploit.status=partial + LLM confidence=0.4 → false_positive."""
        client = _make_llm_client(
            verdict="false_positive",
            confidence=0.4,
            reasoning="PoC uses straw-man contract with no real implementation",
            real_impact="None",
            suggested_severity="info",
        )
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("partial")

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.status == "false_positive"

    @pytest.mark.asyncio
    async def test_skeptic_blocked_always_fp(self):
        """exploit.status=blocked → heuristic gives false_positive."""
        client = MockProvider(side_effect=Exception("unavailable"))
        agent = SkepticAgent(provider=client)
        exploit = _make_exploit("blocked")

        finding = await agent.validate(exploit, None, "test-audit")

        assert finding.status == "false_positive"

    @pytest.mark.asyncio
    async def test_skeptic_run_logs_needs_review_count(self):
        """SkepticAgent.run() counts unverified findings as needs_review in log."""
        client = _make_llm_client(verdict="needs_review", confidence=0.65,
                                   reasoning="ambiguous", real_impact="unclear",
                                   suggested_severity="medium")
        agent = SkepticAgent(provider=client)

        hyp = _make_hypothesis("Maybe vulnerable")
        exploit = _make_exploit("partial", hypothesis_id=hyp.id)

        state = {
            "audit_id": "nr-test",
            "hypotheses": [hyp],
            "exploits": [exploit],
        }
        new_state = await agent.run(state)

        findings = new_state["validated_findings"]
        assert len(findings) == 1
        assert findings[0].status == "unverified"


# ─────────────────────────────────────────────────────────────────────────────
# 8. after_skeptic routing
# ─────────────────────────────────────────────────────────────────────────────

class TestAfterSkepticRouting:
    def test_routes_to_scribe_on_validated(self):
        from audit_agent.graph.edges import after_skeptic
        from audit_agent.models.schemas import ValidatedFinding

        finding = ValidatedFinding(
            audit_id="x", title="Bug", severity="high",
            confidence=0.9, status="validated", description="real bug",
        )
        state = {"validated_findings": [finding]}
        assert after_skeptic(state) == "scribe"

    def test_routes_to_scribe_on_unverified(self):
        from audit_agent.graph.edges import after_skeptic
        from audit_agent.models.schemas import ValidatedFinding

        finding = ValidatedFinding(
            audit_id="x", title="Maybe Bug", severity="medium",
            confidence=0.6, status="unverified", description="unclear",
        )
        state = {"validated_findings": [finding]}
        assert after_skeptic(state) == "scribe"

    def test_routes_to_end_on_all_false_positive(self):
        from audit_agent.graph.edges import after_skeptic
        from audit_agent.models.schemas import ValidatedFinding

        finding = ValidatedFinding(
            audit_id="x", title="FP", severity="info",
            confidence=0.9, status="false_positive", description="not a bug",
        )
        state = {"validated_findings": [finding]}
        assert after_skeptic(state) == "end"

    def test_routes_to_end_on_empty(self):
        from audit_agent.graph.edges import after_skeptic
        state = {"validated_findings": []}
        assert after_skeptic(state) == "end"


# ─────────────────────────────────────────────────────────────────────────────
# Source-based validation (Part 4)
# ─────────────────────────────────────────────────────────────────────────────

class TestSkepticSourceFallback:
    def test_read_scope_sources_returns_content(self, tmp_path):
        """_read_scope_sources reads in-scope .sol files correctly."""
        sol = tmp_path / "Vault.sol"
        sol.write_text("contract Vault { function withdraw() external {} }")
        agent = SkepticAgent()
        state = {
            "target_dir": str(tmp_path),
            "scope_in_scope": [str(sol)],
            "scope_context_files": [],
        }
        result = agent._read_scope_sources(state)
        assert "Vault.sol" in result
        assert "[IN SCOPE]" in result
        assert "withdraw" in result

    def test_read_scope_sources_relative_path(self, tmp_path):
        """_read_scope_sources resolves relative paths via target_dir fallback."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        sol = src_dir / "Token.sol"
        sol.write_text("contract Token { mapping(address => uint256) balances; }")
        agent = SkepticAgent()
        state = {
            "target_dir": str(tmp_path),
            "scope_in_scope": ["src/Token.sol"],
            "scope_context_files": [],
        }
        result = agent._read_scope_sources(state)
        assert "Token.sol" in result
        assert "balances" in result

    def test_read_scope_sources_dependency_labelled(self, tmp_path):
        """Context/dependency files are labelled [DEPENDENCY]."""
        dep = tmp_path / "SafeMath.sol"
        dep.write_text("library SafeMath { function add(uint a, uint b) internal pure returns (uint) {} }")
        agent = SkepticAgent()
        state = {
            "target_dir": str(tmp_path),
            "scope_in_scope": [],
            "scope_context_files": [str(dep)],
        }
        result = agent._read_scope_sources(state)
        assert "[DEPENDENCY]" in result
        assert "SafeMath" in result

    def test_read_scope_sources_returns_empty_for_missing_files(self, tmp_path):
        """_read_scope_sources returns empty string when files don't exist."""
        agent = SkepticAgent()
        state = {
            "target_dir": str(tmp_path),
            "scope_in_scope": ["/nonexistent/path/Vault.sol"],
            "scope_context_files": [],
        }
        result = agent._read_scope_sources(state)
        assert result == ""

    def test_read_scope_sources_respects_total_limit(self, tmp_path):
        """_read_scope_sources output stays within 15000 char limit."""
        big_content = "// " + "x" * 20000
        for i in range(3):
            (tmp_path / f"Contract{i}.sol").write_text(big_content)
        agent = SkepticAgent()
        state = {
            "target_dir": str(tmp_path),
            "scope_in_scope": [str(tmp_path / f"Contract{i}.sol") for i in range(3)],
            "scope_context_files": [],
        }
        result = agent._read_scope_sources(state)
        assert len(result) <= 15000

    def test_skeptic_system_prompt_instructs_source_code_first(self):
        """The updated system prompt must instruct the LLM to use source code as primary evidence."""
        from audit_agent.agents.skeptic import _SKEPTIC_SYSTEM
        assert "SOURCE CODE IS YOUR PRIMARY EVIDENCE" in _SKEPTIC_SYSTEM
        assert "status=partial/error" in _SKEPTIC_SYSTEM
        assert "IGNORE for verdict" in _SKEPTIC_SYSTEM

    @pytest.mark.asyncio
    async def test_run_uses_fallback_when_read_sources_returns_empty(self, tmp_path):
        """run() calls _read_scope_sources when _read_sources returns empty string."""
        sol = tmp_path / "src" / "Vault.sol"
        sol.parent.mkdir()
        sol.write_text("contract Vault { function withdraw() external {} }")

        exploit = ExploitResult(
            hypothesis_id="h1",
            test_type="unit",
            status="partial",
            poc_code="// poc",
        )
        hyp_tool_call = ToolCall(
            id="1", name="validate_exploit",
            arguments={"verdict": "needs_review", "confidence": 0.5,
                       "reasoning": "source shows the bug", "real_impact": "fund loss",
                       "suggested_severity": "high"},
        )
        client = MockProvider([LLMResponse(
            text="", tool_calls=[hyp_tool_call],
            input_tokens=10, output_tokens=50, stop_reason="tool_use", model="test",
        )])
        agent = SkepticAgent(provider=client)

        # Provide an absolute scope path that won't match glob (forcing fallback)
        state = {
            "audit_id": "src-fallback-test",
            "exploits": [exploit],
            "hypotheses": [],
            "target_dir": str(tmp_path),
            # In-scope path matches exactly — _read_sources will find it
            "scope_in_scope": [str(sol)],
            "scope_context_files": [],
        }
        result_state = await agent.run(state)
        assert len(result_state["validated_findings"]) == 1
