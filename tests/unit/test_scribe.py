"""Tests for Phase 5: ScribeAgent platform-specific report generation.

All tests are offline — LLM mocked, no Docker, no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.agents.scribe import (
    ScribeAgent,
    _deterministic_fields,
    _render_needs_review,
    render_report,
)
from audit_agent.models.schemas import ExploitResult, Hypothesis, ValidatedFinding


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_finding(
    title: str = "Reentrancy in withdraw()",
    severity: str = "critical",
    status: str = "validated",
    hypothesis_id: str = "hyp-123",
) -> ValidatedFinding:
    return ValidatedFinding(
        audit_id="test-audit",
        hypothesis_id=hypothesis_id,
        title=title,
        vulnerability_class="SWC-107",
        severity=severity,  # type: ignore[arg-type]
        confidence=0.92,
        status=status,  # type: ignore[arg-type]
        affected_functions=["vulnerablebank.withdraw(uint256)"],
        description="Attacker drains contract balance via reentrancy.",
        poc_code="// SPDX-License-Identifier: UNLICENSED\npragma solidity ^0.8.20;",
        execution_trace="[PASS] test_exploit_withdraw (gas: 54321)",
        skeptic_notes="PoC passed with real logic, straw-man check passed.",
    )


def _make_hypothesis(title: str = "Reentrancy in withdraw()") -> Hypothesis:
    return Hypothesis(
        audit_id="test-audit",
        title=title,
        vulnerability_class="SWC-107",
        affected_functions=["vulnerablebank.withdraw(uint256)"],
        attack_scenario="Attacker calls withdraw recursively before balance update",
        required_preconditions=["attacker has deposited ETH"],
        estimated_severity="critical",
        priority_score=90.0,
    )


def _make_exploit(status: str = "exploited", hypothesis_id: str = "hyp-123") -> ExploitResult:
    return ExploitResult(
        hypothesis_id=hypothesis_id,
        test_type="unit",
        status=status,  # type: ignore[arg-type]
        poc_code="// SPDX-License-Identifier: UNLICENSED\npragma solidity ^0.8.20;\n// PoC code",
        execution_trace="[PASS] test_exploit_withdraw (gas: 54321)",
        gas_used=54321,
        reproduction_steps=[
            "[PASS] test_exploit_withdraw (gas: 54321)",
            "→ CALL VulnerableBank.withdraw(1 ether)",
        ],
    )


def _make_llm_client(
    summary: str = "Critical reentrancy vulnerability in withdraw().",
    vulnerability_detail: str = "The withdraw() function violates CEI.",
    impact: str = "Full drain of contract ETH balance.",
    recommendation: str = "Add nonReentrant modifier and apply CEI pattern.",
) -> MockProvider:
    return MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="generate_report", arguments={
            "summary": summary,
            "vulnerability_detail": vulnerability_detail,
            "impact": impact,
            "recommendation": recommendation,
        })],
        input_tokens=0,
        output_tokens=0,
        stop_reason="tool_use",
        model="test-model",
    )])


# ─────────────────────────────────────────────────────────────────────────────
# 1. render_report — Code4rena format
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderReportC4:
    def _fields(self) -> dict[str, str]:
        return {
            "summary": "The withdraw() function is vulnerable to reentrancy.",
            "vulnerability_detail": "Violates CEI: external call before state update.",
            "impact": "Full drain of contract balance.",
            "recommendation": "Add nonReentrant and apply CEI.",
        }

    def test_c4_high_severity_prefix(self):
        finding = _make_finding(severity="critical")
        md = render_report(finding, self._fields(), platform="c4")
        assert "## [H]" in md

    def test_c4_medium_severity_prefix(self):
        finding = _make_finding(severity="medium")
        md = render_report(finding, self._fields(), platform="c4")
        assert "## [M]" in md

    def test_c4_has_required_sections(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="c4")
        for section in ["Summary", "Vulnerability Detail", "Impact",
                         "Proof of Concept", "Recommended Mitigation Steps"]:
            assert section in md

    def test_c4_includes_poc_code_block(self):
        finding = _make_finding()
        md = render_report(
            finding, self._fields(), platform="c4",
            poc_code="pragma solidity ^0.8.20; contract X {}"
        )
        assert "```solidity" in md
        assert "pragma solidity" in md

    def test_c4_no_poc_shows_placeholder(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="c4", poc_code=None)
        assert "No PoC available" in md


# ─────────────────────────────────────────────────────────────────────────────
# 2. render_report — Immunefi format
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderReportImmunefi:
    def _fields(self) -> dict[str, str]:
        return {
            "summary": "Critical reentrancy.",
            "vulnerability_detail": "CEI violation.",
            "impact": "Full drain.",
            "recommendation": "Use nonReentrant.",
        }

    def test_immunefi_has_title_section(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="immunefi")
        assert "## Title" in md
        assert "Reentrancy in withdraw()" in md

    def test_immunefi_has_severity_section(self):
        finding = _make_finding(severity="critical")
        md = render_report(finding, self._fields(), platform="immunefi")
        assert "## Severity" in md
        assert "Critical" in md

    def test_immunefi_has_required_sections(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="immunefi")
        for section in ["Title", "Severity", "Description", "Proof of Concept",
                         "Impact", "Recommended Fix"]:
            assert section in md

    def test_immunefi_includes_poc_block(self):
        finding = _make_finding()
        md = render_report(
            finding, self._fields(), platform="immunefi",
            poc_code="pragma solidity ^0.8.20;"
        )
        assert "```solidity" in md


# ─────────────────────────────────────────────────────────────────────────────
# 3. render_report — Sherlock format
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderReportSherlock:
    def _fields(self) -> dict[str, str]:
        return {
            "summary": "High reentrancy.",
            "vulnerability_detail": "CEI violated.",
            "impact": "Direct fund loss.",
            "recommendation": "Add guard.",
        }

    def test_sherlock_has_required_sections(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="sherlock")
        for section in ["Summary", "Vulnerability Detail", "Impact",
                         "Code Snippet", "Tool used", "Recommendation"]:
            assert section in md

    def test_sherlock_tool_used_includes_foundry_when_poc(self):
        finding = _make_finding()
        md = render_report(
            finding, self._fields(), platform="sherlock",
            poc_code="pragma solidity ^0.8.20;"
        )
        assert "Foundry" in md

    def test_sherlock_tool_used_manual_only_without_poc(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="sherlock", poc_code=None)
        assert "Manual Review" in md
        assert "Foundry" not in md


# ─────────────────────────────────────────────────────────────────────────────
# 4. render_report — Generic format
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderReportGeneric:
    def _fields(self) -> dict[str, str]:
        return {
            "summary": "Reentrancy vulnerability.",
            "vulnerability_detail": "CEI violated.",
            "impact": "Fund drain.",
            "recommendation": "Add guard.",
        }

    def test_generic_has_confidence_line(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="generic")
        assert "92%" in md

    def test_generic_has_affected_functions(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="generic")
        assert "vulnerablebank.withdraw(uint256)" in md

    def test_generic_includes_skeptic_notes(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="generic")
        assert "Skeptic Notes" in md
        assert "straw-man check passed" in md

    def test_cantina_uses_generic_renderer(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="cantina")
        assert "## Summary" in md

    def test_spearbit_uses_generic_renderer(self):
        finding = _make_finding()
        md = render_report(finding, self._fields(), platform="spearbit")
        assert "## Summary" in md


# ─────────────────────────────────────────────────────────────────────────────
# 5. Deterministic field fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicFields:
    def test_fallback_uses_hypothesis_attack_scenario(self):
        finding = _make_finding()
        hyp = _make_hypothesis()
        exploit = _make_exploit()
        fields = _deterministic_fields(finding, hyp, exploit)
        assert "recursively" in fields["vulnerability_detail"]

    def test_fallback_without_hypothesis(self):
        finding = _make_finding()
        fields = _deterministic_fields(finding, None, None)
        assert isinstance(fields["summary"], str)
        assert len(fields["summary"]) > 0

    def test_fallback_exploit_status_in_impact(self):
        finding = _make_finding()
        exploit = _make_exploit(status="exploited")
        fields = _deterministic_fields(finding, None, exploit)
        assert "confirmed" in fields["impact"].lower() or "critical" in fields["impact"].lower()

    def test_fallback_includes_recommendation(self):
        finding = _make_finding()
        fields = _deterministic_fields(finding, None, None)
        assert fields["recommendation"] != ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. ScribeAgent LLM field generation
# ─────────────────────────────────────────────────────────────────────────────

class TestScribeLLMFields:
    @pytest.mark.asyncio
    async def test_llm_fields_used_in_report(self):
        client = _make_llm_client(
            summary="Precise LLM summary here.",
            vulnerability_detail="LLM vuln detail.",
            impact="LLM impact.",
            recommendation="LLM recommendation.",
        )
        agent = ScribeAgent(provider=client)
        finding = _make_finding()
        hyp = _make_hypothesis()
        exploit = _make_exploit()

        fields = await agent._generate_fields(finding, hyp, exploit)

        assert fields["summary"] == "Precise LLM summary here."
        assert fields["vulnerability_detail"] == "LLM vuln detail."

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        client = MockProvider(side_effect=Exception("auth error"))
        agent = ScribeAgent(provider=client)
        finding = _make_finding()

        fields = await agent._generate_fields(finding, None, None)

        # Should return deterministic fields, not raise
        assert "summary" in fields
        assert "vulnerability_detail" in fields
        assert "impact" in fields
        assert "recommendation" in fields


# ─────────────────────────────────────────────────────────────────────────────
# 7. ScribeAgent.run() — LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

class TestScribeAgentRun:
    @pytest.mark.asyncio
    async def test_run_populates_reports(self):
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        finding = _make_finding(hypothesis_id="hyp-1")
        hyp = _make_hypothesis()
        hyp = hyp.model_copy(update={"id": "hyp-1"})
        exploit = _make_exploit(hypothesis_id="hyp-1")

        state = {
            "audit_id": "scribe-test",
            "platform": "c4",
            "validated_findings": [finding],
            "hypotheses": [hyp],
            "exploits": [exploit],
        }
        new_state = await agent.run(state)

        assert "reports" in new_state
        assert len(new_state["reports"]) == 1
        report_md = list(new_state["reports"].values())[0]
        assert "## [H]" in report_md  # c4 format

    @pytest.mark.asyncio
    async def test_run_skips_false_positives_but_includes_unverified(self):
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        fp_finding = _make_finding(status="false_positive")
        ur_finding = _make_finding(status="unverified", title="Needs Triage")
        vf_finding = _make_finding(status="validated", title="Real Bug")

        state = {
            "audit_id": "filter-test",
            "platform": "generic",
            "validated_findings": [fp_finding, ur_finding, vf_finding],
            "hypotheses": [],
            "exploits": [],
        }
        new_state = await agent.run(state)

        # validated + unverified produce reports; false_positive is skipped
        assert len(new_state["reports"]) == 2
        report_bodies = list(new_state["reports"].values())
        assert any("Real Bug" in r for r in report_bodies)
        assert any("Needs Triage" in r for r in report_bodies)
        # false_positive must NOT appear
        assert not any("false_positive" in r for r in report_bodies)

    @pytest.mark.asyncio
    async def test_run_empty_no_validated_findings(self):
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        state = {
            "audit_id": "empty-test",
            "platform": "sherlock",
            "validated_findings": [],
            "hypotheses": [],
            "exploits": [],
        }
        new_state = await agent.run(state)

        assert new_state["reports"] == {}

    @pytest.mark.asyncio
    async def test_run_renders_immunefi_format(self):
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        finding = _make_finding(hypothesis_id="h1")
        hyp = _make_hypothesis()
        hyp = hyp.model_copy(update={"id": "h1"})
        exploit = _make_exploit(hypothesis_id="h1")

        state = {
            "audit_id": "immunefi-test",
            "platform": "immunefi",
            "validated_findings": [finding],
            "hypotheses": [hyp],
            "exploits": [exploit],
        }
        new_state = await agent.run(state)

        report = list(new_state["reports"].values())[0]
        assert "# Vulnerability Report" in report
        assert "## Title" in report

    @pytest.mark.asyncio
    async def test_run_renders_sherlock_format(self):
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        finding = _make_finding(hypothesis_id="h2")
        hyp = _make_hypothesis()
        hyp = hyp.model_copy(update={"id": "h2"})
        exploit = _make_exploit(hypothesis_id="h2")

        state = {
            "audit_id": "sherlock-test",
            "platform": "sherlock",
            "validated_findings": [finding],
            "hypotheses": [hyp],
            "exploits": [exploit],
        }
        new_state = await agent.run(state)

        report = list(new_state["reports"].values())[0]
        assert "## Tool used" in report

    @pytest.mark.asyncio
    async def test_run_preserves_state_fields(self):
        client = MockProvider(side_effect=Exception("err"))
        agent = ScribeAgent(provider=client)

        state = {
            "audit_id": "preserve-test",
            "platform": "generic",
            "validated_findings": [],
            "hypotheses": [],
            "exploits": [],
            "other_field": "preserved",
        }
        new_state = await agent.run(state)

        assert new_state.get("other_field") == "preserved"

    @pytest.mark.asyncio
    async def test_run_multiple_findings_multiple_reports(self):
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        f1 = _make_finding(title="Bug A", hypothesis_id="h-a")
        f2 = _make_finding(title="Bug B", hypothesis_id="h-b")

        state = {
            "audit_id": "multi-test",
            "platform": "generic",
            "validated_findings": [f1, f2],
            "hypotheses": [],
            "exploits": [],
        }
        new_state = await agent.run(state)

        assert len(new_state["reports"]) == 2



# ─────────────────────────────────────────────────────────────────────────────
# Needs-review section
# ─────────────────────────────────────────────────────────────────────────────

class TestScribeNeedsReview:
    def test_render_needs_review_includes_section_header(self):
        finding = _make_finding(status="unverified", title="Suspicious access control")
        md = _render_needs_review(finding, None, None)
        assert "Findings Requiring Manual Review" in md
        assert "Suspicious access control" in md

    def test_render_needs_review_includes_hypothesis_scenario(self):
        finding = _make_finding(status="unverified")
        hyp = _make_hypothesis("Maybe reentrancy")
        md = _render_needs_review(finding, hyp, None)
        assert "Attacker calls withdraw" in md

    def test_render_needs_review_without_hypothesis_graceful(self):
        finding = _make_finding(status="unverified")
        md = _render_needs_review(finding, None, None)
        assert "No scenario available" in md

    def test_render_needs_review_includes_partial_poc(self):
        finding = _make_finding(status="unverified")
        exploit = _make_exploit(status="partial")
        md = _render_needs_review(finding, None, exploit)
        assert "pragma solidity" in md or "PoC" in md

    @pytest.mark.asyncio
    async def test_scribe_includes_needs_review_in_reports(self):
        """ScribeAgent.run() must create a report entry for unverified findings."""
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        confirmed = _make_finding(title="Confirmed Bug", status="validated",
                                   hypothesis_id="hyp-a")
        unverified = _make_finding(title="Needs Triage", status="unverified",
                                    hypothesis_id="hyp-b")

        state = {
            "audit_id": "nr-scribe-test",
            "platform": "generic",
            "validated_findings": [confirmed, unverified],
            "hypotheses": [],
            "exploits": [],
        }
        new_state = await agent.run(state)

        assert len(new_state["reports"]) == 2
        reports = new_state["reports"]
        # The unverified finding's report should contain the review header
        unverified_report = reports.get(unverified.id, "")
        assert "Manual Review" in unverified_report

    @pytest.mark.asyncio
    async def test_scribe_only_false_positives_no_reports(self):
        """When all findings are false_positive, no reports should be generated."""
        client = _make_llm_client()
        agent = ScribeAgent(provider=client)

        fp = _make_finding(title="Not a Bug", status="false_positive")

        state = {
            "audit_id": "fp-test",
            "platform": "generic",
            "validated_findings": [fp],
            "hypotheses": [],
            "exploits": [],
        }
        new_state = await agent.run(state)

        assert len(new_state["reports"]) == 0
