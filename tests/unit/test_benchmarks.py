"""Tests for Phase 6: benchmark framework — metrics, matching, runner, learning loop.

All tests are offline — no Docker, no network, no API keys.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.learning.feedback import AuditFeedback, FeedbackEvent, LearningLoop
from audit_agent.models.schemas import Hypothesis, ValidatedFinding
from benchmarks.matching import (
    GroundTruthFinding,
    finding_matches,
    match_hypotheses_to_ground_truth,
    match_score,
)
from benchmarks.metrics import (
    report_format_valid,
    time_per_audit,
)
from benchmarks.runner import (
    BenchmarkContest,
    BenchmarkReport,
    BenchmarkDataset,
    GroundTruthFindingModel,
    compute_metrics,
    load_dataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_hypothesis(
    title: str = "Reentrancy in withdraw()",
    vuln_class: str = "SWC-107",
    affected_functions: list[str] | None = None,
) -> Hypothesis:
    return Hypothesis(
        audit_id="test-audit",
        title=title,
        vulnerability_class=vuln_class,
        affected_functions=affected_functions or ["vault.withdraw(uint256)"],
        attack_scenario="Attacker drains vault via reentrancy",
        required_preconditions=["deposited ETH"],
        estimated_severity="critical",
        priority_score=90.0,
    )


def _make_gt(
    title: str = "Reentrancy allows full drain",
    vuln_class: str = "SWC-107",
    affected_functions: list[str] | None = None,
) -> GroundTruthFinding:
    return GroundTruthFinding(
        title=title,
        severity="high",
        vulnerability_class=vuln_class,
        affected_functions=affected_functions or ["vault.withdraw(uint256)"],
        description="Classic CEI violation",
    )


def _make_validated_finding(status: str = "validated", confidence: float = 0.95) -> ValidatedFinding:
    return ValidatedFinding(
        audit_id="test",
        title="Reentrancy",
        severity="critical",
        confidence=confidence,
        status=status,  # type: ignore[arg-type]
        description="Found reentrancy",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. report_format_valid
# ─────────────────────────────────────────────────────────────────────────────

class TestReportFormatValidation:
    def test_valid_c4_report(self):
        md = (
            "## [H] Title\n\n"
            "### Summary\nSummary text\n\n"
            "### Vulnerability Detail\nDetail\n\n"
            "### Impact\nImpact\n\n"
            "### Proof of Concept\n```solidity\n// poc\n```\n\n"
            "### Recommended Mitigation Steps\nFix it\n"
        )
        assert report_format_valid(md, "c4") is True

    def test_invalid_c4_missing_section(self):
        md = "## [H] Title\n### Summary\n### Impact\n"
        assert report_format_valid(md, "c4") is False

    def test_valid_immunefi_report(self):
        md = (
            "# Vulnerability Report\n\n"
            "## Title\nSomething\n\n"
            "## Severity\nCritical\n\n"
            "## Description\nDetail\n\n"
            "## Proof of Concept\n```\n// poc\n```\n\n"
            "## Impact\nFull drain\n\n"
            "## Recommended Fix\nFix CEI\n"
        )
        assert report_format_valid(md, "immunefi") is True

    def test_valid_sherlock_report(self):
        md = (
            "## Summary\nS\n\n"
            "## Vulnerability Detail\nD\n\n"
            "## Impact\nI\n\n"
            "## Code Snippet\n```solidity\n// x\n```\n\n"
            "## Tool used\nManual Review\n\n"
            "## Recommendation\nFix it\n"
        )
        assert report_format_valid(md, "sherlock") is True

    def test_unknown_platform_uses_generic(self):
        md = "## Summary\n## Vulnerability Detail\n## Impact\n## Recommendation\n"
        assert report_format_valid(md, "unknown_platform") is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. finding_matches — exact class
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchingExactClass:
    def test_same_swc_and_function_matches(self):
        hyp = _make_hypothesis(vuln_class="SWC-107", affected_functions=["vault.withdraw(uint256)"])
        gt = _make_gt(vuln_class="SWC-107", affected_functions=["vault.withdraw(uint256)"])
        assert finding_matches(hyp, gt) is True

    def test_different_class_unrelated_title_no_match(self):
        hyp = _make_hypothesis(
            title="tx.origin authentication", vuln_class="SWC-115",
            affected_functions=["vault.withdraw(uint256)"],
        )
        gt = _make_gt(vuln_class="SWC-107", affected_functions=["vault.withdraw(uint256)"])
        # Same function, but incompatible class and no title overlap → no match.
        assert finding_matches(hyp, gt) is False

    def test_class_match_alone_not_sufficient(self):
        # Same class, completely different functions → location-match missing.
        hyp = _make_hypothesis(
            title="Unchecked call return value",
            vuln_class="SWC-107",
            affected_functions=["alpha.execute()"],
        )
        gt = _make_gt(
            title="Missing access control on admin setter",
            vuln_class="SWC-107",
            affected_functions=["beta.configure()"],
        )
        assert finding_matches(hyp, gt) is False

    def test_class_match_with_fn_overlap_matches(self):
        hyp = _make_hypothesis(vuln_class="SWC-107", affected_functions=["vault.withdraw(uint256)"])
        gt = _make_gt(vuln_class="SWC-107", affected_functions=["vault.deposit()", "vault.withdraw()"])
        assert finding_matches(hyp, gt) is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. finding_matches — function overlap
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchingFunctionOverlap:
    def test_shared_function_name_matches(self):
        hyp = _make_hypothesis(
            vuln_class="oracle-manipulation",
            affected_functions=["oracle.getPrice(address)"]
        )
        gt = _make_gt(
            vuln_class="oracle-manipulation",
            affected_functions=["oracle.getPrice(address)", "pool.borrow()"]
        )
        assert finding_matches(hyp, gt) is True

    def test_function_name_token_matching(self):
        # "vulnerablebank.withdraw(uint256)" shares "withdraw" with "vault.withdraw()"
        hyp = _make_hypothesis(affected_functions=["vulnerablebank.withdraw(uint256)"])
        gt = _make_gt(affected_functions=["vault.withdraw()"])
        # withdraw token matches
        score = match_score(hyp, gt)
        assert score > 0.0

    def test_no_function_overlap_reduces_score(self):
        # Same class but completely unrelated titles and no shared function tokens
        hyp = _make_hypothesis(
            title="Integer overflow in fee calculation",
            vuln_class="SWC-107",
            affected_functions=["token.computeFee(uint256)"],
        )
        gt = _make_gt(
            title="Missing pause guard on bridge",
            vuln_class="SWC-107",
            affected_functions=["bridge.lockTokens(uint256)"],
        )
        score = match_score(hyp, gt)
        # Class match = 0.5, fn overlap = 0.0, title overlap ≈ 0.0
        assert score < 0.6


# ─────────────────────────────────────────────────────────────────────────────
# 7. finding_matches — no match
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchingNoMatch:
    def test_completely_different_does_not_match(self):
        hyp = _make_hypothesis(
            title="Gas optimization in loop",
            vuln_class="gas",
            affected_functions=["token.transfer(address,uint256)"],
        )
        gt = _make_gt(
            title="Reentrancy drains vault",
            vuln_class="SWC-107",
            affected_functions=["vault.withdraw(uint256)"],
        )
        assert finding_matches(hyp, gt) is False

    def test_score_zero_on_all_mismatches(self):
        hyp = _make_hypothesis(vuln_class="gas", affected_functions=["a.foo()"])
        gt = _make_gt(vuln_class="SWC-107", affected_functions=["b.bar()"])
        score = match_score(hyp, gt)
        assert score < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetLoading:
    def test_load_dvd_dataset(self):
        ds = load_dataset("damn_vulnerable_defi")
        assert isinstance(ds, BenchmarkDataset)
        assert ds.name == "damn_vulnerable_defi"
        assert len(ds.contests) == 5

    def test_load_ethernaut_dataset(self):
        ds = load_dataset("ethernaut_levels")
        assert isinstance(ds, BenchmarkDataset)
        assert len(ds.contests) == 5

    def test_each_dvd_contest_has_findings(self):
        ds = load_dataset("damn_vulnerable_defi")
        for contest in ds.contests:
            assert len(contest.official_findings) > 0, f"{contest.id} has no findings"

    def test_public_ctf_datasets_are_smoke_and_contaminated(self):
        for name in ("damn_vulnerable_defi", "ethernaut_levels"):
            for contest in load_dataset(name).contests:
                assert contest.split == "smoke"
                assert contest.contaminated is True

    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_dataset")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Benchmark runner e2e (mocked audit)
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkRunnerE2E:
    @pytest.mark.asyncio
    async def test_run_benchmark_with_mock_runner(self):
        """End-to-end: mock audit returns a hypothesis that matches GT."""
        from benchmarks.runner import run_benchmark

        # GT for dvd-unstoppable: denial-of-service in flashLoan
        matching_hyp = _make_hypothesis(
            title="Flash loan blocked by direct token transfer",
            vuln_class="denial-of-service",
            affected_functions=["unstoppablelender.flashloan(uint256)"],
        )

        async def mock_runner(contest, audit_id):
            validated = _make_validated_finding(status="validated", confidence=0.95)
            return {
                "hypotheses": [matching_hyp],
                "validated_findings": [validated],
                "tokens_used": 25_000,
                "exploits": [],
            }

        report = await run_benchmark("damn_vulnerable_defi", limit=1, audit_runner=mock_runner, split="smoke")

        assert isinstance(report, BenchmarkReport)
        assert report.num_contests == 1
        assert report.per_contest[0].recall_hypotheses_at_30 == 1.0  # hypothesis matched the GT
        # The validated finding has no hypothesis link and generic fields → no GT match.
        assert report.per_contest[0].precision_gt == 0.0

    @pytest.mark.asyncio
    async def test_run_benchmark_zero_recall_on_miss(self):
        """Mock audit returns a completely wrong hypothesis — recall = 0."""
        from benchmarks.runner import run_benchmark

        wrong_hyp = _make_hypothesis(
            title="Gas inefficiency in loop",
            vuln_class="gas",
            affected_functions=["completely.different()"],
        )

        async def mock_runner(contest, audit_id):
            return {
                "hypotheses": [wrong_hyp],
                "validated_findings": [],
                "tokens_used": 10_000,
                "exploits": [],
            }

        report = await run_benchmark("damn_vulnerable_defi", limit=1, audit_runner=mock_runner, split="smoke")
        assert report.per_contest[0].recall_hypotheses_at_30 == 0.0

    @pytest.mark.asyncio
    async def test_run_benchmark_respects_limit(self):
        """Limit=2 should only process 2 contests even for a 5-contest dataset."""
        from benchmarks.runner import run_benchmark

        processed: list[str] = []

        async def mock_runner(contest, audit_id):
            processed.append(contest.id)
            return {"hypotheses": [], "validated_findings": [], "tokens_used": 0, "exploits": []}

        await run_benchmark("damn_vulnerable_defi", limit=2, audit_runner=mock_runner, split="smoke")
        assert len(processed) == 2

    @pytest.mark.asyncio
    async def test_run_benchmark_default_split_is_test(self):
        """DVD is smoke-only, so the default (split=test) selects nothing."""
        from benchmarks.runner import run_benchmark

        processed: list[str] = []

        async def mock_runner(contest, audit_id):
            processed.append(contest.id)
            return {}

        report = await run_benchmark("damn_vulnerable_defi", audit_runner=mock_runner)
        assert processed == []
        assert report.num_contests == 0

    @pytest.mark.asyncio
    async def test_compute_metrics_directly(self):
        hyp = _make_hypothesis(vuln_class="SWC-107", affected_functions=["vault.withdraw(uint256)"])
        gt_model = GroundTruthFindingModel(
            title="Reentrancy", severity="high", vulnerability_class="SWC-107",
            affected_functions=["vault.withdraw(uint256)"]
        )
        contest = BenchmarkContest(
            id="test", name="Test", source_dir=".", split="smoke", official_findings=[gt_model]
        )
        validated = _make_validated_finding(status="validated", confidence=0.95)
        final_state = {
            "hypotheses": [hyp],
            "validated_findings": [validated],
            "tokens_used": 50_000,
        }
        metrics = compute_metrics(final_state, contest, elapsed_seconds=12.5)
        assert metrics.recall_hypotheses_at_30 == 1.0
        assert metrics.time_s == 12.5
        assert metrics.tokens == 50_000


# ─────────────────────────────────────────────────────────────────────────────
# 10. LearningLoop — confirmed finding
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningLoopConfirmed:
    @pytest.mark.asyncio
    async def test_confirmed_stored_in_patterns(self):
        loop = LearningLoop()
        event = FeedbackEvent(
            finding_id=str(uuid.uuid4()),
            audit_id="audit-1",
            title="Reentrancy in withdraw()",
            vulnerability_class="SWC-107",
            affected_functions=["vault.withdraw(uint256)"],
            description="Classic CEI violation",
            severity="critical",
        )
        fb = AuditFeedback(audit_id="audit-1", confirmed=[event])
        await loop.process_feedback("audit-1", fb)

        assert len(loop.confirmed_patterns) == 1
        assert loop.confirmed_patterns[0]["title"] == "Reentrancy in withdraw()"

    @pytest.mark.asyncio
    async def test_multiple_confirmed_all_stored(self):
        loop = LearningLoop()
        events = [
            FeedbackEvent(
                finding_id=str(uuid.uuid4()),
                audit_id="audit-1",
                title=f"Finding {i}",
                vulnerability_class="SWC-107",
                affected_functions=["f.g()"],
                description="",
                severity="high",
            )
            for i in range(3)
        ]
        fb = AuditFeedback(audit_id="audit-1", confirmed=events)
        await loop.process_feedback("audit-1", fb)
        assert len(loop.confirmed_patterns) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 11. LearningLoop — missed finding → KB + Semgrep rule
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningLoopMissed:
    @pytest.mark.asyncio
    async def test_missed_added_to_knowledge_base(self):
        loop = LearningLoop()
        event = FeedbackEvent(
            finding_id=str(uuid.uuid4()),
            audit_id="audit-2",
            title="Oracle manipulation missed",
            vulnerability_class="oracle-manipulation",
            affected_functions=["oracle.getPrice(address)"],
            description="TWAP can be manipulated",
            severity="high",
        )
        fb = AuditFeedback(audit_id="audit-2", missed=[event])
        await loop.process_feedback("audit-2", fb)

        assert len(loop.knowledge_base) == 1
        assert loop.knowledge_base[0]["type"] == "missed_finding"

    @pytest.mark.asyncio
    async def test_missed_generates_semgrep_rule(self):
        loop = LearningLoop()
        event = FeedbackEvent(
            finding_id=str(uuid.uuid4()),
            audit_id="audit-2",
            title="Reentrancy missed",
            vulnerability_class="SWC-107",
            affected_functions=["vault.withdraw(uint256)"],
            description="CEI violation missed",
            severity="critical",
        )
        fb = AuditFeedback(audit_id="audit-2", missed=[event])
        await loop.process_feedback("audit-2", fb)

        assert len(loop.semgrep_rules) == 1
        rule = loop.semgrep_rules[0]
        assert "rules:" in rule
        assert "solidity" in rule

    @pytest.mark.asyncio
    async def test_false_positive_stored_in_fp_set(self):
        loop = LearningLoop()
        event = FeedbackEvent(
            finding_id=str(uuid.uuid4()),
            audit_id="audit-3",
            title="Straw-man reentrancy",
            vulnerability_class="SWC-107",
            affected_functions=["token.transfer()"],
            description="Not a real issue",
            severity="info",
            notes="Straw-man contract in test",
        )
        fb = AuditFeedback(audit_id="audit-3", false_positives=[event])
        await loop.process_feedback("audit-3", fb)

        assert len(loop.fp_set) == 1
        assert loop.fp_set[0]["title"] == "Straw-man reentrancy"

    @pytest.mark.asyncio
    async def test_mixed_feedback_all_processed(self):
        loop = LearningLoop()
        make_event = lambda title, vc: FeedbackEvent(
            finding_id=str(uuid.uuid4()), audit_id="a",
            title=title, vulnerability_class=vc,
            affected_functions=[], description="", severity="medium"
        )
        fb = AuditFeedback(
            audit_id="a",
            confirmed=[make_event("Good finding", "SWC-107")],
            false_positives=[make_event("Wrong finding", "SWC-115")],
            missed=[make_event("Missed finding", "oracle-manipulation")],
        )
        await loop.process_feedback("a", fb)
        assert len(loop.confirmed_patterns) == 1
        assert len(loop.fp_set) == 1
        assert len(loop.knowledge_base) == 1
        assert len(loop.semgrep_rules) == 1
