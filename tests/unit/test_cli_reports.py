"""Tests for CLI report-saving functionality.

All tests are offline — no Docker, no network, no real LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audit_agent.cli import _save_reports
from audit_agent.models.schemas import ExploitResult, Hypothesis, ValidatedFinding


# ─────────────────────────────────────────────────────────────────────────────
# Import helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_save_reports():
    return _save_reports


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_validated(title: str = "Bug", status: str = "validated") -> ValidatedFinding:
    return ValidatedFinding(
        audit_id="test-audit",
        title=title,
        severity="high",
        confidence=0.9,
        status=status,  # type: ignore[arg-type]
        description="A real bug.",
    )


def _make_hypothesis(title: str = "Hypothesis") -> Hypothesis:
    return Hypothesis(
        audit_id="test-audit",
        title=title,
        attack_scenario="Attacker drains funds.",
        estimated_severity="high",
    )


def _make_exploit() -> ExploitResult:
    return ExploitResult(
        hypothesis_id="h1",
        test_type="unit",
        status="exploited",
        poc_code="pragma solidity ^0.8.20;",
    )


def _make_state(
    reports: dict,
    validated=None,
    hypotheses=None,
    exploits=None,
) -> dict:
    return {
        "reports": reports,
        "validated_findings": validated or [],
        "hypotheses": hypotheses or [],
        "exploits": exploits or [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. _save_reports — files are created
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveReports:
    def test_reports_saved_to_disk(self, tmp_path):
        save = _get_save_reports()
        finding = _make_validated()
        state = _make_state(
            reports={finding.id: "# Report\n\nFound a bug."},
            validated=[finding],
        )
        save(state, "audit-123", "/contracts", "audit", "generic", tmp_path, MagicMock())

        run_dir = tmp_path / "audit-123"
        md_files = list(run_dir.glob("*.md"))
        # At least the individual report + SUMMARY.md
        assert len(md_files) >= 2

    def test_summary_contains_all_reports(self, tmp_path):
        save = _get_save_reports()
        f1 = _make_validated(title="Bug A")
        f2 = _make_validated(title="Bug B")
        state = _make_state(
            reports={
                f1.id: "# Bug A report",
                f2.id: "# Bug B report",
            },
            validated=[f1, f2],
        )
        save(state, "audit-456", "/contracts", "audit", "generic", tmp_path, MagicMock())

        summary = (tmp_path / "audit-456" / "SUMMARY.md").read_text()
        assert "# Bug A report" in summary
        assert "# Bug B report" in summary

    def test_debug_files_valid_json(self, tmp_path):
        save = _get_save_reports()
        hyp = _make_hypothesis()
        exploit = _make_exploit()
        validated = _make_validated()
        state = _make_state(
            reports={validated.id: "# Report"},
            validated=[validated],
            hypotheses=[hyp],
            exploits=[exploit],
        )
        save(state, "audit-789", "/contracts", "audit", "generic", tmp_path, MagicMock())

        debug_dir = tmp_path / "audit-789" / "debug"
        for fname in ("hypotheses.json", "exploits.json", "validated_findings.json"):
            data = json.loads((debug_dir / fname).read_text())
            assert isinstance(data, list)

    def test_custom_reports_dir(self, tmp_path):
        save = _get_save_reports()
        custom_dir = tmp_path / "custom-output"
        state = _make_state(reports={"id1": "# Custom report"})
        save(state, "audit-abc", "/c", "audit", "generic", custom_dir, MagicMock())

        assert (custom_dir / "audit-abc" / "SUMMARY.md").exists()

    def test_empty_reports_creates_summary_with_zeros(self, tmp_path):
        save = _get_save_reports()
        state = _make_state(reports={})
        save(state, "audit-empty", "/c", "review", "c4", tmp_path, MagicMock())

        summary = (tmp_path / "audit-empty" / "SUMMARY.md").read_text()
        assert "True positives: 0" in summary
        assert "Needs review: 0" in summary

    def test_summary_counts_tp_nr_fp(self, tmp_path):
        save = _get_save_reports()
        tp = _make_validated(status="validated")
        nr = _make_validated(status="unverified")
        fp = _make_validated(status="false_positive")
        state = _make_state(
            reports={tp.id: "# TP", nr.id: "# NR"},
            validated=[tp, nr, fp],
        )
        save(state, "audit-counts", "/c", "audit", "generic", tmp_path, MagicMock())

        summary = (tmp_path / "audit-counts" / "SUMMARY.md").read_text()
        assert "True positives: 1" in summary
        assert "Needs review: 1" in summary
        assert "False positives: 1" in summary
