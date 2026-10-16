"""Unit tests for Pydantic models."""

from __future__ import annotations

from audit_agent.models.schemas import (
    ExploitResult,
    ForkAnalysis,
    Hypothesis,
    Invariant,
    ProtocolSummary,
    StaticFinding,
    ValidatedFinding,
)


def test_protocol_summary_defaults() -> None:
    ps = ProtocolSummary(
        name="Foo",
        category="DEX",
        key_mechanisms=[],
        user_roles=[],
        sol_files=[],
    )
    assert ps.raw_notes == ""


def test_fork_analysis_not_fork() -> None:
    fa = ForkAnalysis(is_fork=False)
    assert fa.similarity_score == 0.0
    assert fa.base_protocol is None


def test_static_finding_defaults() -> None:
    sf = StaticFinding(
        audit_id="abc",
        tool="slither",
        title="reentrancy",
        severity="high",
        description="re-entrancy detected",
    )
    assert sf.id  # UUID auto-generated
    assert sf.raw == {}


def test_hypothesis_priority_default() -> None:
    h = Hypothesis(
        audit_id="abc",
        title="Test",
        attack_scenario="attacker calls foo",
        estimated_severity="critical",
    )
    assert h.priority_score == 0.0
    assert h.status == "pending"


def test_exploit_result() -> None:
    er = ExploitResult(
        hypothesis_id="hyp-1",
        test_type="unit",
        status="exploited",
        poc_code="// test code",
    )
    assert er.reproduction_steps == []


def test_invariant_roundtrip() -> None:
    inv = Invariant(
        audit_id="abc",
        invariant_type="global",
        natural_language="Token balance never exceeds total supply",
        source="natspec",
    )
    data = inv.model_dump()
    restored = Invariant(**data)
    assert restored.natural_language == inv.natural_language
