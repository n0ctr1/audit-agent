"""Unit tests for the Validator — fully offline, no external deps."""

from __future__ import annotations

from audit_agent.knowledge.ingestion.validator import (
    Validator,
    ValidationResult,
    validate_raw_dict,
    QUALITY_REJECT_THRESHOLD,
)
from audit_agent.knowledge.models import (
    CodeSnippet,
    DescriptionStructured,
    Finding,
    Fix,
    ProtocolMeta,
)


def _make_good_finding(**overrides) -> Finding:
    defaults = dict(
        title="Price manipulation via TWAP oracle allows draining lending pool",
        severity="high",
        source="code4rena",
        url="https://github.com/code-423n4/2023-example/blob/main/H-01.md",
        vulnerability_class="Oracle Manipulation",
        custom_tags=["price-manipulation", "twap"],
        protocol=ProtocolMeta(name="Bastion", category="lending"),
        description_structured=DescriptionStructured(
            summary="An attacker exploits a short TWAP window to manipulate oracle price and drain funds.",
            root_cause="The TWAP window is only 30 minutes, allowing economic manipulation.",
            attack_vector="Flash loan + large swap across multiple blocks.",
            impact="Full drain of lending pool liquidity.",
            preconditions=["Flash loan provider available"],
        ),
        code_snippets=[
            CodeSnippet(file="Oracle.sol", lines="87-112", code="uint224 price = ...;")
        ],
        poc="See Foundry test TWAPManipulationTest",
        fix=Fix(description="Increase TWAP window to 2 hours.", code_diff="MIN_PERIOD = 7200"),
    )
    defaults.update(overrides)
    return Finding(**defaults)


def test_good_finding_passes():
    v = Validator()
    result = v.validate(_make_good_finding())
    assert result.valid
    assert result.quality_score >= QUALITY_REJECT_THRESHOLD
    assert result.rejection_reason is None


def test_missing_summary_rejected():
    f = _make_good_finding(
        description_structured=DescriptionStructured(summary="", root_cause="some root cause")
    )
    v = Validator()
    result = v.validate(f)
    assert not result.valid
    assert "summary" in (result.rejection_reason or "")


def test_short_title_rejected():
    f = _make_good_finding(title="Vuln")
    v = Validator()
    result = v.validate(f)
    assert not result.valid
    assert "title" in (result.rejection_reason or "")


def test_invalid_severity_rejected():
    f = _make_good_finding()
    # Manually set invalid severity bypassing Pydantic
    object.__setattr__(f, "severity", "critical")  # valid
    f2 = Finding(
        **{**f.model_dump(), "severity": "high"}
    )
    # Now test with a direct dict that has an invalid severity
    _, err = validate_raw_dict({"title": "T" * 20, "severity": "extreme", "source": "code4rena"})
    assert err is not None


def test_quality_score_increases_with_detail():
    v = Validator()

    minimal = Finding(
        title="A " * 6,  # barely long enough
        severity="low",
        source="code4rena",
        description_structured=DescriptionStructured(
            summary="A " * 16,  # barely long enough
            root_cause="",
        ),
    )
    full = _make_good_finding()

    r_min = v.validate(minimal)
    r_full = v.validate(full)

    assert r_full.quality_score > r_min.quality_score


def test_critical_severity_gets_higher_score():
    v = Validator()
    high = _make_good_finding(severity="high")
    crit = _make_good_finding(severity="critical")
    assert v.validate(crit).quality_score >= v.validate(high).quality_score


def test_fingerprint_same_for_same_content():
    v = Validator()
    f1 = _make_good_finding()
    f2 = _make_good_finding()
    assert v.fingerprint(f1) == v.fingerprint(f2)


def test_fingerprint_different_for_different_content():
    v = Validator()
    f1 = _make_good_finding(title="Reentrancy in withdraw function allows fund drain")
    f2 = _make_good_finding(title="Integer overflow in reward calculation loses user funds")
    assert v.fingerprint(f1) != v.fingerprint(f2)


def test_fingerprint_case_insensitive():
    v = Validator()
    f1 = _make_good_finding(title="PRICE MANIPULATION via twap oracle allows draining lending pool")
    f2 = _make_good_finding(title="price manipulation via TWAP oracle allows draining lending pool")
    # Same after lowercasing
    assert v.fingerprint(f1) == v.fingerprint(f2)


def test_no_code_snippets_warns_but_can_pass():
    v = Validator()
    f = _make_good_finding(code_snippets=[])
    result = v.validate(f)
    assert result.valid
    assert any("snippet" in w for w in result.warnings)


def test_validate_raw_dict_valid():
    data = {
        "title": "Some interesting vulnerability with detail",
        "severity": "high",
        "source": "code4rena",
        "description_structured": {"summary": "A real vulnerability summary here."},
    }
    finding, err = validate_raw_dict(data)
    assert finding is not None
    assert err is None


def test_validate_raw_dict_invalid():
    _, err = validate_raw_dict({"title": "Missing severity field"})
    assert err is not None
