"""Unit tests for the Structurer.

All tests run offline — the Anthropic API call is mocked.
Fixture .md files in tests/fixtures/findings/ are used as raw input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.knowledge.ingestion.structurer import (
    Structurer,
    extract_severity_from_markdown,
    _coerce_severity,
)
from audit_agent.knowledge.models import Finding

FIXTURES = Path(__file__).parent.parent / "fixtures" / "findings"


def _make_mock_client(tool_input: dict[str, Any]) -> MockProvider:
    """Return a MockProvider that returns the given tool input via extract_finding."""
    return MockProvider([LLMResponse(
        text="",
        tool_calls=[ToolCall(id="1", name="extract_finding", arguments=tool_input)],
        input_tokens=0,
        output_tokens=0,
        stop_reason="tool_use",
        model="test-model",
    )])


_SAMPLE_TOOL_OUTPUT: dict[str, Any] = {
    "title": "Price manipulation via TWAP oracle allows draining lending pool",
    "severity": "high",
    "vulnerability_class": "Oracle Manipulation",
    "custom_tags": ["price-manipulation", "twap", "oracle"],
    "protocol_name": "Bastion",
    "protocol_category": "lending",
    "summary": "An attacker can manipulate the 30-minute TWAP oracle via large swaps and drain the lending pool.",
    "root_cause": "The TWAP window is only 30 minutes, insufficient to prevent economically-viable manipulation.",
    "attack_vector": "Flash loan to fund large Uniswap swaps over multiple blocks to skew TWAP, then borrow against inflated collateral.",
    "impact": "Attacker drains up to the full liquidity of the lending pool.",
    "preconditions": ["Attacker needs flash loan provider", "Protocol uses on-chain TWAP oracle"],
    "code_snippets": [
        {
            "file": "contracts/Oracle.sol",
            "lines": "87-112",
            "code": "uint224 price = uint224((price0Cumulative - price0CumulativeLast) / timeElapsed);",
        }
    ],
    "poc": "See test TWAPManipulationTest.testTWAPManipulation()",
    "fix_description": "Increase TWAP window to 2+ hours and add circuit breaker.",
    "fix_code_diff": "uint32 constant MIN_PERIOD = 7200; // was 1800",
    "references": ["https://github.com/bastion-protocol/bastion-core/blob/main/contracts/Oracle.sol#L87"],
}


@pytest.mark.asyncio
async def test_structure_returns_finding():
    client = _make_mock_client(_SAMPLE_TOOL_OUTPUT)
    structurer = Structurer(provider=client)

    raw_text = (FIXTURES / "c4-2023-bastion-h01.md").read_text()
    finding = await structurer.structure(raw_text, source="code4rena", url="https://github.com/example")

    assert isinstance(finding, Finding)
    assert finding.title == _SAMPLE_TOOL_OUTPUT["title"]
    assert finding.severity == "high"
    assert finding.source == "code4rena"
    assert "twap" in finding.custom_tags
    assert finding.protocol.name == "Bastion"
    assert finding.protocol.category == "lending"
    assert "30 minutes" in finding.description_structured.root_cause
    assert len(finding.code_snippets) == 1
    assert finding.code_snippets[0].file == "contracts/Oracle.sol"


@pytest.mark.asyncio
async def test_structure_preserves_raw_text():
    raw_text = (FIXTURES / "c4-2023-blueberry-h02.md").read_text()
    client = _make_mock_client({**_SAMPLE_TOOL_OUTPUT, "title": "Reentrancy in withdraw()"})
    structurer = Structurer(provider=client)

    finding = await structurer.structure(raw_text, source="code4rena")
    assert finding.raw_text == raw_text


@pytest.mark.asyncio
async def test_structure_with_custom_id():
    client = _make_mock_client(_SAMPLE_TOOL_OUTPUT)
    structurer = Structurer(provider=client)

    raw = "# Test finding"
    finding = await structurer.structure(raw, source="code4rena", finding_id="c4-2023-test-h01")
    assert finding.id == "c4-2023-test-h01"


@pytest.mark.asyncio
async def test_structure_raises_on_missing_tool_call():
    # Return LLMResponse with no tool_calls → triggers "did not call extract_finding"
    client = MockProvider([LLMResponse(
        text="some text response",
        tool_calls=[],
        input_tokens=0,
        output_tokens=0,
        stop_reason="end_turn",
        model="test-model",
    )])

    structurer = Structurer(provider=client)
    with pytest.raises(ValueError, match="did not call extract_finding"):
        await structurer.structure("some text", source="code4rena")


def test_extract_severity_from_markdown_high():
    text = "# H-01: Some Title\n\n## Severity: High\n\nBody..."
    assert extract_severity_from_markdown(text) == "high"


def test_extract_severity_from_markdown_medium():
    text = "# M-02: Medium Finding\n\n## Severity: Medium"
    assert extract_severity_from_markdown(text) == "medium"


def test_extract_severity_from_markdown_fallback():
    text = "# Random Report without severity keywords"
    result = extract_severity_from_markdown(text)
    assert result in ("critical", "high", "medium", "low", "info")


def test_coerce_severity_variants():
    assert _coerce_severity("HIGH") == "high"
    assert _coerce_severity("H") == "high"
    assert _coerce_severity("M") == "medium"
    assert _coerce_severity("C") == "critical"
    assert _coerce_severity("unknown") == "medium"


@pytest.mark.asyncio
async def test_structure_all_fixtures():
    """Smoke test: structurer can process all fixture files without crashing."""
    for fixture_path in sorted(FIXTURES.glob("*.md")):
        raw_text = fixture_path.read_text()
        sev = extract_severity_from_markdown(raw_text)
        assert sev in ("critical", "high", "medium", "low", "info"), (
            f"Bad severity from {fixture_path.name}: {sev}"
        )
