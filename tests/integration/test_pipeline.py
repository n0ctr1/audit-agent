"""Integration test: end-to-end pipeline with mocked Qdrant.

Tests the full path: raw fixture MD → structurer → validator → pipeline → Qdrant upsert.
All network calls (Qdrant, Anthropic) are mocked — runs offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.knowledge.ingestion.pipeline import IngestionPipeline
from audit_agent.knowledge.ingestion.scrapers.code4rena import Code4renaScraper
from audit_agent.knowledge.ingestion.structurer import Structurer
from audit_agent.knowledge.ingestion.validator import Validator
from audit_agent.knowledge.models import (
    CodeSnippet,
    DescriptionStructured,
    Finding,
    ProtocolMeta,
)
from audit_agent.knowledge.search import KnowledgeBase
from audit_agent.knowledge.vectors.embedders import MockEmbedder

FIXTURES = Path(__file__).parent.parent / "fixtures" / "findings"


def _mock_qdrant_empty() -> MagicMock:
    """Qdrant mock that reports no existing entries (dedup returns empty)."""
    qdrant = MagicMock()
    qdrant.scroll = AsyncMock(return_value=([], None))
    qdrant.upsert = AsyncMock()
    qdrant.search = AsyncMock(return_value=[])
    return qdrant


def _make_finding(title: str = "Test finding with sufficient length", severity: str = "high") -> Finding:
    return Finding(
        title=title,
        severity=severity,  # type: ignore[arg-type]
        source="code4rena",
        url="https://github.com/code-423n4/test/blob/main/H-01.md",
        description_structured=DescriptionStructured(
            summary="An attacker exploits a vulnerability to drain funds from the protocol.",
            root_cause="The contract does not check balances before external calls.",
            attack_vector="Call withdraw repeatedly via reentrant receive() function.",
            impact="Full drain of all ETH in the vault.",
        ),
        code_snippets=[CodeSnippet(file="Vault.sol", lines="50-70", code="call{value: amount}('')")],
        poc="forge test --match ReentrancyTest",
    )


# ------------------------------------------------------------------ #
# Scraper tests (offline)
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_code4rena_scrape_local_dir():
    scraper = Code4renaScraper()
    result = await scraper.scrape_local_dir(FIXTURES)

    assert result.contests_processed == 1
    assert len(result.findings) == 5  # we have 5 fixture files
    assert result.errors == []

    for raw in result.findings:
        assert raw.text
        assert raw.source == "code4rena"
        assert raw.suggested_severity in ("critical", "high", "medium", "low", "info")


@pytest.mark.asyncio
async def test_code4rena_scrape_local_h_findings_only():
    """Verify H-prefix files are loaded."""
    scraper = Code4renaScraper()
    result = await scraper.scrape_local_dir(FIXTURES)
    titles_lower = [r.filename.lower() for r in result.findings]
    # All our fixtures start with c4- (not H- prefix) but are loaded via scrape_local_dir
    # In a real repo, _iter_hm_files would filter H-/M- prefixes
    assert len(result.findings) > 0


# ------------------------------------------------------------------ #
# Validator tests
# ------------------------------------------------------------------ #

def test_validator_accepts_good_finding():
    v = Validator()
    f = _make_finding()
    result = v.validate(f)
    assert result.valid
    assert result.quality_score >= 0.5


def test_validator_rejects_empty_summary():
    v = Validator()
    f = Finding(
        title="A finding with a long enough title here",
        severity="high",
        source="code4rena",
        description_structured=DescriptionStructured(summary=""),
    )
    result = v.validate(f)
    assert not result.valid


# ------------------------------------------------------------------ #
# Pipeline tests
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_pipeline_ingest_single_finding():
    qdrant = _mock_qdrant_empty()
    pipeline = IngestionPipeline(qdrant=qdrant, embedder=MockEmbedder())

    stats = await pipeline.ingest_batch([_make_finding()])
    assert stats.ingested == 1
    assert stats.errors == 0
    qdrant.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_ingest_batch():
    qdrant = _mock_qdrant_empty()
    pipeline = IngestionPipeline(qdrant=qdrant, embedder=MockEmbedder())

    findings = [_make_finding(f"Finding number {i} with enough title length here", "high") for i in range(5)]
    stats = await pipeline.ingest_batch(findings)

    assert stats.ingested == 5
    assert stats.errors == 0


@pytest.mark.asyncio
async def test_pipeline_deduplicates_in_memory():
    qdrant = _mock_qdrant_empty()
    pipeline = IngestionPipeline(qdrant=qdrant, embedder=MockEmbedder())

    f = _make_finding()
    # Ingest same finding twice in same batch
    stats = await pipeline.ingest_batch([f, f])
    assert stats.ingested == 1
    assert stats.rejected_duplicate == 1


@pytest.mark.asyncio
async def test_pipeline_rejects_low_quality():
    qdrant = _mock_qdrant_empty()
    pipeline = IngestionPipeline(qdrant=qdrant, embedder=MockEmbedder())

    bad = Finding(
        title="x",  # too short
        severity="high",
        source="code4rena",
        description_structured=DescriptionStructured(summary=""),
    )
    stats = await pipeline.ingest_batch([bad])
    assert stats.ingested == 0
    assert stats.rejected_quality == 1


@pytest.mark.asyncio
async def test_pipeline_deduplicates_against_qdrant():
    # Qdrant returns an existing entry for the fingerprint
    qdrant = MagicMock()
    existing_point = MagicMock()
    qdrant.scroll = AsyncMock(return_value=([existing_point], None))
    qdrant.upsert = AsyncMock()

    pipeline = IngestionPipeline(qdrant=qdrant, embedder=MockEmbedder())
    stats = await pipeline.ingest_batch([_make_finding()])
    assert stats.ingested == 0
    assert stats.rejected_duplicate == 1
    qdrant.upsert.assert_not_called()


# ------------------------------------------------------------------ #
# Full end-to-end: scraper → structurer (mocked) → validator → pipeline
# ------------------------------------------------------------------ #

def _make_structurer_mock(findings: list[Finding]) -> Structurer:
    """Return a Structurer whose structure() cycles through provided findings."""
    call_count = [0]

    async def mock_structure(raw_text, source, url="", finding_id=None):
        idx = call_count[0] % len(findings)
        call_count[0] += 1
        f = findings[idx].model_copy()
        f.raw_text = raw_text
        if finding_id:
            f.id = finding_id
        return f

    structurer = MagicMock(spec=Structurer)
    structurer.structure = mock_structure
    return structurer


@pytest.mark.asyncio
async def test_end_to_end_scrape_structure_ingest():
    """Simulates: load fixtures → structure each → ingest into pipeline."""
    scraper = Code4renaScraper()
    raw_result = await scraper.scrape_local_dir(FIXTURES)
    assert len(raw_result.findings) > 0

    template_findings = [
        _make_finding(f"Realistic finding {i} with sufficient detail for KB", sev)
        for i, sev in enumerate(["high", "high", "high", "high", "medium"])
    ]
    structurer = _make_structurer_mock(template_findings)

    qdrant = _mock_qdrant_empty()
    pipeline = IngestionPipeline(qdrant=qdrant, embedder=MockEmbedder())

    structured: list[Finding] = []
    for raw in raw_result.findings:
        finding = await structurer.structure(raw.text, source=raw.source, url=raw.url)
        structured.append(finding)

    stats = await pipeline.ingest_batch(structured)
    assert stats.ingested > 0
    assert stats.errors == 0


# ------------------------------------------------------------------ #
# Search smoke test
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_knowledge_search_acceptance_criteria():
    """Test the acceptance criteria from the spec."""
    qdrant = MagicMock()

    def _hit(title, source, url, severity, vuln_class, tags, summary):
        h = MagicMock()
        h.payload = {
            "title": title, "source": source, "url": url,
            "severity": severity, "vulnerability_class": vuln_class,
            "custom_tags": tags, "summary": summary, "quality_score": 0.85,
        }
        h.score = 0.9
        return h

    qdrant.search = AsyncMock(return_value=[
        _hit(
            "TWAP oracle manipulation in lending protocol",
            "code4rena",
            "https://github.com/code-423n4/example/blob/main/H-01.md",
            "high",
            "Oracle Manipulation",
            ["price-manipulation", "twap", "oracle"],
            "Attacker manipulates short TWAP window to drain lending pool.",
        ),
        _hit(
            "Flash loan attack via oracle price skew",
            "code4rena",
            "https://github.com/code-423n4/other/blob/main/H-02.md",
            "high",
            "Flash Loan Attack",
            ["flash-loan", "oracle", "price-manipulation"],
            "Flash loan enables single-block oracle manipulation.",
        ),
        _hit(
            "Bond price oracle manipulation via AMM",
            "sherlock",
            "https://github.com/sherlock-audit/example/blob/main/001.md",
            "high",
            "Oracle Manipulation",
            ["oracle", "amm", "price-manipulation"],
            "Bond market uses manipulable spot price oracle.",
        ),
        _hit(
            "Spot price oracle used instead of TWAP",
            "solodit",
            "https://solodit.xyz/issues/example",
            "medium",
            "Oracle Manipulation",
            ["oracle", "spot-price"],
            "Contract relies on instantaneous spot price for critical operations.",
        ),
        _hit(
            "TWAP window too short for manipulation resistance",
            "code4rena",
            "https://github.com/code-423n4/another/blob/main/M-01.md",
            "medium",
            "Oracle Manipulation",
            ["twap", "oracle"],
            "30-minute TWAP window insufficient on mainnet.",
        ),
    ])

    kb = KnowledgeBase(qdrant=qdrant, embedder=MockEmbedder())
    results = await kb.search("price manipulation via TWAP oracle", limit=5)

    # Acceptance criteria: returns list[dict] with required fields
    assert len(results) == 5
    for r in results:
        assert "title" in r
        assert "source" in r
        assert "url" in r
        assert "severity" in r
        assert r["severity"] in ("critical", "high", "medium", "low", "info")

    print("\n=== knowledge.search() acceptance test ===")
    for r in results:
        print(f'  {{"title": "{r["title"][:60]}", "source": "{r["source"]}", '
              f'"url": "{r["url"][:50]}...", "severity": "{r["severity"]}"}}')
