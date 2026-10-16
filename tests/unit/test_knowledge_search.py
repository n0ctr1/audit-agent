"""Unit tests for the KnowledgeBase search interface — Qdrant mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from audit_agent.knowledge.search import KnowledgeBase
from audit_agent.knowledge.vectors.embedders import MockEmbedder


def _make_qdrant_hit(payload: dict[str, Any], score: float = 0.9) -> MagicMock:
    hit = MagicMock()
    hit.payload = payload
    hit.score = score
    return hit


def _sample_payload(**overrides) -> dict[str, Any]:
    base = {
        "title": "Price manipulation via TWAP oracle",
        "source": "code4rena",
        "url": "https://github.com/code-423n4/example/blob/main/H-01.md",
        "severity": "high",
        "vulnerability_class": "Oracle Manipulation",
        "custom_tags": ["price-manipulation", "twap"],
        "summary": "An attacker uses a flash loan to skew TWAP and drain a lending pool.",
        "quality_score": 0.85,
    }
    base.update(overrides)
    return base


@pytest.fixture
def mock_qdrant() -> MagicMock:
    qdrant = MagicMock()
    qdrant.search = AsyncMock(return_value=[
        _make_qdrant_hit(_sample_payload(), 0.95),
        _make_qdrant_hit(_sample_payload(title="Reentrancy in withdraw", severity="high"), 0.82),
        _make_qdrant_hit(_sample_payload(title="Access control missing on initialize", severity="high"), 0.75),
        _make_qdrant_hit(_sample_payload(title="Flash loan oracle attack", severity="high"), 0.71),
        _make_qdrant_hit(_sample_payload(title="Integer overflow in reward", severity="medium"), 0.65),
    ])
    return qdrant


@pytest.mark.asyncio
async def test_search_returns_list(mock_qdrant):
    kb = KnowledgeBase(qdrant=mock_qdrant, embedder=MockEmbedder())
    results = await kb.search("price manipulation via TWAP oracle", limit=5)
    assert isinstance(results, list)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_search_result_schema(mock_qdrant):
    kb = KnowledgeBase(qdrant=mock_qdrant, embedder=MockEmbedder())
    results = await kb.search("price manipulation via TWAP oracle", limit=5)
    for r in results:
        assert "title" in r
        assert "source" in r
        assert "url" in r
        assert "severity" in r
        assert "vulnerability_class" in r
        assert "custom_tags" in r
        assert "summary" in r


@pytest.mark.asyncio
async def test_search_respects_limit(mock_qdrant):
    kb = KnowledgeBase(qdrant=mock_qdrant, embedder=MockEmbedder())
    results = await kb.search("reentrancy", limit=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_search_filters_by_min_quality():
    qdrant = MagicMock()
    qdrant.search = AsyncMock(return_value=[
        _make_qdrant_hit(_sample_payload(quality_score=0.9), 0.95),
        _make_qdrant_hit(_sample_payload(title="Low quality", quality_score=0.1), 0.8),
    ])
    kb = KnowledgeBase(qdrant=qdrant, embedder=MockEmbedder())
    results = await kb.search("oracle", limit=10, min_quality=0.5)
    assert all(r["title"] != "Low quality" for r in results)


@pytest.mark.asyncio
async def test_search_builds_severity_filter(mock_qdrant):
    kb = KnowledgeBase(qdrant=mock_qdrant, embedder=MockEmbedder())
    await kb.search("reentrancy", limit=5, severity="high")
    # Verify qdrant.search was called with a filter
    call_kwargs = mock_qdrant.search.call_args.kwargs
    assert call_kwargs.get("query_filter") is not None


@pytest.mark.asyncio
async def test_search_no_filter_when_no_args(mock_qdrant):
    kb = KnowledgeBase(qdrant=mock_qdrant, embedder=MockEmbedder())
    await kb.search("oracle manipulation", limit=5)
    call_kwargs = mock_qdrant.search.call_args.kwargs
    assert call_kwargs.get("query_filter") is None


@pytest.mark.asyncio
async def test_mock_embedder_produces_1024d_vector():
    emb = MockEmbedder()
    vecs = await emb.embed_text(["price manipulation via TWAP"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 1024


@pytest.mark.asyncio
async def test_mock_embedder_deterministic():
    emb = MockEmbedder()
    v1 = await emb.embed_text(["same text"])
    v2 = await emb.embed_text(["same text"])
    assert v1 == v2


@pytest.mark.asyncio
async def test_mock_embedder_different_inputs_differ():
    emb = MockEmbedder()
    v1 = await emb.embed_text(["reentrancy"])
    v2 = await emb.embed_text(["oracle manipulation"])
    assert v1 != v2
