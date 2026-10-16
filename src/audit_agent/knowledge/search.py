"""Knowledge base search interface.

Wraps Qdrant semantic search with severity / source / tag filters.
Returns list[dict] matching the acceptance criteria format.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

from audit_agent.knowledge.vectors.embedders import Embedder, MockEmbedder
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.search")

COLLECTION = "historical_findings"

Severity = Literal["critical", "high", "medium", "low", "info"]
Source = Literal["code4rena", "solodit", "sherlock", "immunefi", "rekt", "defihacklabs", "arxiv", "manual"]


class KnowledgeBase:
    """High-level search interface over the Qdrant historical_findings collection."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: Optional[Embedder] = None,
        collection: str = COLLECTION,
    ) -> None:
        self._qdrant = qdrant
        self._embedder: Embedder = embedder or MockEmbedder()
        self._collection = collection

    async def search(
        self,
        query: str,
        limit: int = 10,
        severity: Optional[Severity | list[Severity]] = None,
        source: Optional[Source | list[Source]] = None,
        vulnerability_class: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_quality: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic search over historical findings.

        Args:
            query: Natural language query, e.g. "price manipulation via TWAP oracle".
            limit: Maximum number of results.
            severity: Filter by severity level(s).
            source: Filter by data source(s).
            vulnerability_class: Filter by SWC number or category name (substring match).
            tags: Filter by custom tags (any match).
            min_quality: Minimum quality_score threshold.

        Returns:
            List of dicts with keys: title, source, url, severity,
            vulnerability_class, custom_tags, summary.
        """
        log.debug("knowledge.search", query=query[:80], limit=limit)

        vectors = await self._embedder.embed_text([query])
        query_vector = vectors[0]

        qdrant_filter = self._build_filter(severity, source, vulnerability_class, tags)

        hits = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit * 2,  # over-fetch to apply quality filter
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            if payload.get("quality_score", 0.0) < min_quality:
                continue
            results.append({
                "title": payload.get("title", ""),
                "source": payload.get("source", ""),
                "url": payload.get("url", ""),
                "severity": payload.get("severity", ""),
                "vulnerability_class": payload.get("vulnerability_class", ""),
                "custom_tags": payload.get("custom_tags", []),
                "summary": payload.get("summary", ""),
                "score": hit.score,
            })
            if len(results) >= limit:
                break

        log.info("knowledge.search_done", query=query[:60], results=len(results))
        return results

    def _build_filter(
        self,
        severity: Optional[Severity | list[Severity]],
        source: Optional[Source | list[Source]],
        vulnerability_class: Optional[str],
        tags: Optional[list[str]],
    ) -> Optional[Filter]:
        must: list[Any] = []

        if severity:
            sevs = [severity] if isinstance(severity, str) else severity
            must.append(FieldCondition(key="severity", match=MatchAny(any=sevs)))

        if source:
            srcs = [source] if isinstance(source, str) else source
            must.append(FieldCondition(key="source", match=MatchAny(any=srcs)))

        if vulnerability_class:
            must.append(
                FieldCondition(
                    key="vulnerability_class",
                    match=MatchValue(value=vulnerability_class),
                )
            )

        if tags:
            must.append(FieldCondition(key="custom_tags", match=MatchAny(any=tags)))

        return Filter(must=must) if must else None
