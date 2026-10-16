"""Ingestion pipeline: structured Finding → embed → upsert Qdrant.

Deduplicates by content fingerprint before upserting.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from audit_agent.knowledge.ingestion.validator import Validator
from audit_agent.knowledge.models import Finding
from audit_agent.knowledge.vectors.embedders import Embedder, MockEmbedder
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.pipeline")

COLLECTION = "historical_findings"
BATCH_SIZE = 32


@dataclass
class PipelineStats:
    ingested: int = 0
    rejected_quality: int = 0
    rejected_duplicate: int = 0
    errors: int = 0
    error_details: list[str] = field(default_factory=list)


def _build_embedding_text(finding: Finding) -> str:
    """Combine title + summary + root_cause + tags into a single embeddable string."""
    parts = [
        finding.title,
        finding.description_structured.summary,
        finding.description_structured.root_cause,
        finding.description_structured.attack_vector,
        " ".join(finding.custom_tags),
        finding.vulnerability_class,
    ]
    return " ".join(p for p in parts if p).strip()


def _build_code_embedding_text(finding: Finding) -> str:
    """Combine code snippets for code embedding."""
    snippets = " ".join(s.code for s in finding.code_snippets if s.code)
    return (snippets + " " + finding.poc).strip() or finding.title


class IngestionPipeline:
    """End-to-end pipeline: validated Finding → Qdrant."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: Optional[Embedder] = None,
        collection: str = COLLECTION,
    ) -> None:
        self._qdrant = qdrant
        self._embedder: Embedder = embedder or MockEmbedder()
        self._collection = collection
        self._validator = Validator()
        self._seen_fingerprints: set[str] = set()  # in-memory dedup within a run

    async def ingest_batch(self, findings: list[Finding]) -> PipelineStats:
        """Ingest a list of pre-structured Finding objects."""
        stats = PipelineStats()

        valid: list[Finding] = []
        for f in findings:
            result = self._validator.validate(f)
            if not result.valid:
                stats.rejected_quality += 1
                log.debug("pipeline.rejected", title=f.title, reason=result.rejection_reason)
                continue

            fp = self._validator.fingerprint(f)
            if fp in self._seen_fingerprints:
                stats.rejected_duplicate += 1
                log.debug("pipeline.dedup_in_memory", title=f.title)
                continue
            self._seen_fingerprints.add(fp)

            f.quality_score = result.quality_score
            valid.append(f)

        if not valid:
            return stats

        # Check Qdrant for existing fingerprints (cross-run dedup)
        valid = await self._filter_existing(valid, stats)
        if not valid:
            return stats

        # Embed in batches
        for i in range(0, len(valid), BATCH_SIZE):
            batch = valid[i : i + BATCH_SIZE]
            try:
                await self._embed_and_upsert(batch, stats)
            except Exception as exc:
                for f in batch:
                    stats.errors += 1
                    stats.error_details.append(f"{f.title}: {exc}")
                log.error("pipeline.batch_error", error=str(exc))

        return stats

    async def _filter_existing(
        self, findings: list[Finding], stats: PipelineStats
    ) -> list[Finding]:
        """Remove findings already in Qdrant by checking payload fingerprint."""
        try:
            results: list[Finding] = []
            for f in findings:
                fp = self._validator.fingerprint(f)
                existing, _ = await self._qdrant.scroll(
                    collection_name=self._collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="fingerprint", match=MatchValue(value=fp))]
                    ),
                    limit=1,
                    with_payload=False,
                )
                if existing:
                    stats.rejected_duplicate += 1
                    log.debug("pipeline.dedup_qdrant", title=f.title, fingerprint=fp)
                else:
                    results.append(f)
            return results
        except Exception as exc:
            log.warning("pipeline.dedup_check_failed", error=str(exc))
            return findings  # proceed without dedup on error

    async def _embed_and_upsert(
        self, findings: list[Finding], stats: PipelineStats
    ) -> None:
        text_inputs = [_build_embedding_text(f) for f in findings]
        code_inputs = [_build_code_embedding_text(f) for f in findings]

        text_vecs, code_vecs = await asyncio.gather(
            self._embedder.embed_text(text_inputs),
            self._embedder.embed_code(code_inputs),
        )

        # Combine text + code embeddings by averaging
        combined: list[list[float]] = []
        for tv, cv in zip(text_vecs, code_vecs):
            avg = [(t + c) / 2.0 for t, c in zip(tv, cv)]
            combined.append(avg)

        points: list[PointStruct] = []
        for f, vec, text_input in zip(findings, combined, text_inputs):
            f.embedding_text = text_input
            fp = self._validator.fingerprint(f)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "finding_id": f.id,
                        "title": f.title,
                        "severity": f.severity,
                        "source": f.source,
                        "url": f.url,
                        "vulnerability_class": f.vulnerability_class,
                        "custom_tags": f.custom_tags,
                        "protocol_name": f.protocol.name,
                        "protocol_category": f.protocol.category,
                        "summary": f.description_structured.summary,
                        "root_cause": f.description_structured.root_cause,
                        "quality_score": f.quality_score,
                        "fingerprint": fp,
                        "has_poc": bool(f.poc),
                    },
                )
            )

        await self._qdrant.upsert(collection_name=self._collection, points=points)
        stats.ingested += len(points)
        log.info("pipeline.upserted", count=len(points))
