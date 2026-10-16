"""Qdrant collection setup."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.qdrant")

_COLLECTIONS = {
    "vulnerability_patterns": VectorParams(size=1024, distance=Distance.COSINE),
    "historical_findings": VectorParams(size=1024, distance=Distance.COSINE),
    "project_invariants": VectorParams(size=1024, distance=Distance.COSINE),
    "fork_fingerprints": VectorParams(size=1024, distance=Distance.COSINE),
}


async def setup_collections(client: AsyncQdrantClient) -> None:
    log.info("Qdrant: setting up collections")
    response = await client.get_collections()
    existing = {c.name for c in response.collections}

    for name, params in _COLLECTIONS.items():
        if name not in existing:
            await client.create_collection(collection_name=name, vectors_config=params)
            log.info("Qdrant: created collection", name=name)
        else:
            log.debug("Qdrant: collection already exists", name=name)

    log.info("Qdrant: collection setup complete")
