"""Embedding providers for knowledge base vectors.

Real embeddings use Voyage AI (voyage-3 / voyage-code-3).
MockEmbedder produces deterministic 1024-d vectors from text hashes — offline-safe for tests.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def embed_text(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_code(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbedder:
    """Deterministic embedder for offline tests. NOT for production."""

    DIM = 1024

    def _hash_embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # Expand 32 bytes → 1024 floats via trig spreading
        floats: list[float] = []
        for i in range(self.DIM):
            byte_val = h[i % 32]
            floats.append(math.sin(byte_val * (i + 1) * 0.01))
        # L2-normalise
        norm = math.sqrt(sum(x * x for x in floats)) or 1.0
        return [x / norm for x in floats]

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    async def embed_code(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed("code:" + t) for t in texts]


class VoyageEmbedder:
    """Production embedder backed by Voyage AI.

    Requires:
        pip install voyageai
        VOYAGE_API_KEY env var
    """

    TEXT_MODEL = "voyage-3"
    CODE_MODEL = "voyage-code-3"
    DIM = 1024

    def __init__(self, api_key: str) -> None:
        try:
            import voyageai  # type: ignore[import]
        except ImportError as e:
            raise ImportError("voyageai not installed: pip install voyageai") from e
        self._client = voyageai.AsyncClient(api_key=api_key)

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self.TEXT_MODEL)
        return result.embeddings  # type: ignore[no-any-return]

    async def embed_code(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self.CODE_MODEL)
        return result.embeddings  # type: ignore[no-any-return]


def get_embedder(voyage_api_key: str = "") -> Embedder:
    """Return real or mock embedder depending on whether the key is set."""
    if voyage_api_key:
        return VoyageEmbedder(voyage_api_key)
    return MockEmbedder()
