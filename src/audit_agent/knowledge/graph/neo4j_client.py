"""Async-friendly Neo4j client wrapper."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.neo4j")


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self._driver.close()

    async def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return [dict(record) async for record in result]

    async def run_write(self, query: str, **params: Any) -> None:
        async with self._driver.session() as session:
            await session.execute_write(lambda tx: tx.run(query, **params))

    async def health_check(self) -> bool:
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as exc:
            log.warning("Neo4j health check failed", error=str(exc))
            return False

    @classmethod
    def from_settings(cls) -> "Neo4jClient":
        from audit_agent.config import get_settings

        s = get_settings()
        return cls(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
