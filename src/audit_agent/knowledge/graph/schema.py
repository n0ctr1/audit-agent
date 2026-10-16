"""Neo4j schema and index setup."""

from __future__ import annotations

from audit_agent.knowledge.graph.neo4j_client import Neo4jClient
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.neo4j.schema")

_CONSTRAINTS = [
    "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT statevar_id IF NOT EXISTS FOR (s:StateVar) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT modifier_id IF NOT EXISTS FOR (m:Modifier) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
]

_INDEXES = [
    "CREATE INDEX contract_name_idx IF NOT EXISTS FOR (c:Contract) ON (c.name)",
    "CREATE INDEX function_signature_idx IF NOT EXISTS FOR (f:Function) ON (f.signature)",
    "CREATE INDEX function_surface_idx IF NOT EXISTS FOR (f:Function) ON (f.attack_surface_score)",
    "CREATE INDEX function_contract_idx IF NOT EXISTS FOR (f:Function) ON (f.contract_id)",
]


async def setup_schema(client: Neo4jClient) -> None:
    log.info("Neo4j: setting up schema constraints and indexes")
    for stmt in _CONSTRAINTS + _INDEXES:
        try:
            await client.run(stmt)
        except Exception as exc:
            log.warning("Neo4j: schema statement failed", stmt=stmt[:80], error=str(exc))
    log.info("Neo4j: schema setup complete")
