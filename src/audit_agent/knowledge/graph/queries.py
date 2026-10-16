"""Reusable Cypher query templates for the code graph.

All queries are parameterized — no string interpolation of user data.
"""

from __future__ import annotations

from typing import Any

from audit_agent.knowledge.graph.neo4j_client import Neo4jClient


# ── Read queries ──────────────────────────────────────────────────────────────

HIGHEST_SURFACE_FUNCTIONS = """
MATCH (f:Function)-[:DEFINED_IN]->(c:Contract)
WHERE c.is_in_scope = true
RETURN f.id AS id, f.name AS name, f.signature AS sig,
       f.attack_surface_score AS score, c.name AS contract
ORDER BY f.attack_surface_score DESC
LIMIT $limit
"""

FUNCTION_SUBGRAPH = """
MATCH path = (e:Function {id: $entry_id})-[:CALLS*1..$depth]->(f:Function)
RETURN DISTINCT f
"""

FUNCTIONS_WITH_EXTERNAL_CALLS = """
MATCH (f:Function)-[:CALLS_EXTERNAL]->(ec:ExternalCall)
WHERE ec.target_type = 'user_controlled'
RETURN f.id AS function_id, f.name AS name, ec.call_type AS call_type,
       ec.line AS line
ORDER BY f.attack_surface_score DESC
"""

STATE_INTERACTIONS_FOR_FUNCTIONS = """
MATCH (f:Function)-[r:READS|WRITES]->(v:StateVar)
WHERE f.id IN $function_ids
RETURN f.id AS function_id, type(r) AS rel_type, v.name AS var_name, v.type AS var_type
"""

FUNCTIONS_WITHOUT_REENTRANCY_GUARD = """
MATCH (f:Function)-[:CALLS_EXTERNAL]->(ec:ExternalCall)
WHERE NOT exists {
    MATCH (f)-[:USES_MODIFIER]->(m:Modifier)
    WHERE m.name CONTAINS 'nonReentrant' OR m.name CONTAINS 'reentrancy'
}
AND f.visibility IN ['external', 'public']
RETURN f.id AS id, f.name AS name, f.attack_surface_score AS score,
       collect(ec.call_type) AS call_types
ORDER BY f.attack_surface_score DESC
"""

CALL_PATHS_TO_STATE_VAR = """
MATCH path = (entry:Function)-[:CALLS*0..$depth]->(writer:Function)-[:WRITES]->(v:StateVar {id: $var_id})
WHERE entry.visibility IN ['external', 'public']
RETURN entry.id AS entry_id, entry.name AS entry_name,
       [node IN nodes(path) | node.name] AS path_names,
       v.name AS var_name
ORDER BY length(path)
LIMIT $limit
"""

CONTRACT_INHERITANCE_TREE = """
MATCH path = (c:Contract {id: $contract_id})-[:INHERITS_FROM*0..5]->(parent:Contract)
RETURN [node IN nodes(path) | node.name] AS chain
"""

PRIVILEGED_FUNCTIONS = """
MATCH (f:Function)-[:DEFINED_IN]->(c:Contract)
WHERE f.is_privileged = true AND c.is_in_scope = true
RETURN f.id, f.name, f.signature, c.name AS contract
"""

GRAPH_STATS = """
MATCH (c:Contract) WITH count(c) AS contracts
MATCH (f:Function) WITH contracts, count(f) AS functions
MATCH ()-[r]->() WITH contracts, functions, count(r) AS relationships
RETURN contracts, functions, relationships
"""


# ── Helper wrappers ───────────────────────────────────────────────────────────

async def get_highest_surface_functions(
    client: Neo4jClient, limit: int = 20
) -> list[dict[str, Any]]:
    return await client.run(HIGHEST_SURFACE_FUNCTIONS, limit=limit)


async def get_function_subgraph(
    client: Neo4jClient, entry_id: str, depth: int = 8
) -> list[dict[str, Any]]:
    return await client.run(FUNCTION_SUBGRAPH, entry_id=entry_id, depth=depth)


async def get_reentrancy_candidates(client: Neo4jClient) -> list[dict[str, Any]]:
    return await client.run(FUNCTIONS_WITHOUT_REENTRANCY_GUARD)


async def get_graph_stats(client: Neo4jClient) -> dict[str, int]:
    rows = await client.run(GRAPH_STATS)
    return rows[0] if rows else {"contracts": 0, "functions": 0, "relationships": 0}
