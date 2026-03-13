"""MemClawz v6 — Graphiti temporal knowledge graph integration.

Provides entity-relationship tracking, temporal fact management,
and contradiction detection via Neo4j + Graphiti.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

# Singleton instance
_graphiti: Graphiti | None = None
_init_lock = asyncio.Lock()


async def get_graphiti() -> Graphiti:
    """Get or create the Graphiti singleton instance."""
    global _graphiti
    if _graphiti is not None:
        return _graphiti

    async with _init_lock:
        if _graphiti is not None:
            return _graphiti
        _graphiti = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
        try:
            await _graphiti.build_indices_and_constraints()
            logger.info("Graphiti initialized and indices built")
        except Exception as e:
            logger.warning(f"Graphiti index build warning (may already exist): {e}")
        return _graphiti


async def close_graphiti():
    """Close the Graphiti connection."""
    global _graphiti
    if _graphiti is not None:
        await _graphiti.close()
        _graphiti = None


async def add_episode(
    content: str,
    agent_id: str = "main",
    source: str = "api",
    timestamp: datetime | None = None,
    group_id: str = "yoniclaw",
) -> dict[str, Any]:
    """Add a conversation episode to the temporal knowledge graph.

    Graphiti extracts entities and relationships automatically from the content,
    builds temporal edges, and handles contradiction detection.

    Args:
        content: The text content (conversation, fact, observation).
        agent_id: Which agent produced this content.
        source: Source identifier (api, lcm, compaction, etc.).
        timestamp: When this episode occurred (defaults to now).
        group_id: Group/namespace for the episode.

    Returns:
        Dict with episode details and extracted entities/edges.
    """
    g = await get_graphiti()
    ts = timestamp or datetime.now(timezone.utc)

    try:
        result = await g.add_episode(
            name=f"{agent_id}:{source}:{ts.isoformat()[:19]}",
            episode_body=content,
            source=EpisodeType.message,
            source_description=f"Agent {agent_id} via {source}",
            reference_time=ts,
            group_id=group_id,
        )
        logger.info(f"Graphiti episode added: agent={agent_id}, source={source}")
        return {
            "status": "ok",
            "episode_name": f"{agent_id}:{source}:{ts.isoformat()[:19]}",
            "result": str(result) if result else "added",
        }
    except Exception as e:
        logger.error(f"Graphiti add_episode error: {e}")
        return {"status": "error", "error": str(e)}


async def search(
    query: str,
    num_results: int = 10,
    group_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search the temporal knowledge graph.

    Uses Graphiti's hybrid search (semantic + BM25 + temporal).

    Args:
        query: Search query string.
        num_results: Max results to return.
        group_ids: Optional group/namespace filter.

    Returns:
        List of edge dicts with source, target, fact, timestamps.
    """
    g = await get_graphiti()

    try:
        edges = await g.search(
            query=query,
            num_results=num_results,
            group_ids=group_ids or ["yoniclaw"],
        )
        results = []
        for edge in edges:
            results.append({
                "uuid": str(edge.uuid) if hasattr(edge, "uuid") else None,
                "fact": edge.fact if hasattr(edge, "fact") else str(edge),
                "source_node": edge.source_node_uuid if hasattr(edge, "source_node_uuid") else None,
                "target_node": edge.target_node_uuid if hasattr(edge, "target_node_uuid") else None,
                "created_at": str(edge.created_at) if hasattr(edge, "created_at") else None,
                "valid_at": str(edge.valid_at) if hasattr(edge, "valid_at") else None,
                "invalid_at": str(edge.invalid_at) if hasattr(edge, "invalid_at") else None,
                "expired": bool(edge.invalid_at) if hasattr(edge, "invalid_at") else False,
            })
        return results
    except Exception as e:
        logger.error(f"Graphiti search error: {e}")
        return []


async def get_entity(name: str, group_id: str = "yoniclaw") -> dict[str, Any] | None:
    """Look up an entity node and its relationships by name.

    Uses a direct Cypher query against Neo4j since Graphiti doesn't
    have a direct entity lookup method.
    """
    g = await get_graphiti()

    try:
        driver = g.driver
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($name)
        OPTIONAL MATCH (n)-[r]-(m:Entity)
        RETURN n, collect({rel: type(r), target: m.name, props: properties(r)}) as relationships
        LIMIT 5
        """
        async with driver.session() as session:
            result = await session.run(query, {"name": name})
            records = [record async for record in result]

        entities = []
        for record in records:
            node = record["n"]
            rels = record["relationships"]
            entities.append({
                "name": node.get("name"),
                "uuid": node.get("uuid"),
                "group_id": node.get("group_id"),
                "created_at": str(node.get("created_at", "")),
                "summary": node.get("summary", ""),
                "relationships": [r for r in rels if r.get("target")],
            })
        return entities[0] if entities else None
    except Exception as e:
        logger.error(f"Graphiti entity lookup error: {e}")
        return None


async def health_check() -> dict[str, Any]:
    """Check Graphiti/Neo4j health."""
    try:
        g = await get_graphiti()
        driver = g.driver
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            record = await result.single()
            node_count_result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            node_count = await node_count_result.single()
            edge_count_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            edge_count = await edge_count_result.single()

        return {
            "status": "ok",
            "neo4j_connected": True,
            "node_count": node_count["cnt"] if node_count else 0,
            "edge_count": edge_count["cnt"] if edge_count else 0,
        }
    except Exception as e:
        return {
            "status": "error",
            "neo4j_connected": False,
            "error": str(e),
        }
