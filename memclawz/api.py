"""MemClawz v6 — Shared Memory Bus (REST API).

Extends v5 with:
  - Composite scoring for search
  - Graphiti temporal graph integration
  - Session/daily/weekly compaction endpoints
  - Multi-claw federation endpoints
  - Reflection endpoint
  - Enhanced health check
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from mem0 import Memory

from .config import MEM0_CONFIG, API_HOST, API_PORT, GRAPHITI_ENABLED, FEDERATION_ENABLED
from .scoring import score_results
from .federation import (
    NodeRegistration,
    FederationPushRequest,
    FederationPullRequest,
    FederationSyncRequest,
    process_push,
    process_pull,
    federation_status,
    registry,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init Graphiti on startup, close on shutdown."""
    if GRAPHITI_ENABLED:
        try:
            from .graphiti_layer import get_graphiti
            await get_graphiti()
            logger.info("Graphiti initialized on startup")
        except Exception as e:
            logger.warning(f"Graphiti init failed (non-fatal): {e}")
    yield
    if GRAPHITI_ENABLED:
        try:
            from .graphiti_layer import close_graphiti
            await close_graphiti()
        except Exception:
            pass


app = FastAPI(title="MemClawz v6 API", version="6.0.0", lifespan=lifespan)

# Initialize Mem0
mem = Memory.from_config(MEM0_CONFIG)


# --- Models ---

class AddMemoryRequest(BaseModel):
    content: str
    user_id: str = "yoni"
    agent_id: str = "main"
    memory_type: str = "fact"
    metadata: dict | None = None


class UpdateMemoryRequest(BaseModel):
    content: str


class CompactSessionRequest(BaseModel):
    session_id: str
    messages: list[dict] = []
    agent_id: str = "main"
    force: bool = False


class ReflectionRequest(BaseModel):
    hours: int = 24
    max_memories: int = 100


# --- Helpers ---

def _unwrap(result):
    """Unwrap Mem0 results — handles both list and {'results': [...]} formats."""
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    if isinstance(result, list):
        return result
    return []


# --- Health ---

@app.get("/health")
async def health():
    """Enhanced health check including Neo4j and federation status."""
    health_data = {
        "status": "ok",
        "version": "6.0.0",
        "qdrant": "ok",
        "neo4j": "disabled",
        "graphiti": "disabled",
        "federation": "disabled",
    }

    if GRAPHITI_ENABLED:
        try:
            from .graphiti_layer import health_check
            gh = await health_check()
            health_data["neo4j"] = "ok" if gh.get("neo4j_connected") else "error"
            health_data["graphiti"] = gh.get("status", "unknown")
            health_data["graph_nodes"] = gh.get("node_count", 0)
            health_data["graph_edges"] = gh.get("edge_count", 0)
        except Exception as e:
            health_data["neo4j"] = "error"
            health_data["graphiti"] = f"error: {e}"

    if FEDERATION_ENABLED:
        try:
            fed = federation_status()
            health_data["federation"] = "ok"
            health_data["federation_nodes"] = fed.get("node_count", 0)
        except Exception as e:
            health_data["federation"] = f"error: {e}"

    return health_data


# --- Core Memory Endpoints (v5 compatible + v6 enhancements) ---

@app.get("/api/v1/search")
async def search(
    q: str,
    user_id: str = "yoni",
    agent_id: str | None = None,
    memory_type: str | None = None,
    limit: int = Query(default=20, le=100),
    use_composite: bool = True,
):
    """Semantic search with composite scoring (v6).

    Set use_composite=false to get raw cosine similarity (v5 behavior).
    """
    results = _unwrap(mem.search(q, user_id=user_id, limit=limit))

    # Post-filter by metadata
    if agent_id:
        results = [r for r in results if r.get("metadata", {}).get("agent") == agent_id]
    if memory_type:
        results = [r for r in results if r.get("metadata", {}).get("type") == memory_type]

    # Apply composite scoring
    if use_composite:
        results = score_results(results)

    return {"results": results, "count": len(results), "scoring": "composite" if use_composite else "cosine"}


@app.post("/api/v1/add")
async def add_memory(req: AddMemoryRequest):
    """Add a memory — feeds to both Qdrant (via Mem0) AND Graphiti (v6)."""
    meta = req.metadata or {}
    meta.update({
        "agent": req.agent_id,
        "type": req.memory_type,
        "source": meta.get("source", "api"),
    })
    result = mem.add(req.content, user_id=req.user_id, metadata=meta)

    # Feed to Graphiti (async, non-blocking)
    graphiti_result = None
    if GRAPHITI_ENABLED:
        try:
            from .graphiti_layer import add_episode
            graphiti_result = await add_episode(
                content=req.content,
                agent_id=req.agent_id,
                source=meta.get("source", "api"),
            )
        except Exception as e:
            logger.warning(f"Graphiti add failed (non-fatal): {e}")
            graphiti_result = {"status": "error", "error": str(e)}

    return {
        "mem0": result,
        "graphiti": graphiti_result,
    }


@app.get("/api/v1/memories")
async def list_memories(
    user_id: str = "yoni",
    agent_id: str | None = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
):
    """List all memories, optionally filtered by agent."""
    all_mems = _unwrap(mem.get_all(user_id=user_id, limit=10000))
    if agent_id:
        all_mems = [m for m in all_mems if m.get("metadata", {}).get("agent") == agent_id]
    return {
        "memories": all_mems[offset : offset + limit],
        "total": len(all_mems),
    }


@app.get("/api/v1/agents")
async def list_agents():
    """List all agents with memory counts."""
    all_mems = _unwrap(mem.get_all(user_id="yoni", limit=10000))
    agents: dict = {}
    for m in all_mems:
        agent = m.get("metadata", {}).get("agent", "unknown")
        if agent not in agents:
            agents[agent] = {"count": 0, "types": {}}
        agents[agent]["count"] += 1
        mtype = m.get("metadata", {}).get("type", "unknown")
        agents[agent]["types"][mtype] = agents[agent]["types"].get(mtype, 0) + 1
    return agents


@app.get("/api/v1/stats")
async def stats():
    """System statistics."""
    all_mems = _unwrap(mem.get_all(user_id="yoni", limit=10000))
    types: dict = {}
    agents: dict = {}
    sources: dict = {}
    for m in all_mems:
        meta = m.get("metadata", {})
        t = meta.get("type", "unknown")
        a = meta.get("agent", "unknown")
        s = meta.get("source", "unknown")
        types[t] = types.get(t, 0) + 1
        agents[a] = agents.get(a, 0) + 1
        sources[s] = sources.get(s, 0) + 1
    return {
        "total_memories": len(all_mems),
        "by_type": types,
        "by_agent": agents,
        "by_source": sources,
    }


@app.delete("/api/v1/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    try:
        mem.delete(memory_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "deleted"}


@app.put("/api/v1/memory/{memory_id}")
async def update_memory(memory_id: str, req: UpdateMemoryRequest):
    """Update a memory."""
    try:
        mem.update(memory_id, req.content)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "updated"}


# --- Graphiti Endpoints (v6) ---

@app.get("/api/v1/graph/search")
async def graph_search(
    q: str,
    num_results: int = Query(default=10, le=50),
    group_id: str = "yoniclaw",
):
    """Search the Graphiti temporal knowledge graph."""
    if not GRAPHITI_ENABLED:
        raise HTTPException(status_code=503, detail="Graphiti not enabled")
    from .graphiti_layer import search
    results = await search(q, num_results=num_results, group_ids=[group_id])
    return {"results": results, "count": len(results)}


@app.get("/api/v1/graph/entity/{name}")
async def graph_entity(name: str, group_id: str = "yoniclaw"):
    """Look up an entity and its relationships in the knowledge graph."""
    if not GRAPHITI_ENABLED:
        raise HTTPException(status_code=503, detail="Graphiti not enabled")
    from .graphiti_layer import get_entity
    entity = await get_entity(name, group_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return entity


# --- Compaction Endpoints (v6) ---

@app.post("/api/v1/compact/session")
async def compact_session(req: CompactSessionRequest):
    """Trigger session compaction."""
    from .compactor import SessionCompactor
    compactor = SessionCompactor(mem)
    result = compactor.compact_session(
        session_id=req.session_id,
        messages=req.messages,
        agent_id=req.agent_id,
        force=req.force,
    )
    return result


@app.post("/api/v1/compact/daily")
async def compact_daily(date: str | None = None):
    """Trigger daily digest generation."""
    from .compactor import DailyDigest
    digest = DailyDigest(mem)
    result = digest.generate(date)
    return result


@app.post("/api/v1/compact/weekly")
async def compact_weekly():
    """Trigger weekly merge."""
    from .compactor import WeeklyMerge
    merger = WeeklyMerge(mem)
    result = merger.merge()
    return result


@app.get("/api/v1/compact/status")
async def compact_status():
    """Get compaction health and status."""
    from .compactor import get_compaction_status
    return get_compaction_status()


# --- Reflection Endpoint (v6) ---

@app.post("/api/v1/reflect")
async def reflect(req: ReflectionRequest | None = None):
    """Trigger sleep-time reflection analysis."""
    from .reflection import ReflectionEngine
    engine = ReflectionEngine(mem)
    hours = req.hours if req else 24
    max_mem = req.max_memories if req else 100
    result = engine.reflect(hours=hours, max_memories=max_mem)
    return result


# --- Federation Endpoints (v6) ---

@app.post("/api/v1/federation/register")
async def fed_register(req: NodeRegistration):
    """Register a remote MemClawz node."""
    if not FEDERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Federation not enabled")
    return registry.register(req)


@app.post("/api/v1/federation/push")
async def fed_push(req: FederationPushRequest):
    """Receive memories from a remote node."""
    if not FEDERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Federation not enabled")
    return process_push(mem, req)


@app.post("/api/v1/federation/pull")
async def fed_pull(req: FederationPullRequest):
    """Send memories to a remote node."""
    if not FEDERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Federation not enabled")
    return process_pull(mem, req)


@app.post("/api/v1/federation/sync")
async def fed_sync(req: FederationSyncRequest):
    """Bidirectional sync with a remote node."""
    if not FEDERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Federation not enabled")

    # Push their memories to us
    push_req = FederationPushRequest(
        node_id=req.node_id,
        node_key=req.node_key,
        memories=req.memories,
    )
    push_result = process_push(mem, push_req)

    # Pull our memories for them
    pull_req = FederationPullRequest(
        node_id=req.node_id,
        node_key=req.node_key,
        since=req.since,
        limit=req.limit,
    )
    pull_result = process_pull(mem, pull_req)

    return {
        "status": "synced",
        "push": push_result,
        "pull": pull_result,
    }


@app.get("/api/v1/federation/status")
async def fed_status():
    """Get federation health status."""
    if not FEDERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Federation not enabled")
    return federation_status()


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
