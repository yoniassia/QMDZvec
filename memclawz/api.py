"""MemClawz v7 — Shared Memory Bus (REST API).

Extends v6 with:
  - Auto-enrichment layer (Phase 1)
  - Temporal validity tracking (Phase 2)  
  - RDF-lite triples extraction (Phase 3)
  - Status lifecycle: 8-state memory lifecycle (Phase 4)
  - Hybrid search: BM25 + vector + lifecycle scoring (Phase 5)
  - Expanded memory types: 12 types covering full action cycle (Phase 6)
  - Google Gemini integration with OpenAI fallback
"""
import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from mem0 import Memory

from .config import MEM0_CONFIG, API_HOST, API_PORT, GRAPHITI_ENABLED, FEDERATION_ENABLED
from .scoring import score_results
from .router import MemClawzRouter
from .v7_extensions import (
    apply_lifecycle_filter,
    apply_hybrid_scoring,
    enhance_memory_with_lifecycle,
    process_search_v7,
    get_v7_stats,
    transition_memory_status,
    bulk_update_outdated,
)
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


app = FastAPI(title="MemClawz v7 API", version="7.0.0", lifespan=lifespan)

# Initialize Mem0
mem = Memory.from_config(MEM0_CONFIG)
router_engine = MemClawzRouter(mem)


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


def _memory_agent(record: dict) -> str:
    """Agent from nested metadata or top-level agent_id/agent (direct-insert compatibility)."""
    meta = record.get("metadata") or {}
    return meta.get("agent") or record.get("agent_id") or record.get("agent") or "unknown"


def _memory_type(record: dict) -> str:
    """Type from nested metadata or top-level memory_type/type (direct-insert compatibility)."""
    meta = record.get("metadata") or {}
    return meta.get("type") or record.get("memory_type") or record.get("type") or "unknown"


def _memory_source(record: dict) -> str:
    """Source from nested metadata or top-level source (direct-insert compatibility)."""
    meta = record.get("metadata") or {}
    return meta.get("source") or record.get("source") or "unknown"


def _memory_text(record: dict) -> str:
    """Memory text from top-level memory or data (legacy)."""
    return record.get("memory") or record.get("data") or ""


def _direct_qdrant_upsert(
    content: str,
    user_id: str,
    agent_id: str,
    memory_type: str,
    metadata: dict | None = None,
) -> dict:
    """Write one memory directly to Qdrant with OpenAI embedding and v7 enrichment.
    Returns {"status": "ok", "id": point_id, "method": "direct_qdrant", "enrichment": enrichment_data}.
    Shared by /api/v1/add (primary path) and /api/v1/add-direct."""
    import uuid
    from openai import OpenAI
    from qdrant_client.models import PointStruct
    from qdrant_client import QdrantClient
    from .config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
    from .enrichment import enrich_memory, compute_validity_end

    # Phase 1: Auto-enrichment (with graceful fallback)
    try:
        enrichment = enrich_memory(content)
        logger.info(f"Memory enriched: type={enrichment['type']}, weight={enrichment['weight']}")
    except Exception as e:
        logger.warning(f"Enrichment failed, using defaults: {e}")
        enrichment = {
            "type": memory_type,  # Keep user-specified type
            "weight": 0.5,
            "title": content[:50],
            "summary": content[:200], 
            "tags": [],
            "validity_hours": None,
            "triples": []
        }

    # Keep user-specified type as override (don't overwrite explicit agent choices)
    enriched_type = memory_type if memory_type != "fact" else enrichment["type"]
    
    # Phase 2: Temporal validity
    ts_valid_start = datetime.now(timezone.utc).isoformat()
    ts_valid_end = compute_validity_end(enrichment["validity_hours"])

    oai = OpenAI()
    emb_resp = oai.embeddings.create(input=content, model="text-embedding-3-small")
    vector = emb_resp.data[0].embedding
    point_id = str(uuid.uuid4())
    content_hash = hashlib.md5(content.encode()).hexdigest()
    meta = metadata or {}
    source = meta.get("source", "api")

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    payload = {
        "memory": content,
        "agent_id": agent_id,
        "memory_type": enriched_type,
        "hash": content_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "source": source,
        # v7: Enhanced metadata with enrichment + temporal validity + triples
        "metadata": {
            "agent": agent_id, 
            "type": enriched_type, 
            "source": source,
            # Enrichment data
            "importance": enrichment["weight"],
            "title": enrichment["title"],
            "summary": enrichment["summary"], 
            "tags": enrichment["tags"],
            # Temporal validity
            "ts_valid_start": ts_valid_start,
            "ts_valid_end": ts_valid_end,
            # Phase 3: RDF triples
            "triples": enrichment["triples"]
        },
        # Top-level for compatibility
        "importance": enrichment["weight"],
        "ts_valid_start": ts_valid_start,
        "ts_valid_end": ts_valid_end,
        "triples": enrichment["triples"],
        # Phase 4: Lifecycle status
        "status": "active",
        "status_updated_at": datetime.now(timezone.utc).isoformat(),
    }
    qc.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
    return {
        "status": "ok", 
        "id": point_id, 
        "method": "direct_qdrant",
        "enrichment": enrichment,
        "validity": {"start": ts_valid_start, "end": ts_valid_end}
    }


async def _run_mem0_background(content: str, user_id: str, meta: dict, timeout_seconds: float = 90.0):
    """Run mem.add() in a thread; log warning on failure or timeout. Does not block."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(mem.add, content, user_id=user_id, metadata=meta),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Mem0 extraction timed out (background); direct insert already succeeded.")
    except Exception as e:
        logger.warning("Mem0 extraction failed (background): %s", e)


# --- Health ---

@app.get("/health")
async def health():
    """Enhanced health check including Neo4j and federation status."""
    health_data = {
        "status": "ok",
        "version": "7.0.0",
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
    include_expired: bool = False,
    use_hybrid: bool = True,
    include_deleted: bool = False,
):
    """Semantic search with composite scoring, hybrid BM25, lifecycle filtering, and temporal validity (v7).

    Set use_composite=false to get raw cosine similarity (v5 behavior).
    Set include_expired=true to include expired memories.
    Set use_hybrid=true (default) for BM25 + vector hybrid scoring.
    Set include_deleted=true to include deleted/superseded memories.
    """
    from .enrichment import is_memory_expired
    
    results = _unwrap(mem.search(q, user_id=user_id, limit=limit))

    # Post-filter by metadata or top-level (direct-insert compatibility)
    if agent_id:
        results = [r for r in results if _memory_agent(r) == agent_id]
    if memory_type:
        results = [r for r in results if _memory_type(r) == memory_type]

    # Phase 2: Filter expired memories unless explicitly requested
    if not include_expired:
        filtered_results = []
        for r in results:
            meta = r.get("metadata", {})
            ts_valid_end = meta.get("ts_valid_end") or r.get("ts_valid_end")
            if not is_memory_expired(ts_valid_end):
                filtered_results.append(r)
            else:
                r["expired"] = True
        results = filtered_results

    # Phase 4: Lifecycle filtering (exclude deleted/superseded by default)
    results = apply_lifecycle_filter(results, include_deleted=include_deleted)

    # Apply composite scoring
    if use_composite:
        results = score_results(results)

    # Phase 5: Hybrid BM25 + vector scoring
    if use_hybrid:
        results = apply_hybrid_scoring(query=q, results=results, top_k=limit)

    # Add validity info to each result
    for r in results:
        meta = r.get("metadata", {})
        r["validity"] = {
            "start": meta.get("ts_valid_start") or r.get("ts_valid_start"),
            "end": meta.get("ts_valid_end") or r.get("ts_valid_end"),
            "expired": is_memory_expired(meta.get("ts_valid_end") or r.get("ts_valid_end"))
        }

    return {
        "results": results, 
        "count": len(results), 
        "scoring": "hybrid" if use_hybrid else ("composite" if use_composite else "cosine"),
        "included_expired": include_expired,
        "included_deleted": include_deleted,
    }


@app.post("/api/v1/add")
async def add_memory(req: AddMemoryRequest):
    """Add a memory — direct Qdrant write (primary, fast), then Mem0 extraction and Graphiti in background.
    Returns quickly; callers see success via direct_insert. Mem0/Graphiti failures only logged."""
    meta = req.metadata or {}
    meta.update({
        "agent": req.agent_id,
        "type": req.memory_type,
        "source": meta.get("source", "api"),
    })

    # Primary path: direct Qdrant write (fast, blocks only for embed + upsert)
    try:
        direct_insert = _direct_qdrant_upsert(
            content=req.content,
            user_id=req.user_id,
            agent_id=req.agent_id,
            memory_type=req.memory_type,
            metadata=meta,
        )
    except Exception as e:
        logger.warning("Direct Qdrant insert failed: %s", e)
        direct_insert = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=503, detail=f"Direct insert failed: {e}")

    # Mem0 extraction: background only; do not block the request
    asyncio.create_task(_run_mem0_background(req.content, req.user_id, meta))

    # Graphiti: non-blocking (fire-and-forget) with v7 triples support
    if GRAPHITI_ENABLED:
        async def _graphiti_background():
            try:
                from .graphiti_layer import add_episode
                # Extract triples from direct_insert enrichment result
                triples = direct_insert.get("enrichment", {}).get("triples", [])
                await add_episode(
                    content=req.content,
                    agent_id=req.agent_id,
                    source=meta.get("source", "api"),
                    triples=triples,
                )
            except Exception as e:
                logger.warning("Graphiti add failed (non-fatal): %s", e)
        asyncio.create_task(_graphiti_background())

    return {
        "direct_insert": direct_insert,
        "mem0": {"status": "background"},
        "graphiti": {"status": "background"} if GRAPHITI_ENABLED else None,
    }


@app.post("/api/v1/add-direct")
async def add_memory_direct(req: AddMemoryRequest):
    """Fast-path: skip Mem0 LLM extraction, write directly to Qdrant + Graphiti.
    Use for bootstrap/bulk writes where content is already clean facts."""
    meta = req.metadata or {}
    meta.setdefault("source", "api")
    direct_insert = _direct_qdrant_upsert(
        content=req.content,
        user_id=req.user_id,
        agent_id=req.agent_id,
        memory_type=req.memory_type,
        metadata=meta,
    )

    # Feed to Graphiti (async, non-blocking)
    graphiti_result = None
    if GRAPHITI_ENABLED:
        try:
            from .graphiti_layer import add_episode
            graphiti_result = await add_episode(
                content=req.content, agent_id=req.agent_id, source=meta.get("source", "api"),
            )
        except Exception as e:
            graphiti_result = {"status": "error", "error": str(e)}

    return {"status": "ok", "id": direct_insert["id"], "method": "direct_qdrant", "graphiti": graphiti_result}


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
        all_mems = [m for m in all_mems if _memory_agent(m) == agent_id]
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
        agent = _memory_agent(m)
        if agent not in agents:
            agents[agent] = {"count": 0, "types": {}}
        agents[agent]["count"] += 1
        mtype = _memory_type(m)
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
        t = _memory_type(m)
        a = _memory_agent(m)
        s = _memory_source(m)
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


@app.get("/api/v1/route")
async def route_task(task: str, include_context: bool = True):
    """Route a task to the right agent."""
    return router_engine.route(task, include_context=include_context)


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
@app.post("/api/v1/reflection")
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


# --- v7 Lifecycle & Stats Endpoints ---

@app.get("/api/v1/v7/stats")
async def v7_stats():
    """Get v7 feature statistics — lifecycle, hybrid search, expanded types."""
    return get_v7_stats()


@app.post("/api/v1/memory/{memory_id}/transition")
async def memory_transition(memory_id: str, from_status: str, to_status: str):
    """Transition a memory's lifecycle status.
    
    Valid states: active, confirmed, outdated, archived, contradicted, merged, superseded, deleted
    """
    result = transition_memory_status(memory_id, from_status, to_status)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/v1/lifecycle/bulk-outdated")
async def lifecycle_bulk_outdated(threshold_days: int = Query(default=30, ge=1)):
    """Find and mark memories older than threshold_days as outdated."""
    return bulk_update_outdated(threshold_days)


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
