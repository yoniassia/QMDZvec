"""MemClawz v5 — Shared Memory Bus (REST API)."""
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from mem0 import Memory

from .config import MEM0_CONFIG, API_HOST, API_PORT

app = FastAPI(title="MemClawz v5 API", version="5.0.0")

# Initialize Mem0
mem = Memory.from_config(MEM0_CONFIG)


class AddMemoryRequest(BaseModel):
    content: str
    user_id: str = "yoni"
    agent_id: str = "main"
    memory_type: str = "fact"
    metadata: dict | None = None


class UpdateMemoryRequest(BaseModel):
    content: str


@app.get("/health")
async def health():
    return {"status": "ok", "version": "5.0.0"}


def _unwrap(result):
    """Unwrap Mem0 results — handles both list and {'results': [...]} formats."""
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    if isinstance(result, list):
        return result
    return []


@app.get("/api/v1/search")
async def search(
    q: str,
    user_id: str = "yoni",
    agent_id: str | None = None,
    memory_type: str | None = None,
    limit: int = Query(default=20, le=100),
):
    """Semantic search across all memories."""
    results = _unwrap(mem.search(q, user_id=user_id, limit=limit))

    # Post-filter by metadata
    if agent_id:
        results = [r for r in results if r.get("metadata", {}).get("agent") == agent_id]
    if memory_type:
        results = [r for r in results if r.get("metadata", {}).get("type") == memory_type]

    return {"results": results, "count": len(results)}


@app.post("/api/v1/add")
async def add_memory(req: AddMemoryRequest):
    """Add a memory from any agent."""
    meta = req.metadata or {}
    meta.update({
        "agent": req.agent_id,
        "type": req.memory_type,
        "source": "api",
    })
    result = mem.add(req.content, user_id=req.user_id, metadata=meta)
    return result


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


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
