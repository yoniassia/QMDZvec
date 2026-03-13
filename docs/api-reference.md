# MemClawz v5 API Reference

Base URL: `http://localhost:3500`

## Health Check

```
GET /health
→ {"status": "ok", "version": "5.0.0"}
```

## Search Memories

```
GET /api/v1/search?q=<query>&user_id=yoni&agent_id=<agent>&memory_type=<type>&limit=20
→ {"results": [...], "count": N}
```

## Add Memory

```
POST /api/v1/add
Body: {"content": "...", "user_id": "yoni", "agent_id": "main", "memory_type": "fact", "metadata": {}}
→ Mem0 add result
```

## List Memories

```
GET /api/v1/memories?user_id=yoni&agent_id=<agent>&limit=50&offset=0
→ {"memories": [...], "total": N}
```

## List Agents

```
GET /api/v1/agents
→ {"agent_name": {"count": N, "types": {"fact": N, ...}}, ...}
```

## Stats

```
GET /api/v1/stats
→ {"total_memories": N, "by_type": {...}, "by_agent": {...}, "by_source": {...}}
```

## Delete Memory

```
DELETE /api/v1/memory/<memory_id>
→ {"status": "deleted"}
```

## Update Memory

```
PUT /api/v1/memory/<memory_id>
Body: {"content": "new content"}
→ {"status": "updated"}
```
