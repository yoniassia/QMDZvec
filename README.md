# MemClawz v6 🧠

AI Agent Fleet Memory System — composite scoring, compaction engine, Graphiti temporal knowledge graph, multi-claw federation, sleep-time reflection, MCP server.

## What's New in v6

- **Composite Scoring** — Weighted blend of semantic similarity + recency decay + importance + access frequency (replaces raw cosine)
- **Compaction Engine** — Three-tier: session compactor, daily digest, weekly merge with deduplication
- **Graphiti Integration** — Neo4j temporal knowledge graph for entity relationships, contradiction detection, temporal fact management
- **Multi-Claw Federation** — HTTP push/pull protocol for sharing memories across fleet (YoniClaw master + remote nodes)
- **Sleep-Time Reflection** — LLM-driven pattern detection, insight generation, and MEMORY.md update proposals
- **Enhanced MCP Server** — New tools: compact_session, reflect, memory_stats

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MemClawz v6 Master (YoniClaw)       │
│                                                   │
│  FastAPI :3500  │  Qdrant :6333  │  Neo4j :7687  │
│                                                   │
│  Mem0 + Composite Scoring + Graphiti              │
│  Compaction: Session │ Daily │ Weekly             │
│  Reflection Engine                                │
│  Federation: register / push / pull / sync        │
└─────────────────────────────────────────────────┘
         ▲              ▲              ▲
    Clawdet         MoneyClaw      WhiteRabbit
```

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys and Neo4j settings

# Start services
systemctl --user start neo4j
systemctl --user start memclawz-api
systemctl --user start memclawz-watcher
systemctl --user start memclawz-cron
```

## API Endpoints

### Core (v5 compatible)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (Qdrant, Neo4j, Federation) |
| GET | `/api/v1/search?q=...` | Semantic search with composite scoring |
| POST | `/api/v1/add` | Add memory (direct Qdrant + background Mem0 + Graphiti) |
| POST | `/api/v1/add-direct` | Fast-path: skip Mem0 extraction, direct Qdrant + Graphiti |
| GET | `/api/v1/memories` | List memories |
| GET | `/api/v1/agents` | List agents with counts |
| GET | `/api/v1/stats` | System statistics |

### Graph (v6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/graph/search?q=...` | Graphiti temporal graph search |
| GET | `/api/v1/graph/entity/{name}` | Entity relationships |

### Compaction (v6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/compact/session` | Trigger session compaction |
| POST | `/api/v1/compact/daily` | Generate daily digest |
| POST | `/api/v1/compact/weekly` | Run weekly merge |
| GET | `/api/v1/compact/status` | Compaction health |

### Reflection (v6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/reflect` | Trigger reflection analysis |

### Federation (v6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/federation/register` | Register a remote node |
| POST | `/api/v1/federation/push` | Push memories from remote |
| POST | `/api/v1/federation/pull` | Pull memories to remote |
| POST | `/api/v1/federation/sync` | Bidirectional sync |
| GET | `/api/v1/federation/status` | Federation health |

## Composite Scoring

```
score = (w_semantic × similarity + w_recency × decay + w_importance × weight) × access_boost
```

- **Semantic similarity**: 50% weight (cosine from Qdrant)
- **Recency decay**: 30% weight (exponential, 90-day half-life)
- **Importance**: 20% weight (type-based: decisions > preferences > facts > events)
- **Access boost**: up to 1.5× for frequently accessed memories
- **Persistent types** (decisions, preferences, relationships): minimum 40% recency floor

## Project Structure

```
memclawz/
├── __init__.py
├── config.py           # All configuration (Qdrant, Neo4j, Federation, etc.)
├── api.py              # FastAPI REST API (v6 endpoints)
├── scoring.py          # Composite relevance scoring
├── compactor.py        # Session/daily/weekly compaction
├── graphiti_layer.py   # Neo4j + Graphiti temporal graph
├── federation.py       # Multi-claw federation protocol
├── reflection.py       # Sleep-time reflection engine
├── watcher.py          # LCM → Mem0 → Qdrant + Graphiti pipeline
├── compaction_cron.py  # Automated compaction scheduler
├── classifier.py       # Memory type classification
├── contradiction.py    # Contradiction detection (+ Graphiti)
├── decay.py            # Legacy decay (kept for compatibility)
├── mcp_server.py       # MCP STDIO server (v6 tools)
├── importer.py         # Bulk import utilities
└── utils.py            # Shared utilities
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `memclawz-api` | 3500 | REST API |
| `memclawz-watcher` | — | LCM auto-extract pipeline |
| `memclawz-cron` | — | Compaction scheduler (30-min) |
| `memclawz-mcp` | stdio | MCP server |
| Neo4j | 7474/7687 | Graph database |
| Qdrant | 6333 | Vector database |

## Current Status (March 2026)

- **Version:** v6.0.0 (ClawHub: memclawz@6.0.1)
- **Total Memories:** 3,772 across 16 agents
- **Agent Distribution:** main (2,826), peopleclaw (257), infraclaw (73), appsclaw (73), tradeclaw (84), qaclaw (118), tradingdataclaw (17), commsclaw (64), moneyclawx (72), cmoclaw (45), devopsoci (32), coinresearchclaw (58), quantclaw (30), paperclipclaw (10), coinsclaw (10), openclaw-brain (3)
- **Memory Types:** indexed_chunk (2,031), session_transcript (457), knowledge (382), conversation_summary (561), fact (50), historical_memory (146), unknown (134), bootstrap (4), decision (1), event (3), test (3)
- **Qdrant:** 3,772 vectors, healthy (systemd `Restart=always`)
- **Neo4j/Graphiti:** Currently disabled (238 nodes, 532 edges from earlier runs)
- **Federation:** 2 nodes registered
- **API:** Healthy on port 3500
- **MCP Server:** Available via stdio
- **Two-Way Memory:** All 16 agents now read/write to shared memory bus
- **Fleet Sync:** Daily automated sync at 03:00 UTC

## Known Issues & Fixes

### Mem0 `add()` Returns Empty (FIXED 2026-03-14)
**Problem:** Mem0 uses an LLM extraction pipeline that rejects already-clean/pre-formatted text as having "nothing to extract." Calling `mem.add()` returns `results: []` for structured facts.

**Fix:** Added direct Qdrant upsert fallback in `/api/v1/add` endpoint. When `mem.add()` returns empty results, the API generates OpenAI embeddings and upserts directly to Qdrant with proper payload structure (`memory`, `hash`, `user_id`, `created_at`, plus flattened `agent`, `type`, `source`).

**File:** `memclawz/api.py` — search for `direct_qdrant` fallback logic.

### Empty `memory` Field in 3,636 Records (FIXED 2026-03-14)
**Problem:** Early direct Qdrant inserts stored content only in nested `metadata.memory` but left the top-level `memory` field empty. This caused search results to return records with `"memory": ""`.

**Fix:** Batch repair script (`reattribute_memories.py`) extracted content from `metadata.memory` and backfilled the top-level `memory` field across 3,636 records. All records now have proper top-level `memory` field for search and retrieval.

### Agent Tags in Stats (FIXED 2026-03-14)
**Problem:** Direct Qdrant inserts weren't populating the `agent_id` field in Mem0's expected metadata structure, causing `by_agent` stats to undercount direct inserts.

**Fix:** Updated direct Qdrant upsert logic to include top-level `agent_id` and `memory_type` fields in the payload structure, ensuring proper agent attribution in stats and search results.

### Memory Redistribution (COMPLETED 2026-03-14)
Moved 2,882 memories out of "main" agent to proper domain agents:
- infraclaw (1,016), cmoclaw (913), tradeclaw (553), qaclaw (182), peopleclaw (111), tradingdataclaw (54), appsclaw (37), commsclaw (31), paperclipclaw (22), devopsoci (19), coinresearchclaw (14), moneyclawx (4)
- 763 memories retained in main (orchestration content only)

## Memory Protocol for Agents

Agents writing to MemClawz should use the `/api/v1/add` endpoint:

```bash
curl -X POST http://localhost:3500/api/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What was learned or decided",
    "agent_id": "agent-name",
    "memory_type": "decision|event|fact|procedure|insight"
  }'
```

**Memory types:**
- `decision` — Choices made (architecture, tools, approach)
- `event` — What happened (deployed X, fixed Y)
- `fact` — Discovered information (endpoints, versions, pricing)
- `procedure` — How something was done (deploy steps, build process)
- `insight` — Lessons learned (what worked, what didn't)

**Canonical memory order:** Local files first → MemClawz second → LCM third.

## License

MIT
