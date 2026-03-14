# MemClawz v7 🧠

AI Agent Fleet Memory System — auto-enrichment, hybrid search, memory lifecycle, Paperclip integration, composite scoring, compaction engine, Graphiti temporal knowledge graph, multi-claw federation, sleep-time reflection, MCP server.

## What's New in v7

- **Auto-Enrichment Layer** — Automatic metadata extraction (entities, categories, temporal markers, importance scoring) on every memory write
- **Hybrid Search** — BM25 keyword scoring alongside vector search for improved relevance (pure Python, no external deps)
- **Memory Lifecycle** — 8-state status management (active → verified → archived → deprecated → superseded → merged → disputed → deleted) with transition validation
- **Expanded Memory Types** — 15 types: decision, event, fact, procedure, insight, preference, relationship, goal, constraint, hypothesis, observation, question, answer, correction, meta
- **Paperclip Integration** — `plugin-memclawz` for agent orchestrator: auto-context injection before spawn, memory-aware routing, post-task writeback
- **v7 API Extensions** — `/api/v1/lifecycle/*`, `/api/v1/hybrid-search`, `/api/v1/enrich` endpoints

## What's in v6

- **Composite Scoring** — Weighted blend of semantic similarity + recency decay + importance + access frequency
- **Compaction Engine** — Three-tier: session compactor, daily digest, weekly merge with deduplication
- **Graphiti Integration** — Neo4j temporal knowledge graph for entity relationships, contradiction detection
- **Multi-Claw Federation** — HTTP push/pull protocol for sharing memories across fleet
- **Sleep-Time Reflection** — LLM-driven pattern detection, insight generation, MEMORY.md update proposals
- **Enhanced MCP Server** — Tools: compact_session, reflect, memory_stats

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                MemClawz v7 Master (YoniClaw)                │
│                                                             │
│  FastAPI :3500  │  Qdrant :6333  │  Neo4j :7687             │
│                                                             │
│  Enrichment → Hybrid Search → Lifecycle                     │
│  Mem0 + Composite Scoring + Graphiti                        │
│  Compaction: Session │ Daily │ Weekly                       │
│  Reflection Engine │ Federation                             │
│                                                             │
│  Paperclip Plugin: context injection + routing + writeback  │
└─────────────────────────────────────────────────────────────┘
         ▲              ▲              ▲
    Clawdet         MoneyClaw      WhiteRabbit
         ▲
      OCI ARM
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

### Core
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (Qdrant, Neo4j, Federation) |
| GET | `/api/v1/search?q=...` | Semantic search with composite scoring |
| POST | `/api/v1/add` | Add memory (direct Qdrant + background Mem0 + Graphiti + auto-enrichment) |
| POST | `/api/v1/add-direct` | Fast-path: skip Mem0, direct Qdrant + Graphiti |
| GET | `/api/v1/memories` | List memories |
| GET | `/api/v1/agents` | List agents with counts |
| GET | `/api/v1/stats` | System statistics |

### v7: Enrichment & Hybrid Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/enrich` | Auto-enrich a memory with metadata extraction |
| GET | `/api/v1/hybrid-search?q=...` | Combined BM25 + vector search |

### v7: Memory Lifecycle
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/lifecycle/transition` | Transition memory to new state |
| GET | `/api/v1/lifecycle/status/{id}` | Get memory lifecycle status |
| POST | `/api/v1/lifecycle/archive-stale` | Archive memories older than threshold |
| POST | `/api/v1/lifecycle/supersede` | Mark memory as superseded by another |

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

## Paperclip Integration (v7)

MemClawz integrates with Paperclip (Agent Orchestrator) via `plugin-memclawz`:

### Auto-Context Injection
Before spawning any agent session, the plugin searches MemClawz for the top 5 relevant memories and prepends them to the prompt as a `## Relevant Memory Context` section. 3s timeout, graceful fallback.

### Memory-Aware Routing
`scoreAgentsByMemory(task, agentIds[])` scores each agent by how many relevant memories they have for a given task. Used for intelligent agent selection.

### Post-Task Writeback
On `session_complete`, `session_error`, `pr_merged`, `ci_failed`, `changes_requested`, and `review_requested` events, the plugin auto-writes a summary to MemClawz via `/api/v1/add-direct`. 5s timeout, fire-and-forget.

**Plugin location:** `packages/plugins/notifier-memclawz/` in the Paperclip monorepo.

## Composite Scoring

```
score = (w_semantic × similarity + w_recency × decay + w_importance × weight) × access_boost
```

- **Semantic similarity**: 50% weight (cosine from Qdrant)
- **Recency decay**: 30% weight (exponential, 90-day half-life)
- **Importance**: 20% weight (type-based: decisions > preferences > facts > events)
- **Access boost**: up to 1.5× for frequently accessed memories
- **Persistent types** (decisions, preferences, relationships): minimum 40% recency floor

## Hybrid Search (v7)

Combines BM25 keyword scoring with vector similarity for better recall:

```
final_score = α × vector_score + (1 - α) × bm25_score
```

Default α = 0.7 (70% semantic, 30% keyword). Useful for exact term matches that pure vector search might miss (agent IDs, error codes, port numbers, file paths).

## Memory Lifecycle (v7)

Eight states with validated transitions:

```
active → verified → archived
  ↓         ↓         ↓
deprecated  merged   deleted
  ↓
superseded
  ↓
disputed
```

- **active**: Default state for new memories
- **verified**: Confirmed by human or cross-reference
- **archived**: No longer actively relevant but preserved
- **deprecated**: Outdated, newer info exists
- **superseded**: Replaced by a specific newer memory (linked)
- **merged**: Combined into another memory during compaction
- **disputed**: Conflicting information detected
- **deleted**: Soft-deleted, excluded from search

Auto-archival via `POST /api/v1/lifecycle/archive-stale` for memories older than configurable threshold.

## Auto-Enrichment (v7)

Every memory written via `/api/v1/add` is automatically enriched with:

- **Entities**: Extracted names, agent IDs, server IPs, domains, file paths
- **Categories**: Infrastructure, trading, research, people, crypto, etc.
- **Temporal markers**: Dates, relative time references
- **Importance score**: 0.0–1.0 based on content analysis
- **Content hash**: For deduplication

Enrichment is non-blocking (runs async after write confirmation).

## Project Structure

```
memclawz/
├── __init__.py
├── config.py           # All configuration
├── api.py              # FastAPI REST API
├── scoring.py          # Composite relevance scoring
├── enrichment.py       # v7: Auto-enrichment layer
├── hybrid_search.py    # v7: BM25 + vector hybrid search
├── lifecycle.py        # v7: Memory status lifecycle
├── v7_extensions.py    # v7: API integration helpers
├── compactor.py        # Session/daily/weekly compaction
├── graphiti_layer.py   # Neo4j + Graphiti temporal graph
├── federation.py       # Multi-claw federation protocol
├── reflection.py       # Sleep-time reflection engine
├── watcher.py          # LCM → Mem0 → Qdrant + Graphiti pipeline
├── compaction_cron.py  # Automated compaction scheduler
├── classifier.py       # Memory type classification
├── contradiction.py    # Contradiction detection
├── decay.py            # Legacy decay (compatibility)
├── mcp_server.py       # MCP STDIO server
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

## Current Status (March 14, 2026)

- **Version:** v7.0.0
- **Total Memories:** 3,774 across 16 agents
- **Fleet:** 5 servers (YoniClaw, White Rabbit, Clawdet, MoneyClaw, OCI ARM)
- **Agents:** 21 registered in Paperclip, 16 with active memories
- **Qdrant:** Healthy (systemd `Restart=always`)
- **Neo4j/Graphiti:** 238 nodes, 532 edges
- **Federation:** 2 nodes registered, daily sync at 03:00 UTC
- **Paperclip Plugin:** Live on White Rabbit (context injection + writeback)
- **Two-Way Memory:** All agents read/write to shared memory bus

## Changelog

### v7.0.0 (2026-03-14)
- Auto-enrichment layer (entities, categories, temporal markers, importance)
- Hybrid search (BM25 + vector)
- Memory lifecycle management (8 states, transition validation)
- Expanded memory types (15 types)
- Paperclip `plugin-memclawz` integration (context injection, routing, writeback)
- Prompt-builder MemClawz context injection in Paperclip core
- Agent memory scoring for routing decisions

### v6.0.0 (2026-03-13)
- Composite scoring (semantic × recency × importance × access)
- 3-tier compaction engine
- Graphiti/Neo4j temporal knowledge graph
- Multi-claw federation protocol
- Sleep-time reflection engine
- Direct Qdrant write path (fix for Mem0 empty extraction)
- Memory redistribution (2,882 memories to domain agents)
- Record repair (3,636 malformed records fixed)

## Memory Protocol for Agents

```bash
curl -X POST http://localhost:3500/api/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What was learned or decided",
    "agent_id": "agent-name",
    "memory_type": "decision|event|fact|procedure|insight"
  }'
```

**Memory types (v7):** decision, event, fact, procedure, insight, preference, relationship, goal, constraint, hypothesis, observation, question, answer, correction, meta

**Canonical memory order:** Local files first → MemClawz second → LCM third.

## License

MIT
