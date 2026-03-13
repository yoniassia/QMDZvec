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
| POST | `/api/v1/add` | Add memory (feeds Qdrant + Graphiti) |
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

## License

MIT
