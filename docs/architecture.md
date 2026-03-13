# MemClawz v5 Architecture

## Overview

MemClawz is the memory layer for the YoniClaw AI agent fleet. It provides:
- **Auto-extraction** from LCM conversation summaries
- **Shared memory bus** via REST API
- **Typed classification** of memories (fact, decision, preference, etc.)
- **Contradiction detection** to identify superseded information
- **Relevance decay** scoring with time-based half-life
- **MCP server** for Model Context Protocol integration

## Data Flow

```
LCM Database (conversations)
    ↓ watcher (every 30 min)
    ↓ classify_memory()
    ↓
Mem0 (manages embeddings + metadata)
    ↓
Qdrant (vector storage)
    ↑
REST API ←→ All Agents (search, add, list)
    ↑
MCP Server ←→ MCP Clients (Claude Desktop, etc.)
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Config | `config.py` | Centralized settings, credential loading |
| Watcher | `watcher.py` | LCM → Mem0 auto-extract pipeline |
| API | `api.py` | FastAPI shared memory bus (port 3500) |
| Classifier | `classifier.py` | Heuristic + LLM memory type classification |
| Contradiction | `contradiction.py` | Detect superseded/contradicting memories |
| Decay | `decay.py` | Time-based relevance scoring |
| MCP Server | `mcp_server.py` | Model Context Protocol integration |
| Importer | `importer.py` | Bulk import from markdown/sqlite/jsonl |

## Memory Types

| Type | Description | Decay Floor |
|------|-------------|-------------|
| decision | Choices made | 50% (persistent) |
| preference | User preferences | 50% (persistent) |
| relationship | People, contacts | 50% (persistent) |
| insight | Learned lessons | Normal |
| procedure | How-to / workflows | Normal |
| fact | General facts | Normal |
| event | Time-specific events | Normal |

## Storage

- **Vector DB**: Qdrant (localhost:6333, collection: `yoniclaw_memories`)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **LLM**: Anthropic Claude Sonnet for classification and memory extraction
- **State**: `~/.memclawz/last_sync.json` tracks watcher position
