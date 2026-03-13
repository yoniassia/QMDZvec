# MemClawz v5 🧠

**AI Agent Fleet Memory System** — auto-extract from conversations, shared memory bus, typed memories, contradiction detection, relevance decay, MCP server.

Built on [Mem0](https://github.com/mem0ai/mem0) + [Qdrant](https://qdrant.tech/).

## Features

- 🔄 **Live Auto-Extract Pipeline** — watches LCM conversation summaries, extracts and classifies memories automatically
- 🌐 **Shared Memory Bus** — REST API for all agents to search, add, and manage memories
- 🏷️ **Typed Classification** — 7 memory types (fact, decision, preference, procedure, relationship, event, insight)
- ⚔️ **Contradiction Detection** — identifies superseded information when new memories conflict with existing ones
- 📉 **Relevance Decay** — time-based scoring with configurable half-life (90 days), persistent types resist decay
- 🔌 **MCP Server** — Model Context Protocol integration for Claude Desktop and other MCP clients
- 📥 **Bulk Import** — import from markdown, SQLite, or JSONL sources

## Architecture

```
LCM Database (conversations)
    ↓ watcher (every 30 min)
    ↓ classify + extract
    ↓
Mem0 (embeddings + metadata)
    ↓
Qdrant (vector storage, 1536-dim cosine)
    ↑
REST API (port 3500) ←→ All Agents
    ↑
MCP Server (stdio) ←→ MCP Clients
```

## Quick Start

```bash
cd ~/memclawz
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start API
uvicorn memclawz.api:app --host 0.0.0.0 --port 3500

# Start watcher (separate terminal)
python -m memclawz.watcher
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/search?q=...` | GET | Semantic search |
| `/api/v1/add` | POST | Add memory |
| `/api/v1/memories` | GET | List memories |
| `/api/v1/agents` | GET | Agent memory counts |
| `/api/v1/stats` | GET | System statistics |
| `/api/v1/memory/{id}` | DELETE | Delete memory |
| `/api/v1/memory/{id}` | PUT | Update memory |

## Memory Types

| Type | Description | Decay |
|------|-------------|-------|
| `decision` | Choices made | Persistent (floor 50%) |
| `preference` | User preferences | Persistent (floor 50%) |
| `relationship` | People, contacts | Persistent (floor 50%) |
| `insight` | Learned lessons | Normal (90-day half-life) |
| `procedure` | Workflows, how-tos | Normal |
| `fact` | General facts | Normal |
| `event` | Time-specific events | Normal |

## Configuration

Copy `.env.example` to `.env` and set your API keys. Or export them as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Systemd Services

```bash
# Link and enable
ln -sf ~/memclawz/systemd/memclawz-api.service ~/.config/systemd/user/
ln -sf ~/memclawz/systemd/memclawz-watcher.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memclawz-api memclawz-watcher
```

## MCP Integration

Add to your MCP client config:

```json
{
  "mcpServers": {
    "memclawz": {
      "command": "/home/yoniclaw/memclawz/.venv/bin/python",
      "args": ["-m", "memclawz.mcp_server"]
    }
  }
}
```

## Stack

- **Vector DB**: Qdrant
- **Memory Layer**: Mem0
- **Embeddings**: OpenAI `text-embedding-3-small`
- **LLM**: Anthropic Claude Sonnet
- **API**: FastAPI + Uvicorn
- **Protocol**: MCP (Model Context Protocol)

## License

MIT
