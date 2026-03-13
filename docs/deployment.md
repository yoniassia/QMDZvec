# MemClawz v5 Deployment Guide

## Prerequisites

- Python 3.11+
- Qdrant running on localhost:6333
- OpenAI API key (for embeddings)
- Anthropic API key (for LLM classification)

## Quick Start

```bash
cd ~/memclawz
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and edit config
cp .env.example .env
nano .env

# Run API
python -m uvicorn memclawz.api:app --host 0.0.0.0 --port 3500

# Run watcher (separate terminal)
python -m memclawz.watcher
```

## Systemd Services

```bash
# Link service files
mkdir -p ~/.config/systemd/user
ln -sf ~/memclawz/systemd/memclawz-api.service ~/.config/systemd/user/
ln -sf ~/memclawz/systemd/memclawz-watcher.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now memclawz-api
systemctl --user enable --now memclawz-watcher

# Check status
systemctl --user status memclawz-api
systemctl --user status memclawz-watcher
```

## MCP Integration

Add to your MCP client config (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "memclawz": {
      "command": "/home/yoniclaw/memclawz/.venv/bin/python",
      "args": ["-m", "memclawz.mcp_server"],
      "cwd": "/home/yoniclaw/memclawz"
    }
  }
}
```

## Verify

```bash
curl http://localhost:3500/health
curl "http://localhost:3500/api/v1/stats"
curl "http://localhost:3500/api/v1/search?q=eToro+SuperApp"
```
