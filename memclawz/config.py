"""Centralized configuration for MemClawz v6"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv(Path(__file__).parent.parent / ".env")

# Credential loading — prefer env vars, fall back to credential files
CREDENTIALS_DIR = Path(os.getenv("MEMCLAWZ_CREDENTIALS_DIR", "/home/yoniclaw/.credentials"))


def _load_key(env_var: str, file_name: str | None = None) -> str:
    """Load API key from env var or credential file."""
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    if file_name:
        path = CREDENTIALS_DIR / file_name
        if path.exists():
            val = path.read_text().strip()
            os.environ[env_var] = val
            return val
    return ""


OPENAI_API_KEY = _load_key("OPENAI_API_KEY", "openai-api-key.txt")
ANTHROPIC_API_KEY = _load_key("ANTHROPIC_API_KEY", "anthropic-api-key.txt")
GOOGLE_API_KEY = _load_key("GOOGLE_API_KEY", "google-api-key.txt")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "yoniclaw_memories")

# Neo4j / Graphiti (v6)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # Auth disabled for local
GRAPHITI_ENABLED = os.getenv("GRAPHITI_ENABLED", "true").lower() in ("true", "1", "yes")
GRAPHITI_GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "yoniclaw")

# LCM
LCM_DB_PATH = os.getenv("LCM_DB_PATH", "/home/yoniclaw/.openclaw/lcm.db")
AGENTS_DIR = os.getenv("AGENTS_DIR", "/home/yoniclaw/.openclaw/agents")

# Workspace
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/home/yoniclaw/.openclaw/workspace"))

# Watcher
STATE_DIR = Path(os.getenv("MEMCLAWZ_STATE_DIR", os.path.expanduser("~/.memclawz")))
STATE_DIR.mkdir(parents=True, exist_ok=True)
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "1800"))  # 30 min

# API
API_PORT = int(os.getenv("MEMCLAWZ_API_PORT", "3500"))
API_HOST = os.getenv("MEMCLAWZ_API_HOST", "0.0.0.0")

# Compaction thresholds (v6)
COMPACTION_THRESHOLD_WARN = float(os.getenv("COMPACTION_THRESHOLD_WARN", "0.6"))
COMPACTION_THRESHOLD_URGENT = float(os.getenv("COMPACTION_THRESHOLD_URGENT", "0.8"))
COMPACTION_INTERVAL = int(os.getenv("COMPACTION_INTERVAL", "1800"))  # 30 min

# Federation (v6)
FEDERATION_ENABLED = os.getenv("FEDERATION_ENABLED", "true").lower() in ("true", "1", "yes")
FEDERATION_ROLE = os.getenv("FEDERATION_ROLE", "master")  # master or node

# Enrichment (v7)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Mem0 config
MEM0_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": COLLECTION_NAME,
        },
    },
    "llm": {
        "provider": "anthropic",
        "config": {"model": "claude-sonnet-4-20250514", "temperature": 0.1},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
}

# Mem0 config without LLM (for search-only / MCP server)
MEM0_CONFIG_LITE = {
    "vector_store": MEM0_CONFIG["vector_store"],
    "embedder": MEM0_CONFIG["embedder"],
}
