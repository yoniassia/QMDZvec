"""Centralized configuration for MemClawz v5"""
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

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "yoniclaw_memories")

# LCM
LCM_DB_PATH = os.getenv("LCM_DB_PATH", "/home/yoniclaw/.openclaw/lcm.db")
AGENTS_DIR = os.getenv("AGENTS_DIR", "/home/yoniclaw/.openclaw/agents")

# Watcher
STATE_DIR = Path(os.getenv("MEMCLAWZ_STATE_DIR", os.path.expanduser("~/.memclawz")))
STATE_DIR.mkdir(parents=True, exist_ok=True)
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "1800"))  # 30 min

# API
API_PORT = int(os.getenv("MEMCLAWZ_API_PORT", "3500"))
API_HOST = os.getenv("MEMCLAWZ_API_HOST", "0.0.0.0")

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
