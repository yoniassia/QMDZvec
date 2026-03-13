"""Shared utilities for MemClawz."""
import json
import os
from pathlib import Path
from datetime import datetime


def load_json(path: str | Path) -> dict:
    """Load JSON file, return empty dict if missing or empty."""
    path = Path(path)
    if path.exists():
        text = path.read_text().strip()
        if text:
            return json.loads(text)
    return {}


def save_json(data: dict, path: str | Path) -> None:
    """Save dict as JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def utcnow_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.utcnow().isoformat()


def extract_agent_from_session(session_id: str, agents_dir: str) -> str:
    """Map a session_id to the agent that owns it."""
    if not os.path.isdir(agents_dir):
        return "main"
    for agent in os.listdir(agents_dir):
        sessions_dir = os.path.join(agents_dir, agent, "sessions")
        if os.path.exists(os.path.join(sessions_dir, f"{session_id}.jsonl")):
            return agent
    return "main"
