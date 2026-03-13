"""Relevance scoring with time-based decay."""
from datetime import datetime
import math

# Base relevance weights by memory type
TYPE_WEIGHTS = {
    "decision": 1.0,
    "preference": 0.95,
    "relationship": 0.9,
    "insight": 0.9,
    "procedure": 0.85,
    "fact": 0.8,
    "event": 0.7,
}

# Types that resist decay (floor at 50%)
PERSISTENT_TYPES = {"decision", "preference", "relationship"}

# Half-life in days
HALF_LIFE_DAYS = 90


def calculate_relevance(memory: dict, access_log: dict | None = None) -> float:
    """Calculate relevance score with exponential decay.

    Args:
        memory: Memory dict with 'metadata' and 'id' keys.
        access_log: Optional dict mapping memory_id → access count.

    Returns:
        Float relevance score (higher = more relevant).
    """
    meta = memory.get("metadata", {})

    # Base relevance by type
    base = TYPE_WEIGHTS.get(meta.get("type", "fact"), 0.8)

    # Recency decay
    created = meta.get("extracted_at", meta.get("date", ""))
    recency = _compute_recency(created)

    # Access frequency boost (capped at 2x)
    access_count = (access_log or {}).get(memory.get("id", ""), 0)
    access_boost = min(1.0 + (access_count * 0.1), 2.0)

    # Persistent types don't decay below 50%
    if meta.get("type") in PERSISTENT_TYPES:
        recency = max(recency, 0.5)

    return base * recency * access_boost


def _compute_recency(created_str: str) -> float:
    """Compute recency factor from ISO date string."""
    if not created_str:
        return 0.5
    try:
        created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        age_days = (datetime.utcnow() - created_dt.replace(tzinfo=None)).days
        return math.exp(-0.693 * age_days / HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.5
