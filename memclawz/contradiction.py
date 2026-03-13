"""Contradiction detection for incoming memories.

v6: Enhanced with optional Graphiti temporal checks.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


def check_contradiction(mem_instance, new_memory: str, user_id: str = "yoni", threshold: float = 0.85) -> list[dict]:
    """Check if new memory contradicts existing ones.

    Returns list of potential contradictions with suggested actions.
    """
    similar = mem_instance.search(new_memory, user_id=user_id, limit=5)
    if isinstance(similar, dict):
        similar = similar.get("results", [])
    contradictions = []

    for existing in similar:
        score = existing.get("score", 0)
        if score > threshold:
            existing_text = existing.get("memory", "")
            if is_update(existing_text, new_memory):
                contradictions.append(
                    {
                        "existing_id": existing.get("id"),
                        "existing_text": existing_text,
                        "action": "supersede",
                        "score": score,
                    }
                )

    return contradictions


async def check_contradiction_graphiti(query: str, group_id: str = "yoniclaw") -> list[dict]:
    """Check for temporal contradictions using Graphiti.

    Graphiti automatically handles contradiction detection via
    temporal edge invalidation. This function searches for
    invalidated edges related to the query.
    """
    try:
        from .graphiti_layer import search
        from .config import GRAPHITI_ENABLED

        if not GRAPHITI_ENABLED:
            return []

        results = await search(query, num_results=10, group_ids=[group_id])
        contradictions = []
        for r in results:
            if r.get("expired") or r.get("invalid_at"):
                contradictions.append({
                    "fact": r.get("fact", ""),
                    "valid_at": r.get("valid_at"),
                    "invalid_at": r.get("invalid_at"),
                    "action": "expired",
                })
        return contradictions
    except Exception as e:
        logger.warning(f"Graphiti contradiction check failed: {e}")
        return []


def is_update(old_text: str, new_text: str) -> bool:
    """Check if new text is an update to old text (same topic, different value)."""
    old_lower = old_text.lower()
    new_lower = new_text.lower()

    old_words = set(old_lower.split())
    new_words = set(new_lower.split())
    shared_words = old_words & new_words
    topic_overlap = len(shared_words) / max(len(old_words), 1)

    exact_match = old_lower.strip() == new_lower.strip()

    return topic_overlap > 0.5 and not exact_match
