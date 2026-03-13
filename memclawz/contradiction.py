"""Contradiction detection for incoming memories."""


def check_contradiction(mem_instance, new_memory: str, user_id: str = "yoni", threshold: float = 0.85) -> list[dict]:
    """Check if new memory contradicts existing ones.

    Returns list of potential contradictions with suggested actions.
    """
    similar = mem_instance.search(new_memory, user_id=user_id, limit=5)
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
