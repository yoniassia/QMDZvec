"""Memory type classification — heuristic + LLM fallback."""

MEMORY_TYPES = {
    "fact": "A factual statement about a person, project, system, or concept",
    "decision": "A choice or decision that was made",
    "preference": "A user preference or style choice",
    "procedure": "Steps to accomplish something, a workflow or process",
    "relationship": "Information about a person, contact, or organizational relationship",
    "event": "Something that happened at a specific time",
    "insight": "A learned lesson, pattern, or strategic insight",
}

# Keyword → type mapping (fast heuristic)
_KEYWORD_MAP = {
    "decision": ["decided", "decision", "chose", "agreed", "resolved", "ruling"],
    "preference": ["prefers", "preference", "likes", "wants", "favorite", "style"],
    "procedure": ["how to", "steps to", "process", "workflow", "recipe", "guide"],
    "relationship": ["is a", "works at", "phone", "email", "contact", "reports to", "married"],
    "event": ["deployed", "launched", "happened", "completed", "shipped", "released", "ipo"],
    "insight": ["learned", "lesson", "insight", "realized", "pattern", "takeaway"],
}


def classify_heuristic(text: str) -> str | None:
    """Fast keyword-based classification. Returns None if uncertain."""
    text_lower = text.lower()
    for mem_type, keywords in _KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return mem_type
    return None


def classify_with_llm(text: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Use LLM for accurate classification."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Classify this memory into exactly one type: "
                    f"{', '.join(MEMORY_TYPES.keys())}\n\n"
                    f"Memory: {text[:500]}\n\n"
                    f"Respond with ONLY the type name, nothing else."
                ),
            }
        ],
    )
    result = response.content[0].text.strip().lower()
    return result if result in MEMORY_TYPES else "fact"


def classify_memory(text: str, use_llm: bool = False) -> str:
    """Classify memory text. Uses heuristics first, LLM fallback if enabled."""
    result = classify_heuristic(text)
    if result:
        return result
    if use_llm:
        try:
            return classify_with_llm(text)
        except Exception:
            pass
    return "fact"
