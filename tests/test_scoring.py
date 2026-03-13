"""Tests for composite scoring."""
from datetime import datetime, timezone, timedelta
from memclawz.scoring import composite_score, score_results, _compute_recency


def test_basic_composite_score():
    """High semantic similarity, recent memory = high score."""
    score = composite_score(
        semantic_similarity=0.95,
        created_at=datetime.now(timezone.utc).isoformat(),
        importance=0.9,
        memory_type="decision",
    )
    assert 0.8 <= score <= 1.0


def test_old_memory_decays():
    """Old memory should score lower than recent."""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=180)

    recent_score = composite_score(0.9, now.isoformat(), memory_type="fact")
    old_score = composite_score(0.9, old.isoformat(), memory_type="fact")
    assert recent_score > old_score


def test_persistent_types_resist_decay():
    """Decisions and preferences should not decay below floor."""
    very_old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    decision_score = composite_score(0.8, very_old, memory_type="decision")
    event_score = composite_score(0.8, very_old, memory_type="event")
    assert decision_score > event_score


def test_access_boost():
    """Frequently accessed memories get a boost."""
    base = composite_score(0.8, datetime.now(timezone.utc).isoformat(), access_count=0)
    boosted = composite_score(0.8, datetime.now(timezone.utc).isoformat(), access_count=10)
    assert boosted > base


def test_access_boost_capped():
    """Access boost should be capped at 1.5x."""
    score = composite_score(0.3, datetime.now(timezone.utc).isoformat(), access_count=100)
    assert score <= 1.0


def test_score_results_sorting():
    """Results should be sorted by composite score descending."""
    results = [
        {"score": 0.5, "metadata": {"type": "event", "extracted_at": (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()}},
        {"score": 0.9, "metadata": {"type": "decision", "extracted_at": datetime.now(timezone.utc).isoformat()}},
        {"score": 0.7, "metadata": {"type": "fact", "extracted_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()}},
    ]
    scored = score_results(results)
    scores = [r["composite_score"] for r in scored]
    assert scores == sorted(scores, reverse=True)


def test_missing_created_at():
    """Should handle missing created_at gracefully."""
    score = composite_score(0.8, None, memory_type="fact")
    assert 0.0 <= score <= 1.0


def test_recency_now_is_one():
    """Recency of something just created should be ~1.0."""
    r = _compute_recency(datetime.now(timezone.utc).isoformat())
    assert r > 0.99


def test_score_never_exceeds_one():
    """Score should never exceed 1.0."""
    score = composite_score(1.0, datetime.now(timezone.utc).isoformat(), importance=1.0, access_count=50, memory_type="decision")
    assert score <= 1.0
