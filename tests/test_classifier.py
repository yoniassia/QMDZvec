"""Tests for memory classifier."""
from memclawz.classifier import classify_heuristic, classify_memory


def test_decision_classification():
    assert classify_heuristic("We decided to use Qdrant for vector storage") == "decision"
    assert classify_heuristic("The team agreed on a weekly standup schedule") == "decision"


def test_preference_classification():
    assert classify_heuristic("Yoni prefers dark mode for all dashboards") == "preference"
    assert classify_heuristic("He likes concise responses over verbose ones") == "preference"


def test_procedure_classification():
    assert classify_heuristic("How to deploy the API: run docker compose up") == "procedure"
    assert classify_heuristic("Steps to configure Qdrant backup") == "procedure"


def test_relationship_classification():
    assert classify_heuristic("Haim is a security contact at +972-587181010") == "relationship"
    assert classify_heuristic("Patricia works at eToro as PA") == "relationship"


def test_event_classification():
    assert classify_heuristic("eToro launched the SuperApp in March 2025") == "event"
    assert classify_heuristic("We deployed the new API yesterday") == "event"


def test_fallback_to_fact():
    assert classify_memory("The sky is blue and water is wet") == "fact"
    assert classify_memory("Qdrant uses HNSW for approximate nearest neighbor search") == "fact"


def test_classify_memory_without_llm():
    # Should use heuristic only by default
    result = classify_memory("We decided to migrate to PostgreSQL")
    assert result == "decision"
