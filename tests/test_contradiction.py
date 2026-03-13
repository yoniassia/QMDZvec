"""Tests for contradiction detection."""
from memclawz.contradiction import is_update


def test_is_update_same_topic_different_value():
    old = "The API runs on port 3000"
    new = "The API runs on port 3500"
    assert is_update(old, new) is True


def test_is_not_update_exact_match():
    text = "Yoni prefers dark mode"
    assert is_update(text, text) is False


def test_is_not_update_different_topic():
    old = "The weather in Tel Aviv is sunny"
    new = "Qdrant uses HNSW indexing for vectors"
    assert is_update(old, new) is False


def test_is_update_partial_overlap():
    old = "eToro IPO valuation is 4 billion dollars"
    new = "eToro IPO valuation is 5.6 billion dollars"
    assert is_update(old, new) is True
