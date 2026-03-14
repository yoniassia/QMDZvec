"""Unit tests for memclawz.hybrid_search module."""
import pytest
import math
from unittest.mock import patch, MagicMock

from memclawz.hybrid_search import (
    tokenize, compute_tf, compute_idf, bm25_score, prepare_corpus_stats,
    hybrid_search, explain_hybrid_score, batch_hybrid_search,
    DEFAULT_K1, DEFAULT_B, DEFAULT_HYBRID_WEIGHTS, STATUS_WEIGHTS
)


@pytest.mark.unit
class TestTokenization:
    """Test tokenization functions."""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello world test"
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "test"]
    
    def test_tokenize_mixed_case(self):
        """Test tokenization converts to lowercase."""
        text = "Hello WORLD Test"
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "test"]
    
    def test_tokenize_punctuation(self):
        """Test tokenization handles punctuation."""
        text = "Hello, world! How are you?"
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "how", "are", "you"]
    
    def test_tokenize_numbers(self):
        """Test tokenization includes numbers."""
        text = "Version 3.10 release 2024"
        tokens = tokenize(text)
        assert "3" in tokens or "10" in tokens  # might split 3.10
        assert "2024" in tokens
    
    def test_tokenize_short_tokens_filtered(self):
        """Test short tokens (< 2 chars) are filtered out."""
        text = "I am a developer"
        tokens = tokenize(text)
        assert "am" in tokens  # 2 chars, should be included
        # "I" and "a" are 1 char, should be filtered
        assert "i" not in tokens
        assert "a" not in tokens
    
    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        assert tokenize("") == []
        assert tokenize(None) == []
        assert tokenize("   ") == []
    
    def test_tokenize_special_characters(self):
        """Test tokenization with special characters."""
        text = "user@domain.com path/to/file api-endpoint"
        tokens = tokenize(text)
        # Should extract alphanumeric parts
        assert "user" in tokens
        assert "domain" in tokens
        assert "com" in tokens
        assert "path" in tokens
        assert "to" in tokens
        assert "file" in tokens
        assert "api" in tokens
        assert "endpoint" in tokens


@pytest.mark.unit
class TestTermFrequency:
    """Test term frequency computation."""
    
    def test_compute_tf_basic(self):
        """Test basic TF computation."""
        tokens = ["hello", "world", "hello", "test"]
        tf = compute_tf(tokens)
        
        assert tf["hello"] == 0.5  # 2/4
        assert tf["world"] == 0.25  # 1/4
        assert tf["test"] == 0.25  # 1/4
    
    def test_compute_tf_empty(self):
        """Test TF computation for empty token list."""
        assert compute_tf([]) == {}
    
    def test_compute_tf_single_token(self):
        """Test TF computation for single token."""
        tokens = ["hello"]
        tf = compute_tf(tokens)
        assert tf["hello"] == 1.0


@pytest.mark.unit
class TestInverseDocumentFrequency:
    """Test IDF computation."""
    
    def test_compute_idf_basic(self):
        """Test basic IDF computation."""
        documents = [
            ["hello", "world"],
            ["hello", "test"],
            ["world", "test"]
        ]
        idf = compute_idf(documents)
        
        # All terms appear in 2/3 documents
        assert "hello" in idf
        assert "world" in idf
        assert "test" in idf
        
        # IDF = log(N/df) = log(3/2) ≈ 0.405
        expected_idf = math.log(3/2)
        assert abs(idf["hello"] - expected_idf) < 0.001
    
    def test_compute_idf_rare_term(self):
        """Test IDF for term appearing in only one document."""
        documents = [
            ["hello", "rare"],
            ["hello", "world"],
            ["hello", "test"]
        ]
        idf = compute_idf(documents)
        
        # "hello" appears in all 3 docs: log(3/3) = 0
        assert idf["hello"] == 0.0
        
        # "rare" appears in 1 doc: log(3/1) = log(3) ≈ 1.099
        expected_rare_idf = math.log(3)
        assert abs(idf["rare"] - expected_rare_idf) < 0.001
    
    def test_compute_idf_empty(self):
        """Test IDF computation for empty document list."""
        assert compute_idf([]) == {}
    
    def test_compute_idf_empty_documents(self):
        """Test IDF computation with empty documents."""
        documents = [[], ["hello"], []]
        idf = compute_idf(documents)
        
        # "hello" appears in 1/3 documents
        expected_idf = math.log(3/1)
        assert abs(idf["hello"] - expected_idf) < 0.001


@pytest.mark.unit
class TestBM25Scoring:
    """Test BM25 scoring algorithm."""
    
    def test_bm25_score_basic(self):
        """Test basic BM25 scoring."""
        query = "hello world"
        document = "hello world test"
        
        score = bm25_score(query, document)
        assert score > 0  # Should have positive relevance
    
    def test_bm25_score_exact_match(self):
        """Test BM25 score for exact query match."""
        query = "hello world"
        document = "hello world"
        
        score = bm25_score(query, document)
        assert score > 0
    
    def test_bm25_score_no_match(self):
        """Test BM25 score when query terms don't appear in document."""
        query = "python java"
        document = "hello world test"
        
        score = bm25_score(query, document)
        assert score == 0.0
    
    def test_bm25_score_partial_match(self):
        """Test BM25 score with partial query match."""
        query = "hello python"  # only "hello" matches
        document = "hello world test"
        
        score1 = bm25_score(query, document)
        
        query = "hello world"  # both match
        score2 = bm25_score(query, document)
        
        assert score2 > score1  # more matches should score higher
    
    def test_bm25_score_with_corpus_stats(self):
        """Test BM25 scoring with provided corpus statistics."""
        documents = ["hello world", "hello test", "world test"]
        corpus_stats = prepare_corpus_stats(documents)
        
        query = "hello"
        document = "hello world"
        
        score = bm25_score(query, document, corpus_stats)
        assert score > 0
    
    def test_bm25_score_parameters(self):
        """Test BM25 scoring with different k1 and b parameters."""
        query = "hello world"
        document = "hello world test hello"
        
        # Default parameters
        score1 = bm25_score(query, document)
        
        # Higher k1 (less term frequency saturation)
        score2 = bm25_score(query, document, k1=2.0)
        
        # Lower k1 (more term frequency saturation)  
        score3 = bm25_score(query, document, k1=1.0)
        
        assert score1 > 0
        assert score2 > 0
        assert score3 > 0
    
    def test_bm25_score_empty_inputs(self):
        """Test BM25 scoring with empty inputs."""
        assert bm25_score("", "document") == 0.0
        assert bm25_score("query", "") == 0.0
        assert bm25_score("", "") == 0.0
        assert bm25_score(None, "document") == 0.0


@pytest.mark.unit
class TestCorpusStatistics:
    """Test corpus statistics preparation."""
    
    def test_prepare_corpus_stats_basic(self):
        """Test basic corpus statistics preparation."""
        documents = ["hello world", "hello test", "world python"]
        stats = prepare_corpus_stats(documents)
        
        assert "idf" in stats
        assert "avgdl" in stats
        assert "doc_count" in stats
        
        assert stats["doc_count"] == 3
        assert stats["avgdl"] == 2.0  # average document length
        
        # Check IDF values
        idf = stats["idf"]
        assert "hello" in idf
        assert "world" in idf
        assert "test" in idf
        assert "python" in idf
    
    def test_prepare_corpus_stats_empty(self):
        """Test corpus stats for empty document list."""
        stats = prepare_corpus_stats([])
        
        assert stats["idf"] == {}
        assert stats["avgdl"] == 0
        assert stats.get("doc_count", 0) == 0
    
    def test_prepare_corpus_stats_single_document(self):
        """Test corpus stats for single document."""
        documents = ["hello world test"]
        stats = prepare_corpus_stats(documents)
        
        assert stats["doc_count"] == 1
        assert stats["avgdl"] == 3  # 3 tokens
        
        # All terms appear in 1/1 document, so IDF = log(1/1) = 0
        idf = stats["idf"]
        for term in ["hello", "world", "test"]:
            assert idf[term] == 0.0


@pytest.mark.unit
class TestHybridSearch:
    """Test hybrid search functionality."""
    
    def test_hybrid_search_basic(self):
        """Test basic hybrid search."""
        query = "test api"
        vector_results = [
            {
                "memory": "test api endpoint functionality",
                "score": 0.8,
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "importance": 0.9,
                    "memory_type": "fact",
                    "access_count": 5
                }
            },
            {
                "memory": "unrelated content about dogs",
                "score": 0.6,
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z", 
                    "importance": 0.5,
                    "memory_type": "fact",
                    "access_count": 1
                }
            }
        ]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results, top_k=10)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # First result should have hybrid scoring fields
        first = results[0]
        assert "hybrid_score" in first
        assert "keyword_score" in first
        assert "raw_bm25" in first
        assert "recency_score" in first
        assert "importance_score" in first
    
    def test_hybrid_search_empty_results(self):
        """Test hybrid search with empty results."""
        results = hybrid_search("test", [], top_k=10)
        assert results == []
    
    def test_hybrid_search_custom_weights(self):
        """Test hybrid search with custom weights."""
        query = "test"
        vector_results = [{
            "memory": "test content",
            "score": 0.8,
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "importance": 0.8,
                "memory_type": "fact",
                "access_count": 0
            }
        }]
        
        custom_weights = {
            "w_semantic": 0.5,
            "w_keyword": 0.3,
            "w_recency": 0.1,
            "w_importance": 0.1
        }
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results, weights=custom_weights)
        
        assert len(results) == 1
        assert "hybrid_score" in results[0]
    
    def test_hybrid_search_lifecycle_status_weighting(self):
        """Test hybrid search applies lifecycle status weights."""
        query = "test"
        vector_results = [
            {
                "memory": "confirmed memory",
                "score": 0.7,
                "metadata": {
                    "status": "confirmed",
                    "created_at": "2024-01-01T00:00:00Z",
                    "importance": 0.8,
                    "memory_type": "fact",
                    "access_count": 0
                }
            },
            {
                "memory": "deleted memory", 
                "score": 0.8,
                "metadata": {
                    "status": "deleted",
                    "created_at": "2024-01-01T00:00:00Z",
                    "importance": 0.8,
                    "memory_type": "fact",
                    "access_count": 0
                }
            }
        ]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results, top_k=10)
        
        # Confirmed memory should rank higher than deleted (status_multiplier)
        confirmed_result = next(r for r in results if "confirmed" in r["memory"])
        deleted_result = next(r for r in results if "deleted" in r["memory"])
        
        assert confirmed_result["status_multiplier"] == STATUS_WEIGHTS["confirmed"]
        assert deleted_result["status_multiplier"] == STATUS_WEIGHTS["deleted"]
        assert confirmed_result["hybrid_score"] > deleted_result["hybrid_score"]
    
    def test_hybrid_search_top_k_limiting(self):
        """Test hybrid search respects top_k parameter."""
        query = "test"
        vector_results = [
            {"memory": f"test content {i}", "score": 0.8 - i*0.1, "metadata": {}}
            for i in range(10)
        ]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results, top_k=5)
        
        assert len(results) == 5
    
    def test_hybrid_search_sorting(self):
        """Test hybrid search sorts by hybrid_score descending."""
        query = "test"
        vector_results = [
            {
                "memory": "lower score content",
                "score": 0.6, 
                "metadata": {"importance": 0.5, "memory_type": "fact", "access_count": 0}
            },
            {
                "memory": "higher score test content", 
                "score": 0.9,
                "metadata": {"importance": 0.9, "memory_type": "decision", "access_count": 10}
            }
        ]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results, top_k=10)
        
        # Results should be sorted by hybrid_score descending
        for i in range(len(results) - 1):
            assert results[i]["hybrid_score"] >= results[i + 1]["hybrid_score"]


@pytest.mark.unit  
class TestBatchHybridSearch:
    """Test batch hybrid search functionality."""
    
    def test_batch_hybrid_search_basic(self):
        """Test basic batch hybrid search."""
        queries = ["test api", "python code"]
        vector_results_list = [
            [{"memory": "test api endpoint", "score": 0.8, "metadata": {}}],
            [{"memory": "python programming code", "score": 0.9, "metadata": {}}]
        ]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = batch_hybrid_search(queries, vector_results_list, top_k=10)
        
        assert len(results) == 2
        assert len(results[0]) == 1  # First query results
        assert len(results[1]) == 1  # Second query results
        
        # Each result should have hybrid scoring
        assert "hybrid_score" in results[0][0]
        assert "hybrid_score" in results[1][0]
    
    def test_batch_hybrid_search_mismatched_lengths(self):
        """Test batch hybrid search with mismatched input lengths."""
        queries = ["test"]
        vector_results_list = [[], []]  # 2 result lists, 1 query
        
        with pytest.raises(ValueError, match="must have same length"):
            batch_hybrid_search(queries, vector_results_list)
    
    def test_batch_hybrid_search_empty(self):
        """Test batch hybrid search with empty inputs."""
        results = batch_hybrid_search([], [])
        assert results == []


@pytest.mark.unit
class TestExplainHybridScore:
    """Test hybrid score explanation functionality."""
    
    def test_explain_hybrid_score_basic(self):
        """Test basic score explanation."""
        result = {
            "memory": "test content for explanation",
            "hybrid_score": 0.85,
            "score": 0.8,  # semantic
            "keyword_score": 0.7,
            "raw_bm25": 2.5,
            "recency_score": 0.9,
            "importance_score": 0.8,
            "status_multiplier": 1.2,
            "status": "confirmed",
            "metadata": {
                "memory_type": "fact",
                "agent_id": "test-agent",
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
        
        explanation = explain_hybrid_score(result, "test query")
        
        assert explanation["query"] == "test query"
        assert explanation["final_score"] == 0.85
        assert "test content" in explanation["content_preview"]
        
        # Check components
        components = explanation["components"]
        assert components["semantic"] == 0.8
        assert components["keyword_normalized"] == 0.7
        assert components["raw_bm25"] == 2.5
        
        # Check multipliers
        multipliers = explanation["multipliers"]
        assert multipliers["status"] == 1.2
        assert multipliers["status_name"] == "confirmed"
        
        # Check metadata
        metadata = explanation["metadata"]
        assert metadata["memory_type"] == "fact"
        assert metadata["agent_id"] == "test-agent"
    
    def test_explain_hybrid_score_missing_fields(self):
        """Test score explanation with missing fields."""
        result = {"memory": "test content"}  # minimal result
        
        explanation = explain_hybrid_score(result, "test")
        
        # Should handle missing fields gracefully
        assert explanation["final_score"] == 0  # default
        assert explanation["components"]["semantic"] == 0
        assert explanation["multipliers"]["status"] == 1.0
        assert explanation["metadata"]["memory_type"] == "unknown"


@pytest.mark.unit
class TestHybridSearchEdgeCases:
    """Test edge cases and error handling."""
    
    def test_hybrid_search_no_query(self):
        """Test hybrid search with empty query."""
        vector_results = [{"memory": "test", "score": 0.8, "metadata": {}}]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
            results = hybrid_search("", vector_results)
        
        # Should still return results
        assert len(results) == 1
    
    def test_hybrid_search_no_content(self):
        """Test hybrid search with results missing content."""
        query = "test"
        vector_results = [
            {"score": 0.8, "metadata": {}},  # no 'memory' or 'content' field
            {"memory": "", "score": 0.7, "metadata": {}},  # empty content
        ]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results)
        
        # Should handle gracefully
        assert len(results) == 2
        for result in results:
            assert "hybrid_score" in result
            assert "keyword_score" in result
    
    def test_status_weights_coverage(self):
        """Test all expected status values have weights."""
        expected_statuses = {
            "confirmed", "active", "outdated", "archived",
            "contradicted", "merged", "superseded", "deleted"
        }
        
        for status in expected_statuses:
            assert status in STATUS_WEIGHTS
            assert isinstance(STATUS_WEIGHTS[status], (int, float))
    
    def test_default_hybrid_weights_sum(self):
        """Test default hybrid weights are reasonable."""
        weights = DEFAULT_HYBRID_WEIGHTS
        
        # Check all expected weights exist
        required_weights = {"w_semantic", "w_keyword", "w_recency", "w_importance"}
        for weight_key in required_weights:
            assert weight_key in weights
            assert 0 <= weights[weight_key] <= 1
        
        # Sum should be 1.0 (or close to it)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_tokenize_performance(self):
        """Test tokenization performance doesn't break on large text."""
        # Large text (simulating real-world content)
        large_text = " ".join(["word" + str(i) for i in range(1000)])
        
        tokens = tokenize(large_text)
        
        # Should handle large text without errors
        assert len(tokens) == 1000
        assert all(token.startswith("word") for token in tokens)
    
    def test_bm25_normalization(self):
        """Test BM25 score normalization in hybrid search."""
        query = "test"
        
        # Create document with high BM25 score (repeated terms)
        high_bm25_doc = " ".join(["test"] * 100)  # 100 repetitions
        vector_results = [{
            "memory": high_bm25_doc,
            "score": 0.5,
            "metadata": {"importance": 0.5, "memory_type": "fact", "access_count": 0}
        }]
        
        with patch('memclawz.scoring.composite_score', return_value=0.8):
                results = hybrid_search(query, vector_results)
        
        # Keyword score should be normalized between 0 and 1
        keyword_score = results[0].get("keyword_score", 0)
        assert 0 <= keyword_score <= 1