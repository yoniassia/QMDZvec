"""Unit tests for memclawz.v7_extensions module."""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from memclawz.v7_extensions import (
    STATUS_WEIGHTS, apply_lifecycle_filter, apply_hybrid_scoring,
    get_lifecycle_manager, enhance_memory_with_lifecycle, get_v7_stats,
    process_search_v7, transition_memory_status, bulk_update_outdated
)
from memclawz.lifecycle import DEFAULT_STATUS


@pytest.mark.unit
class TestStatusWeights:
    """Test status weights export and values."""
    
    def test_status_weights_exported(self):
        """Test STATUS_WEIGHTS is properly exported."""
        assert STATUS_WEIGHTS is not None
        assert isinstance(STATUS_WEIGHTS, dict)
    
    def test_status_weights_coverage(self):
        """Test all expected statuses have weights."""
        expected_statuses = {
            "confirmed", "active", "outdated", "archived",
            "contradicted", "merged", "superseded", "deleted"
        }
        
        for status in expected_statuses:
            assert status in STATUS_WEIGHTS
            assert isinstance(STATUS_WEIGHTS[status], (int, float))
            assert STATUS_WEIGHTS[status] >= 0
    
    def test_status_weights_hierarchy(self):
        """Test status weights follow logical hierarchy."""
        # High confidence statuses should have higher weights
        assert STATUS_WEIGHTS["confirmed"] > STATUS_WEIGHTS["active"]
        assert STATUS_WEIGHTS["active"] > STATUS_WEIGHTS["outdated"]
        assert STATUS_WEIGHTS["active"] > STATUS_WEIGHTS["archived"]
        
        # Problematic statuses should have very low weights
        assert STATUS_WEIGHTS["contradicted"] < STATUS_WEIGHTS["active"]
        assert STATUS_WEIGHTS["superseded"] < STATUS_WEIGHTS["active"]
        assert STATUS_WEIGHTS["merged"] < STATUS_WEIGHTS["active"]
        
        # Deleted should have zero or near-zero weight
        assert STATUS_WEIGHTS["deleted"] <= 0.1


@pytest.mark.unit
class TestLifecycleFilter:
    """Test lifecycle filtering functionality."""
    
    def test_apply_lifecycle_filter_basic(self):
        """Test basic lifecycle filtering."""
        results = [
            {"metadata": {"status": "active"}, "memory": "active memory"},
            {"metadata": {"status": "deleted"}, "memory": "deleted memory"},
            {"metadata": {"status": "confirmed"}, "memory": "confirmed memory"},
            {"metadata": {"status": "contradicted"}, "memory": "contradicted memory"},
        ]
        
        filtered = apply_lifecycle_filter(results)
        
        # By default: include contradicted, exclude deleted and superseded
        assert len(filtered) == 3
        statuses = [r["metadata"]["status"] for r in filtered]
        assert "active" in statuses
        assert "confirmed" in statuses
        assert "contradicted" in statuses
        assert "deleted" not in statuses
    
    def test_apply_lifecycle_filter_include_deleted(self):
        """Test lifecycle filter with include_deleted=True."""
        results = [
            {"metadata": {"status": "active"}, "memory": "active memory"},
            {"metadata": {"status": "deleted"}, "memory": "deleted memory"},
        ]
        
        filtered = apply_lifecycle_filter(results, include_deleted=True)
        
        assert len(filtered) == 2
        statuses = [r["metadata"]["status"] for r in filtered]
        assert "deleted" in statuses
    
    def test_apply_lifecycle_filter_exclude_contradicted(self):
        """Test lifecycle filter with include_contradicted=False."""
        results = [
            {"metadata": {"status": "active"}, "memory": "active memory"},
            {"metadata": {"status": "contradicted"}, "memory": "contradicted memory"},
        ]
        
        filtered = apply_lifecycle_filter(results, include_contradicted=False)
        
        assert len(filtered) == 1
        assert filtered[0]["metadata"]["status"] == "active"
    
    def test_apply_lifecycle_filter_include_superseded(self):
        """Test lifecycle filter with include_superseded=True."""
        results = [
            {"metadata": {"status": "active"}, "memory": "active memory"},
            {"metadata": {"status": "superseded"}, "memory": "superseded memory"},
        ]
        
        filtered = apply_lifecycle_filter(results, include_superseded=True)
        
        assert len(filtered) == 2
        statuses = [r["metadata"]["status"] for r in filtered]
        assert "superseded" in statuses
    
    def test_apply_lifecycle_filter_payload_format(self):
        """Test lifecycle filter with payload format (Qdrant style)."""
        results = [
            {"payload": {"status": "active"}, "memory": "active memory"},
            {"payload": {"status": "deleted"}, "memory": "deleted memory"},
        ]
        
        filtered = apply_lifecycle_filter(results)
        
        assert len(filtered) == 1
        assert filtered[0]["payload"]["status"] == "active"
    
    def test_apply_lifecycle_filter_missing_status(self):
        """Test lifecycle filter with missing status (should use default)."""
        results = [
            {"metadata": {}, "memory": "no status memory"},
            {"metadata": {"status": "confirmed"}, "memory": "confirmed memory"},
        ]
        
        filtered = apply_lifecycle_filter(results)
        
        # Should include both (missing status treated as DEFAULT_STATUS)
        assert len(filtered) == 2
    
    def test_apply_lifecycle_filter_empty_results(self):
        """Test lifecycle filter with empty results."""
        filtered = apply_lifecycle_filter([])
        assert filtered == []


@pytest.mark.unit
class TestHybridScoring:
    """Test hybrid scoring integration."""
    
    @patch('memclawz.v7_extensions.hybrid_search')
    def test_apply_hybrid_scoring_basic(self, mock_hybrid_search):
        """Test basic hybrid scoring application."""
        query = "test query"
        results = [{"memory": "test content", "score": 0.8}]
        
        mock_hybrid_search.return_value = [
            {"memory": "test content", "score": 0.8, "hybrid_score": 0.85}
        ]
        
        scored = apply_hybrid_scoring(query, results)
        
        mock_hybrid_search.assert_called_once_with(
            query=query,
            vector_results=results,
            top_k=20,
            weights={'w_semantic': 0.4, 'w_keyword': 0.15, 'w_recency': 0.25, 'w_importance': 0.2}
        )
        
        assert len(scored) == 1
        assert scored[0]["hybrid_score"] == 0.85
    
    @patch('memclawz.v7_extensions.hybrid_search')
    def test_apply_hybrid_scoring_custom_weights(self, mock_hybrid_search):
        """Test hybrid scoring with custom weights."""
        query = "test"
        results = [{"memory": "content", "score": 0.7}]
        custom_weights = {"w_semantic": 0.6, "w_keyword": 0.4}
        
        mock_hybrid_search.return_value = results
        
        apply_hybrid_scoring(query, results, weights=custom_weights, top_k=10)
        
        mock_hybrid_search.assert_called_once_with(
            query=query,
            vector_results=results,
            top_k=10,
            weights=custom_weights
        )
    
    @patch('memclawz.v7_extensions.hybrid_search')
    def test_apply_hybrid_scoring_empty_query(self, mock_hybrid_search):
        """Test hybrid scoring with empty query."""
        results = [{"memory": "content", "score": 0.8}]
        
        scored = apply_hybrid_scoring("", results)
        
        # Should return original results (limited by top_k)
        assert scored == results
        mock_hybrid_search.assert_not_called()
    
    @patch('memclawz.v7_extensions.hybrid_search')
    def test_apply_hybrid_scoring_exception_fallback(self, mock_hybrid_search):
        """Test hybrid scoring falls back gracefully on exception."""
        query = "test"
        results = [{"memory": "content", "score": 0.8}]
        
        mock_hybrid_search.side_effect = Exception("Hybrid search failed")
        
        scored = apply_hybrid_scoring(query, results, top_k=5)
        
        # Should return original results (truncated to top_k)
        assert scored == results
    
    def test_apply_hybrid_scoring_empty_results(self):
        """Test hybrid scoring with empty results."""
        scored = apply_hybrid_scoring("test", [])
        assert scored == []


@pytest.mark.unit
class TestLifecycleManager:
    """Test lifecycle manager access."""
    
    @patch('memclawz.v7_extensions.MemoryLifecycle')
    def test_get_lifecycle_manager(self, mock_lifecycle_class):
        """Test getting lifecycle manager instance."""
        mock_instance = Mock()
        mock_lifecycle_class.return_value = mock_instance
        
        manager = get_lifecycle_manager()
        
        assert manager == mock_instance
        mock_lifecycle_class.assert_called_once()


@pytest.mark.unit
class TestMemoryEnhancement:
    """Test memory enhancement for lifecycle."""
    
    def test_enhance_memory_with_lifecycle_basic(self):
        """Test basic memory enhancement."""
        memory_data = {
            "memory": "test content",
            "agent_id": "test-agent"
        }
        
        enhanced = enhance_memory_with_lifecycle(memory_data)
        
        # Should add lifecycle fields
        assert "status" in enhanced
        assert enhanced["status"] == DEFAULT_STATUS
        assert "status_updated_at" in enhanced
        assert "lifecycle_metadata" in enhanced
        
        # Should preserve original fields
        assert enhanced["memory"] == "test content"
        assert enhanced["agent_id"] == "test-agent"
    
    def test_enhance_memory_with_lifecycle_existing_status(self):
        """Test memory enhancement preserves existing status."""
        memory_data = {
            "memory": "test content",
            "status": "confirmed"
        }
        
        enhanced = enhance_memory_with_lifecycle(memory_data)
        
        # Should preserve existing status
        assert enhanced["status"] == "confirmed"
        assert "status_updated_at" in enhanced
    
    def test_enhance_memory_with_lifecycle_existing_metadata(self):
        """Test memory enhancement preserves existing lifecycle metadata."""
        existing_metadata = {"custom_field": "value"}
        memory_data = {
            "memory": "test content",
            "lifecycle_metadata": existing_metadata
        }
        
        enhanced = enhance_memory_with_lifecycle(memory_data)
        
        # Should preserve existing metadata
        assert enhanced["lifecycle_metadata"] == existing_metadata
    
    def test_enhance_memory_with_lifecycle_timestamp_format(self):
        """Test memory enhancement creates valid ISO timestamp."""
        memory_data = {"memory": "test"}
        
        enhanced = enhance_memory_with_lifecycle(memory_data)
        
        timestamp = enhanced["status_updated_at"]
        
        # Should be valid ISO format
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime)
        assert parsed.tzinfo is not None


@pytest.mark.unit
class TestV7Stats:
    """Test v7 statistics generation."""
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_get_v7_stats_success(self, mock_get_manager):
        """Test successful v7 stats generation."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        mock_lifecycle_stats = {
            "active": 100,
            "confirmed": 50,
            "deleted": 5
        }
        mock_manager.get_lifecycle_stats.return_value = mock_lifecycle_stats
        
        stats = get_v7_stats()
        
        # Check structure
        assert stats["version"] == "7.0"
        assert "features" in stats
        assert "lifecycle_stats" in stats
        assert "memory_types" in stats
        assert "status_weights" in stats
        assert "hybrid_weights" in stats
        
        # Check features
        features = stats["features"]
        assert features["lifecycle"] is True
        assert features["hybrid_search"] is True
        assert features["expanded_types"] is True
        
        # Check lifecycle stats
        assert stats["lifecycle_stats"] == mock_lifecycle_stats
        
        # Check memory types
        memory_types = stats["memory_types"]
        assert "v6_types" in memory_types
        assert "v7_types" in memory_types
        assert "persistent_types" in memory_types
        
        # Check v6 types
        v6_types = memory_types["v6_types"]
        expected_v6 = ["decision", "preference", "relationship", "insight", "procedure", "fact", "event"]
        for t in expected_v6:
            assert t in v6_types
        
        # Check v7 types
        v7_types = memory_types["v7_types"]
        expected_v7 = ["intention", "plan", "commitment", "action", "outcome", "cancellation"]
        for t in expected_v7:
            assert t in v7_types
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_get_v7_stats_lifecycle_failure(self, mock_get_manager):
        """Test v7 stats when lifecycle manager fails."""
        mock_get_manager.side_effect = Exception("Lifecycle error")
        
        stats = get_v7_stats()
        
        # Should return error state
        assert stats["version"] == "7.0"
        assert stats["features"]["lifecycle"] is False
        assert stats["features"]["hybrid_search"] is True  # Pure Python, should work
        assert stats["features"]["expanded_types"] is True  # Just dict updates
        assert "error" in stats


@pytest.mark.unit
class TestProcessSearchV7:
    """Test integrated v7 search processing."""
    
    @patch('memclawz.v7_extensions.apply_lifecycle_filter')
    @patch('memclawz.v7_extensions.apply_hybrid_scoring')
    def test_process_search_v7_full_pipeline(self, mock_hybrid, mock_lifecycle):
        """Test full v7 search processing pipeline."""
        query = "test query"
        raw_results = [
            {"memory": "test content", "score": 0.8, "metadata": {"status": "active"}},
            {"memory": "deleted content", "score": 0.7, "metadata": {"status": "deleted"}}
        ]
        
        # Mock lifecycle filter removes deleted
        mock_lifecycle.return_value = [raw_results[0]]
        
        # Mock hybrid scoring enhances results
        mock_hybrid.return_value = [
            {"memory": "test content", "score": 0.8, "hybrid_score": 0.85, "metadata": {"status": "active"}}
        ]
        
        results = process_search_v7(query, raw_results, top_k=10)
        
        # Check both stages were called
        mock_lifecycle.assert_called_once_with(
            raw_results,
            include_deleted=False,
            include_contradicted=True,
            include_superseded=False
        )
        
        mock_hybrid.assert_called_once_with(
            query=query,
            results=[raw_results[0]],  # After lifecycle filtering
            weights=None,
            top_k=10
        )
        
        # Check final result
        assert len(results) == 1
        assert results[0]["hybrid_score"] == 0.85
    
    @patch('memclawz.v7_extensions.apply_lifecycle_filter')
    @patch('memclawz.v7_extensions.apply_hybrid_scoring')
    def test_process_search_v7_lifecycle_disabled(self, mock_hybrid, mock_lifecycle):
        """Test v7 processing with lifecycle disabled."""
        query = "test"
        raw_results = [{"memory": "test", "score": 0.8}]
        
        mock_hybrid.return_value = raw_results
        
        results = process_search_v7(
            query, raw_results, 
            apply_lifecycle=False,
            apply_hybrid=True
        )
        
        # Lifecycle filter should not be called
        mock_lifecycle.assert_not_called()
        
        # Hybrid scoring should still be called
        mock_hybrid.assert_called_once_with(
            query=query,
            results=raw_results,
            weights=None,
            top_k=20
        )
    
    @patch('memclawz.v7_extensions.apply_lifecycle_filter')
    @patch('memclawz.v7_extensions.apply_hybrid_scoring')
    def test_process_search_v7_hybrid_disabled(self, mock_hybrid, mock_lifecycle):
        """Test v7 processing with hybrid scoring disabled."""
        query = "test"
        raw_results = [{"memory": "test", "score": 0.8}]
        
        mock_lifecycle.return_value = raw_results
        
        results = process_search_v7(
            query, raw_results,
            apply_lifecycle=True,
            apply_hybrid=False,
            top_k=5
        )
        
        # Lifecycle filter should be called
        mock_lifecycle.assert_called_once()
        
        # Hybrid scoring should not be called
        mock_hybrid.assert_not_called()
        
        # Should just truncate results
        assert len(results) <= 5
    
    def test_process_search_v7_empty_query_no_hybrid(self):
        """Test v7 processing with empty query skips hybrid scoring."""
        raw_results = [{"memory": "test", "score": 0.8}]
        
        with patch('memclawz.v7_extensions.apply_lifecycle_filter', return_value=raw_results):
            results = process_search_v7("", raw_results, apply_hybrid=True)
        
        # Should return results without hybrid scoring
        assert results == raw_results


@pytest.mark.unit  
class TestTransitionMemoryStatus:
    """Test memory status transition wrapper."""
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_transition_memory_status_success(self, mock_get_manager):
        """Test successful memory status transition."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.transition.return_value = True
        
        result = transition_memory_status("test-id", "active", "confirmed")
        
        assert result["status"] == "success"
        assert result["memory_id"] == "test-id"
        assert result["from_status"] == "active"
        assert result["to_status"] == "confirmed"
        assert "Transitioned" in result["message"]
        
        mock_manager.transition.assert_called_once_with("test-id", "active", "confirmed")
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_transition_memory_status_failure(self, mock_get_manager):
        """Test failed memory status transition."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.transition.return_value = False
        
        result = transition_memory_status("test-id", "active", "confirmed")
        
        assert result["status"] == "error"
        assert result["memory_id"] == "test-id"
        assert "Failed to transition" in result["message"]
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_transition_memory_status_exception(self, mock_get_manager):
        """Test memory status transition with exception."""
        mock_get_manager.side_effect = Exception("Database error")
        
        result = transition_memory_status("test-id", "active", "confirmed")
        
        assert result["status"] == "error"
        assert result["memory_id"] == "test-id"
        assert "Exception during transition" in result["message"]
        assert "error" in result


@pytest.mark.unit
class TestBulkUpdateOutdated:
    """Test bulk outdated update functionality."""
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_bulk_update_outdated_success(self, mock_get_manager):
        """Test successful bulk outdated update."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        # Mock candidates found
        mock_candidates = [
            {"id": "old-1", "current_status": "active"},
            {"id": "old-2", "current_status": "active"},
            {"id": "old-3", "current_status": "confirmed"},  # Won't be updated
        ]
        mock_manager.bulk_check_outdated.return_value = mock_candidates
        
        # Mock successful transitions for active memories
        mock_manager.transition.side_effect = lambda mid, from_s, to_s: from_s == "active"
        
        result = bulk_update_outdated(threshold_days=30)
        
        assert result["status"] == "success"
        assert result["candidates_found"] == 3
        assert result["updated_count"] == 2  # Only active memories updated
        assert result["threshold_days"] == 30
        assert len(result["errors"]) == 0
        
        # Check transition calls
        assert mock_manager.transition.call_count == 2
        mock_manager.transition.assert_any_call("old-1", "active", "outdated")
        mock_manager.transition.assert_any_call("old-2", "active", "outdated")
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_bulk_update_outdated_with_failures(self, mock_get_manager):
        """Test bulk outdated update with some transition failures."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        mock_candidates = [
            {"id": "old-1", "current_status": "active"},
            {"id": "old-2", "current_status": "active"},
        ]
        mock_manager.bulk_check_outdated.return_value = mock_candidates
        
        # Mock first transition succeeds, second fails
        mock_manager.transition.side_effect = [True, False]
        
        result = bulk_update_outdated()
        
        assert result["status"] == "success"
        assert result["candidates_found"] == 2
        assert result["updated_count"] == 1
        assert len(result["errors"]) == 1
        assert "Failed to update old-2" in result["errors"][0]
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_bulk_update_outdated_no_candidates(self, mock_get_manager):
        """Test bulk outdated update with no candidates."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.bulk_check_outdated.return_value = []
        
        result = bulk_update_outdated(threshold_days=60)
        
        assert result["status"] == "success"
        assert result["candidates_found"] == 0
        assert result["updated_count"] == 0
        assert result["threshold_days"] == 60
    
    @patch('memclawz.v7_extensions.get_lifecycle_manager')
    def test_bulk_update_outdated_exception(self, mock_get_manager):
        """Test bulk outdated update with exception."""
        mock_get_manager.side_effect = Exception("Database error")
        
        result = bulk_update_outdated()
        
        assert result["status"] == "error"
        assert "Bulk update failed" in result["message"]
        assert "error" in result


@pytest.mark.unit
class TestV7ExtensionsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_apply_lifecycle_filter_mixed_metadata_formats(self):
        """Test lifecycle filter handles mixed metadata formats."""
        results = [
            {"metadata": {"status": "active"}, "memory": "metadata format"},
            {"payload": {"status": "confirmed"}, "memory": "payload format"},
            {"memory": "no status format"},  # No metadata or payload
        ]
        
        filtered = apply_lifecycle_filter(results)
        
        # Should handle all formats
        assert len(filtered) == 3
    
    def test_enhance_memory_with_lifecycle_empty_input(self):
        """Test memory enhancement with minimal input."""
        enhanced = enhance_memory_with_lifecycle({})
        
        # Should add lifecycle fields to empty dict
        assert "status" in enhanced
        assert "status_updated_at" in enhanced
        assert "lifecycle_metadata" in enhanced
    
    def test_process_search_v7_all_disabled(self):
        """Test v7 processing with all enhancements disabled."""
        raw_results = [{"memory": "test", "score": 0.8}] * 10
        
        results = process_search_v7(
            "test", raw_results,
            apply_lifecycle=False,
            apply_hybrid=False,
            top_k=5
        )
        
        # Should just truncate results
        assert len(results) == 5
        assert results == raw_results[:5]
    
    def test_v7_extensions_imports(self):
        """Test v7 extensions module imports work correctly."""
        from memclawz.v7_extensions import (
            STATUS_WEIGHTS, apply_lifecycle_filter, apply_hybrid_scoring,
            get_lifecycle_manager, enhance_memory_with_lifecycle, get_v7_stats
        )
        
        # All imports should be available
        assert STATUS_WEIGHTS is not None
        assert callable(apply_lifecycle_filter)
        assert callable(apply_hybrid_scoring)
        assert callable(get_lifecycle_manager)
        assert callable(enhance_memory_with_lifecycle)
        assert callable(get_v7_stats)
    
    def test_status_weights_consistency_with_hybrid_search(self):
        """Test STATUS_WEIGHTS is consistent with hybrid_search module."""
        # Import from hybrid_search to compare
        from memclawz.hybrid_search import STATUS_WEIGHTS as hs_status_weights
        
        # Should be the same object or have same values
        assert STATUS_WEIGHTS == hs_status_weights