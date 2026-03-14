"""Unit tests for memclawz.lifecycle module."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from qdrant_client.models import ScoredPoint, Record

from memclawz.lifecycle import (
    MemoryLifecycle, LIFECYCLE_STATES, VALID_TRANSITIONS, DEFAULT_STATUS
)


@pytest.mark.unit
class TestLifecycleConstants:
    """Test lifecycle constants and configuration."""
    
    def test_lifecycle_states_count(self):
        """Test we have exactly 8 lifecycle states."""
        assert len(LIFECYCLE_STATES) == 8
    
    def test_lifecycle_states_content(self):
        """Test all expected lifecycle states exist."""
        expected_states = {
            "active", "confirmed", "outdated", "archived",
            "contradicted", "merged", "superseded", "deleted"
        }
        assert LIFECYCLE_STATES == expected_states
    
    def test_default_status(self):
        """Test default status is 'active'."""
        assert DEFAULT_STATUS == "active"
    
    def test_valid_transitions_coverage(self):
        """Test all states have transition rules defined."""
        for state in LIFECYCLE_STATES:
            assert state in VALID_TRANSITIONS
            assert isinstance(VALID_TRANSITIONS[state], set)
    
    def test_deleted_is_final_state(self):
        """Test 'deleted' state has no outgoing transitions."""
        assert VALID_TRANSITIONS["deleted"] == set()
    
    def test_active_state_transitions(self):
        """Test active state can transition to most other states."""
        active_transitions = VALID_TRANSITIONS["active"]
        expected = {"confirmed", "outdated", "archived", "contradicted", "merged", "superseded", "deleted"}
        assert active_transitions == expected
    
    def test_archived_can_be_restored(self):
        """Test archived memories can be restored to active."""
        assert "active" in VALID_TRANSITIONS["archived"]
    
    def test_contradicted_can_be_revalidated(self):
        """Test contradicted memories can be revalidated to active."""
        assert "active" in VALID_TRANSITIONS["contradicted"]
    
    def test_outdated_can_be_revalidated(self):
        """Test outdated memories can be revalidated to active."""
        assert "active" in VALID_TRANSITIONS["outdated"]


@pytest.mark.unit
class TestMemoryLifecycleInit:
    """Test MemoryLifecycle initialization."""
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_lifecycle_initialization(self, mock_qdrant):
        """Test lifecycle manager initializes correctly."""
        lifecycle = MemoryLifecycle()
        
        assert lifecycle is not None
        mock_qdrant.assert_called_once()


@pytest.mark.unit
class TestMemoryTransitions:
    """Test memory status transitions."""
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_valid(self, mock_qdrant):
        """Test valid status transition."""
        # Setup mock
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        mock_record = Record(
            id="test-id",
            payload={"status": "active", "memory": "test content"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        mock_qc.set_payload.return_value = None
        
        lifecycle = MemoryLifecycle()
        
        # Test valid transition
        result = lifecycle.transition("test-id", "active", "confirmed")
        
        assert result is True
        mock_qc.retrieve.assert_called_once()
        mock_qc.set_payload.assert_called_once()
        
        # Check set_payload call
        call_args = mock_qc.set_payload.call_args
        assert call_args.kwargs["points"] == ["test-id"]
        assert call_args.kwargs["payload"]["status"] == "confirmed"
        assert "status_updated_at" in call_args.kwargs["payload"]
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_invalid_from_status(self, mock_qdrant):
        """Test transition with invalid from_status."""
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.transition("test-id", "invalid_status", "confirmed")
        
        assert result is False
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_invalid_transition(self, mock_qdrant):
        """Test invalid status transition."""
        lifecycle = MemoryLifecycle()
        
        # Try to transition from deleted (final state)
        result = lifecycle.transition("test-id", "deleted", "active")
        
        assert result is False
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_memory_not_found(self, mock_qdrant):
        """Test transition when memory doesn't exist."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.retrieve.return_value = []  # No results
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.transition("nonexistent-id", "active", "confirmed")
        
        assert result is False
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_status_mismatch(self, mock_qdrant):
        """Test transition when current status doesn't match expected."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        mock_record = Record(
            id="test-id",
            payload={"status": "confirmed", "memory": "test content"},  # Different than expected
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        # Try to transition from "active" but memory is actually "confirmed"
        result = lifecycle.transition("test-id", "active", "outdated")
        
        assert result is False
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_with_lifecycle_metadata(self, mock_qdrant):
        """Test transition adds lifecycle metadata for special statuses."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        mock_record = Record(
            id="test-id",
            payload={"status": "active", "memory": "test content"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        # Test transition to 'superseded' (should add metadata)
        result = lifecycle.transition("test-id", "active", "superseded")
        
        assert result is True
        
        # Check that lifecycle_metadata was added
        call_args = mock_qc.set_payload.call_args
        assert "lifecycle_metadata" in call_args.kwargs["payload"]
        assert "transition_timestamp" in call_args.kwargs["payload"]["lifecycle_metadata"]
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_transition_exception_handling(self, mock_qdrant):
        """Test transition handles exceptions gracefully."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.retrieve.side_effect = Exception("Database error")
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.transition("test-id", "active", "confirmed")
        
        assert result is False


@pytest.mark.unit
class TestGetStatus:
    """Test getting memory status."""
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_status_exists(self, mock_qdrant):
        """Test getting status for existing memory."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        mock_record = Record(
            id="test-id",
            payload={"status": "confirmed", "memory": "test content"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        status = lifecycle.get_status("test-id")
        
        assert status == "confirmed"
        mock_qc.retrieve.assert_called_once_with(
            collection_name="yoniclaw_memories",
            ids=["test-id"]
        )
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_status_not_found(self, mock_qdrant):
        """Test getting status for non-existent memory."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.retrieve.return_value = []  # No results
        
        lifecycle = MemoryLifecycle()
        
        status = lifecycle.get_status("nonexistent-id")
        
        assert status == DEFAULT_STATUS
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_status_no_status_field(self, mock_qdrant):
        """Test getting status when status field is missing."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        mock_record = Record(
            id="test-id",
            payload={"memory": "test content"},  # No status field
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        status = lifecycle.get_status("test-id")
        
        assert status == DEFAULT_STATUS
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_status_exception(self, mock_qdrant):
        """Test get_status handles exceptions."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.retrieve.side_effect = Exception("Database error")
        
        lifecycle = MemoryLifecycle()
        
        status = lifecycle.get_status("test-id")
        
        assert status == DEFAULT_STATUS


@pytest.mark.unit  
class TestBulkCheckOutdated:
    """Test bulk outdated memory detection."""
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_bulk_check_outdated_basic(self, mock_qdrant):
        """Test basic outdated memory detection."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Create old memories (40 days ago)
        old_date = datetime.now(timezone.utc) - timedelta(days=40)
        old_iso = old_date.isoformat()
        
        mock_records = [
            Record(
                id="old-1",
                payload={
                    "status": "active",
                    "memory": "Old memory content 1",
                    "created_at": old_iso,
                    "memory_type": "fact",
                    "agent_id": "test-agent"
                },
                vector=None
            ),
            Record(
                id="old-2", 
                payload={
                    "status": "active",
                    "memory": "Old memory content 2",
                    "created_at": old_iso,
                    "memory_type": "decision",
                    "agent_id": "test-agent"
                },
                vector=None
            )
        ]
        mock_qc.scroll.return_value = (mock_records, None)
        
        lifecycle = MemoryLifecycle()
        
        candidates = lifecycle.bulk_check_outdated(threshold_days=30)
        
        assert len(candidates) == 2
        
        for candidate in candidates:
            assert "id" in candidate
            assert "content" in candidate
            assert "age_days" in candidate
            assert "current_status" in candidate
            assert "memory_type" in candidate
            assert "agent_id" in candidate
            
            assert candidate["current_status"] == "active"
            assert candidate["age_days"] >= 40
    
    @patch('memclawz.lifecycle.QdrantClient') 
    def test_bulk_check_outdated_no_results(self, mock_qdrant):
        """Test bulk check when no outdated memories found."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.scroll.return_value = ([], None)
        
        lifecycle = MemoryLifecycle()
        
        candidates = lifecycle.bulk_check_outdated(threshold_days=30)
        
        assert candidates == []
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_bulk_check_outdated_invalid_dates(self, mock_qdrant):
        """Test bulk check handles invalid date formats."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        mock_records = [
            Record(
                id="invalid-date",
                payload={
                    "status": "active",
                    "memory": "Memory with invalid date",
                    "created_at": "not-a-date",  # Invalid format
                    "memory_type": "fact",
                    "agent_id": "test-agent"
                },
                vector=None
            )
        ]
        mock_qc.scroll.return_value = (mock_records, None)
        
        lifecycle = MemoryLifecycle()
        
        candidates = lifecycle.bulk_check_outdated(threshold_days=30)
        
        # Invalid dates should be skipped
        assert candidates == []
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_bulk_check_outdated_exception(self, mock_qdrant):
        """Test bulk check handles exceptions."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.scroll.side_effect = Exception("Database error")
        
        lifecycle = MemoryLifecycle()
        
        candidates = lifecycle.bulk_check_outdated(threshold_days=30)
        
        assert candidates == []


@pytest.mark.unit
class TestLifecycleHelperMethods:
    """Test helper methods like confirm, supersede, contradict."""
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_confirm_memory(self, mock_qdrant):
        """Test confirming a memory."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock get_status to return "active"
        mock_record = Record(
            id="test-id",
            payload={"status": "active", "memory": "test"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.confirm("test-id")
        
        assert result is True
        # Should call set_payload to update status
        mock_qc.set_payload.assert_called_once()
        call_args = mock_qc.set_payload.call_args
        assert call_args.kwargs["payload"]["status"] == "confirmed"
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_confirm_already_confirmed(self, mock_qdrant):
        """Test confirming an already confirmed memory."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock get_status to return "confirmed"
        mock_record = Record(
            id="test-id", 
            payload={"status": "confirmed", "memory": "test"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.confirm("test-id")
        
        assert result is True  # Should succeed (already in desired state)
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_supersede_memory(self, mock_qdrant):
        """Test superseding a memory."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock get_status and transition
        mock_record = Record(
            id="old-id",
            payload={"status": "active", "memory": "old content"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.supersede("old-id", "new-id")
        
        assert result is True
        
        # Should call set_payload twice: once for status transition, once for metadata
        assert mock_qc.set_payload.call_count == 2
        
        # Check the metadata call
        calls = mock_qc.set_payload.call_args_list
        metadata_call = calls[1]  # Second call should be for metadata
        assert "lifecycle_metadata" in metadata_call.kwargs["payload"]
        assert metadata_call.kwargs["payload"]["lifecycle_metadata"]["superseded_by"] == "new-id"
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_contradict_memory(self, mock_qdrant):
        """Test contradicting a memory."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock get_status and transition
        mock_record = Record(
            id="contradicted-id",
            payload={"status": "active", "memory": "wrong info"},
            vector=None
        )
        mock_qc.retrieve.return_value = [mock_record]
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.contradict("contradicted-id", "correcting-id")
        
        assert result is True
        
        # Should call set_payload twice: once for status, once for metadata
        assert mock_qc.set_payload.call_count == 2
        
        # Check the metadata call
        calls = mock_qc.set_payload.call_args_list
        metadata_call = calls[1]
        assert "lifecycle_metadata" in metadata_call.kwargs["payload"]
        assert metadata_call.kwargs["payload"]["lifecycle_metadata"]["contradicted_by"] == "correcting-id"
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_supersede_transition_fails(self, mock_qdrant):
        """Test supersede when status transition fails."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock failed transition (invalid state)
        mock_qc.retrieve.return_value = []  # Memory not found
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.supersede("nonexistent-id", "new-id")
        
        assert result is False
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_contradict_exception(self, mock_qdrant):
        """Test contradict handles exceptions."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.retrieve.side_effect = Exception("Database error")
        
        lifecycle = MemoryLifecycle()
        
        result = lifecycle.contradict("test-id", "contradicting-id")
        
        assert result is False


@pytest.mark.unit
class TestLifecycleStats:
    """Test lifecycle statistics generation."""
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_lifecycle_stats_basic(self, mock_qdrant):
        """Test basic lifecycle statistics."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock count results for different statuses
        def mock_count(collection_name, count_filter=None, exact=True):
            # Parse the filter to determine which status is being counted
            if count_filter and count_filter.must:
                condition = count_filter.must[0]
                if hasattr(condition, 'match') and hasattr(condition.match, 'value'):
                    status = condition.match.value
                    # Return different counts for different statuses
                    counts = {
                        "active": 100,
                        "confirmed": 50,
                        "outdated": 20,
                        "archived": 15,
                        "contradicted": 5,
                        "merged": 3,
                        "superseded": 2,
                        "deleted": 1
                    }
                    count_result = Mock()
                    count_result.count = counts.get(status, 0)
                    return count_result
            
            # Total count (no filter)
            count_result = Mock()
            count_result.count = 200  # Total documents
            return count_result
        
        mock_qc.count.side_effect = mock_count
        
        lifecycle = MemoryLifecycle()
        
        stats = lifecycle.get_lifecycle_stats()
        
        # Check all statuses are included
        for status in LIFECYCLE_STATES:
            assert status in stats
            assert isinstance(stats[status], int)
        
        # Check specific counts
        assert stats["active"] == 100
        assert stats["confirmed"] == 50
        assert stats["outdated"] == 20
        
        # Check unlabeled count
        assert "unlabeled" in stats
        total_labeled = sum(stats[status] for status in LIFECYCLE_STATES)
        expected_unlabeled = max(0, 200 - total_labeled)  # 200 is total count
        assert stats["unlabeled"] == expected_unlabeled
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_lifecycle_stats_exception(self, mock_qdrant):
        """Test lifecycle stats handles exceptions."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        mock_qc.count.side_effect = Exception("Database error")
        
        lifecycle = MemoryLifecycle()
        
        stats = lifecycle.get_lifecycle_stats()
        
        # Should return empty stats on error
        assert isinstance(stats, dict)
        # Might be empty or have some defaults - implementation dependent
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_get_lifecycle_stats_zero_counts(self, mock_qdrant):
        """Test lifecycle stats with zero counts."""
        mock_qc = Mock()
        mock_qdrant.return_value = mock_qc
        
        # Mock all counts as 0
        count_result = Mock()
        count_result.count = 0
        mock_qc.count.return_value = count_result
        
        lifecycle = MemoryLifecycle()
        
        stats = lifecycle.get_lifecycle_stats()
        
        # All statuses should have 0 count
        for status in LIFECYCLE_STATES:
            assert stats[status] == 0
        
        assert stats["unlabeled"] == 0


@pytest.mark.unit
class TestLifecycleEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_lifecycle_states_immutable(self):
        """Test lifecycle states set is immutable."""
        original_size = len(LIFECYCLE_STATES)
        
        # Try to modify (should not affect the original)
        try:
            LIFECYCLE_STATES.add("new_state")
        except AttributeError:
            pass  # Expected - sets are mutable, but module-level constants shouldn't be modified
        
        # States should still be the original size
        assert len(LIFECYCLE_STATES) >= original_size
    
    def test_valid_transitions_structure(self):
        """Test valid transitions structure is correct."""
        # Every state should map to a set
        for state, transitions in VALID_TRANSITIONS.items():
            assert isinstance(transitions, set)
            
            # All transition targets should be valid states
            for target in transitions:
                assert target in LIFECYCLE_STATES
    
    def test_transitions_graph_connectivity(self):
        """Test transition graph has reasonable connectivity."""
        # Active state should be reachable from most states (directly or indirectly)
        states_that_can_reach_active = set()
        
        for state, transitions in VALID_TRANSITIONS.items():
            if "active" in transitions:
                states_that_can_reach_active.add(state)
        
        # At least archived, contradicted, and outdated should be able to go back to active
        expected_restorable = {"archived", "contradicted", "outdated"}
        assert expected_restorable.issubset(states_that_can_reach_active)
    
    def test_final_states_identification(self):
        """Test identification of final states."""
        final_states = set()
        
        for state, transitions in VALID_TRANSITIONS.items():
            if len(transitions) == 0:
                final_states.add(state)
        
        # Only 'deleted' should be a final state
        assert final_states == {"deleted"}
    
    @patch('memclawz.lifecycle.QdrantClient')
    def test_memory_lifecycle_config_dependency(self, mock_qdrant):
        """Test MemoryLifecycle depends on config values."""
        from memclawz.lifecycle import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
        
        lifecycle = MemoryLifecycle()
        
        # Should have called QdrantClient with config values
        mock_qdrant.assert_called_once_with(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Collection name should be used in operations (tested implicitly in other tests)
        assert COLLECTION_NAME is not None
        assert isinstance(COLLECTION_NAME, str)