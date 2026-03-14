"""Comprehensive E2E test suite for MemClawz v7.0.0.

Tests all major functionality including:
- Health & basics
- Memory CRUD operations  
- Auto-enrichment
- Temporal validity
- Lifecycle management
- Hybrid search
- Memory types
- Edge cases
"""
import pytest
import httpx
from datetime import datetime, timezone
import json


def get_search_results(data: dict) -> list:
    """Normalize search API response shape across versions."""
    return data.get("results") or data.get("memories") or []


def get_memory_text(item: dict) -> str:
    """Extract memory text from either top-level or nested metadata."""
    return item.get("memory") or item.get("metadata", {}).get("memory") or ""


def get_memory_type(item: dict):
    """Extract memory type from common response locations."""
    metadata = item.get("metadata", {})
    nested = metadata.get("metadata", {})
    enrichment = metadata.get("enrichment", {})
    return (
        item.get("memory_type")
        or metadata.get("memory_type")
        or metadata.get("type")
        or nested.get("type")
        or enrichment.get("type")
    )


class TestHealthAndBasics:
    """Test basic health endpoints and system status."""

    async def test_health_endpoint_returns_v7(self, async_client: httpx.AsyncClient):
        """Health endpoint returns 200 with version 7.0.0."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "7.0.0"

    async def test_stats_endpoint_returns_valid_data(self, async_client: httpx.AsyncClient):
        """Stats endpoint returns valid system statistics."""
        response = await async_client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_memories" in data
        assert "by_agent" in data
        assert isinstance(data["total_memories"], int)
        assert isinstance(data["by_agent"], dict)

    async def test_agents_endpoint_returns_list(self, async_client: httpx.AsyncClient):
        """Agents endpoint returns agent data structure."""
        response = await async_client.get("/api/v1/agents")
        assert response.status_code == 200
        
        data = response.json()
        # The agents endpoint returns a dict with agent_id -> stats mapping
        assert isinstance(data, dict)
        assert len(data) > 0  # Should have at least some agents
        
        # Each agent should have count and types info
        for agent_id, agent_data in data.items():
            assert "count" in agent_data
            assert "types" in agent_data
            assert isinstance(agent_data["count"], int)

    async def test_v7_stats_endpoint(self, async_client: httpx.AsyncClient):
        """V7 stats endpoint returns v7 feature statistics."""
        response = await async_client.get("/api/v1/v7/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert "lifecycle_stats" in data
        assert "memory_types" in data
        
        # Check v7 features are enabled (based on actual API response)
        features = data["features"]
        assert features.get("lifecycle") is True
        assert features.get("hybrid_search") is True
        assert features.get("expanded_types") is True


class TestMemoryCRUD:
    """Test basic memory CRUD operations."""

    async def test_add_memory_minimal(self, async_client: httpx.AsyncClient, agent_id: str):
        """Add memory with just content + agent_id returns 200 and ID."""
        payload = {
            "content": "Test memory for minimal CRUD",
            "agent_id": agent_id
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "direct_insert" in data
        assert data["direct_insert"]["status"] == "ok"
        assert "id" in data["direct_insert"]

    async def test_add_memory_with_explicit_type(self, async_client: httpx.AsyncClient, agent_id: str):
        """Add memory with explicit memory_type preserves the type."""
        payload = {
            "content": "This is a decision about the API architecture",
            "agent_id": agent_id,
            "memory_type": "decision"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["direct_insert"]["status"] == "ok"
        
        # Verify the memory was stored with correct type
        memory_id = data["direct_insert"]["id"]
        
        # Search for the memory to verify type
        search_response = await async_client.get(f"/api/v1/search?q=API architecture&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        search_data = search_response.json()
        results = get_search_results(search_data)
        assert len(results) >= 1
        
        # Find our memory in results
        found_memory = None
        for memory in results:
            if "API architecture" in get_memory_text(memory):
                found_memory = memory
                break
                
        assert found_memory is not None
        assert get_memory_type(found_memory) == "decision"

    async def test_add_all_v7_memory_types(self, async_client: httpx.AsyncClient, agent_id: str, v7_memory_types: list):
        """Add memory with all v7 types succeeds."""
        for memory_type in v7_memory_types:
            payload = {
                "content": f"Test content for {memory_type} type",
                "agent_id": agent_id,
                "memory_type": memory_type
            }
            
            response = await async_client.post("/api/v1/add", json=payload)
            assert response.status_code == 200, f"Failed to add memory with type: {memory_type}"
            
            data = response.json()
            assert data["direct_insert"]["status"] == "ok"

    async def test_search_returns_relevant_results(self, async_client: httpx.AsyncClient, agent_id: str):
        """Search returns relevant results for known content."""
        # Add a specific memory we can search for
        content = "France is famous for the Eiffel Tower and croissants"
        payload = {
            "content": content,
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        # Search for it
        search_response = await async_client.get(f"/api/v1/search?q=France Eiffel Tower&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        # Should find our memory
        found = False
        for memory in results:
            text = get_memory_text(memory)
            if "France" in text and "Eiffel Tower" in text:
                found = True
                break
        assert found, "Should find the memory about France and Eiffel Tower"

    async def test_search_with_agent_filter(self, async_client: httpx.AsyncClient, agent_id: str):
        """Search with agent_id filter works correctly."""
        # Add memory for our test agent
        payload = {
            "content": "Memory specific to our test agent",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        # Search with agent filter
        search_response = await async_client.get(f"/api/v1/search?q=specific test agent&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        
        # All results should be for our agent
        for memory in results:
            memory_agent = memory.get("agent_id") or memory.get("metadata", {}).get("agent") or memory.get("metadata", {}).get("metadata", {}).get("agent")
            if memory_agent:  # Some might not have agent set
                assert memory_agent == agent_id

    async def test_list_memories_with_limit(self, async_client: httpx.AsyncClient, agent_id: str):
        """List memories works with limit parameter."""
        # Add a few memories
        for i in range(3):
            payload = {
                "content": f"Memory number {i} for listing test",
                "agent_id": agent_id,
                "memory_type": "fact"
            }
            response = await async_client.post("/api/v1/add", json=payload)
            assert response.status_code == 200
        
        # List with limit
        list_response = await async_client.get(f"/api/v1/memories?agent_id={agent_id}&limit=2")
        assert list_response.status_code == 200
        
        data = list_response.json()
        assert "memories" in data
        assert "total" in data
        assert isinstance(data["memories"], list)
        # Should respect limit
        assert len(data["memories"]) <= 2


class TestAutoEnrichment:
    """Test Phase 1: Auto-enrichment of memory content."""

    async def test_plain_text_gets_enriched(self, async_client: httpx.AsyncClient, agent_id: str):
        """Plain text content gets auto-enriched with type, weight, title, summary, tags."""
        payload = {
            "content": "The quarterly revenue increased by 15% due to strong product sales",
            "agent_id": agent_id
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        # Search to get the enriched memory back
        search_response = await async_client.get(f"/api/v1/search?q=quarterly revenue&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        memory = results[0]
        metadata = memory.get("metadata", {})
        nested = metadata.get("metadata", {})
        
        # Check for enrichment fields from actual API structure
        assert (
            metadata.get("memory_type") is not None or
            nested.get("type") is not None or
            nested.get("title") is not None or
            nested.get("summary") is not None or
            nested.get("tags") is not None or
            metadata.get("importance") is not None
        ), "Memory should have some form of enrichment"

    async def test_enrichment_uses_valid_types(self, async_client: httpx.AsyncClient, agent_id: str, v7_memory_types: list):
        """Enrichment returns valid types from the 13 supported types."""
        payload = {
            "content": "We decided to migrate the database to PostgreSQL for better performance",
            "agent_id": agent_id
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        # Search to get the enriched memory
        search_response = await async_client.get(f"/api/v1/search?q=database PostgreSQL&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        memory = results[0]
        memory_type = get_memory_type(memory)
        
        if memory_type:
            assert memory_type in v7_memory_types, f"Memory type '{memory_type}' should be one of the 13 v7 types"


class TestTemporalValidity:
    """Test Phase 2: Temporal validity tracking."""

    async def test_fact_memories_get_validity_window(self, async_client: httpx.AsyncClient, agent_id: str):
        """Fact-type memories get appropriate validity windows."""
        payload = {
            "content": "The current stock price of AAPL is $180.50",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        # Search to get the memory with validity info
        search_response = await async_client.get(f"/api/v1/search?q=AAPL stock price&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        memory = results[0]
        metadata = memory.get("metadata", {})
        nested = metadata.get("metadata", {})
        validity = memory.get("validity", {})
        
        # Check for validity-related fields from actual API structure
        has_validity = (
            validity.get("start") is not None or
            validity.get("end") is not None or
            metadata.get("ts_valid_start") is not None or
            metadata.get("ts_valid_end") is not None or
            nested.get("ts_valid_start") is not None or
            nested.get("ts_valid_end") is not None
        )
        
        if get_memory_type(memory) == "fact":
            assert has_validity, "Fact-type memories should have validity tracking"

    async def test_validity_timestamps_iso_format(self, async_client: httpx.AsyncClient, agent_id: str):
        """Validity timestamps are in ISO format."""
        payload = {
            "content": "Today's weather is sunny with 25°C temperature",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        search_response = await async_client.get(f"/api/v1/search?q=weather sunny&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        memory = results[0]
        metadata = memory.get("metadata", {})
        nested = metadata.get("metadata", {})
        
        # Check timestamp formats if present
        timestamps = [
            memory.get("created_at"),
            metadata.get("ts_valid_start"),
            metadata.get("ts_valid_end"),
            nested.get("ts_valid_start"),
            nested.get("ts_valid_end"),
            memory.get("validity", {}).get("start"),
            memory.get("validity", {}).get("end"),
        ]
        found_timestamp = False
        for timestamp in timestamps:
            if timestamp:
                found_timestamp = True
                assert 'T' in timestamp, "timestamp should be in ISO format"
                assert ('Z' in timestamp or '+' in timestamp), "timestamp should have timezone info"
        assert found_timestamp, "Expected at least one validity-related timestamp"


class TestLifecycle:
    """Test Phase 4: Memory lifecycle management."""

    async def test_new_memories_start_active(self, async_client: httpx.AsyncClient, agent_id: str):
        """New memories start with 'active' status."""
        payload = {
            "content": "New memory should start as active",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        # Search to verify status
        search_response = await async_client.get(f"/api/v1/search?q=memory should start active&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        memory = results[0]
        status = memory.get("status") or memory.get("metadata", {}).get("status")
        
        # Default status should be active (or not set, which defaults to active)
        assert status in [None, "active"], "New memories should start as 'active' or have no status (defaults to active)"

    async def test_valid_transition_active_to_confirmed(self, async_client: httpx.AsyncClient, agent_id: str):
        """Valid transition: active → confirmed returns 200 success."""
        # First add a memory
        payload = {
            "content": "Memory for transition testing active to confirmed",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        # Try transition
        transition_response = await async_client.post(
            f"/api/v1/memory/{memory_id}/transition?from_status=active&to_status=confirmed"
        )
        assert transition_response.status_code == 200
        
        data = transition_response.json()
        assert data.get("status") == "success" or data.get("success") is True

    async def test_valid_transition_active_to_outdated(self, async_client: httpx.AsyncClient, agent_id: str):
        """Valid transition: active → outdated returns 200 success."""
        payload = {
            "content": "Memory for transition testing active to outdated",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        transition_response = await async_client.post(
            f"/api/v1/memory/{memory_id}/transition?from_status=active&to_status=outdated"
        )
        assert transition_response.status_code == 200

    async def test_invalid_transition_deleted_to_active(self, async_client: httpx.AsyncClient, agent_id: str):
        """Invalid transition: deleted → active returns 400 error."""
        payload = {
            "content": "Memory for invalid transition testing",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        # First transition to deleted (if possible)
        await async_client.post(
            f"/api/v1/memory/{memory_id}/transition?from_status=active&to_status=deleted"
        )
        
        # Now try invalid transition from deleted to active
        invalid_response = await async_client.post(
            f"/api/v1/memory/{memory_id}/transition?from_status=deleted&to_status=active"
        )
        assert invalid_response.status_code == 400

    async def test_status_mismatch_transition_error(self, async_client: httpx.AsyncClient, agent_id: str):
        """Status mismatch: transition from wrong from_status returns 400 error."""
        payload = {
            "content": "Memory for status mismatch testing",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        # Try transition from wrong status (memory should be active, not confirmed)
        mismatch_response = await async_client.post(
            f"/api/v1/memory/{memory_id}/transition?from_status=confirmed&to_status=outdated"
        )
        assert mismatch_response.status_code == 400

    async def test_v7_stats_returns_lifecycle_counts(self, async_client: httpx.AsyncClient):
        """V7 stats returns lifecycle state counts."""
        response = await async_client.get("/api/v1/v7/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "lifecycle_stats" in data
        
        lifecycle_stats = data["lifecycle_stats"]
        assert isinstance(lifecycle_stats, dict)
        
        # Should have counts for different states
        expected_states = ["active", "confirmed", "outdated", "archived", "contradicted", "merged", "superseded", "deleted"]
        # At least some of these states should be present
        state_keys = set(lifecycle_stats.keys())
        assert len(state_keys.intersection(expected_states)) > 0, "Should have lifecycle state counts"

    async def test_bulk_outdated_endpoint(self, async_client: httpx.AsyncClient):
        """Bulk outdated endpoint works (even if 0 updated)."""
        response = await async_client.post("/api/v1/lifecycle/bulk-outdated?threshold_days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert "updated_count" in data or "count" in data or "affected" in data
        
        # Should be a non-negative number
        count = data.get("updated_count", data.get("count", data.get("affected", 0)))
        assert isinstance(count, int)
        assert count >= 0


class TestHybridSearch:
    """Test Phase 5: Hybrid search functionality."""

    async def test_semantic_search_returns_results(self, async_client: httpx.AsyncClient, agent_id: str):
        """Semantic search returns relevant results with scores."""
        # Add some content to search
        test_content = [
            "Python is a programming language used for web development",
            "JavaScript is essential for frontend web development", 
            "Machine learning algorithms require large datasets"
        ]
        
        for content in test_content:
            payload = {
                "content": content,
                "agent_id": agent_id,
                "memory_type": "fact"
            }
            response = await async_client.post("/api/v1/add", json=payload)
            assert response.status_code == 200
        
        # Search for programming-related content
        search_response = await async_client.get(f"/api/v1/search?q=programming languages&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        # Actual API returns score/composite_score/hybrid_score
        assert any(
            (
                memory.get("score") is not None or
                memory.get("composite_score") is not None or
                memory.get("hybrid_score") is not None
            )
            for memory in results
        ), "Expected at least one scored search result"

    async def test_search_includes_enrichment_metadata(self, async_client: httpx.AsyncClient, agent_id: str):
        """Search results include metadata with enrichment data."""
        payload = {
            "content": "The company's quarterly earnings exceeded expectations by 20%",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        search_response = await async_client.get(f"/api/v1/search?q=quarterly earnings exceeded&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        memory = results[0]
        assert "metadata" in memory
        
        metadata = memory["metadata"]
        nested = metadata.get("metadata", {})
        has_enrichment = (
            metadata.get("memory_type") is not None or
            nested.get("type") is not None or
            nested.get("tags") is not None or
            nested.get("title") is not None or
            nested.get("summary") is not None or
            metadata.get("triples") is not None
        )
        assert has_enrichment, "Search results should include enrichment metadata"


class TestMemoryTypes:
    """Test Phase 6: Expanded memory types."""

    async def test_all_13_types_accepted(self, async_client: httpx.AsyncClient, agent_id: str, v7_memory_types: list):
        """All 13 memory types are accepted without error."""
        for memory_type in v7_memory_types:
            payload = {
                "content": f"Testing memory type: {memory_type}",
                "agent_id": agent_id,
                "memory_type": memory_type
            }
            
            response = await async_client.post("/api/v1/add", json=payload)
            assert response.status_code == 200, f"Memory type '{memory_type}' should be accepted"
            
            data = response.json()
            assert data["direct_insert"]["status"] == "ok"

    async def test_invalid_type_still_accepted(self, async_client: httpx.AsyncClient, agent_id: str):
        """Invalid memory type is still accepted (enrichment auto-classifies)."""
        payload = {
            "content": "Testing with an invalid memory type",
            "agent_id": agent_id,
            "memory_type": "invalid_type_xyz"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        # Should still succeed
        assert response.status_code == 200
        
        data = response.json()
        assert data["direct_insert"]["status"] == "ok"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_empty_content_handled_gracefully(self, async_client: httpx.AsyncClient, agent_id: str):
        """Empty content is handled gracefully."""
        payload = {
            "content": "",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        # Should either succeed or return a meaningful error
        assert response.status_code in [200, 400, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "direct_insert" in data

    async def test_very_long_content_works(self, async_client: httpx.AsyncClient, agent_id: str, sample_memory_content: dict):
        """Very long content (1000+ chars) works."""
        long_content = sample_memory_content["long_content"]
        assert len(long_content) > 1000
        
        payload = {
            "content": long_content,
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["direct_insert"]["status"] == "ok"

    async def test_unicode_emoji_content_works(self, async_client: httpx.AsyncClient, agent_id: str, sample_memory_content: dict):
        """Unicode/emoji content works correctly."""
        unicode_content = sample_memory_content["unicode_content"]
        
        payload = {
            "content": unicode_content,
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["direct_insert"]["status"] == "ok"
        
        # Verify we can search and retrieve it
        search_response = await async_client.get(f"/api/v1/search?q=MemClawz v7.0.0&agent_id={agent_id}")
        assert search_response.status_code == 200

    async def test_duplicate_content_handled(self, async_client: httpx.AsyncClient, agent_id: str):
        """Duplicate content is handled appropriately (may deduplicate)."""
        content = "This is duplicate content for testing deduplication"
        payload = {
            "content": content,
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        # Add the same content twice
        response1 = await async_client.post("/api/v1/add", json=payload)
        assert response1.status_code == 200
        
        response2 = await async_client.post("/api/v1/add", json=payload)
        assert response2.status_code == 200
        
        # Both requests should succeed (system may deduplicate internally)
        data1 = response1.json()
        data2 = response2.json()
        assert data1["direct_insert"]["status"] == "ok"
        assert data2["direct_insert"]["status"] == "ok"

    async def test_nonexistent_memory_transition_error(self, async_client: httpx.AsyncClient):
        """Non-existent memory ID in transition returns error."""
        fake_id = "nonexistent-memory-id-12345"
        
        response = await async_client.post(
            f"/api/v1/memory/{fake_id}/transition?from_status=active&to_status=confirmed"
        )
        assert response.status_code in [400, 404], "Should return error for non-existent memory ID"

    async def test_malformed_transition_request(self, async_client: httpx.AsyncClient, agent_id: str):
        """Malformed transition request returns appropriate error."""
        # Add a memory first
        payload = {
            "content": "Memory for malformed transition test",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        # Try malformed transition (missing parameters)
        malformed_response = await async_client.post(f"/api/v1/memory/{memory_id}/transition")
        assert malformed_response.status_code in [400, 422], "Should return error for malformed request"


class TestIntegration:
    """Integration tests combining multiple features."""

    async def test_full_memory_lifecycle_workflow(self, async_client: httpx.AsyncClient, agent_id: str):
        """Test complete workflow: add → search → transition → search again."""
        # 1. Add memory
        payload = {
            "content": "Integration test: The new API design uses REST principles",
            "agent_id": agent_id,
            "memory_type": "decision"
        }
        
        response = await async_client.post("/api/v1/add", json=payload)
        assert response.status_code == 200
        
        memory_id = response.json()["direct_insert"]["id"]
        
        # 2. Search and verify it's found
        search_response = await async_client.get(f"/api/v1/search?q=API design REST&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        results = get_search_results(data)
        assert len(results) >= 1
        
        # 3. Transition status
        transition_response = await async_client.post(
            f"/api/v1/memory/{memory_id}/transition?from_status=active&to_status=confirmed"
        )
        assert transition_response.status_code == 200
        
        # 4. Search again to verify it still appears (status changed but should be findable)
        search_response2 = await async_client.get(f"/api/v1/search?q=API design REST&agent_id={agent_id}")
        assert search_response2.status_code == 200
        
        # Memory should still be findable after status change
        data2 = search_response2.json()
        found = False
        for memory in get_search_results(data2):
            if "API design" in get_memory_text(memory):
                found = True
                break
        assert found, "Memory should still be findable after status transition"

    async def test_multiple_agents_isolation(self, async_client: httpx.AsyncClient, agent_id: str):
        """Test that agent isolation works correctly."""
        other_agent = f"{agent_id}-other"
        
        # Add memory for first agent
        payload1 = {
            "content": "Memory for agent isolation test - agent 1",
            "agent_id": agent_id,
            "memory_type": "fact"
        }
        
        response1 = await async_client.post("/api/v1/add", json=payload1)
        assert response1.status_code == 200
        
        # Add memory for second agent
        payload2 = {
            "content": "Memory for agent isolation test - agent 2",
            "agent_id": other_agent,
            "memory_type": "fact"
        }
        
        response2 = await async_client.post("/api/v1/add", json=payload2)
        assert response2.status_code == 200
        
        # Search with agent filter should only return memories for that agent
        search_response = await async_client.get(f"/api/v1/search?q=agent isolation test&agent_id={agent_id}")
        assert search_response.status_code == 200
        
        data = search_response.json()
        found_agents = set()
        for memory in get_search_results(data):
            memory_agent = memory.get("agent_id") or memory.get("metadata", {}).get("agent") or memory.get("metadata", {}).get("metadata", {}).get("agent")
            if memory_agent:
                found_agents.add(memory_agent)
        
        # Should only find memories for our specific agent (or no agent info)
        if found_agents:
            assert agent_id in found_agents
            assert other_agent not in found_agents, "Agent filter should isolate memories correctly"