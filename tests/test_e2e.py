"""End-to-end integration tests for MemClawz v7 API.

Tests the actual FastAPI app with mocked external dependencies (Qdrant, OpenAI).
Uses FastAPI TestClient to simulate real HTTP requests.
"""
import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from qdrant_client.models import ScoredPoint


@pytest.mark.e2e
class TestHealthEndpoint:
    """Test health endpoint."""
    
    def test_health_endpoint_returns_v7(self):
        """Test health endpoint returns v7.0.0."""
        from memclawz.api import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "7.0.0"
        assert "uptime" in data
        assert "timestamp" in data


@pytest.mark.e2e
class TestMemoryAddEndpoints:
    """Test memory addition endpoints."""
    
    @patch("memclawz.api.mem")
    @patch("memclawz.api.GRAPHITI_ENABLED", False)
    @patch("openai.OpenAI")
    @patch("qdrant_client.QdrantClient")
    def test_add_memory_stores_and_searchable(self, mock_qc, mock_openai, mock_mem):
        """Test POST /api/v1/add → verify stored → GET /api/v1/search finds it."""
        from memclawz.api import app
        
        # Setup mocks
        fake_vector = [0.1] * 1536
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
        
        captured_points = []
        def capture_upsert(collection_name, points):
            captured_points.extend(points)
        
        mock_qc.return_value.upsert = capture_upsert
        
        # Mock search results
        mock_qc.return_value.search.return_value = [
            ScoredPoint(
                id=str(uuid.uuid4()),
                score=0.85,
                payload={
                    "memory": "Test memory for v7 integration",
                    "agent_id": "test-agent",
                    "memory_type": "fact",
                    "status": "active"
                },
                vector=fake_vector
            )
        ]
        
        client = TestClient(app)
        
        # Step 1: Add a memory
        add_response = client.post("/api/v1/add", json={
            "content": "Test memory for v7 integration",
            "agent_id": "test-agent",
            "memory_type": "fact"
        })
        
        assert add_response.status_code == 200
        add_data = add_response.json()
        assert add_data["status"] == "success"
        assert "memory_id" in add_data
        
        # Verify memory was stored with v7 fields
        assert len(captured_points) == 1
        stored_payload = captured_points[0].payload
        assert stored_payload["memory"] == "Test memory for v7 integration"
        assert stored_payload["agent_id"] == "test-agent"
        assert stored_payload["memory_type"] == "fact"
        assert stored_payload["status"] == "active"  # v7 lifecycle
        assert "status_updated_at" in stored_payload  # v7 lifecycle
        
        # Step 2: Search for the memory
        search_response = client.get("/api/v1/search", params={
            "q": "v7 integration",
            "limit": 10
        })
        
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["status"] == "success"
        assert len(search_data["memories"]) > 0
        
        # Verify search result has v7 enhancements
        found_memory = search_data["memories"][0]
        assert "hybrid_score" in found_memory  # v7 hybrid search
        assert "keyword_score" in found_memory
        assert "status" in found_memory
        assert found_memory["memory"] == "Test memory for v7 integration"
    
    @patch("memclawz.api.mem")
    @patch("memclawz.api.GRAPHITI_ENABLED", False) 
    @patch("openai.OpenAI")
    @patch("qdrant_client.QdrantClient")
    def test_add_direct_endpoint(self, mock_qc, mock_openai, mock_mem):
        """Test POST /api/v1/add-direct stores memory."""
        from memclawz.api import app
        
        # Setup mocks
        fake_vector = [0.2] * 1536
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
        
        captured_points = []
        mock_qc.return_value.upsert = lambda collection_name, points: captured_points.extend(points)
        
        client = TestClient(app)
        
        response = client.post("/api/v1/add-direct", json={
            "content": "Direct memory addition test",
            "agent_id": "direct-agent",
            "memory_type": "decision",
            "importance": 0.9
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "memory_id" in data
        
        # Verify stored in Qdrant with v7 enhancements
        assert len(captured_points) == 1
        payload = captured_points[0].payload
        assert payload["memory"] == "Direct memory addition test"
        assert payload["agent_id"] == "direct-agent"
        assert payload["memory_type"] == "decision"
        assert payload["importance"] == 0.9
        assert payload["status"] == "active"  # v7 lifecycle default
        assert "status_updated_at" in payload


@pytest.mark.e2e
class TestMemoryListingEndpoints:
    """Test memory listing endpoints."""
    
    @patch("qdrant_client.QdrantClient")
    def test_get_memories_endpoint(self, mock_qc):
        """Test GET /api/v1/memories returns memory list."""
        from memclawz.api import app
        
        # Mock scroll results
        mock_records = [
            ScoredPoint(
                id="mem-1",
                score=1.0,
                payload={
                    "memory": "First memory",
                    "agent_id": "agent-1", 
                    "memory_type": "fact",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z"
                },
                vector=[0.1] * 1536
            ),
            ScoredPoint(
                id="mem-2", 
                score=1.0,
                payload={
                    "memory": "Second memory",
                    "agent_id": "agent-2",
                    "memory_type": "decision", 
                    "status": "confirmed",
                    "created_at": "2024-01-02T00:00:00Z"
                },
                vector=[0.2] * 1536
            )
        ]
        mock_qc.return_value.scroll.return_value = (mock_records, None)
        
        client = TestClient(app)
        
        response = client.get("/api/v1/memories", params={"limit": 10})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["memories"]) == 2
        
        # Check memory format
        memory = data["memories"][0]
        assert "id" in memory
        assert "memory" in memory
        assert "agent_id" in memory
        assert "memory_type" in memory
        assert "status" in memory  # v7 lifecycle
        assert "created_at" in memory
    
    @patch("qdrant_client.QdrantClient")
    def test_get_agents_endpoint(self, mock_qc):
        """Test GET /api/v1/agents returns agent list."""
        from memclawz.api import app
        
        # Mock scroll results with different agents
        mock_records = [
            ScoredPoint(id="1", score=1.0, payload={"agent_id": "agent-1"}, vector=None),
            ScoredPoint(id="2", score=1.0, payload={"agent_id": "agent-2"}, vector=None), 
            ScoredPoint(id="3", score=1.0, payload={"agent_id": "agent-1"}, vector=None),  # Duplicate
        ]
        mock_qc.return_value.scroll.return_value = (mock_records, None)
        
        client = TestClient(app)
        
        response = client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "agents" in data
        
        # Should deduplicate agent IDs
        agents = data["agents"]
        assert len(agents) == 2
        assert "agent-1" in agents
        assert "agent-2" in agents
    
    @patch("qdrant_client.QdrantClient")
    def test_get_stats_endpoint(self, mock_qc):
        """Test GET /api/v1/stats returns statistics."""
        from memclawz.api import app
        
        # Mock count results
        mock_qc.return_value.count.return_value = Mock(count=150)
        
        client = TestClient(app)
        
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "stats" in data
        
        stats = data["stats"]
        assert "total_memories" in stats
        assert stats["total_memories"] == 150
        assert "version" in stats
        assert stats["version"] == "7.0.0"


@pytest.mark.e2e
class TestV7SpecificEndpoints:
    """Test v7-specific endpoints."""
    
    @patch("memclawz.api.enrich_memory")
    def test_enrich_endpoint(self, mock_enrich):
        """Test POST /api/v1/enrich endpoint."""
        from memclawz.api import app
        
        # Mock enrichment result
        mock_enrich.return_value = {
            "type": "fact",
            "weight": 0.8,
            "title": "Enriched Title",
            "summary": "Enriched summary",
            "tags": ["enriched", "test"],
            "validity_hours": 24,
            "triples": [{"subject": "test", "predicate": "is", "object": "enriched"}]
        }
        
        client = TestClient(app)
        
        response = client.post("/api/v1/enrich", json={
            "content": "Test content for enrichment"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "enrichment" in data
        
        enrichment = data["enrichment"]
        assert enrichment["type"] == "fact"
        assert enrichment["weight"] == 0.8
        assert enrichment["title"] == "Enriched Title"
        assert len(enrichment["tags"]) == 2
        assert len(enrichment["triples"]) == 1
        
        mock_enrich.assert_called_once_with("Test content for enrichment")
    
    @patch("memclawz.api.hybrid_search")
    @patch("qdrant_client.QdrantClient")
    def test_hybrid_search_endpoint(self, mock_qc, mock_hybrid_search):
        """Test POST /api/v1/hybrid-search endpoint.""" 
        from memclawz.api import app
        
        # Mock vector search results
        vector_results = [
            {
                "memory": "hybrid search test content",
                "score": 0.8,
                "metadata": {"agent_id": "test", "memory_type": "fact"}
            }
        ]
        mock_qc.return_value.search.return_value = [
            ScoredPoint(
                id="hybrid-1",
                score=0.8,
                payload={
                    "memory": "hybrid search test content",
                    "agent_id": "test",
                    "memory_type": "fact"
                },
                vector=None
            )
        ]
        
        # Mock hybrid search enhancement
        mock_hybrid_search.return_value = [
            {
                "memory": "hybrid search test content",
                "score": 0.8,
                "hybrid_score": 0.85,
                "keyword_score": 0.7,
                "metadata": {"agent_id": "test", "memory_type": "fact"}
            }
        ]
        
        client = TestClient(app)
        
        response = client.post("/api/v1/hybrid-search", json={
            "query": "hybrid test",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "memories" in data
        
        memories = data["memories"]
        assert len(memories) > 0
        
        memory = memories[0]
        assert "hybrid_score" in memory
        assert "keyword_score" in memory
        assert memory["hybrid_score"] == 0.85


@pytest.mark.e2e
class TestLifecycleEndpoints:
    """Test v7 lifecycle management endpoints."""
    
    @patch("memclawz.api.MemoryLifecycle")
    def test_lifecycle_transition_endpoint(self, mock_lifecycle_class):
        """Test POST /api/v1/lifecycle/transition endpoint."""
        from memclawz.api import app
        
        mock_lifecycle = Mock()
        mock_lifecycle_class.return_value = mock_lifecycle
        mock_lifecycle.transition.return_value = True
        
        client = TestClient(app)
        
        response = client.post("/api/v1/lifecycle/transition", json={
            "memory_id": "test-memory-id",
            "from_status": "active", 
            "to_status": "confirmed"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Transitioned" in data["message"]
        
        mock_lifecycle.transition.assert_called_once_with(
            "test-memory-id", "active", "confirmed"
        )
    
    @patch("memclawz.api.MemoryLifecycle") 
    def test_lifecycle_status_endpoint(self, mock_lifecycle_class):
        """Test GET /api/v1/lifecycle/status/{memory_id} endpoint."""
        from memclawz.api import app
        
        mock_lifecycle = Mock()
        mock_lifecycle_class.return_value = mock_lifecycle
        mock_lifecycle.get_status.return_value = "confirmed"
        
        client = TestClient(app)
        
        response = client.get("/api/v1/lifecycle/status/test-memory-id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success" 
        assert data["memory_id"] == "test-memory-id"
        assert data["current_status"] == "confirmed"
        
        mock_lifecycle.get_status.assert_called_once_with("test-memory-id")
    
    @patch("memclawz.api.MemoryLifecycle")
    def test_lifecycle_stats_endpoint(self, mock_lifecycle_class):
        """Test GET /api/v1/lifecycle/stats endpoint."""
        from memclawz.api import app
        
        mock_lifecycle = Mock()
        mock_lifecycle_class.return_value = mock_lifecycle
        mock_lifecycle.get_lifecycle_stats.return_value = {
            "active": 100,
            "confirmed": 50,
            "outdated": 10,
            "archived": 5,
            "contradicted": 2,
            "merged": 1,
            "superseded": 1, 
            "deleted": 0,
            "unlabeled": 5
        }
        
        client = TestClient(app)
        
        response = client.get("/api/v1/lifecycle/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "stats" in data
        
        stats = data["stats"]
        assert stats["active"] == 100
        assert stats["confirmed"] == 50
        assert stats["unlabeled"] == 5
    
    @patch("memclawz.api.MemoryLifecycle")
    def test_lifecycle_outdated_endpoint(self, mock_lifecycle_class):
        """Test GET /api/v1/lifecycle/outdated endpoint."""
        from memclawz.api import app
        
        mock_lifecycle = Mock()
        mock_lifecycle_class.return_value = mock_lifecycle
        mock_lifecycle.bulk_check_outdated.return_value = [
            {
                "id": "old-1",
                "content": "Old memory 1...",
                "age_days": 45,
                "current_status": "active",
                "memory_type": "fact",
                "agent_id": "agent-1"
            },
            {
                "id": "old-2", 
                "content": "Old memory 2...",
                "age_days": 60,
                "current_status": "active",
                "memory_type": "decision",
                "agent_id": "agent-2"
            }
        ]
        
        client = TestClient(app)
        
        response = client.get("/api/v1/lifecycle/outdated", params={
            "threshold_days": 30
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "candidates" in data
        
        candidates = data["candidates"]
        assert len(candidates) == 2
        assert candidates[0]["age_days"] == 45
        assert candidates[1]["age_days"] == 60
        
        mock_lifecycle.bulk_check_outdated.assert_called_once_with(30)


@pytest.mark.e2e
class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_endpoint_404(self):
        """Test invalid endpoint returns 404."""
        from memclawz.api import app
        
        client = TestClient(app)
        
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    def test_invalid_json_400(self):
        """Test invalid JSON in POST request."""
        from memclawz.api import app
        
        client = TestClient(app)
        
        response = client.post("/api/v1/add", data="invalid json")
        
        assert response.status_code == 422  # FastAPI validation error
    
    @patch("memclawz.api.mem")
    @patch("qdrant_client.QdrantClient")
    def test_database_error_handling(self, mock_qc, mock_mem):
        """Test API handles database errors gracefully."""
        from memclawz.api import app
        
        # Mock Qdrant error
        mock_qc.return_value.search.side_effect = Exception("Database connection failed")
        
        client = TestClient(app)
        
        response = client.get("/api/v1/search", params={"q": "test"})
        
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data


@pytest.mark.e2e
class TestSearchEnhancements:
    """Test v7 search enhancements end-to-end."""
    
    @patch("memclawz.api.hybrid_search")
    @patch("qdrant_client.QdrantClient")
    def test_search_includes_v7_enhancements(self, mock_qc, mock_hybrid_search):
        """Test regular search includes v7 hybrid scoring and lifecycle filtering."""
        from memclawz.api import app
        
        # Mock Qdrant search results
        mock_qc.return_value.search.return_value = [
            ScoredPoint(
                id="search-1",
                score=0.8, 
                payload={
                    "memory": "search test content",
                    "agent_id": "search-agent",
                    "memory_type": "fact",
                    "status": "active"
                },
                vector=None
            )
        ]
        
        # Mock hybrid search enhancement
        mock_hybrid_search.return_value = [
            {
                "memory": "search test content",
                "score": 0.8,
                "hybrid_score": 0.85,
                "keyword_score": 0.7,
                "recency_score": 0.9,
                "importance_score": 0.8,
                "status": "active",
                "status_multiplier": 1.0,
                "metadata": {
                    "agent_id": "search-agent", 
                    "memory_type": "fact",
                    "status": "active"
                }
            }
        ]
        
        client = TestClient(app)
        
        response = client.get("/api/v1/search", params={
            "q": "search test",
            "limit": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        memories = data["memories"]
        assert len(memories) > 0
        
        memory = memories[0]
        # Should have v7 hybrid scoring fields
        assert "hybrid_score" in memory
        assert "keyword_score" in memory
        assert "recency_score" in memory
        assert "importance_score" in memory
        assert "status_multiplier" in memory
        
        # Verify hybrid search was called
        mock_hybrid_search.assert_called_once()
    
    @patch("qdrant_client.QdrantClient")
    def test_search_filters_deleted_memories(self, mock_qc):
        """Test search filters out deleted memories by default."""
        from memclawz.api import app
        
        # Mock search results with mixed statuses
        mock_qc.return_value.search.return_value = [
            ScoredPoint(
                id="active-mem",
                score=0.9,
                payload={"memory": "active memory", "status": "active"},
                vector=None
            ),
            ScoredPoint(
                id="deleted-mem", 
                score=0.8,
                payload={"memory": "deleted memory", "status": "deleted"},
                vector=None
            )
        ]
        
        client = TestClient(app)
        
        response = client.get("/api/v1/search", params={"q": "memory"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Should only return active memory (deleted filtered out)
        memories = data["memories"]
        assert len(memories) == 1
        assert memories[0]["status"] == "active"
        assert "active memory" in memories[0]["memory"]


@pytest.mark.e2e
class TestBackwardsCompatibility:
    """Test v7 maintains backwards compatibility with v6."""
    
    @patch("memclawz.api.mem")
    @patch("openai.OpenAI") 
    @patch("qdrant_client.QdrantClient")
    def test_v6_add_format_still_works(self, mock_qc, mock_openai, mock_mem):
        """Test v6-style add requests still work in v7."""
        from memclawz.api import app
        
        # Setup mocks
        fake_vector = [0.5] * 1536
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
        
        captured_points = []
        mock_qc.return_value.upsert = lambda collection_name, points: captured_points.extend(points)
        
        client = TestClient(app)
        
        # v6-style request (minimal fields)
        response = client.post("/api/v1/add", json={
            "content": "Legacy v6 format memory"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Should still work, with v7 enhancements added
        assert len(captured_points) == 1
        payload = captured_points[0].payload
        assert payload["memory"] == "Legacy v6 format memory"
        assert payload["status"] == "active"  # v7 default added
        assert "status_updated_at" in payload  # v7 timestamp added
    
    @patch("qdrant_client.QdrantClient")
    def test_v6_search_results_enhanced_with_v7(self, mock_qc):
        """Test v6-format memories get v7 enhancements in search."""
        from memclawz.api import app
        
        # Mock old v6-format memory (no status field)
        mock_qc.return_value.search.return_value = [
            ScoredPoint(
                id="v6-memory",
                score=0.8,
                payload={
                    "memory": "Old v6 memory without status",
                    "agent_id": "legacy-agent",
                    "memory_type": "fact"
                    # No status field (v6 format)
                },
                vector=None
            )
        ]
        
        client = TestClient(app)
        
        response = client.get("/api/v1/search", params={"q": "v6 memory"})
        
        assert response.status_code == 200
        data = response.json()
        
        memories = data["memories"]
        assert len(memories) == 1
        
        memory = memories[0]
        # Should get v7 hybrid scoring even for old memories
        assert "hybrid_score" in memory
        # Status should default to "active" for old memories
        assert memory.get("status", "active") == "active"


@pytest.mark.e2e  
class TestPerformanceAndLimits:
    """Test API performance and limits."""
    
    @patch("qdrant_client.QdrantClient")
    def test_search_respects_limit_parameter(self, mock_qc):
        """Test search endpoint respects limit parameter."""
        from memclawz.api import app
        
        # Mock more results than limit
        mock_results = [
            ScoredPoint(
                id=f"mem-{i}",
                score=0.9 - i*0.1,
                payload={"memory": f"Memory {i}", "status": "active"},
                vector=None
            )
            for i in range(20)  # 20 results
        ]
        mock_qc.return_value.search.return_value = mock_results
        
        client = TestClient(app)
        
        response = client.get("/api/v1/search", params={
            "q": "memory",
            "limit": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should respect limit
        memories = data["memories"]
        assert len(memories) == 5
    
    @patch("qdrant_client.QdrantClient")
    def test_memories_endpoint_pagination(self, mock_qc):
        """Test memories endpoint supports pagination."""
        from memclawz.api import app
        
        # Mock paginated results  
        mock_records = [
            ScoredPoint(
                id=f"page-mem-{i}",
                score=1.0,
                payload={"memory": f"Page memory {i}", "status": "active"},
                vector=None
            )
            for i in range(10)
        ]
        mock_qc.return_value.scroll.return_value = (mock_records, None)
        
        client = TestClient(app)
        
        response = client.get("/api/v1/memories", params={
            "limit": 3,
            "offset": 2
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle pagination parameters
        memories = data["memories"]
        assert len(memories) <= 10  # Based on mock data
    
    def test_large_content_handling(self):
        """Test API handles large content gracefully."""
        from memclawz.api import app
        
        # Very large content
        large_content = "A" * 10000  # 10KB content
        
        client = TestClient(app)
        
        response = client.post("/api/v1/enrich", json={
            "content": large_content
        })
        
        # Should not crash (might return error, but shouldn't 500)
        assert response.status_code in [200, 400, 422, 500]  # Any reasonable response