"""Tests for the FastAPI endpoints (requires running Qdrant)."""
import hashlib
import re
import time
import pytest
from unittest.mock import patch, MagicMock


def test_health_endpoint():
    """Test health endpoint returns ok."""
    from fastapi.testclient import TestClient
    from memclawz.api import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "7.0.0"


def test_add_direct_path_payload_has_required_top_level_fields():
    """Direct Qdrant (primary path) payload includes memory, agent_id, memory_type, hash, created_at."""
    from fastapi.testclient import TestClient
    from memclawz.api import app

    captured_payload = {}
    fake_vector = [0.1] * 1536

    def fake_upsert(collection_name, points):
        nonlocal captured_payload
        if points:
            captured_payload = points[0].payload

    with (
        patch("memclawz.api.mem") as mock_mem,
        patch("memclawz.api.GRAPHITI_ENABLED", False),
        patch("openai.OpenAI") as mock_openai,
        patch("qdrant_client.QdrantClient") as mock_qc,
    ):
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
        mock_qc.return_value.upsert = fake_upsert

        client = TestClient(app)
        response = client.post(
            "/api/v1/add",
            json={
                "content": "Test fact for fallback.",
                "user_id": "yoni",
                "agent_id": "my_agent",
                "memory_type": "fact",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["direct_insert"]["status"] == "ok"
    assert data["direct_insert"]["method"] == "direct_qdrant"
    assert captured_payload.get("memory") == "Test fact for fallback."
    assert captured_payload.get("agent_id") == "my_agent"
    assert captured_payload.get("memory_type") == "fact"
    assert captured_payload.get("hash") == hashlib.md5(b"Test fact for fallback.").hexdigest()
    assert "created_at" in captured_payload
    assert re.search(r"Z|\+00:00", captured_payload["created_at"])
    assert captured_payload.get("user_id") == "yoni"
    assert "metadata" in captured_payload
    assert captured_payload["metadata"].get("agent") == "my_agent"
    assert captured_payload["metadata"].get("type") == "fact"


def test_add_returns_success_when_mem_add_sleeps():
    """POST /api/v1/add returns success based on direct insert; does not depend on mem.add() returning.
    (mem.add is run in background; if it slept, success is still from direct path.)"""
    from fastapi.testclient import TestClient
    from memclawz.api import app

    fake_vector = [0.1] * 1536

    def slow_add(*args, **kwargs):
        time.sleep(2)
        return {"results": []}

    with (
        patch("memclawz.api.mem") as mock_mem,
        patch("memclawz.api.GRAPHITI_ENABLED", False),
        patch("openai.OpenAI") as mock_openai,
        patch("qdrant_client.QdrantClient") as mock_qc,
    ):
        mock_mem.add.side_effect = slow_add
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
        mock_qc.return_value.upsert = MagicMock()

        client = TestClient(app)
        response = client.post(
            "/api/v1/add",
            json={"content": "Slow mem.add test.", "user_id": "yoni", "agent_id": "a", "memory_type": "fact"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["direct_insert"]["status"] == "ok"
    assert data["direct_insert"]["method"] == "direct_qdrant"
    assert "id" in data["direct_insert"]
    # Endpoint success is from direct insert; mem.add runs in background (may complete after response)


def test_add_returns_success_when_mem_add_raises():
    """POST /api/v1/add returns success based on direct insert even if mem.add() raises."""
    from fastapi.testclient import TestClient
    from memclawz.api import app

    fake_vector = [0.1] * 1536

    with (
        patch("memclawz.api.mem") as mock_mem,
        patch("memclawz.api.GRAPHITI_ENABLED", False),
        patch("openai.OpenAI") as mock_openai,
        patch("qdrant_client.QdrantClient") as mock_qc,
    ):
        mock_mem.add.side_effect = RuntimeError("Mem0 LLM timeout")
        mock_openai.return_value.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
        mock_qc.return_value.upsert = MagicMock()

        client = TestClient(app)
        response = client.post(
            "/api/v1/add",
            json={"content": "Mem0 fails.", "user_id": "yoni", "agent_id": "a", "memory_type": "fact"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["direct_insert"]["status"] == "ok"
    assert data["direct_insert"]["method"] == "direct_qdrant"
    assert data.get("mem0", {}).get("status") == "background"


def test_stats_by_agent_counts_metadata_and_flattened_records():
    """Stats by_agent counts both metadata-based and flattened (direct-insert) style records."""
    from fastapi.testclient import TestClient
    from memclawz.api import app

    # One record with nested metadata, one with top-level agent_id/memory_type/source only
    mixed_records = [
        {"id": "1", "metadata": {"agent": "claw_a", "type": "fact", "source": "api"}},
        {"id": "2", "agent_id": "claw_a", "memory_type": "fact", "source": "api"},
    ]

    with patch("memclawz.api.mem") as mock_mem:
        mock_mem.get_all.return_value = mixed_records
        client = TestClient(app)
        response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["by_agent"].get("claw_a") == 2
    assert data["total_memories"] == 2


def test_list_memories_agent_filter_includes_flattened_records():
    """list_memories agent_id filter works for flattened (direct-insert) style records."""
    from fastapi.testclient import TestClient
    from memclawz.api import app

    # Mix: metadata.agent, top-level agent_id only, and another agent (should be excluded)
    records = [
        {"id": "1", "memory": "From metadata", "metadata": {"agent": "bot_x", "type": "fact"}},
        {"id": "2", "memory": "From top-level", "agent_id": "bot_x", "memory_type": "fact"},
        {"id": "3", "memory": "Other agent", "agent_id": "other_bot", "memory_type": "fact"},
    ]

    with patch("memclawz.api.mem") as mock_mem:
        mock_mem.get_all.return_value = records
        client = TestClient(app)
        response = client.get("/api/v1/memories?agent_id=bot_x")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["memories"]) == 2
    agents = [
        m.get("metadata", {}).get("agent") or m.get("agent_id") for m in data["memories"]
    ]
    assert set(agents) == {"bot_x"}
