"""Pytest configuration and shared fixtures for MemClawz tests."""
import asyncio
import pytest
import httpx
from datetime import datetime
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the MemClawz API."""
    return "http://localhost:3500"


@pytest.fixture(scope="session")
def agent_id() -> str:
    """Unique agent ID for test isolation."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"e2e-test-{timestamp}"


@pytest.fixture
async def async_client(base_url: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for testing."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


@pytest.fixture(autouse=True)
async def cleanup_test_memories(async_client: httpx.AsyncClient, agent_id: str):
    """Clean up test memories after each test."""
    yield
    
    # Clean up memories created during the test
    try:
        # Get all memories for the test agent
        response = await async_client.get(f"/api/v1/memories?agent_id={agent_id}&limit=1000")
        if response.status_code == 200:
            data = response.json()
            memories = data.get("memories", [])
            
            # Delete each memory
            for memory in memories:
                memory_id = memory.get("id")
                if memory_id:
                    await async_client.delete(f"/api/v1/memory/{memory_id}")
    except Exception as e:
        # Non-fatal cleanup failure
        print(f"Warning: Cleanup failed: {e}")


@pytest.fixture
def sample_memory_content() -> dict:
    """Sample memory content for testing."""
    return {
        "simple_fact": "The capital of France is Paris",
        "unicode_content": "Test with unicode: 🚀 MemClawz v7.0.0 支持中文",
        "long_content": "This is a very long piece of content that exceeds normal length limits. " * 20,
        "empty_content": "",
    }


@pytest.fixture
def v7_memory_types() -> list:
    """All memory types supported in v7."""
    return [
        "fact", "decision", "preference", "relationship", "insight", "procedure",
        "event", "intention", "plan", "commitment", "action", "outcome", "cancellation"
    ]


@pytest.fixture
def lifecycle_states() -> list:
    """Valid lifecycle states."""
    return ["active", "confirmed", "outdated", "archived", "contradicted", "merged", "superseded", "deleted"]


@pytest.fixture
def valid_transitions() -> dict:
    """Valid state transitions mapping."""
    return {
        "active": ["confirmed", "outdated", "archived", "contradicted", "merged", "superseded", "deleted"],
        "confirmed": ["outdated", "archived", "superseded", "deleted"],
        "outdated": ["archived", "deleted", "active"],
        "archived": ["active", "deleted"],
        "contradicted": ["deleted", "active"],
        "merged": ["deleted"],
        "superseded": ["deleted"],
        "deleted": [],
    }
