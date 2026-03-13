"""Tests for the FastAPI endpoints (requires running Qdrant)."""
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
    assert data["version"] == "5.0.0"
