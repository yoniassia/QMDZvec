from unittest.mock import patch

from fastapi.testclient import TestClient

from memclawz.router import MemClawzRouter


def test_router_routes_to_infraclaw():
    router = MemClawzRouter()
    result = router.route("deploy app to server")
    assert result["agent_id"] == "infraclaw"
    assert result["session_key"] == "agent:infraclaw:webchat:main"
    assert result["fallback"] is None


def test_router_routes_to_tradeclaw():
    router = MemClawzRouter()
    result = router.route("check BTC price")
    assert result["agent_id"] == "tradeclaw"
    assert result["session_key"] == "agent:tradeclaw:webchat:main"


def test_router_routes_to_qaclaw():
    router = MemClawzRouter()
    result = router.route("run test cycle on clawqa")
    assert result["agent_id"] == "qaclaw"
    assert result["model"] == "anthropic/claude-opus-4-6"
    assert result["fallback"] is None


def test_router_routes_to_commsclaw():
    router = MemClawzRouter()
    result = router.route("send email to Yoav")
    assert result["agent_id"] == "commsclaw"
    assert result["session_key"] == "agent:commsclaw:webchat:main"


def test_router_routes_to_coinresearchclaw():
    router = MemClawzRouter()
    result = router.route("research DOGE coin")
    assert result["agent_id"] == "coinresearchclaw"
    assert result["session_key"] == "agent:coinresearchclaw:webchat:main"


def test_router_falls_back_to_main_without_memory():
    router = MemClawzRouter()
    result = router.route("do the thing with the stuff")
    assert result["agent_id"] == "main"
    assert result["session_key"] == "main"
    assert result["confidence"] == 0.3
    assert result["reason"] == "Semantic fallback"
    assert result["fallback"] == "main"


def test_route_api_endpoint():
    fake_result = {
        "agent_id": "infraclaw",
        "session_key": "agent:infraclaw:webchat:main",
        "emoji": "🏗️",
        "model": "openai/gpt-5.4",
        "confidence": 0.92,
        "reason": "Matched domains: ['servers']",
        "memory_context": [],
        "fallback": None,
    }

    with patch("memclawz.api.router_engine.route", return_value=fake_result) as mock_route:
        from memclawz.api import app

        client = TestClient(app)
        response = client.get("/api/v1/route", params={"task": "deploy app to server"})

    assert response.status_code == 200
    assert response.json() == fake_result
    mock_route.assert_called_once_with("deploy app to server", include_context=True)
