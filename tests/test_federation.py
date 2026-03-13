"""Tests for federation module."""
import tempfile
from pathlib import Path
from memclawz.federation import FederationRegistry, NodeRegistration


def test_register_node():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        reg = FederationRegistry(Path(f.name))

    node = NodeRegistration(
        node_id="test-claw",
        node_url="http://localhost:3501",
        node_key="secret123",
        description="Test node",
    )
    result = reg.register(node)
    assert result["status"] == "registered"
    assert result["node_id"] == "test-claw"


def test_authenticate_node():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        reg = FederationRegistry(Path(f.name))

    node = NodeRegistration(
        node_id="auth-test",
        node_url="http://localhost:3502",
        node_key="mysecret",
    )
    reg.register(node)

    assert reg.authenticate("auth-test", "mysecret") is True
    assert reg.authenticate("auth-test", "wrongkey") is False
    assert reg.authenticate("nonexistent", "mysecret") is False


def test_list_nodes_hides_keys():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        reg = FederationRegistry(Path(f.name))

    node = NodeRegistration(
        node_id="list-test",
        node_url="http://localhost:3503",
        node_key="topsecret",
    )
    reg.register(node)
    nodes = reg.list_nodes()
    assert "list-test" in nodes
    assert "key_hash" not in nodes["list-test"]


def test_update_sync_stats():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        reg = FederationRegistry(Path(f.name))

    node = NodeRegistration(
        node_id="stats-test",
        node_url="http://localhost:3504",
        node_key="key",
    )
    reg.register(node)
    reg.update_sync_stats("stats-test", "push", 5)
    info = reg.get_node("stats-test")
    assert info["push_count"] == 5
    assert info["last_sync"] is not None
