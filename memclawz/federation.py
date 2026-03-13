"""MemClawz v6 — Multi-Claw Federation.

Allows multiple OpenClaw instances to share memories via HTTP push/pull.
YoniClaw acts as master node; remote claws register and sync.
"""
import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import STATE_DIR
from .utils import load_json, save_json

logger = logging.getLogger(__name__)

FEDERATION_REGISTRY_FILE = STATE_DIR / "federation_registry.json"
FEDERATION_LOG_FILE = STATE_DIR / "federation_sync_log.json"


# --- Models ---

class NodeRegistration(BaseModel):
    node_id: str
    node_url: str
    node_key: str  # shared secret
    description: str = ""


class FederationPushRequest(BaseModel):
    node_id: str
    node_key: str
    memories: list[dict[str, Any]]


class FederationPullRequest(BaseModel):
    node_id: str
    node_key: str
    since: str | None = None  # ISO timestamp
    agents: list[str] | None = None
    memory_types: list[str] | None = None
    limit: int = 100


class FederationSyncRequest(BaseModel):
    node_id: str
    node_key: str
    memories: list[dict[str, Any]] = []
    since: str | None = None
    limit: int = 100


# --- Registry ---

class FederationRegistry:
    """Manages registered federation nodes."""

    def __init__(self, registry_path: Path = FEDERATION_REGISTRY_FILE):
        self.path = registry_path

    def _load(self) -> dict:
        return load_json(self.path)

    def _save(self, data: dict):
        save_json(data, self.path)

    def register(self, node: NodeRegistration) -> dict:
        """Register a new node or update existing."""
        data = self._load()
        nodes = data.get("nodes", {})

        # Hash the key for storage (don't store plaintext)
        key_hash = hashlib.sha256(node.node_key.encode()).hexdigest()

        nodes[node.node_id] = {
            "node_url": node.node_url,
            "key_hash": key_hash,
            "description": node.description,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_sync": None,
            "push_count": 0,
            "pull_count": 0,
        }
        data["nodes"] = nodes
        self._save(data)
        logger.info(f"Federation node registered: {node.node_id}")
        return {"status": "registered", "node_id": node.node_id}

    def authenticate(self, node_id: str, node_key: str) -> bool:
        """Verify a node's authentication key."""
        data = self._load()
        nodes = data.get("nodes", {})
        node = nodes.get(node_id)
        if not node:
            return False
        key_hash = hashlib.sha256(node_key.encode()).hexdigest()
        return hmac.compare_digest(key_hash, node.get("key_hash", ""))

    def get_node(self, node_id: str) -> dict | None:
        data = self._load()
        return data.get("nodes", {}).get(node_id)

    def list_nodes(self) -> dict:
        data = self._load()
        nodes = data.get("nodes", {})
        # Strip key hashes for API response
        safe = {}
        for nid, info in nodes.items():
            safe[nid] = {k: v for k, v in info.items() if k != "key_hash"}
        return safe

    def update_sync_stats(self, node_id: str, direction: str, count: int):
        """Update sync statistics for a node."""
        data = self._load()
        nodes = data.get("nodes", {})
        if node_id in nodes:
            nodes[node_id]["last_sync"] = datetime.now(timezone.utc).isoformat()
            if direction == "push":
                nodes[node_id]["push_count"] = nodes[node_id].get("push_count", 0) + count
            elif direction == "pull":
                nodes[node_id]["pull_count"] = nodes[node_id].get("pull_count", 0) + count
            data["nodes"] = nodes
            self._save(data)


# --- Federation Operations ---

registry = FederationRegistry()


def authenticate_node(node_id: str, node_key: str) -> bool:
    """Authenticate a federation node."""
    return registry.authenticate(node_id, node_key)


def process_push(mem_instance, req: FederationPushRequest) -> dict:
    """Process incoming memories from a remote node.

    Deduplicates by checking similarity before adding.
    """
    if not authenticate_node(req.node_id, req.node_key):
        return {"status": "error", "error": "Authentication failed"}

    added = 0
    skipped = 0
    errors = 0

    for memory in req.memories:
        content = memory.get("content", "")
        if not content or len(content) < 10:
            skipped += 1
            continue

        try:
            # Check for near-duplicates
            existing = mem_instance.search(content, user_id="yoni", limit=3)
            if isinstance(existing, dict):
                existing = existing.get("results", [])

            is_duplicate = any(
                r.get("score", 0) > 0.92
                for r in existing
            )

            if is_duplicate:
                skipped += 1
                continue

            # Add with federation metadata
            meta = memory.get("metadata", {})
            meta.update({
                "source": f"federation:{req.node_id}",
                "agent": memory.get("agent", "main"),
                "type": memory.get("type", "fact"),
                "federated_from": req.node_id,
                "federated_at": datetime.now(timezone.utc).isoformat(),
                "original_timestamp": memory.get("timestamp", ""),
            })

            mem_instance.add(content, user_id="yoni", metadata=meta)
            added += 1

        except Exception as e:
            logger.error(f"Federation push error for memory: {e}")
            errors += 1

    registry.update_sync_stats(req.node_id, "push", added)

    return {
        "status": "ok",
        "added": added,
        "skipped": skipped,
        "errors": errors,
    }


def process_pull(mem_instance, req: FederationPullRequest) -> dict:
    """Process a pull request from a remote node.

    Returns memories matching the filter criteria.
    """
    if not authenticate_node(req.node_id, req.node_key):
        return {"status": "error", "error": "Authentication failed"}

    try:
        all_mems = mem_instance.get_all(user_id="yoni", limit=10000)
        if isinstance(all_mems, dict):
            all_mems = all_mems.get("results", [])

        # Filter by timestamp
        if req.since:
            since_dt = datetime.fromisoformat(req.since.replace("Z", "+00:00"))
            filtered = []
            for m in all_mems:
                meta = m.get("metadata", {})
                created = meta.get("extracted_at", meta.get("created_at", ""))
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        if created_dt >= since_dt:
                            filtered.append(m)
                    except ValueError:
                        pass
            all_mems = filtered

        # Filter by agent
        if req.agents:
            all_mems = [
                m for m in all_mems
                if m.get("metadata", {}).get("agent") in req.agents
            ]

        # Filter by type
        if req.memory_types:
            all_mems = [
                m for m in all_mems
                if m.get("metadata", {}).get("type") in req.memory_types
            ]

        # Apply limit
        results = all_mems[:req.limit]
        registry.update_sync_stats(req.node_id, "pull", len(results))

        return {
            "status": "ok",
            "memories": results,
            "count": len(results),
            "total_available": len(all_mems),
        }

    except Exception as e:
        logger.error(f"Federation pull error: {e}")
        return {"status": "error", "error": str(e)}


def federation_status() -> dict:
    """Get federation health status."""
    nodes = registry.list_nodes()
    return {
        "status": "ok",
        "node_count": len(nodes),
        "nodes": nodes,
        "role": "master",
    }
