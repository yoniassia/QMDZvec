"""Task routing for MemClawz agents."""

from __future__ import annotations

from typing import Any

AGENT_REGISTRY = {
    "infraclaw": {
        "session_key": "agent:infraclaw:webchat:main",
        "domains": ["servers", "dns", "deploys", "docker", "security", "monitoring", "infrastructure", "caddy", "nginx", "ssh", "systemd", "fleet", "hetzner"],
        "emoji": "🏗️",
        "model": "openai/gpt-5.4",
    },
    "tradeclaw": {
        "session_key": "agent:tradeclaw:webchat:main",
        "domains": ["markets", "trading", "defi", "portfolio", "bots", "crypto price", "btc", "eth", "stocks", "binance", "hyperliquid"],
        "emoji": "💰",
        "model": "openai/gpt-5.4",
    },
    "appsclaw": {
        "session_key": "agent:appsclaw:webchat:main",
        "domains": ["app store", "apps", "swagger", "etoro apps", "liquidation map", "app building"],
        "emoji": "📱",
        "model": "openai/gpt-5.4",
    },
    "tradingdataclaw": {
        "session_key": "agent:tradingdataclaw:webchat:main",
        "domains": ["market data", "alphaear", "data pipelines", "quantclaw", "backtesting", "vectorbt", "data modules"],
        "emoji": "📡",
        "model": "openai/gpt-5.4",
    },
    "commsclaw": {
        "session_key": "agent:commsclaw:webchat:main",
        "domains": ["email", "social", "calls", "communications", "send email", "agentmail"],
        "emoji": "📬",
        "model": "openai/gpt-5.4",
    },
    "coinresearchclaw": {
        "session_key": "agent:coinresearchclaw:webchat:main",
        "domains": ["coin research", "due diligence", "crypto dd", "token analysis", "coin scan", "memos"],
        "emoji": "🧠",
        "model": "openai/gpt-5.4",
    },
    "coinsclaw": {
        "session_key": "agent:coinsclaw:webchat:main",
        "domains": ["crypto listings", "coin tracking", "mica", "whitepaper", "listing status"],
        "emoji": "🪙",
        "model": "openai/gpt-5.4",
    },
    "peopleclaw": {
        "session_key": "agent:peopleclaw:webchat:main",
        "domains": ["hr", "team", "culture", "hiring", "people", "org chart", "employees"],
        "emoji": "👥",
        "model": "openai/gpt-5.4",
    },
    "paperclipclaw": {
        "session_key": "agent:paperclipclaw:webchat:main",
        "domains": ["fleet orchestration", "agent tooling", "paperclip", "agent management", "fleet status"],
        "emoji": "📎",
        "model": "openai/gpt-5.4",
    },
    "qaclaw": {
        "session_key": "agent:qaclaw:webchat:main",
        "domains": ["qa", "testing", "applause", "playwright", "bugs", "test cycles", "clawqa"],
        "emoji": "🧪",
        "model": "anthropic/claude-opus-4-6",
    },
}

DEFAULT_AGENT_INFO = {
    "session_key": "main",
    "emoji": "🦞",
    "model": "anthropic/claude-opus-4-6",
}


class MemClawzRouter:
    def __init__(self, mem_instance=None):
        self.mem = mem_instance
        self.registry = AGENT_REGISTRY

    def _search_memories(self, task: str) -> list[dict[str, Any]]:
        if not self.mem:
            return []
        results = self.mem.search(task, user_id="yoni", limit=5)
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        if isinstance(results, list):
            return results
        return []

    def route(self, task: str, include_context: bool = True) -> dict[str, Any]:
        """Route a task to the right agent."""
        task_lower = task.lower()

        scores: dict[str, int] = {}
        for agent_id, info in self.registry.items():
            score = sum(1 for d in info["domains"] if d in task_lower)
            if score > 0:
                scores[agent_id] = score

        if scores:
            best = max(scores, key=scores.get)
            confidence = min(scores[best] / 3.0, 1.0)
        else:
            if self.mem:
                results = self._search_memories(task)
                agent_counts: dict[str, int] = {}
                for result in results:
                    agent = result.get("metadata", {}).get("agent", "main")
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                if agent_counts:
                    best = max(agent_counts, key=agent_counts.get)
                    confidence = 0.6
                else:
                    best = "main"
                    confidence = 0.3
            else:
                best = "main"
                confidence = 0.3

        memory_context: list[dict[str, Any]] = []
        if include_context and self.mem:
            try:
                results = self._search_memories(task)
                memory_context = [
                    {
                        "memory": result.get("memory", ""),
                        "agent": result.get("metadata", {}).get("agent", ""),
                        "score": result.get("score", 0),
                    }
                    for result in results
                ]
            except Exception:
                pass

        info = self.registry.get(best, DEFAULT_AGENT_INFO)
        matched_domains = [d for d in info.get("domains", []) if d in task_lower]

        return {
            "agent_id": best,
            "session_key": info.get("session_key", f"agent:{best}:webchat:main"),
            "emoji": info.get("emoji", "🦞"),
            "model": info.get("model", "anthropic/claude-opus-4-6"),
            "confidence": confidence,
            "reason": f"Matched domains: {matched_domains}" if scores else "Semantic fallback",
            "memory_context": memory_context,
            "fallback": "main" if confidence < 0.5 else None,
        }
