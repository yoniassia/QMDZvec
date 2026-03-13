"""MemClawz v6 MCP Server — exposes fleet memory to MCP clients via STDIO.

v6 additions: composite scoring, graph search, compaction tools.
"""
import json
import sys
from mem0 import Memory

from .config import MEM0_CONFIG_LITE
from .scoring import score_results

mem = Memory.from_config(MEM0_CONFIG_LITE)


def _unwrap(result):
    """Unwrap Mem0 results."""
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    if isinstance(result, list):
        return result
    return []


TOOLS = [
    {
        "name": "search_memory",
        "description": "Search YoniClaw fleet memories semantically with composite scoring",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "agent": {"type": "string", "description": "Filter by agent name"},
                "type": {
                    "type": "string",
                    "description": "Filter by memory type (fact/decision/preference/procedure/relationship/event/insight)",
                },
                "limit": {"type": "integer", "default": 10},
                "use_composite": {"type": "boolean", "default": True, "description": "Use composite scoring (v6)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_memory",
        "description": "Store a new memory in the fleet memory system",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "agent": {"type": "string", "default": "main"},
                "type": {"type": "string", "default": "fact"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_agent_memories",
        "description": "Get all memories for a specific agent",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["agent"],
        },
    },
    {
        "name": "compact_session",
        "description": "Trigger session compaction for a given session",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session identifier"},
                "agent_id": {"type": "string", "default": "main"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "reflect",
        "description": "Trigger sleep-time reflection on recent memories",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "default": 24, "description": "Hours to look back"},
            },
            "required": [],
        },
    },
    {
        "name": "memory_stats",
        "description": "Get memory system statistics and health",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def handle_request(request: dict) -> dict:
    """Handle a single MCP JSON-RPC request."""
    method = request.get("method")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "memclawz", "version": "6.0.0"},
        }

    elif method == "tools/list":
        return {"tools": TOOLS}

    elif method == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        if tool_name == "search_memory":
            results = _unwrap(mem.search(args["query"], user_id="yoni", limit=args.get("limit", 10)))
            # Post-filter
            if args.get("agent"):
                results = [r for r in results if r.get("metadata", {}).get("agent") == args["agent"]]
            if args.get("type"):
                results = [r for r in results if r.get("metadata", {}).get("type") == args["type"]]
            # Composite scoring
            if args.get("use_composite", True):
                results = score_results(results)
            return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}

        elif tool_name == "add_memory":
            result = mem.add(
                args["content"],
                user_id="yoni",
                metadata={
                    "agent": args.get("agent", "main"),
                    "type": args.get("type", "fact"),
                    "source": "mcp",
                },
            )
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

        elif tool_name == "get_agent_memories":
            all_mems = _unwrap(mem.get_all(user_id="yoni"))
            filtered = [m for m in all_mems if m.get("metadata", {}).get("agent") == args["agent"]]
            return {
                "content": [
                    {"type": "text", "text": json.dumps(filtered[: args.get("limit", 20)], indent=2, default=str)}
                ]
            }

        elif tool_name == "compact_session":
            from .compactor import SessionCompactor
            compactor = SessionCompactor()
            result = compactor.compact_session(
                session_id=args["session_id"],
                messages=[],  # MCP caller would need to provide messages
                agent_id=args.get("agent_id", "main"),
            )
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

        elif tool_name == "reflect":
            from .reflection import ReflectionEngine
            engine = ReflectionEngine()
            result = engine.reflect(hours=args.get("hours", 24))
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

        elif tool_name == "memory_stats":
            all_mems = _unwrap(mem.get_all(user_id="yoni", limit=10000))
            types = {}
            agents = {}
            for m in all_mems:
                meta = m.get("metadata", {})
                t = meta.get("type", "unknown")
                a = meta.get("agent", "unknown")
                types[t] = types.get(t, 0) + 1
                agents[a] = agents.get(a, 0) + 1
            stats = {
                "total_memories": len(all_mems),
                "by_type": types,
                "by_agent": agents,
            }
            # Add compaction status
            try:
                from .compactor import get_compaction_status
                stats["compaction"] = get_compaction_status()
            except Exception:
                pass
            return {"content": [{"type": "text", "text": json.dumps(stats, indent=2, default=str)}]}

        return {"error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

    return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
    """Run MCP STDIO transport."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            result = handle_request(request)
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
            print(json.dumps(response, default=str), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": 0,
                "error": {"code": -32603, "message": str(e)},
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
