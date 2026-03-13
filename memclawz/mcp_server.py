"""MemClawz MCP Server — exposes fleet memory to MCP clients via STDIO."""
import json
import sys
from mem0 import Memory

from .config import MEM0_CONFIG_LITE

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
        "description": "Search YoniClaw fleet memories semantically",
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
]


def handle_request(request: dict) -> dict:
    """Handle a single MCP JSON-RPC request."""
    method = request.get("method")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "memclawz", "version": "5.0.0"},
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
