"""
mcp_server.py — Sidecar AI Server: MCP Example
================================================

Exposes your sync agent as an MCP tool — works with Claude Desktop,
Claude Code, Cursor, and any other MCP client.

Replace MyAgent with your actual agent (LangChain, LlamaIndex, etc.).

STDIO (Claude Desktop / Claude Code):
    python examples/mcp_server.py

    Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "my-agent": {
          "command": "python",
          "args": ["/absolute/path/to/examples/mcp_server.py"]
        }
      }
    }

HTTP (remote MCP clients):
    python examples/mcp_server.py --transport http --port 8001
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sidecar_ai import MCPServer, auto_discover


# ── Your agent (replace this) ──────────────────────────────────────────────────

class MyAgent:
    """
    Minimal stub agent. Replace with your real agent class.

    Requirements:
        - __init__(self, vlm_endpoint: str)
        - process_turn(self, user_input: str) -> str
    """

    def __init__(self, vlm_endpoint: str) -> None:
        self.endpoint = vlm_endpoint

    def process_turn(self, user_input: str) -> str:
        # Replace with your agent's real logic
        return f"[{self.endpoint}] Echo: {user_input}"


# ── Server ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sidecar MCP Server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP/SSE")
    parser.add_argument("--port", type=int, default=8001, help="Port for HTTP/SSE")
    args = parser.parse_args()

    server = MCPServer(
        agent_factory=lambda ep: MyAgent(vlm_endpoint=ep),
        sessions=2,
        endpoints=auto_discover(),
        name="my-agent",
    )

    if args.transport == "stdio":
        server.serve()  # stdio — no host/port needed
    else:
        server.serve(transport=args.transport, host=args.host, port=args.port)
