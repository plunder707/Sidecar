"""
single_gpu.py — Sidecar AI Server: Single GPU Example
======================================================

Minimal setup to turn any sync agent into a concurrent API server.
2 sessions share 1 GPU endpoint. Both requests run concurrently —
one waits async while the other is processing, no thread blocking.

Replace MyAgent with your actual agent (ChatAI, LangChain, etc.).

Usage:
    pip install fastapi uvicorn
    python examples/single_gpu.py

Then try:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/chat \\
         -H "Content-Type: application/json" \\
         -d '{"message": "hello"}'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sidecar_ai import AIServer, auto_discover


# ── Your agent (replace this) ──────────────────────────────────────────────────

class MyAgent:
    """
    Minimal stub agent. Replace with your real agent class.

    Requirements:
        - __init__(self, vlm_endpoint: str)  — receives the LLM endpoint URL
        - process_turn(self, user_input: str) -> str  — one call per request
    """

    def __init__(self, vlm_endpoint: str) -> None:
        self.endpoint = vlm_endpoint
        # Initialize your agent here (e.g. load model, start browser, etc.)
        print(f"    Agent ready → {vlm_endpoint}")

    def process_turn(self, user_input: str) -> str:
        # Replace with your agent's real logic
        return f"[{self.endpoint}] Echo: {user_input}"


# ── Server ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Find whatever LLM backend is running locally
    endpoints = auto_discover()
    if not endpoints:
        print("No LLM backend found. Start LM Studio or Ollama, then retry.")
        print("Falling back to http://localhost:1234")

    server = AIServer(
        agent_factory=lambda ep: MyAgent(vlm_endpoint=ep),
        sessions=2,           # 2 concurrent users on 1 GPU
        endpoints=endpoints,  # auto-discovered above
    )

    print("\nEndpoints:")
    print("  GET  http://localhost:8000/health")
    print("  POST http://localhost:8000/chat")
    print("  POST http://localhost:8000/v1/chat/completions  (OpenAI-compatible)\n")

    server.serve(host="127.0.0.1", port=8000)
