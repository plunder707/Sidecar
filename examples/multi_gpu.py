"""
multi_gpu.py — Sidecar AI Server: Multi-GPU Example
====================================================

Two LM Studio (or vLLM) instances on separate GPUs, separate ports.
4 sessions distributed round-robin: sessions 0,2 → GPU 0, sessions 1,3 → GPU 1.

GPU setup:
    GPU 0: LM Studio on port 1234  (or: vLLM --port 1234 --tensor-parallel-size 1)
    GPU 1: LM Studio on port 1235  (or: vLLM --port 1235 --tensor-parallel-size 1)

Alternative (one big model across both GPUs):
    vLLM with --tensor-parallel-size 2 on a single port → use single_gpu.py instead

Usage:
    python examples/multi_gpu.py

Concurrent throughput: 2x vs single GPU (GPU utilization doubles).
Model size limit: same per GPU (48GB total if using separate instances).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sidecar_ai import AIServer, EndpointRouter


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
        print(f"    Agent ready → {vlm_endpoint}")

    def process_turn(self, user_input: str) -> str:
        return f"[{self.endpoint}] Echo: {user_input}"


# ── Server ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Explicit endpoints — one per GPU
    all_endpoints = [
        "http://localhost:1234",  # GPU 0
        "http://localhost:1235",  # GPU 1
    ]

    # Health-check before starting — skip unreachable endpoints
    router = EndpointRouter(all_endpoints)
    live = [ep for ep in all_endpoints if router.is_healthy(ep)]

    if not live:
        print("No endpoints reachable. Start LM Studio on ports 1234 and 1235.")
        sys.exit(1)

    if len(live) < len(all_endpoints):
        missing = set(all_endpoints) - set(live)
        print(f"Warning: {missing} not reachable, using only {live}")

    print(f"Live endpoints: {live}")

    server = AIServer(
        agent_factory=lambda ep: MyAgent(vlm_endpoint=ep),
        sessions=4,      # 4 sessions distributed: 2 per GPU
        endpoints=live,
    )

    print("\nEndpoints:")
    print("  GET  http://localhost:8000/health")
    print("  POST http://localhost:8000/chat")
    print("  POST http://localhost:8000/v1/chat/completions\n")

    server.serve(host="127.0.0.1", port=8000)
