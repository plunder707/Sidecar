# Sidecar  
**Universal Async/Sync Bridge**

Run any async library from synchronous code – safely and cleanly – without `"Event loop is closed"` errors.

Bridge **httpx**, **Playwright**, **FastAPI**, and more into **Flask**, **Django**, CLIs, or scripts.

---
## ✨ Universal Compatibility

Sidecar auto-detects your Python runtime and picks the fastest safe execution engine.

| Python Version | Engine            | Performance       | Notes                                   |
|----------------|-------------------|-------------------|-----------------------------------------|
| **3.14+**      | Sub-interpreters  | **Up to 40× faster** | True parallelism with fast data passing |
| **3.13t**      | Free-threading    | **Near-zero overhead** | No GIL. Fully threaded execution        |
| **3.9 – 3.12** | Process / thread pool | **Stable & compatible** | Works on legacy systems                 |

No config required: you get the best engine available for the interpreter you’re running.

---

## ⚡ Quickstart

### Drop-in single file

Just copy `sidecar.py` into your project and import it.

No extra dependencies. Works in any project layout.

---

## 📣 Sidecar Information

* **Python 3.14 Ready** – PEP 734-style subinterpreters via InterpreterPoolExecutor
* **Free-threading Support (3.13, no-GIL)** – Automatically leverages the no-GIL runtime when available.
* **Deadlock Protection** – Recursive or nested calls are guarded with loop detection to avoid freezes.
* **Context-Aware** – Fully supports `contextvars` so things like request IDs and traces survive the bridge.

---

## ⭐ Key Features

* **Universal Bridge**
  Run async functions from:

  * Standard synchronous scripts
  * Legacy web stacks (Flask, Django, etc.)
  * Jupyter notebooks
  * CLI tools and background workers

* **Progressive Enhancement**
  Automatically upgrades from:

  * Process pool → Thread pool → Sub-interpreter pool
    depending on what your Python version supports.

* **Streaming Support**
  Turn async generators (e.g. LLM token streams) into standard Python iterators you can use in sync code.

* **Zero External Dependencies**
  Uses only the Python standard library. No heavy frameworks, no bloat.

* **Automatic Cleanup**
  Registers `atexit` hooks so background workers are stopped cleanly and won’t hang your CLI or tests.

---

## Motivation - The “Colored Function” Problem

In Python, functions effectively come in two “colors”:

* **Blue** – regular synchronous functions
* **Red** – `async def` functions

Rules:

* Blue ➜ Blue (OK)

* Red ➜ Red (OK)

* **Blue ➜ Red (problematic)**
  You can’t `await` inside a synchronous function. Trying to wire async libraries into sync code leads to:

* “just one more `async`” creeping up the stack

* brittle `asyncio.run(...)` calls

* conflicts with already-running loops (e.g. FastAPI, Jupyter)

This is **async pollution**: one async dependency forces your entire call stack to turn red.

**Sidecar acts as the bridge.**

### Standard Python vs. Sidecar

| Aspect           | Standard Python Async                 | With Sidecar                       |
| ---------------- | ------------------------------------- | ---------------------------------- |
| Call stack       | Async spreads everywhere              | Async stays isolated               |
| Refactoring cost | High (propagate `async` + `await`)    | Minimal (drop-in usage)            |
| Mental model     | Manage event loops and tasks yourself | Just call functions from sync code |
| Typical errors   | “Event loop is closed”, nested loops  | Guarded, loop-safe execution       |

---

## 📋 Usage & Documentation

Sidecar spins up a dedicated background worker (thread / sub-interpreter / process) with its own event loop. You call into it from synchronous code using a small, focused API.

---

## 1. The Basics – `run_sync`

Stop battling:

* `RuntimeError: Event loop is closed`
* `RuntimeError: This event loop is already running`
* Syntax errors from `await` in sync code

Instead, just **run the coroutine**:

```python
from sidecar import run_sync
import httpx


async def fetch_events():
    async with httpx.AsyncClient() as client:
        return await client.get("https://api.github.com/events")


def main() -> None:
    # GOOD: run the coroutine from sync code
    response = run_sync(fetch_events())
    print(f"Status: {response.status_code}")


if __name__ == "__main__":
    main()
```

No explicit event-loop management, no refactor of your call stack to `async`.

---

## 2. Streaming Data – `stream`

Turn any async generator (e.g. from OpenAI / Anthropic SDKs) into a plain iterator. Ideal for server-sent events, chunked responses, and CLI progress streaming.

```python
from sidecar import stream
import asyncio


# Imagine this is from an async LLM SDK
async def mock_llm_stream():
    words = ["Sidecar", " ", "makes", " ", "async", " ", "easy!"]
    for word in words:
        yield word
        await asyncio.sleep(0.05)


def generate_text() -> None:
    print("AI says: ", end="", flush=True)

    # Works like any normal iterable in sync code
    for token in stream(mock_llm_stream()):
        print(token, end="", flush=True)

    print()


if __name__ == "__main__":
    generate_text()
```

No need to expose async all the way up to your route handlers or CLI entry points.

---

## 3. CPU Offloading – `run_cpu`

Push heavy CPU work into a separate worker so your main thread stays responsive (for GUIs, web handlers, or streaming responses).

Sidecar automatically chooses the best engine:

* Sub-interpreters (3.14+)
* Free-threaded runtime (3.13t)
* Process or thread pools (3.9–3.12)

```python
from sidecar import run_cpu


def heavy_computation(data) -> int:
    # Normally this might freeze your UI or block request handling
    return sum(x * x for x in data)


def main() -> None:
    result = run_cpu(heavy_computation, range(10_000_000))
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

You focus on the computation; Sidecar focuses on safe parallel execution.

---

## 4. Flask / Django Integration

Add async capabilities to existing sync web stacks without rewriting them.

```python
from flask import Flask, jsonify
from sidecar import run_sync

app = Flask(__name__)


# Assume this lives in another module
# async def fetch_profile(user_id: int) -> dict: ...


@app.route("/analyze")
def analyze_data():
    # 1. Call your modern async service from sync Flask
    user_data = run_sync(fetch_profile(123))

    # 2. Return standard JSON as usual
    return jsonify(user_data)


if __name__ == "__main__":
    app.run(debug=True)
```

Same pattern works in Django views, management commands, or DRF endpoints.

---

## 🛠️ Advanced Configuration – `Sidecar` Instances

If you need more control (e.g. separate pools per subsystem), create your own bridge instance.

```python
from sidecar import Sidecar


def send_telemetry_event(event_type: str) -> None:
    # Replace with your own logging / metrics implementation
    print(f"[telemetry] {event_type}")


# Isolated bridge for analytics / telemetry
analytics_bridge = Sidecar(
    workers=8,              # Scale up workers for heavy loads
    name="AnalyticsWorker", # Name appears in logs / debuggers
    daemon=True             # Auto-exit when the main app shuts down
)


def record_login() -> None:
    # Fire-and-forget background work
    analytics_bridge.submit(send_telemetry_event, "user_login")
```

Use separate `Sidecar` instances for:

* Analytics vs. request handling
* Background jobs vs. interactive workflows
* Tenant- or feature-specific isolation

---

## 🤖 AI Server — `sidecar_ai.py`

Turn any sync AI agent into a **concurrent, multi-GPU-ready OpenAI-compatible server** — in 3 lines.

### The problem it solves

Sync AI agents that use thread-affine tools (Playwright browser automation, psycopg3, etc.) are stuck at `ThreadPoolExecutor(max_workers=1)` — **one concurrent user, max**. Swapping backends or adding a second GPU requires custom plumbing.

`sidecar_ai.py` fixes this automatically.

### Architecture

```
User A ──┐
User B ──┤──► FastAPI (async) ──► SessionPool ──► AgentSession 0 → ThreadExec → GPU 0
User C ──┤                             │        ──► AgentSession 1 → ThreadExec → GPU 0
User D ──┘                             │        ──► AgentSession 2 → ThreadExec → GPU 1
                                       │        ──► AgentSession 3 → ThreadExec → GPU 1
                                  (async wait                     ▲
                                  if all busy)            EndpointRouter
                                                          (round-robin)
```

Each session has its **own OS thread** — Playwright and other thread-affine tools are safe.
Callers wait **async** (not blocking) when all sessions are busy.

### Quick Start

```python
from sidecar_ai import AIServer, auto_discover

server = AIServer(
    agent_factory=lambda ep: MyAgent(vlm_endpoint=ep),
    sessions=4,                  # concurrent users
    endpoints=auto_discover(),   # finds LM Studio / Ollama / vLLM automatically
)
server.serve()  # OpenAI-compatible API on localhost:8000
```

Your agent needs exactly one method:

```python
class MyAgent:
    def __init__(self, vlm_endpoint: str): ...
    def process_turn(self, user_input: str) -> str: ...
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Pool status, session counts, active endpoints |
| `POST` | `/chat` | Simple: `{"message": "..."}` → `{"response": "..."}` |
| `POST` | `/v1/chat/completions` | Full OpenAI-compatible (system prompt forwarding) |

### Backend Compatibility

| Backend | Default Port | Auto-discovered |
|---------|-------------|-----------------|
| LM Studio | 1234 | ✓ |
| Ollama | 11434 | ✓ |
| vLLM / SGLang | 8000 | ✓ |
| text-gen-webui | 7860 | ✓ |
| OpenAI API | — | Pass URL directly |

### 1 GPU vs 2 GPUs

| Setup | Config | Result |
|-------|--------|--------|
| 1 GPU, 1 LM Studio | `sessions=2, endpoints=auto_discover()` | 2 concurrent users, 1 backend |
| 2 GPUs, 2 LM Studio | `sessions=4, endpoints=["...:1234", "...:1235"]` | 4 concurrent users, 2 backends |
| 2 GPUs, 1 vLLM (tensor parallel) | `sessions=4, endpoints=["...:8000"]` | 4 concurrent users, 1 big model |

### Install

```bash
pip install fastapi uvicorn   # sidecar.py has no deps; sidecar_ai.py adds these
```

See `examples/single_gpu.py` and `examples/multi_gpu.py` for runnable templates.

---
---

## 📝 Notes

### Quick Reference

* `run_sync(awaitable)`
  → Run a coroutine in the background worker and return its result.

* `stream(async_gen_func_or_awaitable)`
  → Consume an async generator as a sync iterator.

* `run_cpu(fn, *args, **kwargs)`
  → Run a CPU-bound callable in a worker (process/thread/sub-interpreter).

* `Sidecar(workers=..., name=..., daemon=True)`
  → Create an isolated bridge instance with its own worker pool.

* `sidecar_instance.submit(fn, *args, **kwargs)`
  → Fire-and-forget / future-style work (no `await` in the caller).

### Behavior & Compatibility

* **Exceptions**
  Exceptions raised inside the async worker (or CPU worker) are propagated back to the caller as normal Python exceptions. Tracebacks are preserved where possible.

* **Picklability (Python 3.9–3.12)**
  On Python 3.9–3.12, CPU work may run in a process pool. Functions and arguments must be picklable (no local closures, no open file handles, etc.).
