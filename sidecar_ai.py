"""
sidecar_ai.py — Universal AI Server Adapter
============================================

Turns any sync agent into a concurrent, multi-GPU-ready OpenAI-compatible
API server. Built on sidecar.py's async/sync bridge.

QUICK START (3 lines):

    from sidecar_ai import AIServer, auto_discover

    server = AIServer(
        agent_factory=lambda ep: MyAgent(vlm_endpoint=ep),
        sessions=4,
        endpoints=auto_discover(),
    )
    server.serve()

AGENT INTERFACE:
    Your agent needs exactly one method:
        def process_turn(self, user_input: str) -> str

    That's it. Works with ChatAI, LangChain, LlamaIndex, or any custom agent.

THREAD ISOLATION:
    Each session gets its own ThreadPoolExecutor(max_workers=1).
    Thread-affine tools (Playwright, psycopg3) are safe — they always
    run on the same OS thread they were created on.

ENDPOINTS AUTO-DETECTED:
    LM Studio (1234), Ollama (11434), vLLM (8000), text-gen-webui (7860)
    Falls back to localhost:1234 if nothing is found.

MULTI-GPU:
    Two LM Studio instances on different ports = automatic load balancing.
    Sessions distributed round-robin across all discovered endpoints.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

__version__ = "1.0.0"
__all__ = [
    "auto_discover",
    "EndpointRouter",
    "AgentSession",
    "SessionPool",
    "AIServer",
]

# ── Optional: FastAPI + uvicorn ────────────────────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


# ── Endpoint Discovery ─────────────────────────────────────────────────────────

#: Default ports for common local LLM backends
DEFAULT_PORTS: List[int] = [1234, 11434, 8000, 7860, 1235]

#: Backend names by port (for display)
_PORT_NAMES: Dict[int, str] = {
    1234: "LM Studio",
    11434: "Ollama",
    8000: "vLLM / SGLang",
    7860: "text-gen-webui",
    1235: "LM Studio (2nd)",
}


def auto_discover(
    ports: List[int] = DEFAULT_PORTS,
    host: str = "localhost",
    timeout: float = 1.0,
) -> List[str]:
    """
    Scan localhost for running LLM backends.

    Probes each port's /v1/models endpoint (OpenAI-compatible).
    Returns list of live base URLs, e.g. ["http://localhost:1234"].

    Args:
        ports:   Ports to check. Defaults cover LM Studio, Ollama, vLLM, webui.
        host:    Host to scan. Default: "localhost".
        timeout: Per-port timeout in seconds.

    Returns:
        List of reachable OpenAI-compatible endpoint base URLs.

    Example:
        endpoints = auto_discover()
        # → ["http://localhost:1234"]  (if LM Studio is running)
    """
    live = []
    for port in ports:
        url = f"http://{host}:{port}/v1/models"
        try:
            req = Request(url, headers={"User-Agent": f"sidecar-ai/{__version__}"})
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    base = f"http://{host}:{port}"
                    name = _PORT_NAMES.get(port, f"port {port}")
                    print(f"  [sidecar-ai] Found {name} at {base}")
                    live.append(base)
        except (URLError, OSError):
            pass
    return live


# ── Endpoint Router ────────────────────────────────────────────────────────────


class EndpointRouter:
    """
    Round-robin load balancer across LLM endpoints. Thread-safe.

    Example:
        router = EndpointRouter(["http://localhost:1234", "http://localhost:1235"])
        ep = router.next()   # alternates per call
    """

    def __init__(self, endpoints: List[str]) -> None:
        if not endpoints:
            raise ValueError("EndpointRouter requires at least one endpoint")
        self._endpoints = list(endpoints)
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        """Get next endpoint (round-robin, thread-safe)."""
        with self._lock:
            ep = self._endpoints[self._idx % len(self._endpoints)]
            self._idx += 1
            return ep

    def is_healthy(self, endpoint: str, timeout: float = 2.0) -> bool:
        """Quick health check — GET /v1/models, returns True if reachable."""
        try:
            req = Request(
                f"{endpoint}/v1/models",
                headers={"User-Agent": f"sidecar-ai/{__version__}"},
            )
            with urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def endpoints(self) -> List[str]:
        """All registered endpoints."""
        return list(self._endpoints)


# ── Agent Session ──────────────────────────────────────────────────────────────


@dataclass
class AgentSession:
    """
    A sync agent wrapped with its own dedicated OS thread.

    Thread isolation is required for Playwright, psycopg3, and any other
    thread-affine library — they must run on the same thread they were
    initialized on. ThreadPoolExecutor(max_workers=1) guarantees this.

    Created by SessionPool.create() — don't instantiate directly.
    """

    agent: Any
    """The sync agent. Must have process_turn(str) -> str."""

    executor: ThreadPoolExecutor
    """Dedicated single-thread executor. Owns the agent's thread."""

    endpoint: str = ""
    """LLM endpoint URL this session is wired to."""

    _id: str = field(default_factory=lambda: uuid.uuid4().hex[:8], repr=False)


# ── Session Pool ───────────────────────────────────────────────────────────────


class SessionPool:
    """
    asyncio.Queue-based pool of agent sessions.

    Sessions are checked out via acquire() and returned automatically,
    even if the request raises an exception. Callers block (async wait)
    if all sessions are busy — natural backpressure.

    Example:
        async with pool.acquire() as session:
            result = await loop.run_in_executor(
                session.executor,
                session.agent.process_turn,
                user_message,
            )
        # session automatically returned to pool here
    """

    def __init__(self, sessions: List[AgentSession]) -> None:
        self._all = list(sessions)
        self._queue: asyncio.Queue[AgentSession] = asyncio.Queue()
        for s in sessions:
            self._queue.put_nowait(s)

    @classmethod
    def create(
        cls,
        agent_factory: Callable[[str], Any],
        n: int,
        router: EndpointRouter,
    ) -> "SessionPool":
        """
        Build N sessions, assigning endpoints round-robin from the router.

        Args:
            agent_factory: Callable(endpoint_url) -> agent instance.
                           Called once per session at startup.
            n:             Number of concurrent sessions.
            router:        EndpointRouter for endpoint assignment.

        Returns:
            SessionPool ready to serve requests.
        """
        sessions = []
        for i in range(n):
            ep = router.next()
            agent = agent_factory(ep)
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"sidecar-session-{i}",
            )
            sessions.append(AgentSession(agent=agent, executor=executor, endpoint=ep))
        return cls(sessions)

    @asynccontextmanager
    async def acquire(self) -> Generator[AgentSession, None, None]:
        """
        Async context manager. Waits for a free session, yields it,
        returns it to the pool on exit (success or exception).
        """
        session = await self._queue.get()
        try:
            yield session
        finally:
            await self._queue.put(session)

    @property
    def size(self) -> int:
        """Total number of sessions in the pool."""
        return len(self._all)

    @property
    def available(self) -> int:
        """Number of sessions currently idle."""
        return self._queue.qsize()

    def shutdown(self, wait: bool = False) -> None:
        """Shutdown all session executors."""
        for session in self._all:
            try:
                session.executor.shutdown(wait=wait, cancel_futures=True)
            except Exception:
                pass


# ── AI Server ──────────────────────────────────────────────────────────────────


class AIServer:
    """
    OpenAI-compatible FastAPI server wrapping any sync agent.

    Drop-in replacement for the chatai2_api.py pattern — but agent-agnostic.
    Works with any object that has process_turn(str) -> str.

    Endpoints:
        GET  /health                  Pool status, active sessions, endpoints
        POST /chat                    Simple: {"message": "..."} → {"response": "..."}
        POST /v1/chat/completions     OpenAI-compatible (system prompt forwarding included)

    Args:
        agent_factory: Callable(vlm_endpoint: str) -> agent.
                       Called once per session at startup.
        sessions:      Number of concurrent sessions (default: 2).
        endpoints:     LLM endpoint URLs. If None, auto_discover() is called.
                       Falls back to http://localhost:1234 if nothing found.

    Example:
        server = AIServer(
            agent_factory=lambda ep: ChatAI(vlm_endpoint=ep),
            sessions=4,
            endpoints=auto_discover(),
        )
        server.serve()
    """

    def __init__(
        self,
        agent_factory: Callable[[str], Any],
        sessions: int = 2,
        endpoints: Optional[List[str]] = None,
    ) -> None:
        if not _HAS_FASTAPI:
            raise ImportError(
                "AIServer requires FastAPI and uvicorn:\n"
                "  pip install fastapi uvicorn"
            )

        if endpoints is None:
            print("[sidecar-ai] Discovering endpoints...")
            endpoints = auto_discover()

        if not endpoints:
            print("[sidecar-ai] No endpoints found, defaulting to http://localhost:1234")
            endpoints = ["http://localhost:1234"]

        self._agent_factory = agent_factory
        self._n_sessions = sessions
        self._endpoints = endpoints
        self._router = EndpointRouter(endpoints)
        self._pool: Optional[SessionPool] = None
        self._app = self._build_app()

    def _build_app(self) -> "FastAPI":
        # ── Pydantic models ────────────────────────────────────────────────────

        class SimpleRequest(BaseModel):
            message: str

        class SimpleResponse(BaseModel):
            response: str

        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatCompletionRequest(BaseModel):
            model: str = "sidecar-ai"
            messages: List[ChatMessage]
            temperature: Optional[float] = 0.7
            max_tokens: Optional[int] = None
            stream: Optional[bool] = False

        class ChatCompletionChoice(BaseModel):
            index: int
            message: ChatMessage
            finish_reason: str = "stop"

        class ChatCompletionUsage(BaseModel):
            prompt_tokens: int = 0
            completion_tokens: int = 0
            total_tokens: int = 0

        class ChatCompletionResponse(BaseModel):
            id: str
            object: str = "chat.completion"
            created: int
            model: str
            choices: List[ChatCompletionChoice]
            usage: ChatCompletionUsage

        # ── FastAPI app ────────────────────────────────────────────────────────

        app = FastAPI(
            title="Sidecar AI Server",
            description="Universal concurrent AI server adapter",
            version=__version__,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.on_event("startup")
        async def _startup() -> None:
            print(
                f"[sidecar-ai] Initializing {self._n_sessions} sessions "
                f"across {len(self._endpoints)} endpoint(s)..."
            )
            self._pool = SessionPool.create(
                self._agent_factory, self._n_sessions, self._router
            )
            print("[sidecar-ai] Ready.")

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            if self._pool:
                self._pool.shutdown(wait=False)

        # ── /health ────────────────────────────────────────────────────────────

        @app.get("/health")
        async def health() -> Dict[str, Any]:
            """Pool status, session counts, and active endpoints."""
            pool = self._pool
            return {
                "status": "ok",
                "sessions_total": pool.size if pool else 0,
                "sessions_available": pool.available if pool else 0,
                "endpoints": self._endpoints,
            }

        # ── /chat ──────────────────────────────────────────────────────────────

        @app.post("/chat", response_model=SimpleResponse)
        async def simple_chat(body: SimpleRequest) -> SimpleResponse:
            """Simple single-message endpoint."""
            if not self._pool:
                raise HTTPException(status_code=503, detail="Server not ready")
            loop = asyncio.get_event_loop()
            try:
                async with self._pool.acquire() as session:
                    response = await loop.run_in_executor(
                        session.executor,
                        session.agent.process_turn,
                        body.message,
                    )
                return SimpleResponse(response=response or "")
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        # ── /v1/chat/completions ───────────────────────────────────────────────

        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
            """
            OpenAI-compatible chat completions endpoint.

            System message is extracted and prepended to the user message
            before calling process_turn() — compatible with GAIA and other
            benchmarks that use system prompts for formatting instructions.
            """
            if not self._pool:
                raise HTTPException(status_code=503, detail="Server not ready")

            if request.stream:
                raise HTTPException(status_code=400, detail="Streaming not yet supported")

            # Extract system message (prepend to user message)
            system_content: Optional[str] = None
            for msg in request.messages:
                if msg.role == "system":
                    system_content = msg.content
                    break

            # Get last user message
            user_message: Optional[str] = None
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_message = msg.content
                    break

            if not user_message:
                raise HTTPException(status_code=400, detail="No user message found")

            if system_content:
                user_message = f"{system_content}\n\n{user_message}"

            loop = asyncio.get_event_loop()
            try:
                async with self._pool.acquire() as session:
                    response_text = await loop.run_in_executor(
                        session.executor,
                        session.agent.process_turn,
                        user_message,
                    )
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:20]}",
                created=int(time.time()),
                model="sidecar-ai",
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response_text or "",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(),
            )

        return app

    def serve(self, host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
        """
        Start the server. Blocks until stopped (Ctrl+C).

        Args:
            host: Bind host. Use "0.0.0.0" to expose on network.
            port: Port (default: 8000).
            **kwargs: Passed through to uvicorn.run().
        """
        print(f"[sidecar-ai] Serving on http://{host}:{port}")
        uvicorn.run(self._app, host=host, port=port, **kwargs)

    @property
    def app(self) -> "FastAPI":
        """The underlying FastAPI app (for ASGI testing or custom integration)."""
        return self._app
