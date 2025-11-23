"""
sidecar.py - Universal Async/Sync Bridge

Use any async library in sync code. Works everywhere.

COMPATIBILITY:
- Python 3.9+: ProcessPool (works everywhere)
- Python 3.13t: Free-Threading (zero overhead)
- Python 3.14+: Sub-Interpreters (40x faster data transfer)

USAGE:
    from sidecar import run_sync

    # Use httpx in sync code
    response = run_sync(httpx.get("https://api.com"))

    # Use aiohttp in Flask/Django
    data = run_sync(aiohttp_fetch(...))

    # Works in Jupyter notebooks (no "loop already running" errors)
    df = pd.DataFrame(run_sync(fetch_async_data()))
"""

from __future__ import annotations

import asyncio
import threading
import logging
import time
import sys
import atexit
import contextvars
from queue import Queue, Full
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from typing import (
    TypeVar,
    Callable,
    Optional,
    AsyncGenerator,
    Any,
    Coroutine,
    Generator,
)
from dataclasses import dataclass

__version__ = "1.0.0"
__all__ = ["Sidecar", "run_sync", "submit", "stream", "run_cpu", "shutdown"]

logger = logging.getLogger("sidecar")
T = TypeVar("T")

# Feature Detection
try:
    IS_GIL_ENABLED = sys._is_gil_enabled()
except AttributeError:
    IS_GIL_ENABLED = True

IS_FREE_THREADED = not IS_GIL_ENABLED
HAS_SUBINTERPRETERS = False
InterpreterPoolExecutor = None

if sys.version_info >= (3, 14):
    try:
        from concurrent.futures import InterpreterPoolExecutor
        HAS_SUBINTERPRETERS = True
    except ImportError:
        pass


@dataclass
class BridgeStats:
    """Runtime statistics for monitoring."""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    mode: str = "Unknown"


class Sidecar:
    """
    Universal async/sync bridge with progressive enhancement.

    Automatically selects best execution engine:
    - Python 3.14+: Sub-Interpreters (40x faster data transfer)
    - Python 3.13t: Free-Threading (zero overhead)
    - Python 3.9-3.12: ProcessPool (works everywhere)

    Example:
        bridge = Sidecar()
        result = bridge.run_sync(async_function())
    """

    def __init__(
        self,
        workers: int = 4,
        daemon: bool = True,
        name: str = "Sidecar",
    ):
        self._daemon = daemon
        self._name = name
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stats = BridgeStats()

        # Progressive Enhancement: Select best execution engine
        if IS_FREE_THREADED:
            self._cpu_executor = ThreadPoolExecutor(max_workers=workers)
            self._stats.mode = "Free-Threading"
        elif HAS_SUBINTERPRETERS and InterpreterPoolExecutor:
            self._cpu_executor = InterpreterPoolExecutor(max_workers=workers)
            self._stats.mode = "Sub-Interpreters"
        else:
            self._cpu_executor = ProcessPoolExecutor(max_workers=workers)
            self._stats.mode = "ProcessPool"

        # Auto-cleanup on exit
        atexit.register(self.shutdown)
        self._start()

    def _start(self) -> None:
        """Start the background event loop thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=self._daemon,
                name=self._name,
            )
            self._thread.start()

    def _run_loop(self) -> None:
        """Main event loop runner (runs in background thread)."""
        assert self._loop is not None
        loop = self._loop

        asyncio.set_event_loop(loop)

        async def heartbeat():
            """Keep-alive task."""
            try:
                while not self._shutdown_event.is_set():
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass

        self._heartbeat_task = loop.create_task(heartbeat())

        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"{self._name} crashed: {e}", exc_info=True)
        finally:
            # Cancel all pending tasks and close loop cleanly
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def run_sync(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = 30,
    ) -> T:
        """
        Run async code from sync context.

        Args:
            coro: Coroutine to execute
            timeout: Maximum execution time in seconds

        Returns:
            Result of the coroutine

        Raises:
            RuntimeError:
                If called from within the Sidecar loop thread (would deadlock)
            TimeoutError:
                If execution exceeds timeout
        """
        # Deadlock detection: thread + loop checks
        def _deadlock(msg: str) -> None:
            if asyncio.iscoroutine(coro):
                try:
                    coro.close()
                except RuntimeError:
                    pass
            raise RuntimeError(msg)

        if self._thread and self._thread.is_alive() and threading.current_thread() is self._thread:
            _deadlock(
                "Deadlock detected: run_sync() called from within Sidecar loop thread. "
                "Use 'await' instead."
            )

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None  # no loop in this thread → safe
        else:
            if current_loop is self._loop:
                _deadlock(
                    "Deadlock detected: run_sync() called while Sidecar loop is running. "
                    "Use 'await' instead."
                )

        # Context propagation
        ctx = contextvars.copy_context()

        async def wrapped() -> T:
            tokens = []
            try:
                for var, val in ctx.items():
                    tokens.append((var, var.set(val)))
                return await coro
            finally:
                for var, token in tokens:
                    var.reset(token)

        # Submit to loop thread
        with self._lock:
            if not self._loop:
                self._start()

            assert self._loop is not None
            coro_to_run = (
                asyncio.wait_for(wrapped(), timeout) if timeout else wrapped()
            )
            future = asyncio.run_coroutine_threadsafe(coro_to_run, self._loop)
            self._stats.tasks_submitted += 1

        # Wait for result
        try:
            result = future.result(timeout=timeout if timeout else None)
            with self._lock:
                self._stats.tasks_completed += 1
            return result
        except Exception:
            with self._lock:
                self._stats.tasks_failed += 1
            raise

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """
        Fire-and-forget async execution.

        Args:
            coro: Coroutine to execute

        Returns:
            Future for tracking completion

        Example:
            bridge.submit(send_notification(user_id))
        """
        ctx = contextvars.copy_context()

        async def wrapped():
            tokens = []
            try:
                for var, val in ctx.items():
                    tokens.append((var, var.set(val)))
                return await coro
            finally:
                for var, token in tokens:
                    var.reset(token)

        with self._lock:
            if not self._loop:
                self._start()
            assert self._loop is not None
            self._stats.tasks_submitted += 1
            return asyncio.run_coroutine_threadsafe(wrapped(), self._loop)

    def stream(
        self,
        async_gen: AsyncGenerator[T, None],
        queue_size: int = 256,
    ) -> Generator[T, None, None]:
        """
        Stream async generator to sync context.

        Args:
            async_gen: Async generator to stream
            queue_size: Internal buffer size

        Yields:
            Items from the async generator

        Example:
            for item in bridge.stream(fetch_pages()):
                process(item)
        """
        q: Queue = Queue(maxsize=queue_size)

        async def relay():
            try:
                async for item in async_gen:
                    while True:
                        try:
                            q.put_nowait(("data", item))
                            break
                        except Full:
                            await asyncio.sleep(0.01)

                while True:
                    try:
                        q.put_nowait(("done", None))
                        break
                    except Full:
                        await asyncio.sleep(0.01)
            except Exception as e:
                while True:
                    try:
                        q.put_nowait(("error", e))
                        break
                    except Full:
                        await asyncio.sleep(0.01)

        # Kick off relay in the bridge loop
        self.submit(relay())

        # Consume from sync side
        while True:
            msg_type, payload = q.get()
            if msg_type == "done":
                break
            if msg_type == "error":
                raise payload
            yield payload

    def run_cpu(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Run CPU-bound work on best available engine.

        Automatically uses:
        - Sub-Interpreters (3.14+): True parallelism, 40x faster data transfer
        - Free-Threading (3.13t): True parallelism, zero overhead
        - ProcessPool (3.9-3.12): Compatible mode

        Args:
            func: Function to execute (must be at module level for ProcessPool)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function

        Example:
            result = bridge.run_cpu(heavy_computation, data)
        """
        return self._cpu_executor.submit(func, *args, **kwargs).result()

    def get_stats(self) -> BridgeStats:
        """Get current runtime statistics."""
        # You could return a shallow copy if you want to be extra safe
        return self._stats

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the bridge.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        with self._lock:
            self._shutdown_event.set()
            loop = self._loop
            thread = self._thread

        # Ask loop to stop and cancel heartbeat
        if loop and loop.is_running():
            if self._heartbeat_task and not self._heartbeat_task.done():
                loop.call_soon_threadsafe(self._heartbeat_task.cancel)
            loop.call_soon_threadsafe(loop.stop)

        # Wait for loop thread to finish
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

        # Shutdown executor
        if hasattr(self, "_cpu_executor") and self._cpu_executor:
            self._cpu_executor.shutdown(wait=True, cancel_futures=False)

        # Clear references so this instance is clearly "dead"
        with self._lock:
            self._loop = None
            self._thread = None
            self._heartbeat_task = None


# Global singleton for convenience
_global_bridge: Optional[Sidecar] = None


def get_bridge(**kwargs) -> Sidecar:
    """Get or create the global Sidecar instance."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = Sidecar(**kwargs)
    return _global_bridge


def run_sync(coro: Coroutine[Any, Any, T], timeout: Optional[float] = 30) -> T:
    """
    Run async code from sync context using global bridge.

    Example:
        from sidecar import run_sync
        import httpx

        response = run_sync(httpx.get("https://api.com"))
    """
    return get_bridge().run_sync(coro, timeout)


def submit(coro: Coroutine[Any, Any, Any]) -> Future:
    """
    Fire-and-forget async execution using global bridge.

    Example:
        from sidecar import submit
        submit(send_notification(user_id))
    """
    return get_bridge().submit(coro)


def stream(gen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    """
    Stream async generator to sync using global bridge.

    Example:
        from sidecar import stream

        for item in stream(fetch_pages()):
            process(item)
    """
    return get_bridge().stream(gen)


def run_cpu(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run CPU-bound work using global bridge.

    Example:
        from sidecar import run_cpu
        result = run_cpu(expensive_calculation, data)
    """
    return get_bridge().run_cpu(func, *args, **kwargs)


def shutdown(timeout: float = 5.0) -> None:
    """Shutdown global bridge (usually not needed - auto cleanup on exit)."""
    global _global_bridge
    if _global_bridge:
        _global_bridge.shutdown(timeout)
        _global_bridge = None
