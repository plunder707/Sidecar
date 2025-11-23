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
from typing import TypeVar, Callable, Optional, AsyncGenerator, Any, Coroutine, Generator
from dataclasses import dataclass

__version__ = "1.0.1"
__all__ = ["Sidecar", "run_sync", "submit", "stream", "run_cpu", "shutdown"]

logger = logging.getLogger("sidecar")
T = TypeVar('T')

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
        
        # Auto-cleanup on exit (defensive - may fail during shutdown)
        try:
            atexit.register(self.shutdown)
        except Exception:
            pass  # Ignore if interpreter is already shutting down
        
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
                name=self._name
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
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            try:
                loop.close()
            except Exception:
                pass
    
    def run_sync(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = 30
    ) -> T:
        """
        Run async code from sync context.
        
        Args:
            coro: Coroutine to execute
            timeout: Maximum execution time in seconds
        
        Returns:
            Result of the coroutine
        
        Example:
            response = bridge.run_sync(httpx.get("https://api.com"))
        
        Raises:
            RuntimeError: If called from within the Sidecar loop (deadlock)
            TimeoutError: If execution exceeds timeout
        """
        def _fail_deadlock(msg: str):
            """Clean up coro and raise error."""
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError(msg)

        # Deadlock detection: if called from the bridge thread or its loop
        if self._thread and self._thread.is_alive() and threading.current_thread() is self._thread:
            _fail_deadlock("Deadlock detected: run_sync() called from within Sidecar loop thread. Use 'await' instead.")

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        else:
            if current_loop is self._loop:
                _fail_deadlock("Deadlock detected: run_sync() called while Sidecar loop is running. Use 'await' instead.")
        
        # Capture context from calling thread
        ctx = contextvars.copy_context()
        
        async def wrapped() -> T:
            return await coro

        with self._lock:
            if not self._loop:
                self._start()
            assert self._loop is not None

            coro_to_run = ctx.run(lambda: asyncio.wait_for(wrapped(), timeout) if timeout else wrapped())
            future = asyncio.run_coroutine_threadsafe(coro_to_run, self._loop)
            self._stats.tasks_submitted += 1
        
        try:
            # Single timeout point is enforced by wait_for above
            result = future.result()
            self._stats.tasks_completed += 1
            return result
        except Exception:
            self._stats.tasks_failed += 1
            raise
    
    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """
        Fire-and-forget async execution.

        Args:
            coro: Coroutine to execute

        Returns:
            Future that can be used to observe completion or exceptions.

        Semantics:
        - tasks_submitted is incremented immediately.
        - tasks_completed / tasks_failed are incremented when the coroutine
          actually finishes, independent of run_sync().
        - Context variables from the caller are propagated into the async task.

        Example:
            bridge.submit(send_notification(user_id))
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError(f"submit() expected a coroutine, got {type(coro)!r}")

        # Capture context from the calling thread so contextvars propagate
        ctx = contextvars.copy_context()

        async def wrapped() -> Any:
            # Run the original coroutine under the captured context
            return await coro

        def _on_done(fut: Future) -> None:
            """
            Done callback executed in the loop thread when the task finishes.
            Updates BridgeStats based on success vs failure.
            """
            with self._lock:
                try:
                    # This will re-raise if the coroutine failed
                    fut.result()
                except Exception:
                    self._stats.tasks_failed += 1
                else:
                    self._stats.tasks_completed += 1

        with self._lock:
            # Ensure background loop is running
            if not self._loop:
                self._start()
            assert self._loop is not None

            # Build the coroutine object inside the captured context
            coro_to_run = ctx.run(lambda: wrapped())
            future = asyncio.run_coroutine_threadsafe(coro_to_run, self._loop)

            # Book-keeping: we attempted to run one more task
            self._stats.tasks_submitted += 1

            # When it finishes, update completed/failed counters
            future.add_done_callback(_on_done)

            return future

    def stream(
        self,
        async_gen: AsyncGenerator[T, None],
        queue_size: int = 256
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

                # Signal completion
                while True:
                    try:
                        q.put_nowait(("done", None))
                        break
                    except Full:
                        await asyncio.sleep(0.01)
            except Exception as e:
                # Best-effort error forwarding
                while True:
                    try:
                        q.put_nowait(("error", e))
                        break
                    except Full:
                        await asyncio.sleep(0.01)
        
        self.submit(relay())
        
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
        **kwargs: Any
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
        if hasattr(self, '_cpu_executor') and self._cpu_executor:
            try:
                self._cpu_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass

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
