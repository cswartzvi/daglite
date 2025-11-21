from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any, Callable, TypeVar, override

from daglite.settings import get_global_settings

from .base import Backend

T = TypeVar("T")

_GLOBAL_THREAD_POOL: ThreadPoolExecutor | None = None


def _get_global_thread_pool() -> ThreadPoolExecutor:
    """Get or create the global thread pool."""
    settings = get_global_settings()
    global _GLOBAL_THREAD_POOL
    if _GLOBAL_THREAD_POOL is None:
        max_workers = settings.max_backend_threads if settings else None
        _GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
    return _GLOBAL_THREAD_POOL


class SequentialBackend(Backend):
    """
    Sequential execution backend that runs tasks in the current process and thread.

    LocalBackend provides simple, synchronous execution with no parallelism. This is
    the default backend and is suitable for:
    - CPU-bound tasks where parallelism doesn't help (due to Python's GIL)
    - Development and debugging (predictable execution order)
    - Tasks with no I/O waits
    - Small workloads where overhead of parallelism isn't worth it

    For I/O-bound tasks or when you want parallelism within map operations,
    consider using ThreadBackend instead.
    """

    @override
    def run_single(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        return fn(**kwargs)

    @override
    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        return [self.run_single(fn, kwargs) for kwargs in calls]


class ThreadBackend(Backend):
    """
    Parallel execution backend using a shared global thread pool.

    ThreadBackend executes multiple task invocations concurrently using Python threads from a
    single global ThreadPoolExecutor. This avoids creating multiple thread pools and allows better
    resource management across the entire application.The global pool size is controlled by
    DagLiteSettings.max_backend_threads. All ThreadBackend instances share the same pool.
    Max_workers parameter in the constructor only limits the number of concurrent tasks submitted
    during map operations.

    Note that due to Python's Global Interpreter Lock (GIL), ThreadBackend provides limited
    benefits for CPU-bound pure Python code. For CPU-bound tasks, consider using a process-based
    backend (future feature) or ensure tasks release the GIL (e.g., NumPy operations, I/O,
    C extensions).

    Best Use Cases:
    - Network requests (HTTP APIs, database queries)
    - File I/O operations
    - Tasks that block on external services
    - Large fan-out operations with I/O-bound tasks

    Args:
        max_workers (int | None):
            Maximum number of threads to run in this backend instance. Note that all
            ThreadBackend instances share the same global pool, so this parameter only affects
            number of chunks submitted during map operations.
    """

    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers

    @override
    def run_single(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        return fn(**kwargs)

    @override
    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        # NOTE:If max_workers is set, limits the number of concurrent tasks to that value,
        # submitting new tasks as previous ones complete. Otherwise, submits all tasks
        # at once (limited only by the global pool size).

        executor = _get_global_thread_pool()

        if self._max_workers is None:
            futures = [executor.submit(fn, **kw) for kw in calls]
            return [f.result() for f in futures]

        # Limit concurrency: keep only max_workers tasks in-flight at any time
        max_concurrent = min(self._max_workers, executor._max_workers)
        results: list[T | None] = [None] * len(calls)
        futures_map: dict[Any, int] = {}  # future -> index in results

        remaining = list(enumerate(calls))
        in_flight: set[Any] = set()

        while remaining or in_flight:
            # Submit new tasks up to the concurrency limit
            while remaining and len(in_flight) < max_concurrent:
                idx, kwargs = remaining.pop(0)
                future = executor.submit(fn, **kwargs)
                futures_map[future] = idx
                in_flight.add(future)

            # Wait for at least one task to complete
            if in_flight:
                done = next(as_completed(in_flight))
                idx = futures_map[done]
                results[idx] = done.result()
                in_flight.remove(done)
                del futures_map[done]

        return results  # type: ignore[return-value]
