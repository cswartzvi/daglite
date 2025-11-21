from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, override

from .base import Backend

T = TypeVar("T")


class ThreadBackend(Backend):
    """
    Parallel execution backend using a thread pool.

    ThreadBackend executes multiple task invocations concurrently using Python threads.
    This is particularly effective for I/O-bound workloads where threads can wait on
    I/O operations without blocking each other.

    Note:
        Due to Python's Global Interpreter Lock (GIL), ThreadBackend provides limited
        benefits for CPU-bound pure Python code. For CPU-bound tasks, consider using
        a process-based backend (future feature) or ensure tasks release the GIL
        (e.g., NumPy operations, I/O, C extensions).

    Best Use Cases:
        - Network requests (HTTP APIs, database queries)
        - File I/O operations
        - Tasks that block on external services
        - Large fan-out operations with I/O-bound tasks

    Args:
        max_workers (int | None): Maximum number of threads in the pool. If None, defaults to
            min(32, os.cpu_count() + 4) following ThreadPoolExecutor's default.

    Examples:
        >>> backend = ThreadBackend(max_workers=10)
        >>> @task(backend="threading")
        >>> def fetch_url(url: str) -> str:
        >>>     return requests.get(url).text
    """

    def __init__(self, max_workers: int | None = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self):
        self._executor.shutdown(wait=True)

    @override
    def run_single(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        return fn(**kwargs)

    @override
    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        futures = [self._executor.submit(fn, **kw) for kw in calls]
        return [f.result() for f in futures]
