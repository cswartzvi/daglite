from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, override

from daglite.engine import Backend

T = TypeVar("T")


class ThreadBackend(Backend):
    """A backend that uses threads to run tasks."""

    def __init__(self, max_workers: int = 8):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @override
    def run_task(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        # You can choose: inline or via executor.
        # For now, run inline to avoid overhead for isolated tasks.
        return fn(**kwargs)

    @override
    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        futures = [self._executor.submit(fn, **kw) for kw in calls]
        return [f.result() for f in futures]
