from typing import Any, Callable, TypeVar, override

from .base import Backend

T = TypeVar("T")


class LocalBackend(Backend):
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
