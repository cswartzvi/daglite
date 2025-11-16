from typing import Any, Callable, TypeVar, override

from daglite.engine import Backend

T = TypeVar("T")


class LocalBackend(Backend):
    """Local execution backend that runs tasks in the current process."""

    @override
    def run_task(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        return fn(**kwargs)

    @override
    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        return [self.run_task(fn, kwargs) for kwargs in calls]
