from __future__ import annotations

import abc
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from uuid import UUID

if TYPE_CHECKING:
    from .tasks import EvaluatableTask
else:
    EvaluatableTask = object  # type: ignore[misc]


T = TypeVar("T")


def evaluate(future: EvaluatableTask[T], backend: Backend | str | None = None) -> T:
    """
    Evaluate a Node[T] expression to a concrete T.

    Args:
        future (daglite.tasks.EvaluatableTask[T]):
            Task future to be evaluated into a concrete value of type T.
        backend (daglite.backends.Backend | str, optional):
            Backend to use for evaluation. If a string is provided, it is used to look up
            a backend via `find_backend()`. If None, the LocalBackend is used.
    """
    from daglite.backends import find_backend
    from daglite.backends.local import LocalBackend

    if backend is None:
        backend = LocalBackend()
    elif isinstance(backend, str):
        backend = find_backend(backend)
    elif not isinstance(backend, Backend):
        raise TypeError(f"backend must be a Backend instance or str, got {type(backend)}")
    engine = Engine(backend=backend)
    return engine.evaluate(future)


@dataclass(frozen=True)
class Engine:
    """
    Engine for evaluating task futures.

    NOTE: This class is intended for internal use by the evaluation engine only. User code should
    should directly interact with either the the `evaluate` function or the `Engine` class.
    """

    backend: Backend
    cache: MutableMapping[UUID, Any] = field(default_factory=dict)
    _evaluating: set[UUID] = field(default_factory=set)

    def evaluate(self, future: EvaluatableTask[T]) -> T:
        """
        Evaluates a task future to a concrete value.

        Args:
            future (daglite.tasks.EvaluatableTask[T]):
                Task future to be evaluated into a concrete value of type T.

        Returns:
            T: Evaluated concrete value.

        Raises:
            RuntimeError: If a circular dependency is detected.
        """
        if future.id in self.cache:
            return self.cache[future.id]

        if future.id in self._evaluating:
            raise RuntimeError(
                f"Circular dependency detected: task {future.id} depends on itself. "
                "Check your task bindings for cycles."
            )

        self._evaluating.add(future.id)
        try:
            result = future._evaluate(self)
            self.cache[future.id] = result
            return result
        finally:
            self._evaluating.discard(future.id)


class Backend(abc.ABC):
    """
    Abstract base class for execution backends.

    Backend Resolution Order:
    When a task is executed, the backend is selected in the following priority order:
    1. Backend explicitly passed to evaluate() function
    2. Backend specified in the task definition (@task decorator)
    3. Backend specified in MapTaskFuture.extend()/zip()/map()/join()
    4. Default LocalBackend (if no backend is specified anywhere)
    """

    @abc.abstractmethod
    def run_task(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        """
        Runs a single function call on the execution backend.

        Args:
            fn (Callable[..., T]):
                Function to be called.
            kwargs (dict[str, Any]):
                Keyword arguments to be passed to fn.
        """
        raise NotImplementedError()

    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        """
        Runs multiple function calls on the execution backend.

        Default implementation: sequentially loop using run_task().
        Subclasses can override this to provide parallel execution.

        Args:
            fn (Callable[..., T]):
                Function to be called multiple times.
            calls (list[dict[str, Any]]):
                List of keyword argument dicts, one per call.

        Returns:
            list[T]: Results from each call, in the same order as calls.
        """
        return [self.run_task(fn, kwargs) for kwargs in calls]
