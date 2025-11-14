from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar, overload

from daglite.nodes import CallNode
from daglite.nodes import Node

P = ParamSpec("P")
R = TypeVar("R")


class Task(Generic[P, R]):
    """A Task wraps a plain Python function with a concrete signature."""

    def __init__(self, fn: Callable[P, R], name: str | None = None) -> None:
        self._fn: Callable[P, R] = fn

        # preserve metadata for nice repr / debugging
        # self.__name__ = name if name is not None else getattr(fn, "__name__", "task")
        # self.__doc__ = fn.__doc__
        # self.__wrapped__ = fn  # helpful for introspection

    # def __call__(self: "Task[..., R]", *args: P.args, **kwargs: P.kwargs) -> R:
    #     """
    #     Call the underlying function directly.

    #     Warning: This will NOT build a graph node, but will execute the function immediately.
    #     """

    #     return self._fn(*args, **kwargs)

    def bind(self, **kwargs: Any) -> Node[R]:
        """
        Build a Node[R] node representing this task with some parameters bound.

        Args:
            kwargs (Mapping[str, Any]):
                Keyword arguments to be passed to the task during graph execution. May contain
                - Python objects (int, str, dict, DataFrame, ...)
                - Other Nodes representing lazy values

        Example:
        >>> @task
        >>> def add(x: int, y: int) -> int: ...
        >>> node = add.bind(x=1, y=2)  # TaskNode[int]
        """

        return CallNode(self, dict(kwargs))


@overload
def task(fn: Callable[P, R]) -> Task[P, R]: ...


@overload
def task(*, name: str | None = None) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(
    fn: Callable[P, R] | None = None, *, name: str | None = None
) -> Task[P, R] | Callable[[Callable[P, R]], Task[P, R]]:
    """
    Decorator to turn a plain function into a Task.

    The wrapped function keeps its concrete signature and can still be called
    directly (for tests, REPL, etc.). Use .bind(...) to build DAG nodes.

    Args:
        fn (Callable[P, R]): The function to wrap (provided when used as `@task`).
        name (str, optional): Custom name for the task, defaults to the function's

    Usage:
    >>> @task
    >>> def my_func(): ...

    >>> @task()
    >>> def my_func(): ...

    >>> @task(name="custom_name")
    >>> def my_func(): ...
    """
    if fn is not None:
        # Used as @task (without parentheses)
        return Task(fn, name=name)

    # Used as @task() or @task(name="...")
    def decorator(f: Callable[P, R]) -> Task[P, R]:
        return Task(f, name=name)

    return decorator
