import abc
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Backend(abc.ABC):
    """
    Abstract base class for task execution backends.

    Backends define how task functions are executed within the DAG. They provide
    two execution modes:

    1. run_single(): Execute a single task call (used by TaskNode)
    2. run_many(): Execute multiple calls of the same task (used by MapTaskNode)

    Examples:
        Sequential execution:
            >>> backend = LocalBackend()
            >>> result = backend.run_single(my_function, {"x": 1, "y": 2})

        Parallel execution:
            >>> backend = ThreadBackend(max_workers=8)
            >>> calls = [{"x": 1}, {"x": 2}, {"x": 3}]
            >>> results = backend.run_many(my_function, calls)
    """

    @abc.abstractmethod
    def run_single(self, fn: Callable[..., T], kwargs: dict[str, Any]) -> T:
        """
        Executes a single function call on this backend.

        Args:
            fn (Callable[..., T]): The function to call.
            kwargs (dict[str, Any]): Keyword arguments for the function.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        """
        Executes a function multiple times with different arguments on this backend.

        Args:
            fn (Callable[..., T]):
                The function to call.
            calls (list[dict[str, Any]]):
                List of keyword argument dicts for each call.
        """
        raise NotImplementedError()
