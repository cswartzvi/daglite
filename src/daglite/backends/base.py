import abc
from concurrent.futures import Future
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Backend(abc.ABC):
    """Abstract base class for task execution backends."""

    @abc.abstractmethod
    def submit(self, fn: Callable[..., T], **kwargs: Any) -> Future[T]:
        """
        Submit a single function call for execution.

        Returns:
            Future representing the execution
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def submit_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[Future[T]]:
        """
        Submit multiple function calls for execution.

        Args:
            fn: Function to execute
            calls: List of keyword argument dicts, one per call

        Returns:
            List of Futures, one per call (in same order as calls)
        """
        raise NotImplementedError()
