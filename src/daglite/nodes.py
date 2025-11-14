import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Mapping, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

if TYPE_CHECKING:
    from .tasks import Task
else:
    Task = object  # type: ignore[misc]


class Node(abc.ABC, Generic[T]):
    """Base representation of a lazy (unevaluated) computation of type T."""

    # NOTE: The following methods are to prevent accidental usage of unevaluated nodes.

    def __bool__(self) -> bool:
        raise TypeError("Lazy value has no truthiness; evaluate it first.")

    def __len__(self) -> int:
        raise TypeError("Lazy value has no length; evaluate it first.")


@dataclass(frozen=True)
class CallNode(Node[T]):
    """
    A node representing the simple invocation of a Task with some bound parameters.

    Args:
        task (daglite.tasks.Task):
            Task to be invoked when this node is evaluated during graph execution.
        kwargs (Mapping[str, Any]):
            Keyword arguments to be passed to the task innovation. The values in kwargs may be
            - Python objects (int, str, dict, DataFrame, ...)
            - Other Nodes representing lazy values
    """

    task: "Task[..., T]"
    kwargs: Mapping[str, Any]
