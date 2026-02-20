"""@workflow decorator and Workflow class for multi-sink task networks."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, overload

from daglite.futures.base import BaseTaskFuture
from daglite.workflow_result import WorkflowResult

P = ParamSpec("P")


@overload
def workflow(func: Callable[P, Any]) -> Workflow[P]: ...


@overload
def workflow(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, Any]], Workflow[P]]: ...


def workflow(
    func: Callable[P, Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Workflow[P] | Callable[[Callable[P, Any]], Workflow[P]]:
    """
    Decorator to convert a Python function into a daglite ``Workflow``.

    Workflows are **multi-sink entry points** for DAG execution.  They wrap a
    function that builds a task graph returning one or more
    ``BaseTaskFuture`` objects (as a single future, tuple, or list) and
    provide convenient methods to evaluate the entire graph in one pass:

    - Call the workflow to invoke the underlying function (returns the raw futures).
    - Use ``.run()`` / ``.run_async()`` to build and evaluate in one step,
      returning a ``WorkflowResult`` indexable by task name or UUID.

    Args:
        func: The function to wrap.  When used without parentheses
            (``@workflow``), this is automatically passed.  When used with
            parentheses (``@workflow()``), this is ``None``.
        name: Custom name for the workflow.  Defaults to the function's
            ``__name__``.
        description: Workflow description.  Defaults to the function's
            docstring.

    Returns:
        Either a ``Workflow`` (when used as ``@workflow``) or a decorator
        function (when used as ``@workflow()``).

    Examples:
        >>> from daglite import task, workflow
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> @task
        ... def mul(x: int, y: int) -> int:
        ...     return x * y

        Single-sink workflow
        >>> @workflow
        ... def my_workflow(x: int, y: int):
        ...     return add(x=x, y=y)
        >>> result = my_workflow.run(2, 3)
        >>> result["add"]
        5

        Multi-sink workflow
        >>> @workflow
        ... def dual_workflow(x: int, y: int):
        ...     return add(x=x, y=y), mul(x=x, y=y)
        >>> result = dual_workflow.run(2, 3)
        >>> result["add"], result["mul"]
        (5, 6)
    """

    def decorator(fn: Callable[P, Any]) -> Workflow[P]:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@workflow` can only be applied to callable functions.")

        return Workflow(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_workflow"),
            description=description if description is not None else getattr(fn, "__doc__", "") or "",
        )

    if func is not None:
        return decorator(func)

    return decorator


@dataclass(frozen=True)
class Workflow(Generic[P]):
    """
    Entry point for building and running a multi-sink task graph.

    Users should **not** directly instantiate this class — use the ``@workflow``
    decorator instead.

    A workflow wraps a function that accepts parameters and returns one or more
    ``BaseTaskFuture`` objects (single future, tuple, or list).  Workflows can
    be invoked two ways:

    1. **Call** — ``workflow(...)`` returns the raw future(s) for manual handling.
    2. **Run** — ``workflow.run(...)`` / ``workflow.run_async(...)`` builds and
       evaluates the DAG in a single step, returning a ``WorkflowResult``.
    """

    func: Callable[P, Any]
    """Workflow function to be wrapped."""

    name: str
    """Name of the workflow."""

    description: str
    """Description of the workflow."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """Call the underlying workflow function, returning the raw future(s)."""
        return self.func(*args, **kwargs)

    def _collect_futures(self, raw: Any) -> list[BaseTaskFuture]:
        """Normalise the decorated function's return value to a list of futures."""
        if isinstance(raw, BaseTaskFuture):
            return [raw]
        if isinstance(raw, (tuple, list)):
            return list(raw)
        raise TypeError(
            f"@workflow function must return a BaseTaskFuture or a tuple/list of them, "
            f"got {type(raw).__name__!r}"
        )

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the underlying workflow function."""
        return inspect.signature(self.func)

    def get_typed_params(self) -> dict[str, type | None]:
        """
        Extract parameter names and their type annotations from the workflow function.

        Returns:
            Dictionary mapping parameter names to their type annotations. If a parameter has no
            annotation, the value is None.
        """
        sig = self.signature
        typed_params: dict[str, type | None] = {}
        for param_name, param in sig.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                typed_params[param_name] = None
            else:
                typed_params[param_name] = param.annotation
        return typed_params

    def has_typed_params(self) -> bool:
        """
        Check if all parameters have type annotations.

        Returns:
            True if all parameters are typed, False otherwise.
        """
        return all(t is not None for t in self.get_typed_params().values())

    def run(self, *args: P.args, **kwargs: P.kwargs) -> WorkflowResult:
        """
        Build and evaluate the workflow synchronously.

        Cannot be called from within an async context.  Use ``.run_async()`` instead.

        Args:
            *args: Positional arguments forwarded to the workflow function.
            **kwargs: Keyword arguments forwarded to the workflow function.

        Returns:
            A ``WorkflowResult`` containing the evaluated outputs of all sink nodes.
        """
        from daglite.engine import evaluate_workflow

        futures = self._collect_futures(self.func(*args, **kwargs))
        return evaluate_workflow(futures)

    async def run_async(self, *args: P.args, **kwargs: P.kwargs) -> WorkflowResult:
        """
        Build and evaluate the workflow asynchronously.

        Args:
            *args: Positional arguments forwarded to the workflow function.
            **kwargs: Keyword arguments forwarded to the workflow function.

        Returns:
            A ``WorkflowResult`` containing the evaluated outputs of all sink nodes.
        """
        from daglite.engine import evaluate_workflow_async

        futures = self._collect_futures(self.func(*args, **kwargs))
        return await evaluate_workflow_async(futures)
