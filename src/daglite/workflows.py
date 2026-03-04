"""@workflow decorator and Workflow class for eager task entry points."""

from __future__ import annotations

import importlib
import inspect
import sys
import typing
from collections.abc import Callable
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


# region Decorator


class _WorkflowDecorator(Protocol):
    """Return type for keyword-args form of ``workflow()``."""

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Coroutine[Any, Any, R]], /
    ) -> AsyncWorkflow[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R], /) -> SyncWorkflow[P, R]: ...

    def __call__(self, func: Any, /) -> Any: ...


@overload
def workflow(  # type: ignore[overload-overlap]
    func: Callable[P, Coroutine[Any, Any, R]], /
) -> AsyncWorkflow[P, R]: ...


@overload
def workflow(func: Callable[P, R], /) -> SyncWorkflow[P, R]: ...


@overload
def workflow(
    *,
    name: str | None = None,
    description: str | None = None,
) -> _WorkflowDecorator: ...


def workflow(
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """
    Decorator to convert a Python function into a daglite workflow.

    Workflows are **named entry points** that wrap a function calling `@task`-decorated functions.
    Tasks execute eagerly inside the workflow — they run immediately and return real values.

    Calling a workflow sets up a managed session that provides backend, cache, plugin, and event
    infrastructure. Sync workflows return the result directly; async workflows return a coroutine
    that the caller must ``await``.

    Args:
        func: The function to wrap. When used without parentheses (`@workflow`), this is
            automatically passed.  When used with parentheses (`@workflow()`), this is `None`.
        name: Custom name for the workflow.  Defaults to the function's `__name__`.
        description: Workflow description.  Defaults to the function's docstring.

    Returns:
        A `SyncWorkflow` or `AsyncWorkflow` (when used as `@workflow`) or a decorator function
        (when used as `@workflow()`).

    Examples:
        >>> from daglite import task, workflow
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> @task
        ... def mul(x: int, y: int) -> int:
        ...     return x * y

        Single-value workflow
        >>> @workflow
        ... def my_workflow(x: int, y: int):
        ...     return add(x=x, y=y)
        >>> my_workflow(2, 3)
        5

        Multi-step workflow
        >>> @workflow
        ... def chain_workflow(x: int, y: int):
        ...     a = add(x=x, y=y)
        ...     return mul(x=a, y=10)
        >>> chain_workflow(2, 3)
        50
    """

    def decorator(fn: Callable[..., Any]) -> SyncWorkflow[Any, Any] | AsyncWorkflow[Any, Any]:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@workflow` can only be applied to callable functions.")

        _name = name if name is not None else getattr(fn, "__name__", "unnamed_workflow")
        _description = description if description is not None else getattr(fn, "__doc__", "") or ""

        if inspect.iscoroutinefunction(fn):
            return AsyncWorkflow(func=fn, name=_name, description=_description)
        return SyncWorkflow(func=fn, name=_name, description=_description)

    if func is not None:
        return decorator(func)

    return decorator


# region Load


def load_workflow(workflow_path: str) -> Workflow:
    """
    Load a workflow from a module path.

    Args:
        workflow_path: Dotted path to the workflow (e.g., 'mymodule.my_workflow').

    Returns:
        The loaded Workflow instance.

    Raises:
        ValueError: If the workflow path is invalid.
        ModuleNotFoundError: If the module cannot be found.
        AttributeError: If the workflow attribute does not exist in the module.
        TypeError: If the loaded object is not a Workflow instance.
    """
    if "." not in workflow_path:
        raise ValueError(
            f"Invalid workflow path: '{workflow_path}'. Expected format: 'module.workflow_name'"
        )

    module_path, attr_name = workflow_path.rsplit(".", 1)

    # Add current directory to Python path if not already there
    cwd = str(Path.cwd())
    if cwd not in sys.path:  # pragma: no cover
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)

    if not hasattr(module, attr_name):
        raise AttributeError(f"Workflow '{attr_name}' not found in module '{module_path}'")

    workflow_obj = getattr(module, attr_name)

    if not isinstance(workflow_obj, (SyncWorkflow, AsyncWorkflow)):
        raise TypeError(
            f"'{workflow_path}' is not a Workflow. Did you forget to use the @workflow decorator?"
        )

    return workflow_obj


# region Workflow types


@dataclass(frozen=True)
class _BaseWorkflow(Generic[P, R]):
    """
    Shared base for sync and async workflow types.

    Workflows are **named entry points** for running a multi-task function
    with full session infrastructure (backend, cache, plugins, events).

    Users should **not** directly instantiate workflow classes — use the `@workflow` decorator.
    """

    func: Callable[P, Any]
    """Workflow function to be wrapped."""

    name: str
    """Name of the workflow."""

    description: str
    """Description of the workflow."""

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the underlying workflow function."""
        return inspect.signature(self.func)

    def get_typed_params(self) -> dict[str, type | None]:
        """
        Extract parameter names and their type annotations from the workflow function.

        Uses ``typing.get_type_hints`` rather than the raw ``inspect.Signature``
        so that stringified annotations (produced by ``from __future__ import
        annotations``) are resolved to their actual types.

        Returns:
            Dictionary mapping parameter names to their type annotations. If a parameter has no
            annotation, the value is None.
        """
        try:
            hints = typing.get_type_hints(self.func)
        except Exception:
            hints = {}

        typed_params: dict[str, type | None] = {}
        for param_name in self.signature.parameters:
            typed_params[param_name] = hints.get(param_name)

        return typed_params

    def has_typed_params(self) -> bool:
        """
        Check if all parameters have type annotations.

        Returns:
            True if all parameters are typed, False otherwise.
        """
        return all(t is not None for t in self.get_typed_params().values())


@dataclass(frozen=True)
class SyncWorkflow(_BaseWorkflow[P, R]):
    """
    Workflow wrapping a synchronous function.

    Calling an instance sets up a `session`, executes the function, and returns its result.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # noqa: D102
        from daglite.session import session

        with session():
            return self.func(*args, **kwargs)


@dataclass(frozen=True)
class AsyncWorkflow(_BaseWorkflow[P, R]):
    """
    Workflow wrapping an async coroutine function.

    Calling an instance returns a coroutine that sets up an `async_session`, executes the function,
    and returns its result.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]:  # noqa: D102
        return self._run(*args, **kwargs)

    async def _run(self, *args: Any, **kwargs: Any) -> R:
        from daglite.session import async_session

        async with async_session():
            result = self.func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result


Workflow = SyncWorkflow[..., Any] | AsyncWorkflow[..., Any]
"""Union of sync and async workflow types."""
