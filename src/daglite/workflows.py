"""@workflow decorator and Workflow class for eager task entry points."""

from __future__ import annotations

import importlib
import inspect
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, ParamSpec, overload

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
    Decorator to convert a Python function into a daglite `Workflow`.

    Workflows are **named entry points** that wrap a function calling `@task`-decorated functions.
    Tasks execute eagerly inside the workflow — they run immediately and return real values.

    - Call the workflow to invoke the underlying function directly (no session).
    - Use `.run()` / `.run_async()` to wrap the call in a managed `session` that provides backend,
      cache, plugin, and event infrastructure.

    Args:
        func: The function to wrap. When used without parentheses (`@workflow`), this is
            automatically passed.  When used with parentheses (`@workflow()`), this is `None`.
        name: Custom name for the workflow.  Defaults to the function's `__name__`.
        description: Workflow description.  Defaults to the function's docstring.

    Returns:
        Either a `Workflow` (when used as `@workflow`) or a decorator function (when used as
        `@workflow()`).

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
        >>> my_workflow.run(2, 3)
        5

        Multi-step workflow
        >>> @workflow
        ... def chain_workflow(x: int, y: int):
        ...     a = add(x=x, y=y)
        ...     return mul(x=a, y=10)
        >>> chain_workflow.run(2, 3)
        50
    """

    def decorator(fn: Callable[P, Any]) -> Workflow[P]:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@workflow` can only be applied to callable functions.")

        return Workflow(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_workflow"),
            description=description
            if description is not None
            else getattr(fn, "__doc__", "") or "",
        )

    if func is not None:
        return decorator(func)

    return decorator


def load_workflow(workflow_path: str) -> Workflow[Any]:
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

    if not isinstance(workflow_obj, Workflow):
        raise TypeError(
            f"'{workflow_path}' is not a Workflow. Did you forget to use the @workflow decorator?"
        )

    return workflow_obj


@dataclass(frozen=True)
class Workflow(Generic[P]):
    """
    Workflows are **named entry points** for running a multi-task function
    with full session infrastructure (backend, cache, plugins, events).

    Users should **not** directly instantiate this class — use the `@workflow` decorator instead.

    A workflow wraps a function that calls `@task`-decorated functions. Tasks execute eagerly
    (returning real values, not futures). Workflows can be invoked two ways:

    1. **Call** — `workflow(...)` runs the function directly, no session setup.
    2. **Run** — `workflow.run(...)` / `workflow.run_async(...)` wraps the function
       in a `session`, providing backend, cache, plugin, and event infrastructure.
    """

    func: Callable[P, Any]
    """Workflow function to be wrapped."""

    name: str
    """Name of the workflow."""

    description: str
    """Description of the workflow."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """Call the underlying workflow function directly — no session setup."""
        return self.func(*args, **kwargs)

    def run(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """
        Run the workflow inside a managed session.

        Sets up backend, cache, plugin, and event infrastructure, calls the
        workflow function, tears everything down, and returns whatever the
        function returns.

        Args:
            *args: Positional arguments forwarded to the workflow function.
            **kwargs: Keyword arguments forwarded to the workflow function.

        Returns:
            The return value of the workflow function.
        """
        from daglite.session import session

        with session():
            return self.func(*args, **kwargs)

    async def run_async(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """
        Run the workflow inside a managed async session.

        Async equivalent of `run()`. Sets up the same infrastructure and
        calls the workflow function with ``await`` if it is a coroutine.

        Args:
            *args: Positional arguments forwarded to the workflow function.
            **kwargs: Keyword arguments forwarded to the workflow function.

        Returns:
            The return value of the workflow function.
        """
        import inspect as _inspect

        from daglite.session import async_session

        async with async_session():
            result = self.func(*args, **kwargs)
            if _inspect.isawaitable(result):
                return await result
            return result

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
