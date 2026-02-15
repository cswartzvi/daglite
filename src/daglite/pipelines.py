from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, overload

from daglite.futures.base import BaseTaskFuture

P = ParamSpec("P")
R = TypeVar("R")

Dag = BaseTaskFuture
"""
Type alias for pipeline return annotations.

Pipelines are entry points — the ``Dag`` alias provides a readable annotation
for functions that build and return a task graph:

    @pipeline
    def compute(x: int, y: int) -> Dag[int]:
        return add(x=x, y=y)

    @pipeline
    def sweep(n: int) -> Dag[int]:
        return double.map(x=list(range(n)))
"""


@overload
def pipeline(func: Callable[P, R]) -> Pipeline[P, R]: ...


@overload
def pipeline(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, R]], Pipeline[P, R]]: ...


def pipeline(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Pipeline[P, R] | Callable[[Callable[P, R]], Pipeline[P, R]]:
    """
    Decorator to convert a Python function into a daglite `Pipeline`.

    Pipelines are **entry points** for DAG execution. They wrap a function that
    builds a task graph (returning a `TaskFuture` or `MapTaskFuture`) and
    provide convenient methods to evaluate it:

    - Call the pipeline to get the underlying future for manual evaluation.
    - Use `.run()` / `.run_async()` to build and evaluate in one step.
    - Run from the command line with `daglite run module.my_pipeline`.

    Args:
        func: The function to wrap. When used without parentheses (`@pipeline`), this is
            automatically passed. When used with parentheses (`@pipeline()`), this is None.
        name: Custom name for the pipeline. Defaults to the function's ``__name__``.
        description: Pipeline description. Defaults to the function's docstring.

    Returns:
        Either a `Pipeline` (when used as `@pipeline`) or a decorator function
        (when used as `@pipeline()`).

    Examples:
        >>> from daglite import pipeline, task
        >>> from daglite.futures import TaskFuture
        >>> @task
        ... def some_task(x: int, y: int) -> int:
        ...     return x + y

        Basic usage
        >>> @pipeline
        ... def my_pipeline(x: int, y: int) -> TaskFuture[int]:
        ...     return some_task(x=x, y=y)
        >>> my_pipeline.run(2, 3)
        5

        With parameters
        >>> @pipeline(name="custom_pipeline", description="Does something cool")
        ... def my_pipeline(x: int, y: int) -> TaskFuture[int]:
        ...     return some_task(x=x, y=y)
        >>> my_pipeline.run(5, 7)
        12
    """

    def decorator(fn: Callable[P, R]) -> Pipeline[P, R]:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@pipeline` can only be applied to callable functions.")

        return Pipeline(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_pipeline"),
            description=description if description is not None else getattr(fn, "__doc__", ""),
        )

    if func is not None:
        # Used as @pipeline (without parentheses)
        return decorator(func)

    return decorator


@dataclass(frozen=True)
class Pipeline(Generic[P, R]):
    """
    Entry point for building and running a task graph.

    Users should **not** directly instantiate this class — use the ``@pipeline``
    decorator instead.

    A pipeline wraps a function that accepts parameters and returns a
    ``TaskFuture`` or ``MapTaskFuture``.  Pipelines can be invoked three ways:

    1. **Call** — ``pipeline(...)`` returns the underlying future for manual
       evaluation or further composition.
    2. **Run** — ``pipeline.run(...)`` / ``pipeline.run_async(...)`` builds and
       evaluates the DAG in a single step.
    3. **CLI** — ``daglite run module.pipeline --param key=value`` executes the
       pipeline from the command line.
    """

    func: Callable[P, R]
    """Pipeline function to be wrapped."""

    name: str
    """Name of the pipeline."""

    description: str
    """Description of the pipeline."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Call the underlying pipeline function.

        Args:
            *args: Positional arguments to pass to the pipeline function.
            **kwargs: Keyword arguments to pass to the pipeline function.

        Returns:
            The result of calling the pipeline function (typically a BaseTaskFuture).
        """
        return self.func(*args, **kwargs)

    def run(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """
        Build and evaluate the pipeline synchronously.

        This is the primary way to execute a pipeline from scripts, notebooks,
        or the REPL.  It calls the underlying function to build the task graph
        and immediately evaluates it.

        Cannot be called from within an async context (e.g., inside an
        ``async def`` or running event loop).  Use ``.run_async()`` instead.

        Args:
            *args: Positional arguments forwarded to the pipeline function.
            **kwargs: Keyword arguments forwarded to the pipeline function.

        Returns:
            The evaluated result of the pipeline.

        Raises:
            RuntimeError: If called from within an async context.

        Examples:
            >>> from daglite import Dag, pipeline, task
            >>> @task
            ... def add(x: int, y: int) -> int:
            ...     return x + y
            >>> @pipeline
            ... def my_pipeline(x: int, y: int) -> Dag[int]:
            ...     return add(x=x, y=y)
            >>> my_pipeline.run(5, 10)
            15
        """
        future = self.func(*args, **kwargs)
        from daglite.engine import evaluate

        return evaluate(future)

    async def run_async(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """
        Build and evaluate the pipeline asynchronously.

        Async counterpart of ``.run()``.  Use this when running inside an
        existing event loop or when the pipeline contains async tasks.

        Args:
            *args: Positional arguments forwarded to the pipeline function.
            **kwargs: Keyword arguments forwarded to the pipeline function.

        Returns:
            The evaluated result of the pipeline.

        Examples:
            >>> import asyncio
            >>> from daglite import Dag, pipeline, task
            >>> @task
            ... def add(x: int, y: int) -> int:
            ...     return x + y
            >>> @pipeline
            ... def my_pipeline(x: int, y: int) -> Dag[int]:
            ...     return add(x=x, y=y)
            >>> asyncio.run(my_pipeline.run_async(5, 10))
            15
        """
        future = self.func(*args, **kwargs)
        from daglite.engine import evaluate_async

        return await evaluate_async(future)

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the underlying pipeline function."""
        return inspect.signature(self.func)

    def get_typed_params(self) -> dict[str, type | None]:
        """
        Extract parameter names and their type annotations from the pipeline function.

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


def load_pipeline(pipeline_path: str) -> Pipeline[Any, Any]:
    """
    Load a pipeline from a module path.

    Args:
        pipeline_path: Dotted path to the pipeline (e.g., 'mymodule.my_pipeline').

    Returns:
        The loaded Pipeline instance.

    Raises:
        ValueError: If the pipeline path is invalid.
        ModuleNotFoundError: If the module cannot be found.
        AttributeError: If the pipeline attribute does not exist in the module.
        TypeError: If the loaded object is not a Pipeline instance.
    """
    # Split into module and attribute
    if "." not in pipeline_path:
        raise ValueError(
            f"Invalid pipeline path: '{pipeline_path}'. Expected format: 'module.pipeline_name'"
        )

    module_path, attr_name = pipeline_path.rsplit(".", 1)

    # Add current directory to Python path if not already there
    cwd = str(Path.cwd())
    if cwd not in sys.path:  # pragma: no cover
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)

    if not hasattr(module, attr_name):
        raise AttributeError(f"Pipeline '{attr_name}' not found in module '{module_path}'")

    pipeline_obj = getattr(module, attr_name)

    if not isinstance(pipeline_obj, Pipeline):
        raise TypeError(
            f"'{pipeline_path}' is not a Pipeline. Did you forget to use the @pipeline decorator?"
        )

    return pipeline_obj
