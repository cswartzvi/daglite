from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


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

    Pipelines are functions that return a `GraphBuilder` (typically a `TaskFuture` or
    `MapTaskFuture`). They serve as entry points for CLI execution and provide a way
    to parameterize DAG construction.

    Args:
        func (Callable[P, R], optional):
            The function to wrap. When used without parentheses (`@pipeline`), this is
            automatically passed. When used with parentheses (`@pipeline()`), this is None.
        name (str, optional):
            Custom name for the pipeline. Defaults to the function's `__name__`.
        description (str, optional):
            Pipeline description. Defaults to the function's docstring.

    Returns:
        Either a `Pipeline` (when used as `@pipeline`) or a decorator function
        (when used as `@pipeline()`).

    Examples:
        Basic usage
        >>> @pipeline
        >>> def my_pipeline(x: int, y: int) -> GraphBuilder:
        >>>     return some_task.bind(x=x, y=y)

        With parameters
        >>> @pipeline(name="custom_pipeline", description="Does something cool")
        >>> def my_pipeline(x: int, y: int) -> GraphBuilder:
        >>>     return some_task.bind(x=x, y=y)
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
    Wraps a Python function as a pipeline that returns a GraphBuilder.

    Users should **not** directly instantiate this class, use the `@pipeline` decorator instead.

    A pipeline function should accept parameters and return a `GraphBuilder` (e.g., `TaskFuture`
    or `MapTaskFuture`). These parameters can be provided via the CLI, allowing for dynamic
    DAG construction based on user input.
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

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the underlying pipeline function."""
        return inspect.signature(self.func)

    def get_typed_params(self) -> dict[str, type | None]:
        """
        Extract parameter names and their type annotations from the pipeline function.

        Returns:
            Dictionary mapping parameter names to their type annotations.
            If a parameter has no annotation, the value is None.
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
