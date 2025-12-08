from __future__ import annotations

import abc
import inspect
import sys
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from functools import cached_property
from inspect import Signature
from typing import Any, Generic, ParamSpec, TypeVar, overload

from typing_extensions import Self, override

from daglite.backends import Backend
from daglite.backends import find_backend
from daglite.exceptions import ParameterError
from daglite.futures import BaseTaskFuture
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")

# region Decorator


@overload
def task(func: Callable[P, R], /) -> Task[P, R]: ...


@overload
def task(
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(  # noqa: D417
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Any:
    """
    Decorator to convert a Python function into a daglite `Task`.

    Tasks are the building blocks of daglite DAGs. They wrap plain Python functions (both sync
    and async) and provide methods for composition and execution.

    This is the recommended way for users to create tasks. Direct instantiation of the `Task`
    or `FixedParamTask` classes is strongly discouraged.

    Args:
        name: Custom name for the task. Defaults to the function's `__name__`. For lambda functions,
              defaults to "unnamed_task".
        description: Task description. Defaults to the function's docstring.
        backend: Backend for executing this task. Can be a registered backend name or a `Backend`
            instance. If None, uses the engine's default backend.

    Returns:
        Either a `Task` (when used as `@task`) or a decorator function (when used as `@task()`).

    Examples:
        >>> # Synchronous function
        >>> @task
        >>> def add(x: int, y: int) -> int:
        >>>     return x + y
        >>>
        >>> # Async function
        >>> @task
        >>> async def fetch_data(url: str) -> dict:
        >>>     async with httpx.AsyncClient() as client:
        >>>         response = await client.get(url)
        >>>         return response.json()
        >>>
        >>> # With parameters
        >>> @task(name="custom_add", backend="threading")
        >>> def add(x: int, y: int) -> int:
        >>>     return x + y
        >>>
        >>> # Lambda functions
        >>> double = task(lambda x: x * 2, name="double")
    """

    def decorator(fn: Any) -> Any:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@task` can only be applied to callable functions.")

        is_async = inspect.iscoroutinefunction(fn)

        # Store original function in module namespace for pickling (multiprocessing backend)
        if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
            module = sys.modules.get(fn.__module__)
            if module is not None:  # pragma: no branch
                private_name = f"__{fn.__name__}_func__"
                setattr(module, private_name, fn)
                fn.__qualname__ = private_name

        return Task(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_task"),
            description=description if description is not None else getattr(fn, "__doc__", ""),
            backend=find_backend(backend),
            is_async=is_async,
        )

    if func is not None:
        # Used as @task (without parentheses)
        return decorator(func)

    return decorator


# region Tasks


@dataclass(frozen=True)
class BaseTask(abc.ABC, Generic[P, R]):
    """Base class for all tasks, providing common functionality for task composition."""

    name: str
    """Name of the task."""

    description: str
    """Description of the task."""

    backend: Backend
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    @cached_property
    @abc.abstractmethod
    def signature(self) -> Signature:
        """Get the signature of the underlying task function."""
        raise NotImplementedError()

    def with_options(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        backend: str | Backend | None = None,
    ) -> Self:
        """
        Create a new task with updated options.

        Args:
            name: New name for the task. If `None`, keeps the existing name.
            description: New description for the task. If `None`, keeps the existing description.
            backend: New backend for the task. If `None`, keeps the existing backend.

        Returns:
            A new `BaseTask` instance with updated options.
        """
        from daglite.backends import find_backend

        name = name if name is not None else self.name
        description = description if description is not None else self.description
        backend = find_backend(backend) if backend is not None else self.backend

        # Collect the remaining fields (assumes this is a dataclass)
        remaining_fields = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"name", "description", "backend"}
        }

        return type(self)(name=name, description=description, backend=backend, **remaining_fields)

    @abc.abstractmethod
    def bind(self, **kwargs: Any | TaskFuture[Any]) -> "TaskFuture[R]":
        """
        Creates a `TaskFuture` future by binding values to the parameters of this task.

        This is the primary way to connect a task with inputs and dependencies. Parameters can
        be concrete values or other TaskFutures, enabling composition of complex DAGs.

        Args:
            **kwargs: Keyword arguments matching the task function's parameters. Values can be
                concrete Python objects or TaskFutures from other tasks.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def product(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> "MapTaskFuture[R]":
        """
        Create a fan-out operation by applying this task over all combinations of sequences.

        This creates a Cartesian product of all provided sequences, calling the task once
        for each combination. Useful for parameter sweeps and batch operations.

        Args:
            **kwargs: Keyword arguments where values are sequences. Each sequence element will be
                combined with elements from other sequences in a Cartesian product. Can include
                TaskFutures that resolve to sequences.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def zip(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> "MapTaskFuture[R]":
        """
        Create a fan-out operation by applying this task to zipped sequences.

        Sequences are zipped element-wise (similar to Python's `zip(`) function), calling
        the task once for each aligned set of elements. All sequences must have the same length.

        Args:
            **kwargs: Keyword arguments where values are equal-length sequences. Elements at the
                same index across sequences are combined in each call. Can include TaskFutures that
                resolve to sequences.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class Task(BaseTask[P, R]):
    """
    Wraps a Python function as a composable task in a DAG.

    Users should **not** directly instantiate this class, use the `@task` decorator instead.
    """

    func: Callable[P, R]
    """Task function to be wrapped into a Task."""

    is_async: bool = False
    """Whether this task's function is an async coroutine function."""

    def __post_init__(self) -> None:
        # Detect if function is async and update is_async field
        if inspect.iscoroutinefunction(self.func):
            object.__setattr__(self, "is_async", True)

    # NOTE: We should not define `__call__` in order to avoid confusing type checkers. We want
    # them to view this object as a `Task[P, R]` and not as a `Callable[P, R]` (which some type
    # checkers would do if we defined `__call__`).

    @cached_property
    @override
    def signature(self) -> Signature:
        return inspect.signature(self.func)

    def fix(self, **kwargs: Any) -> "FixedParamTask[P, R]":
        """
        Fix some parameters of this task, returning a `FixedParamTask`.

        Args:
            **kwargs: Keyword arguments to be fixed for this task. Can be a combination of concrete
                values and TaskFutures.

        Examples:
        >>> def score(x: int, y: int) -> float: ...
        >>>
        >>> base = score.fix(y=seed)
        >>> branch1 = base.bind(x=lazy_x)  # TaskFuture[int]
        >>> branch2 = base.product(x=[1, 2, 3, 4])  # MapTaskFuture[int]
        """
        check_invalid_params(self, kwargs)
        return FixedParamTask(
            name=self.name,
            description=self.description,
            task=self,
            fixed_kwargs=dict(kwargs),
            backend=self.backend,
        )

    @override
    def bind(self, **kwargs: Any | TaskFuture[Any]) -> TaskFuture[R]:
        # NOTE: All validation is done in FixedParamTask.bind()
        return self.fix().bind(**kwargs)

    @override
    def product(self, **kwargs: Any) -> MapTaskFuture[R]:
        # NOTE: All validation is done in FixedParamTask.product()
        return self.fix().product(**kwargs)

    @override
    def zip(self, **kwargs: Any) -> MapTaskFuture[R]:
        # NOTE: All validation is done in FixedParamTask.zip()
        return self.fix().zip(**kwargs)


@dataclass(frozen=True)
class FixedParamTask(BaseTask[P, R]):
    """
    A task with one or more parameters fixed to specific values.

    Users should **not** directly instantiate this class, use the `Task.fix(..)` instead.
    """

    task: Task[Any, R]
    """The underlying task to be called."""

    fixed_kwargs: Mapping[str, Any]
    """The parameters already bound in this FixedParamTask; can contain other TaskFutures."""

    @cached_property
    @override
    def signature(self) -> Signature:
        return self.task.signature

    @override
    def bind(self, **kwargs: Any | TaskFuture[Any]) -> TaskFuture[R]:
        merged = {**self.fixed_kwargs, **kwargs}

        check_invalid_params(self, merged)
        check_missing_params(self, merged)
        check_overlap_params(self, kwargs)

        return TaskFuture(task=self.task, kwargs=merged, backend=self.backend)

    @override
    def product(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        merged = {**self.fixed_kwargs, **kwargs}

        check_invalid_params(self, merged)
        check_missing_params(self, merged)

        check_overlap_params(self, kwargs)
        check_invalid_map_params(self, kwargs)

        return MapTaskFuture(
            task=self.task,
            mode="product",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )

    @override
    def zip(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        merged = {**self.fixed_kwargs, **kwargs}

        check_invalid_params(self, merged)
        check_missing_params(self, merged)

        check_overlap_params(self, kwargs)
        check_invalid_map_params(self, kwargs)

        len_details = {
            len(val)  # type: ignore
            for val in kwargs.values()
            if not isinstance(val, BaseTaskFuture)
        }
        if len(len_details) > 1:
            raise ParameterError(
                f"Mixed lengths for task '{self.name}', pairwise fan-out with `.zip()` requires "
                f"all sequences to have the same length. Found lengths: {sorted(len_details)}"
            )

        return MapTaskFuture(
            task=self.task,
            mode="zip",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )


# region Helpers

# NOTE: The following helper functions are used for parameter validation and extraction. They
# are public, but generally intended for internal use within the task and future classes.


def check_invalid_params(task: BaseTask, kwargs: dict) -> None:
    """
    Checks that all provided parameters are valid for the given task.

    Args:
        task: Task whose parameters are being validated.
        kwargs: Provided arguments to validate.

    Raises:
        ParameterError: If any provided parameters are not in the task's signature.
    """
    if invalid_params := sorted(kwargs.keys() - task.signature.parameters.keys()):
        raise ParameterError(f"Invalid parameters for task '{task.name}': {invalid_params}")


def check_missing_params(task: BaseTask, kwargs: dict) -> None:
    """
    Checks that all required parameters for the given task are provided.

    Args:
        task: Task whose parameters are being validated.
        kwargs: Provided arguments to validate.

    Raises:
        ParameterError: If any required parameters are missing.
    """
    if missing_params := sorted(task.signature.parameters.keys() - kwargs.keys()):
        raise ParameterError(f"Missing parameters for task '{task.name}': {missing_params}")


def check_overlap_params(task: FixedParamTask, kwargs: dict) -> None:
    """
    Checks that no provided parameters overlap with already fixed parameters.

    Args:
        task: `FixedParamTask` whose parameters are being validated.
        kwargs: Provided arguments to validate.

    Raises:
        ParameterError: If any provided parameters overlap with already fixed parameters.
    """
    fixed = task.fixed_kwargs.keys()
    if overlap_params := sorted(fixed & kwargs.keys()):
        raise ParameterError(
            f"Overlapping parameters for task '{task.name}', specified parameters "
            f"were previously bound in `.fix()`: {overlap_params}"
        )


def check_invalid_map_params(task: BaseTask, kwargs: dict) -> None:
    """
    Checks that all provided parameters for a mapping task are iterable.

    Args:
        task: Task whose parameters are being validated.
        kwargs: Provided arguments to validate.

    Raises:
        ParameterError: If any provided parameters are not iterable.
    """
    non_sequences = []
    parameters = task.signature.parameters.keys()
    for key, value in kwargs.items():
        if key in parameters and not isinstance(value, (Iterable, BaseTaskFuture)):
            non_sequences.append(key)
    if non_sequences := sorted(non_sequences):
        raise ParameterError(
            f"Non-iterable parameters for task '{task.name}', "
            f"all parameters must be Iterable or TaskFuture[Iterable] "
            f"(use `.fix()` to set scalar parameters): {non_sequences}"
        )


def get_unbound_param(task: BaseTask, kwargs: dict) -> str:
    """
    Returns the single unbound parameter name for the given task and provided arguments.

    Args:
        task: Task whose unbound parameter is being determined.
        kwargs: Provided arguments to validate.

    Raises:
        ParameterError: If there are zero or multiple unbound parameters.
    """
    unbound = [p for p in task.signature.parameters if p not in kwargs]
    if len(unbound) == 0:
        raise ParameterError(
            f"Task '{task.name}' has no unbound parameters for "
            f"upstream value. All parameters already provided: {list(kwargs.keys())}"
        )
    if len(unbound) > 1:
        raise ParameterError(
            f"Task '{task.name}' must have exactly one "
            f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
            f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
        )
    return unbound[0]
