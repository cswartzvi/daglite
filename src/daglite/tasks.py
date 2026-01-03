from __future__ import annotations

import abc
import inspect
import sys
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from functools import cached_property
from inspect import Signature
from typing import Any, Generic, ParamSpec, TypeVar, overload

from typing_extensions import Self, override

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
    backend_name: str | None = None,
    retries: int | None = None,
    timeout: float | None = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(  # noqa: D417
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend_name: str | None = None,
    retries: int | None = None,
    timeout: float | None = None,
) -> Any:
    """
    Decorator to convert a Python function into a daglite `Task`.

    Tasks are the building blocks of daglite DAGs. They wrap plain Python functions (both sync
    and async) and provide methods for composition and execution.

    This is the recommended way for users to create tasks. Users should **not** directly
    instantiate the `Task` class.

    Args:
        name: Custom name for the task. Defaults to the function's `__name__`. For lambda functions,
            defaults to "unnamed_task".
        description: Task description. Defaults to the function's docstring.
        backend_name: Name of the backend to use for this task. If not provided, uses the default
            global settings backend.
        retries: Number of times to retry the task on failure. Defaults to 0 (no retries).
        timeout: Maximum execution time in seconds. If None, no timeout is enforced.

    Returns:
        Either a `Task` (when used as `@task`) or a decorator function (when used as `@task()`).

    Examples:
        >>> from daglite import task

        Basic usage with synchronous function
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> add.name
        'add'

        Custom parameters
        >>> @task(name="custom_add")
        ... def add_nums(x: int, y: int) -> int:
        ...     return x + y
        >>> add_nums.name
        'custom_add'

        Lambda functions
        >>> double = task(lambda x: x * 2, name="double")
        >>> double.name
        'double'
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
            backend_name=backend_name,
            is_async=is_async,
            retries=retries if retries is not None else 0,
            timeout=timeout,
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

    backend_name: str | None
    """Name of backend to use for this task."""

    retries: int = field(default=0, kw_only=True)
    """Number of times to retry the task on failure."""

    timeout: float | None = field(default=None, kw_only=True)
    """Maximum execution time in seconds. If None, no timeout is enforced."""

    @cached_property
    @abc.abstractmethod
    def signature(self) -> Signature:
        """Signature of the underlying task function."""
        raise NotImplementedError()

    def with_options(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        backend_name: str | None = None,
        retries: int | None = None,
        timeout: float | None = None,
    ) -> Self:
        """
        Create a new task with updated options.

        Args:
            name: New name for the task. If `None`, keeps the existing name.
            description: New description for the task. If `None`, keeps the existing description.
            backend_name: New backend name for the task. If `None`, keeps the existing backend name.
            retries: New number of retries for the task. If `None`, keeps the existing retries.
            timeout: New timeout for the task. If `None`, keeps the existing timeout.

        Returns:
            A new `BaseTask` instance with updated options.
        """

        name = name if name is not None else self.name
        description = description if description is not None else self.description
        backend_name = backend_name if backend_name is not None else self.backend_name
        new_retries = retries if retries is not None else self.retries
        new_timeout = timeout if timeout is not None else self.timeout

        # Collect the remaining fields (assumes this is a dataclass)
        remaining_fields = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"name", "description", "backend_name", "retries", "timeout"}
        }

        return type(self)(
            name=name,
            description=description,
            backend_name=backend_name,
            retries=new_retries,
            timeout=new_timeout,
            **remaining_fields,
        )

    @abc.abstractmethod
    def __call__(self, **kwargs: Any | TaskFuture[Any]) -> "TaskFuture[R]":
        """
        Creates a task future by binding values to the parameters of this task.

        This does NOT execute the task immediately; it returns a future object representing
        evaluation results. See `daglite.engine.evaluate` or `daglite.engine.evaluate_async` for
        more details on evaluating task futures.

        Args:
            **kwargs: Keyword arguments matching the task function's parameters. Must include
                all required parameters. Can include `TaskFuture` objects.

        Returns:
            A `TaskFuture` representing the execution of this task with the provided parameters.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def add(x: int, y: int) -> int:
            ...     return x + y

            Future creation
            >>> add(x=1, y=2)  # doctest: +ELLIPSIS
            TaskFuture(...)

            Future evaluation
            >>> evaluate(add(x=1, y=2))
            3

            Connecting upstream task futures
            >>> @task
            ... def multiply(x: int, factor: int) -> int:
            ...     return x * factor
            >>> future_add = add(x=2, y=3)
            >>> future_multiply = multiply(x=future_add, factor=10)
            >>> evaluate(future_multiply)
            50

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def product(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> "MapTaskFuture[R]":
        """
        Create a fan-out operation by applying this task over all combinations of sequences.

        This creates a Cartesian product of all provided sequences, calling the task once for each
        combination. Useful for parameter sweeps and batch operations.

        Args:
            **kwargs: Keyword arguments where values are sequences. Each sequence element will be
                combined with elements from other sequences in a Cartesian product. Can include
                `TaskFuture` objects that resolve to sequences.

        Returns:
            A `MapTaskFuture` representing the fan-out execution of this task.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def combine(x: int, y: int) -> int:
            ...     return x + y

            All sequences provided:
            >>> future = combine.product(x=[1, 2], y=[10, 20])
            >>> evaluate(future)
            [11, 21, 12, 22]

            Fixed scalar parameter with single sequence:
            >>> future = combine.partial(y=10).product(x=[1, 2, 3])
            >>> evaluate(future)
            [11, 12, 13]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def zip(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> "MapTaskFuture[R]":
        """
        Create a fan-out operation by applying this task to zipped sequences.

        Sequences are zipped element-wise (similar to Python's `zip(`) function), calling the task
        once for each aligned set of elements. All sequences must have the same length or an error
        is raised.

        Args:
            **kwargs: Keyword arguments where values are equal-length sequences. Elements at the
                same index across sequences are combined in each call. Can include `TaskFuture`
                objects that resolve to sequences.

        Returns:
            A `MapTaskFuture` representing the fan-out execution of this task.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def combine(x: int, y: int) -> int:
            ...     return x + y

            All sequences provided:
            >>> future = combine.zip(x=[1, 2, 3], y=[10, 20, 30])
            >>> evaluate(future)
            [11, 22, 33]

            Fixed scalar parameter with single sequence:
            >>> future = combine.partial(y=10).zip(x=[1, 2, 3])
            >>> evaluate(future)
            [11, 12, 13]
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

    @cached_property
    @override
    def signature(self) -> Signature:
        return inspect.signature(self.func)

    @override
    def __call__(self, **kwargs: Any | TaskFuture[Any]) -> TaskFuture[R]:
        return self.partial()(**kwargs)

    def partial(self, **kwargs: Any) -> PartialTask[P, R]:
        """
        Partially apply some parameters of this task, returning a "partial" task.

        Similar to `functools.partial`, but for daglite tasks. This creates a reusable task
        template with some parameters already fixed. It also allows users to create tasks with a
        mix of scalar and mapped parameters.

        Args:
            **kwargs: Keyword arguments to be fixed for this task. Can include literals, variables,
                or even `TaskFuture` objects.

        Returns:
            A `PartialTask` with the specified parameters fixed.

        Examples:
            >>> from daglite import task
            >>> @task
            ... def score(x: int, y: int, z: int) -> float:
            ...     return (x + y) / z

            Create a partial task with `y` and `z` fixed
            >>> seed = 42
            >>> base = score.partial(y=seed, z=0.5)

            Use the partial task with only the remaining parameter
            >>> base(x=1)  # doctest: +ELLIPSIS
            TaskFuture(...)
            >>> base.product(x=[1, 2, 3, 4])  # doctest: +ELLIPSIS
            MapTaskFuture(...)
        """
        check_invalid_params(self, kwargs)
        return PartialTask(
            name=self.name,
            description=self.description,
            task=self,
            fixed_kwargs=dict(kwargs),
            backend_name=self.backend_name,
            retries=self.retries,
            timeout=self.timeout,
        )

    @override
    def product(self, **kwargs: Any) -> MapTaskFuture[R]:
        # Create a PartialTask with no fixed params, then call product
        return self.partial().product(**kwargs)

    @override
    def zip(self, **kwargs: Any) -> MapTaskFuture[R]:
        # Create a PartialTask with no fixed params, then call zip
        return self.partial().zip(**kwargs)


@dataclass(frozen=True)
class PartialTask(BaseTask[P, R]):
    """
    A task with one or more parameters partially applied to specific values.

    This creates a reusable task template that can be called multiple times with different values
    for the remaining parameters. Similar to `functools.partial`.

    Users should **not** directly instantiate this class, use `Task.partial()` instead.
    """

    task: Task[Any, R]
    """The underlying task to be called."""

    fixed_kwargs: Mapping[str, Any]
    """The parameters already bound in this PartialTask; can contain other TaskFutures."""

    @cached_property
    @override
    def signature(self) -> Signature:
        return self.task.signature

    @override
    def __call__(self, **kwargs: Any | TaskFuture[Any]) -> TaskFuture[R]:
        merged = {**self.fixed_kwargs, **kwargs}

        check_invalid_params(self, merged)
        check_missing_params(self, merged)
        check_overlap_params(self, kwargs)

        return TaskFuture(task=self.task, kwargs=merged, backend_name=self.backend_name)

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
            backend_name=self.backend_name,
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
            backend_name=self.backend_name,
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


def check_overlap_params(task: PartialTask, kwargs: dict) -> None:
    """
    Checks that no provided parameters overlap with already fixed parameters.

    Args:
        task: `PartialTask` whose parameters are being validated.
        kwargs: Provided arguments to validate.

    Raises:
        ParameterError: If any provided parameters overlap with already fixed parameters.
    """
    fixed = task.fixed_kwargs.keys()
    if overlap_params := sorted(fixed & kwargs.keys()):
        raise ParameterError(
            f"Overlapping parameters for task '{task.name}', specified parameters "
            f"were previously bound in `.partial()`: {overlap_params}"
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
            f"(use `.partial()` to set scalar parameters): {non_sequences}"
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
            f"(use `.partial()` to set scalar parameters): {unbound[1:]}"
        )
    return unbound[0]
