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
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload

from typing_extensions import Self, override

from daglite._typing import MapMode
from daglite._validation import check_invalid_map_params
from daglite._validation import check_invalid_params
from daglite._validation import check_missing_params
from daglite._validation import check_overlap_params
from daglite._validation import resolve_positional_args
from daglite.datasets.store import DatasetStore
from daglite.exceptions import ParameterError

# NOTE: Tasks are the building blocks of daglite DAGs, however the fluent API for composing
# DAGs allows for tasks to accept downstream types as parameters. To avoid circular imports,
# all downstream types used in type annotations are imported within TYPE_CHECKING blocks.
# If methods within this file need to use downstream types at runtime, they should be imported
# locally within those methods.
if TYPE_CHECKING:
    from daglite.futures import MapTaskFuture
    from daglite.futures import TaskFuture
    from daglite.futures.iter_future import IterTaskFuture
else:
    MapTaskFuture = object
    TaskFuture = object
    IterTaskFuture = object


P = ParamSpec("P")
R = TypeVar("R")


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
    cache: bool = False,
    cache_ttl: int | None = None,
    store: DatasetStore | str | None = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(  # noqa: D417
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend_name: str | None = None,
    retries: int | None = None,
    timeout: float | None = None,
    cache: bool = False,
    cache_ttl: int | None = None,
    store: DatasetStore | str | None = None,
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
        timeout: Maximum execution time in seconds. Must be non-negative if provided.
            If None, no timeout is enforced.
        cache: Whether to enable hash-based caching for this task. When True, the task's output
            will be cached based on a hash of its function source and input parameters.
            Defaults to False.
        cache_ttl: Time-to-live for cached results in seconds. If None, cached results never expire.
            Only used when cache=True.
        store: Default output store for all `.save()` calls on this task. Can be a DatasetStore
            instance or a string path (e.g., directory) to be used for to create a dataset store.
            If None, the default store will be used.

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
    from daglite.datasets.store import DatasetStore

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

        actual_store = DatasetStore(store) if isinstance(store, str) else store

        return Task(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_task"),
            description=description if description is not None else getattr(fn, "__doc__", ""),
            backend_name=backend_name,
            is_async=is_async,
            retries=retries if retries is not None else 0,
            timeout=timeout,
            cache=cache,
            cache_ttl=cache_ttl,
            store=actual_store,
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

    cache: bool = field(default=False, kw_only=True)
    """Whether to enable hash-based caching for this task."""

    cache_ttl: int | None = field(default=None, kw_only=True)
    """Time-to-live for cached results in seconds. If None, cached results never expire."""

    store: DatasetStore | None = field(default=None, kw_only=True)
    """Default dataset store for `.save()` calls on this task."""

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ParameterError(
                f"Task '{self.name}' has invalid retries={self.retries}. Must be non-negative."
            )
        if self.timeout is not None and self.timeout < 0:
            raise ParameterError(
                f"Task '{self.name}' has invalid timeout={self.timeout}. Must be non-negative."
            )
        if self.cache_ttl is not None and self.cache_ttl < 0:
            raise ParameterError(
                f"Task '{self.name}' has invalid cache_ttl={self.cache_ttl}. Must be non-negative."
            )

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
        cache: bool | None = None,
        cache_ttl: int | None = None,
        store: DatasetStore | None = None,
    ) -> Self:
        """
        Create a new task with updated options.

        Args:
            name: New name for the task. If `None`, keeps the existing name.
            description: New description for the task. If `None`, keeps the existing description.
            backend_name: New backend name for the task. If `None`, keeps the existing backend name.
            retries: New number of retries for the task. If `None`, keeps the existing retries.
            timeout: New timeout for the task. If `None`, keeps the existing timeout.
            cache: Whether to enable caching for the task. If `None`, keeps the existing setting.
            cache_ttl: Time-to-live for cached results. If `None`, keeps the existing cache_ttl.
            store: Default dataset store for the task. If `None`, keeps the existing store.

        Returns:
            A new `BaseTask` instance with updated options.
        """

        name = name if name is not None else self.name
        description = description if description is not None else self.description
        backend_name = backend_name if backend_name is not None else self.backend_name
        new_retries = retries if retries is not None else self.retries
        new_timeout = timeout if timeout is not None else self.timeout
        new_cache = cache if cache is not None else self.cache
        new_cache_ttl = cache_ttl if cache_ttl is not None else self.cache_ttl
        new_store = store if store is not None else self.store

        # Collect the remaining fields (assumes this is a dataclass)
        remaining_fields = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in {
                "name",
                "description",
                "backend_name",
                "retries",
                "timeout",
                "cache",
                "cache_ttl",
                "store",
            }
        }

        return type(self)(
            name=name,
            description=description,
            backend_name=backend_name,
            retries=new_retries,
            timeout=new_timeout,
            cache=new_cache,
            cache_ttl=new_cache_ttl,
            store=new_store,
            **remaining_fields,
        )

    @abc.abstractmethod
    def __call__(
        self, *args: Any | TaskFuture[Any], **kwargs: Any | TaskFuture[Any]
    ) -> "TaskFuture[R]":
        """
        Creates a task future by binding values to the parameters of this task.

        This does NOT execute the task immediately; it returns a future object representing
        evaluation results. See `daglite.engine.evaluate` or `daglite.engine.evaluate_async` for
        more details on evaluating task futures.

        Supports both positional and keyword arguments. Positional arguments are resolved
        to keyword arguments using the task function's parameter names.

        Args:
            *args: Positional arguments resolved to parameters by position.
            **kwargs: Keyword arguments matching the task function's parameters. Must include
                all required parameters. Can include `TaskFuture` objects.

        Returns:
            A `TaskFuture` representing the execution of this task with the provided parameters.

        Examples:
            >>> from daglite import task
            >>> @task
            ... def add(x: int, y: int) -> int:
            ...     return x + y

            Future creation (keyword or positional)
            >>> add(x=1, y=2)  # doctest: +ELLIPSIS
            TaskFuture(...)
            >>> add(1, 2)  # doctest: +ELLIPSIS
            TaskFuture(...)

            Future evaluation
            >>> add(x=1, y=2).run()
            3

            Connecting upstream task futures
            >>> @task
            ... def multiply(x: int, factor: int) -> int:
            ...     return x * factor
            >>> future_add = add(x=2, y=3)
            >>> future_multiply = multiply(x=future_add, factor=10)
            >>> future_multiply.run()
            50

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def map(
        self, *, map_mode: MapMode = "zip", **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]
    ) -> MapTaskFuture[R]:
        """
        Create a fan-out operation by applying this task to mapped sequences.

        Args:
            map_mode: Whether to use "product" (Cartesian product) or "zip" (element-wise zip) for
                the fan-out operation. Defaults to "zip". If "zip" is used, all sequences must have
                the same length or an error is raised.
            **kwargs: Keyword arguments where values are sequences. Each sequence element will be
                combined with elements from other sequences (based on the map_mode) and passed as
                parameters to this task.

        Returns:
            A `MapTaskFuture` representing the fan-out execution of this task.

        Examples:
            >>> from daglite import task
            >>> @task
            ... def combine(x: int, y: int) -> int:
            ...     return x + y

            Zip mode (default) - All sequences provided:
            >>> future = combine.map(x=[1, 2, 3], y=[10, 20, 30])
            >>> future.run()
            [11, 22, 33]

            Zip mode (default) - Fixed scalar parameter with single sequence:
            >>> future = combine.partial(y=10).map(x=[1, 2, 3])
            >>> future.run()
            [11, 12, 13]

            Product mode - All sequences provided:
            >>> future = combine.map(x=[1, 2], y=[10, 20], map_mode="product")
            >>> future.run()
            [11, 21, 12, 22]

            Product mode - Fixed scalar parameter with single sequence:
            >>> future = combine.partial(y=10).map(x=[1, 2, 3], map_mode="product")
            >>> future.run()
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
        super().__post_init__()
        # Detect if function is async and update is_async field
        if inspect.iscoroutinefunction(self.func):
            object.__setattr__(self, "is_async", True)

    @cached_property
    @override
    def signature(self) -> Signature:
        return inspect.signature(self.func)

    @override
    def __call__(
        self, *args: Any | TaskFuture[Any], **kwargs: Any | TaskFuture[Any]
    ) -> TaskFuture[R]:
        kwargs = resolve_positional_args(self.signature, args, kwargs, self.name)
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
            >>> base.map(x=[1, 2, 3, 4])  # doctest: +ELLIPSIS
            MapTaskFuture(...)
        """
        check_invalid_params(self.signature, kwargs, self.name)
        return PartialTask(
            name=self.name,
            description=self.description,
            task=self,
            fixed_kwargs=dict(kwargs),
            backend_name=self.backend_name,
            retries=self.retries,
            timeout=self.timeout,
            store=self.store,
            cache=self.cache,
            cache_ttl=self.cache_ttl,
        )

    @override
    def map(self, *, map_mode: MapMode = "zip", **kwargs: Any) -> MapTaskFuture[R]:
        # Create a PartialTask with no fixed params, then call map
        return self.partial().map(**kwargs, map_mode=map_mode)

    def iter(self, *args: Any, **kwargs: Any) -> IterTaskFuture[R]:
        """
        Creates a lazy iterator that yields items one at a time.

        Args:
            *args: Positional arguments resolved to parameters by position.
            **kwargs: Keyword arguments matching the task function's parameters.

        Returns:
            An `IterTaskFuture` representing the lazy iterator invocation.
        """
        kwargs = resolve_positional_args(self.signature, args, kwargs, self.name)
        return self.partial().iter(**kwargs)


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
    def __call__(
        self, *args: Any | TaskFuture[Any], **kwargs: Any | TaskFuture[Any]
    ) -> TaskFuture[R]:
        from daglite.futures import TaskFuture

        # Resolve positional args against unbound parameters (those not already fixed)
        unbound_params = [
            name for name in self.signature.parameters if name not in self.fixed_kwargs
        ]
        unbound_sig = self.signature.replace(
            parameters=[self.signature.parameters[name] for name in unbound_params]
        )
        kwargs = resolve_positional_args(unbound_sig, args, kwargs, self.name)

        merged = {**self.fixed_kwargs, **kwargs}

        check_invalid_params(self.signature, merged, self.name)
        check_missing_params(self.signature, merged, self.name)
        check_overlap_params(dict(self.fixed_kwargs), kwargs, self.name)

        return TaskFuture(
            task=self.task, kwargs=merged, backend_name=self.backend_name, task_store=self.store
        )

    def iter(self, **kwargs: Any) -> IterTaskFuture[R]:
        """
        Create a lazy iterator that yields items one at a time.

        Args:
            **kwargs: Keyword arguments for the remaining unbound parameters.

        Returns:
            An `IterTaskFuture` representing the lazy iterator invocation.
        """
        from daglite.futures.iter_future import IterTaskFuture

        merged = {**self.fixed_kwargs, **kwargs}
        check_invalid_params(self.signature, merged, self.name)
        check_missing_params(self.signature, merged, self.name)
        check_overlap_params(dict(self.fixed_kwargs), kwargs, self.name)
        return IterTaskFuture(
            task=self.task,
            kwargs=merged,
            backend_name=self.backend_name,
            task_store=self.store,
        )

    @override
    def map(
        self, *, map_mode: MapMode = "zip", **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]
    ) -> MapTaskFuture[R]:
        from daglite.futures import MapTaskFuture
        from daglite.futures.base import BaseTaskFuture
        from daglite.futures.iter_future import IterTaskFuture

        merged = {**self.fixed_kwargs, **kwargs}

        check_invalid_params(self.signature, merged, self.name)
        check_missing_params(self.signature, merged, self.name)
        check_overlap_params(dict(self.fixed_kwargs), kwargs, self.name)
        check_invalid_map_params(self.signature, kwargs, self.name)

        # Validate: .iter() sources cannot be mixed with other mapped arguments
        iter_kwargs = {k for k, v in kwargs.items() if isinstance(v, IterTaskFuture)}
        if iter_kwargs and len(kwargs) > len(iter_kwargs):
            raise ParameterError(
                f"`.iter()` must be the only mapped argument for task '{self.name}'. "
                f"Cannot mix iter source with other mapped arguments."
            )

        if map_mode == "zip":
            len_details = {
                len(val)  # type: ignore
                for val in kwargs.values()
                if not isinstance(val, BaseTaskFuture)
            }
            if len(len_details) > 1:
                raise ParameterError(
                    f"Mixed lengths for task '{self.name}', pairwise fan-out with `map` in 'zip' "
                    f"mode requires all sequences to have the same length. Found lengths: "
                    f"{sorted(len_details)}"
                )

        return MapTaskFuture(
            task=self.task,
            mode=map_mode,
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend_name=self.backend_name,
            task_store=self.store,
        )
