from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload

from typing_extensions import Self, override

from daglite.exceptions import ParameterError
from daglite.futures import BaseTaskFuture
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture

# NOTE: Import types only for type checking to avoid circular imports, if you need
# to use them at runtime, import them within methods.
if TYPE_CHECKING:
    from daglite.engine import Backend
else:
    Backend = object

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")

# region Decorator


@overload
def task(func: Callable[P, R]) -> Task[P, R]: ...


@overload
def task(
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Task[P, R] | Callable[[Callable[P, R]], Task[P, R]]:
    """
    Decorator to convert a Python function into a daglite `Task`.

    Tasks are the building blocks of daglite DAGs. They wrap plain Python functions and provide
    methods for composition and execution. This is the recommended way to create tasks. Direct
    instantiation of the `Task` class is discouraged.

    Args:
        func (Callable[P, R], optional):
            The function to wrap. When used without parentheses (`@task`), this is automatically
            passed. When used with parentheses (`@task()`), this is None.
        name (str, optional):
            Custom name for the task. Defaults to the function's `__name__`. For lambda functions,
            defaults to "unnamed_task".
        description (str, optional):
            Task description. Defaults to the function's docstring.
        backend (str | Backend | None):
            Backend for executing this task. Can be a backend name ("sequential", "threading") or a
            Backend instance. If None, uses the engine's default backend.

    Returns:
        Either a `Task` (when used as `@task`) or a decorator function (when used as `@task()`).

    Examples:
        Basic usage
        >>> @task
        >>> def add(x: int, y: int) -> int:
        >>>     return x + y

        With parameters
        >>> @task(name="custom_add", backend="threading")
        >>> def add(x: int, y: int) -> int:
        >>>     return x + y

        Lambda functions
        >>> double = task(lambda x: x * 2, name="double")
    """

    def decorator(fn: Callable[P, R]) -> Task[P, R]:
        from daglite.backends import find_backend

        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@task` can only be applied to callable functions.")

        return Task(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_task"),
            description=description if description is not None else getattr(fn, "__doc__", ""),
            backend=find_backend(backend),
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

    def with_options(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        backend: str | Backend | None = None,
    ) -> "Self":
        """
        Create a new task with updated options.

        Args:
            name (str, optional):
                New name for the task. If `None`, keeps the existing name.
            description (str, optional):
                New description for the task. If `None`, keeps the existing description.
            backend (str | Backend | None, optional):
                New backend for the task. If `None`, keeps the existing backend.

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
            **kwargs (Any):
                Keyword arguments matching the task function's parameters. Values can be concrete
                Python objects or TaskFutures from other tasks.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def product(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> "MapTaskFuture[R]":
        """
        Create a fan-out operation by applying this task over all combinations of sequences.

        This creates a Cartesian product of all provided sequences, calling the task once
        for each combination. Useful for parameter sweeps and batch operations.

        Args:
            **kwargs (Iterable[Any] | TaskFuture[Iterable[Any]]):
                Keyword arguments where values are sequences. Each sequence element will be
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
            **kwargs (Iterable[Any] | TaskFuture[Iterable[Any]]):
                Keyword arguments where values are equal-length sequences. Elements at the same
                index across sequences are combined in each call. Can include TaskFutures that
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

    # NOTE: We should not define `__call__` in order to avoid confusing type checkers. We want
    # them to view this object as a `Task[P, R]` and not as a `Callable[P, R]` (which some type
    # checkers would do if we defined `__call__`).

    def fix(self, **kwargs: Any) -> "FixedParamTask[P, R]":
        """
        Fix some parameters of this task, returning a `FixedParamTask`.

        Args:
            **kwargs (Any):
                Keyword arguments to be fixed for this task. Can be a combination of concrete
                values and TaskFutures.

        Examples:
        >>> def score(x: int, y: int) -> float: ...
        >>>
        >>> base = score.fix(y=seed)
        >>> branch1 = base.bind(x=lazy_x)  # TaskFuture[int]
        >>> branch2 = base.product(x=[1, 2, 3, 4])  # MapTaskFuture[int]
        """
        signature = inspect.signature(self.func)
        if invalid_params := _find_invalid_parameters(signature.parameters, kwargs):
            raise ParameterError(
                f"Invalid parameters for task '{self.name}' in `.fix()`: {invalid_params}"
            )

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

    @override
    def bind(self, **kwargs: Any | TaskFuture[Any]) -> TaskFuture[R]:
        merged = {**self.fixed_kwargs, **kwargs}
        signature = inspect.signature(self.task.func)

        if invalid_params := _find_invalid_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Invalid parameters for task '{self.name}' in `.bind()`: {invalid_params}"
            )

        if missing_params := _find_missing_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Missing parameters for task '{self.name}' in `.bind()`: {missing_params}"
            )

        if overlap_params := _find_overlapping_parameters(self.fixed_kwargs.keys(), kwargs):
            raise ParameterError(
                f"Overlapping parameters for '{self.name}' in `.bind()`, specified parameters "
                f"were previously bound in `.fix()`: {overlap_params}"
            )

        return TaskFuture(task=self.task, kwargs=merged, backend=self.backend)

    @override
    def product(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        signature = inspect.signature(self.task.func)

        merged = {**self.fixed_kwargs, **kwargs}

        if invalid_params := _find_invalid_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Invalid parameters for task '{self.name}' in `.product()`: {invalid_params}"
            )

        if missing_params := _find_missing_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Missing parameters for task '{self.name}' in `.product()`: {missing_params}"
            )

        if overlap_params := _find_overlapping_parameters(self.fixed_kwargs.keys(), kwargs):
            raise ParameterError(
                f"Overlapping parameters for '{self.name}' in `.product()`, specified parameters "
                f"were previously bound in `.fix()`: {overlap_params}"
            )

        if missing_map_params := _find_missing_map_parameters(signature.parameters, kwargs):
            raise ParameterError(
                f"Non-iterable parameters for task '{self.name}' in `.product()`, "
                f"all parameters must be Iterable or TaskFuture[Iterable] "
                f"(use `.fix()` to set scalar parameters): {missing_map_params}"
            )

        return MapTaskFuture(
            task=self.task,
            mode="product",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )

    @override
    def zip(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        signature = inspect.signature(self.task.func)

        merged = {**self.fixed_kwargs, **kwargs}

        if invalid_params := _find_invalid_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Invalid parameters for task '{self.name}' in `.zip()`: {invalid_params}"
            )

        if missing_params := _find_missing_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Missing parameters for task '{self.name}' in `.zip()`: {missing_params}"
            )

        if overlap_params := _find_overlapping_parameters(self.fixed_kwargs.keys(), kwargs):
            raise ParameterError(
                f"Overlapping parameters for '{self.name}' in `.zip()`, specified parameters "
                f"were previously bound in `.fix()`: {overlap_params}"
            )

        if missing_map_params := _find_missing_map_parameters(signature.parameters, kwargs):
            raise ParameterError(
                f"Non-iterable parameters for task '{self.name}' in `.zip()`, "
                f"all parameters must be Iterable or TaskFuture[Iterable] "
                f"(use `.fix()` to set scalar parameters): {missing_map_params}"
            )

        len_details = {
            len(val)  # type: ignore
            for val in kwargs.values()
            if not isinstance(val, BaseTaskFuture)
        }
        if len(len_details) > 1:
            raise ParameterError(
                f"Mixed lengths for scalar parameters in `.zip()`, all sequences must have the "
                f"same length: {kwargs}"
            )

        return MapTaskFuture(
            task=self.task,
            mode="zip",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )


def _find_invalid_parameters(params: Mapping[str, Any], args: dict[str, Any]) -> list[str]:
    return sorted(args.keys() - params.keys())


def _find_missing_parameters(params: Mapping[str, Any], args: dict[str, Any]) -> list[str]:
    return sorted(params.keys() - args.keys())


def _find_missing_map_parameters(params: Mapping[str, Any], args: dict[str, Any]) -> list[str]:
    non_sequences = []
    for key, value in args.items():
        if key in params and not isinstance(value, (Iterable, BaseTaskFuture)):
            non_sequences.append(key)
    return sorted(non_sequences)


def _find_overlapping_parameters(fixed: KeysView[str], new: dict[str, Any]) -> list[str]:
    return sorted(fixed & new.keys())
