from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, Self, TypeVar, overload, override
from uuid import UUID
from uuid import uuid4

from daglite.exceptions import GraphConstructionError
from daglite.exceptions import ParameterError
from daglite.graph.base import GraphBuilder
from daglite.graph.base import ParamInput

if TYPE_CHECKING:
    from daglite.engine import Backend
    from daglite.graph.nodes import MapTaskNode
    from daglite.graph.nodes import TaskNode
else:
    Backend = object
    MapTaskNode = object
    TaskNode = object

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
    def extend(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> "MapTaskFuture[R]":
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
        >>> branch2 = base.extend(x=[1, 2, 3, 4])  # MapTaskFuture[int]
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
    def extend(self, **kwargs: Any) -> MapTaskFuture[R]:
        # NOTE: All validation is done in FixedParamTask.extend()
        return self.fix().extend(**kwargs)

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
        if overlap_params := _find_overlapping_parameters(self.fixed_kwargs.keys(), kwargs):
            raise ParameterError(
                f"Overlapping parameters for '{self.name}' in `.bind()`, specified parameters "
                f"were previously bound in `.fix()`: {overlap_params}"
            )

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

        return TaskFuture(task=self.task, kwargs=merged, backend=self.backend)

    @override
    def extend(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        if overlap_params := _find_overlapping_parameters(self.fixed_kwargs.keys(), kwargs):
            raise ParameterError(
                f"Overlapping parameters for '{self.name}' in `.extend()`, specified parameters "
                f"were previously bound in `.fix()`: {overlap_params}"
            )

        signature = inspect.signature(self.task.func)

        if missing_map_params := _find_missing_map_parameters(signature.parameters, kwargs):
            raise ParameterError(
                f"Non-iterable parameters for task '{self.name}' in `.extend()`, "
                f"all parameters must be Iterable or TaskFuture[Iterable] "
                f"(use `.fix()` to set scalar parameters): {missing_map_params}"
            )

        merged = {**self.fixed_kwargs, **kwargs}

        if invalid_params := _find_invalid_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Invalid parameters for task '{self.name}' in `.extend()`: {invalid_params}"
            )

        if missing_params := _find_missing_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Missing parameters for task '{self.name}' in `.extend()`: {missing_params}"
            )

        return MapTaskFuture(
            task=self.task,
            mode="extend",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )

    @override
    def zip(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        if overlap_params := _find_overlapping_parameters(self.fixed_kwargs.keys(), kwargs):
            raise ParameterError(
                f"Overlapping parameters for '{self.name}' in `.zip()`, specified parameters "
                f"were previously bound in `.fix()`: {overlap_params}"
            )

        signature = inspect.signature(self.task.func)

        if missing_map_params := _find_missing_map_parameters(signature.parameters, kwargs):
            raise ParameterError(
                f"Non-iterable parameters for task '{self.name}' in `.zip()`, "
                f"all parameters must be Iterable or TaskFuture[Iterable] "
                f"(use `.fix()` to set scalar parameters): {missing_map_params}"
            )

        merged = {**self.fixed_kwargs, **kwargs}

        if invalid_params := _find_invalid_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Invalid parameters for task '{self.name}' in `.zip()`: {invalid_params}"
            )

        if missing_params := _find_missing_parameters(signature.parameters, merged):
            raise ParameterError(
                f"Missing parameters for task '{self.name}' in `.zip()`: {missing_params}"
            )

        return MapTaskFuture(
            task=self.task,
            mode="zip",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )


# region Task Futures


class BaseTaskFuture(abc.ABC):
    """Base class for all task futures, representing unevaluated task invocations."""

    @cached_property
    def id(self) -> UUID:
        """Unique identifier for this lazy value instance."""
        return uuid4()

    # NOTE: The following methods are to prevent accidental usage of unevaluated nodes.

    def __bool__(self) -> bool:
        raise TypeError(
            "TaskFutures cannot be used in boolean context. Did you mean to call evaluate() first?"
        )

    def __len__(self) -> int:
        raise TypeError("TaskFutures do not support len(). Did you mean to call evaluate() first?")

    def __repr__(self) -> str:  # pragma : no cover
        return f"<Lazy {id(self):#x}>"


@dataclass(frozen=True)
class TaskFuture(BaseTaskFuture, GraphBuilder, Generic[R]):
    """
    Represents a single task invocation that will produce a value of type R.

    All parameters to the task can be either concrete values or other TaskFutures.
    """

    task: Task[Any, R]
    """Underlying task to be called."""

    kwargs: Mapping[str, Any]
    """Parameters to be passed to the task during execution, can contain other task futures."""

    backend: Backend
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []
        for value in self.kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)  # type: ignore[arg-type]
        return deps

    @override
    def to_graph(self) -> TaskNode:
        from daglite.graph.nodes import TaskNode

        kwargs: dict[str, ParamInput] = {}
        for name, value in self.kwargs.items():
            if isinstance(value, BaseTaskFuture):
                kwargs[name] = ParamInput.from_ref(value.id)
            else:
                kwargs[name] = ParamInput.from_value(value)
        return TaskNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            kwargs=kwargs,
            backend=self.backend,
        )


@dataclass(frozen=True)
class MapTaskFuture(BaseTaskFuture, GraphBuilder, Generic[R]):
    """
    Represents a fan-out task invocation producing a sequence of values of type R.

    Fan-out can be done in either 'extend' (Cartesian product) or 'zip' (pairwise) mode. In both
    modes, some parameters are fixed across all calls, while others are parameters will be iterated
    over. All iterable parameters must be of type `Iterable[Any]` or `TaskFuture[Iterable[Any]]`.
    """

    task: Task[Any, R]
    """Underlying task to be called."""

    mode: str  # "extend" or "zip"
    """Mode of operation ('extend' for Cartesian product, 'zip' for pairwise)."""

    fixed_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to fixed values applied to every call.

    Note that fixed parameters can be a combination of concrete values and TaskFutures.
    """

    mapped_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to sequences to be iterated over during calls.

    Note that sequence parameters can be a combination of concrete values and TaskFutures.
    """

    backend: Backend
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    def map(self, mapped_task: Task[Any, S] | FixedParamTask[Any, S]) -> MapTaskFuture[S]:
        """
        Apply a task to each element of this sequence.

        Args:
            mapped_task (Task[Any, S] | FixedParamTask[Any, S]):
                `Task` with exactly ONE parameter, or a `FixedParamTask` where one parameter
                remains unbound.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound parameter.

        Examples:
        Single-parameter task
        >>> @task
        >>> def double(x: int) -> int:
        >>>     return x * 2
        >>> results = numbers.map(double)

        Multi-parameter task with scalars fixed
        >>> @task
        >>> def scale(x: int, factor: int) -> int:
        >>>     return x * factor
        >>> results = numbers.map(scale.fix(factor=10))
        """
        if isinstance(mapped_task, FixedParamTask):
            sig = inspect.signature(mapped_task.task.func)
            bound_params = set(mapped_task.fixed_kwargs.keys())
            unbound_params = [p for p in sig.parameters if p not in bound_params]
            if len(unbound_params) != 1:
                raise ParameterError(
                    f"Task '{mapped_task.name}' in `.map()` must have exactly one "
                    f"unbound parameter, found {len(unbound_params)} "
                    f"(use `.fix()` to set scalar parameters): {unbound_params}"
                )
            param_name = unbound_params[0]
            actual_task = mapped_task.task
            fixed_kwargs = mapped_task.fixed_kwargs

        else:
            sig = inspect.signature(mapped_task.func)
            params = list(sig.parameters.keys())
            if len(params) != 1:
                raise ParameterError(
                    f"Task '{mapped_task.name}' in `.map()` must have exactly one parameter, "
                    f"found {len(params)} (use `.fix()` to set scalar parameters): {params}"
                )
            param_name = params[0]
            actual_task = mapped_task
            fixed_kwargs = {}

        return MapTaskFuture(
            task=actual_task,
            mode="extend",
            fixed_kwargs=fixed_kwargs,
            mapped_kwargs={param_name: self},
            backend=self.backend,
        )

    def join(self, reducer_task: Task[Any, S] | FixedParamTask[Any, S]) -> TaskFuture[S]:
        """
        Reduce this sequence to a single value.

        Args:
            reducer_task:
                `Task` with exactly ONE parameter (the sequence), or a `FixedParamTask` with
                one unbound parameter.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound parameter.

        Examples:
            # Simple reducer
            >>> @task
            >>> def sum_all(xs: list[int]) -> int:
            >>>    return sum(xs)
            >>> total = numbers.join(sum_all)

            # Reducer with additional parameters
            >>> @task
            >>> def weighted_sum(xs: list[int], weight: float) -> float:
            >>>    return sum(xs) * weight
            >>> total = numbers.join(weighted_sum.fix(weight=0.5))
        """

        if isinstance(reducer_task, FixedParamTask):
            sig = inspect.signature(reducer_task.task.func)
            bound_params = set(reducer_task.fixed_kwargs.keys())
            unbound_params = [p for p in sig.parameters if p not in bound_params]
            if len(unbound_params) != 1:
                raise ParameterError(
                    f"Task '{reducer_task.name}' in `.join()` must have exactly one "
                    f"unbound parameter, found {len(unbound_params)} "
                    f"(use `.fix()` to set scalar parameters): {unbound_params}"
                )
            param_name = unbound_params[0]
            actual_task = reducer_task.task
            fixed_kwargs = reducer_task.fixed_kwargs

        else:
            sig = inspect.signature(reducer_task.func)
            params = list(sig.parameters.keys())

            if len(params) != 1:
                raise ParameterError(
                    f"Task '{reducer_task.name}' in `.join()` must have exactly one parameter, "
                    f"found {len(params)} (use `.fix()` to set scalar parameters): {params}"
                )
            param_name = params[0]
            actual_task = reducer_task
            fixed_kwargs = {}

        merged_kwargs = dict(fixed_kwargs)
        merged_kwargs[param_name] = self

        return TaskFuture(
            task=actual_task,
            kwargs=merged_kwargs,
            backend=self.backend,
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []
        for value in self.fixed_kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)  # type: ignore[arg-type]
        for seq in self.mapped_kwargs.values():
            if isinstance(seq, BaseTaskFuture):
                deps.append(seq)  # type: ignore[arg-type]
        return deps

    @override
    def to_graph(self) -> MapTaskNode:
        from daglite.graph.nodes import MapTaskNode

        if self.mode not in {"extend", "zip"}:
            raise GraphConstructionError(f"Invalid MapTaskFuture mode: '{self.mode}'")

        fixed_kwargs: dict[str, ParamInput] = {}
        mapped_kwargs: dict[str, ParamInput] = {}

        for name, value in self.fixed_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                fixed_kwargs[name] = ParamInput.from_ref(value.id)
            else:
                fixed_kwargs[name] = ParamInput.from_value(value)

        for name, seq in self.mapped_kwargs.items():
            if isinstance(seq, BaseTaskFuture):
                mapped_kwargs[name] = ParamInput.from_sequence_ref(seq.id)
            else:
                mapped_kwargs[name] = ParamInput.from_sequence(seq)

        return MapTaskNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            mode=self.mode,
            fixed_kwargs=fixed_kwargs,
            mapped_kwargs=mapped_kwargs,
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
