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
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from typing_extensions import Self, override

from daglite.exceptions import ParameterError
from daglite.graph.base import GraphBuilder
from daglite.graph.base import ParamInput

# NOTE: Import types only for type checking to avoid circular imports, if you need
# to use them at runtime, import them within methods.
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
def task(func: Callable[P, R], /) -> Task[P, R]: ...


@overload
def task(
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Any:
    """
    Decorator to convert a Python function into a daglite `Task`.

    Tasks are the building blocks of daglite DAGs. They wrap plain Python functions (both sync
    and async) and provide methods for composition and execution. This is the recommended way to
    create tasks. Direct instantiation of the `Task` class is discouraged.

    Args:
        func (Callable[P, R], optional):
            The function to wrap. When used without parentheses (`@task`), this is automatically
            passed. When used with parentheses (`@task()`), this is None. Can be either a
            synchronous function or an async function.
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
        Synchronous function
        >>> @task
        >>> def add(x: int, y: int) -> int:
        >>>     return x + y

        Async function
        >>> @task
        >>> async def fetch_data(url: str) -> dict:
        >>>     async with httpx.AsyncClient() as client:
        >>>         response = await client.get(url)
        >>>         return response.json()

        With parameters
        >>> @task(name="custom_add", backend="threading")
        >>> def add(x: int, y: int) -> int:
        >>>     return x + y

        Lambda functions
        >>> double = task(lambda x: x * 2, name="double")
    """

    def decorator(fn: Any) -> Any:
        from daglite.backends import find_backend

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
        _check_invalid_params(self, kwargs, method="fix")
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

        _check_invalid_params(self, merged, method="bind")
        _check_missing_params(self, merged, method="bind")
        _check_overlap_params(self, kwargs, method="bind")

        return TaskFuture(task=self.task, kwargs=merged, backend=self.backend)

    @override
    def product(self, **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]]) -> MapTaskFuture[R]:
        merged = {**self.fixed_kwargs, **kwargs}

        _check_invalid_params(self, merged, method="product")
        _check_missing_params(self, merged, method="product")

        _check_overlap_params(self, kwargs, method="product")
        _check_invalid_map_params(self, kwargs, method="product")

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

        _check_invalid_params(self, merged, method="zip")
        _check_missing_params(self, merged, method="zip")

        _check_overlap_params(self, kwargs, method="zip")
        _check_invalid_map_params(self, kwargs, method="zip")

        len_details = {
            len(val)  # type: ignore
            for val in kwargs.values()
            if not isinstance(val, BaseTaskFuture)
        }
        if len(len_details) > 1:
            raise ParameterError(
                f"Mixed lengths for task '{self.name}' in `.zip()`, all sequences must have the "
                f"same length. Found lengths: {sorted(len_details)}"
            )

        return MapTaskFuture(
            task=self.task,
            mode="zip",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=self.backend,
        )


# region Futures


class BaseTaskFuture(abc.ABC, GraphBuilder, Generic[R]):
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
class TaskFuture(BaseTaskFuture[R]):
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

    @overload
    def then(self, next_task: Task[Any, S]) -> "TaskFuture[S]": ...

    @overload
    def then(
        self, next_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> "TaskFuture[S]": ...

    def then(
        self,
        next_task: Task[Any, S] | FixedParamTask[Any, S],
        **kwargs: Any,
    ) -> "TaskFuture[S]":
        """
        Chain this task's output as input to another task.

        Additional kwargs are passed to the next task. The upstream output is automatically bound
        to the single remaining unbound parameter.

        Args:
            next_task (Task[Any, S] | FixedParamTask[Any, S]):
                Task or FixedParamTask to chain with this task's output. Must have exactly one
                unbound parameter.
            **kwargs (Any):
                Additional parameters to pass to the next task.

        Returns:
            A new TaskFuture representing the chained computation.

        Raises:
            ParameterError: If the next task has no unbound parameters, multiple unbound
            parameters, or if kwargs overlap with fixed parameters.

        Examples:
            Basic chaining:
            >>> @task
            >>> def fetch(url: str) -> str:
            >>>     return requests.get(url).text
            >>>
            >>> @task
            >>> def parse(html: str) -> dict:
            >>>     return parse_html(html)
            >>>
            >>> result = fetch.bind(url="example.com").then(parse)

            With inline parameters:
            >>> @task
            >>> def transform(data: str, format: str) -> dict:
            >>>     return convert(data, format)
            >>>
            >>> result = fetch.bind(url="example.com").then(transform, format="json")

            With pre-fixed tasks:
            >>> transform_json = transform.fix(format="json")
            >>> result = fetch.bind(url="example.com").then(transform_json)

            Chaining multiple operations:
            >>> result = (
            >>>     fetch.bind(url="example.com")
            >>>     .then(parse)
            >>>     .then(transform, format="json")
            >>>     .then(save, path="output.json")
            >>> )
        """
        if isinstance(next_task, FixedParamTask):
            _check_overlap_params(next_task, kwargs, method="then")
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            actual_task = next_task

        ubound_param = _get_unbound_param(actual_task, all_fixed, method="then")
        return actual_task.bind(**{ubound_param: self}, **all_fixed)

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []
        for value in self.kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)
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
class MapTaskFuture(BaseTaskFuture[R]):
    """
    Represents a fan-out task invocation producing a sequence of values of type R.

    Fan-out can be done in either 'extend' (Cartesian product) or 'zip' (pairwise) mode. In both
    modes, some parameters are fixed across all calls, while others are parameters will be iterated
    over. All iterable parameters must be of type `Iterable[Any]` or `TaskFuture[Iterable[Any]]`.
    """

    task: Task[Any, R]
    """Underlying task to be called."""

    mode: str  # "product" or "zip"
    """Mode of operation ('product' for Cartesian product, 'zip' for pairwise)."""

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

    @overload
    def map(self, mapped_task: Task[Any, S]) -> "MapTaskFuture[S]": ...

    @overload
    def map(
        self, mapped_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> "MapTaskFuture[S]": ...

    def map(
        self, mapped_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> MapTaskFuture[S]:
        """
        Apply a task to each element of this sequence.

        Args:
            mapped_task: Task with exactly ONE parameter, or a FixedParamTask where
                one parameter remains unbound.
            **kwargs: Additional parameters to pass to the task. The sequence element
                will be bound to the single remaining unbound parameter.

        Returns:
            A new MapTaskFuture with the mapped task applied to each element.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound
                parameter, or if kwargs overlap with fixed parameters.

        Examples:
            Single-parameter task:
            >>> @task
            >>> def double(x: int) -> int:
            >>>     return x * 2
            >>>
            >>> numbers = identity.product(x=[1, 2, 3])
            >>> doubled = numbers.map(double)
            >>> # Result: [2, 4, 6]

            Multi-parameter task with inline kwargs:
            >>> @task
            >>> def scale(x: int, factor: int) -> int:
            >>>     return x * factor
            >>>
            >>> scaled = numbers.map(scale, factor=10)
            >>> # Result: [10, 20, 30]

            Chaining map operations:
            >>> result = (
            >>>     numbers
            >>>     .map(scale, factor=2)
            >>>     .map(add, offset=10)
            >>>     .map(square)
            >>> )

            With pre-fixed tasks (still supported):
            >>> scaled = numbers.map(scale.fix(factor=10))
        """
        if isinstance(mapped_task, FixedParamTask):
            _check_overlap_params(mapped_task, kwargs, method="map")
            all_fixed = {**mapped_task.fixed_kwargs, **kwargs}
            actual_task = mapped_task.task
        else:
            all_fixed = kwargs
            actual_task = mapped_task

        unbound_param = _get_unbound_param(actual_task, all_fixed, method="map")
        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs={unbound_param: self},
            backend=self.backend,
        )

    @overload
    def join(self, reducer_task: Task[Any, S]) -> "TaskFuture[S]": ...

    @overload
    def join(
        self, reducer_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> "TaskFuture[S]": ...

    def join(
        self, reducer_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> TaskFuture[S]:
        """
        Reduce this sequence to a single value.

        Args:
            reducer_task: Task with exactly ONE parameter (the sequence), or a
                FixedParamTask with one unbound parameter.
            **kwargs: Additional parameters to pass to the reducer task. The sequence
                will be bound to the single remaining unbound parameter.

        Returns:
            A TaskFuture representing the reduced single value.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound
                parameter, or if kwargs overlap with fixed parameters.

        Examples:
            Simple reducer:
            >>> @task
            >>> def sum_all(xs: list[int]) -> int:
            >>>     return sum(xs)
            >>>
            >>> numbers = square.product(x=[1, 2, 3, 4])
            >>> total = numbers.join(sum_all)
            >>> # Result: 30 (1 + 4 + 9 + 16)

            Reducer with inline kwargs:
            >>> @task
            >>> def weighted_sum(xs: list[int], weight: float) -> float:
            >>>     return sum(xs) * weight
            >>>
            >>> total = numbers.join(weighted_sum, weight=0.5)
            >>> # Result: 15.0

            Complete map-reduce pipeline:
            >>> result = (
            >>>     identity.product(x=[1, 2, 3])
            >>>     .map(scale, factor=2)
            >>>     .map(add, offset=10)
            >>>     .join(sum_with_bonus, bonus=100)
            >>> )

            With pre-fixed tasks (still supported):
            >>> total = numbers.join(weighted_sum.fix(weight=0.5))
        """
        if isinstance(reducer_task, FixedParamTask):
            _check_overlap_params(reducer_task, kwargs, method="join")
            all_fixed = {**reducer_task.fixed_kwargs, **kwargs}
            actual_task = reducer_task.task
        else:
            all_fixed = kwargs
            actual_task = reducer_task

        # Add ubound param to merged kwargs
        unbound_param = _get_unbound_param(actual_task, all_fixed, method="join")
        merged_kwargs = dict(all_fixed)
        merged_kwargs[unbound_param] = self

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
                deps.append(value)
        for seq in self.mapped_kwargs.values():
            if isinstance(seq, BaseTaskFuture):
                deps.append(seq)
        return deps

    @override
    def to_graph(self) -> MapTaskNode:
        from daglite.graph.nodes import MapTaskNode

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


# region Helpers


def _check_invalid_params(task: BaseTask, args: dict, method: str) -> None:
    if invalid_params := sorted(args.keys() - task.signature.parameters.keys()):
        raise ParameterError(
            f"Invalid parameters for task '{task.name}' in `{method}()`: {invalid_params}"
        )


def _check_missing_params(task: BaseTask, args: dict, method: str) -> None:
    if missing_params := sorted(task.signature.parameters.keys() - args.keys()):
        raise ParameterError(
            f"Missing parameters for task '{task.name}' in `{method}()`: {missing_params}"
        )


def _check_overlap_params(task: FixedParamTask, new: dict, method: str) -> None:
    fixed = task.fixed_kwargs.keys()
    if overlap_params := sorted(fixed & new.keys()):
        raise ParameterError(
            f"Overlapping parameters for task '{task.name}' in `{method}()`, specified parameters "
            f"were previously bound in `.fix()`: {overlap_params}"
        )


def _check_invalid_map_params(task: BaseTask, args: dict, method: str) -> None:
    non_sequences = []
    parameters = task.signature.parameters.keys()
    for key, value in args.items():
        if key in parameters and not isinstance(value, (Iterable, BaseTaskFuture)):
            non_sequences.append(key)
    if non_sequences:
        raise ParameterError(
            f"Non-iterable parameters for task '{task.name}' in `{method}()`, "
            f"all parameters must be Iterable or TaskFuture[Iterable] "
            f"(use `.fix()` to set scalar parameters): {non_sequences}"
        )


def _get_unbound_param(task: BaseTask, args: dict, method: str) -> str:
    unbound = [p for p in task.signature.parameters if p not in args]
    if len(unbound) == 0:
        raise ParameterError(
            f"Task '{task.name}' in `{method}()` has no unbound parameters for "
            f"upstream value. All parameters already provided: {list(args.keys())}"
        )
    if len(unbound) > 1:
        raise ParameterError(
            f"Task '{task.name}' in `{method}()` must have exactly one "
            f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
            f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
        )
    return unbound[0]
