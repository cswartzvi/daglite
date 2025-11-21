from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload, override
from uuid import UUID
from uuid import uuid4

from daglite.exceptions import ParameterError
from daglite.graph.base import GraphBuildContext
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphBuildVisiter
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
def task(fn: Callable[P, R]) -> Task[P, R]: ...


@overload
def task(*, name: str | None = None) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def task(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | Backend | None = None,
) -> Task[P, R] | Callable[[Callable[P, R]], Task[P, R]]:
    """
    Decorator to convert a Python function into a daglite `Task`.

    Tasks are the building blocks of daglite DAGs. They wrap plain Python functions and provide
    methods for composition (`.bind`, `.extend`, `.zip`) and execution. This is the recommended way
    to create tasks. Direct instantiation of the `Task` class is discouraged.

    Args:
        fn (Callable[P, R], optional): The function to wrap. When used without parentheses
            (`@task`), this is automatically passed. When used with parentheses (`@task()`),
            this is None.
        name (str, optional): Custom name for the task. Defaults to the function's `__name__`.
            For lambda functions, defaults to "unnamed_task".
        description (str, optional): Task description. Defaults to the function's docstring.
        backend (str | Backend | None): Backend for executing this task. Can be a backend
            name ("local", "threading") or a Backend instance. If None, uses the engine's
            default backend.

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

    # Used as @task() or @task(name="...")
    def decorator(f: Callable[P, R]) -> Task[P, R]:
        from daglite.backends import find_backend

        return Task(
            f,
            name=name if name is not None else getattr(f, "__name__", "unnamed_task"),
            description=description if description is not None else getattr(f, "__doc__", ""),
            backend=find_backend(backend) if isinstance(backend, str) else backend,
        )

    if fn is not None:
        # Used as @task (without parentheses)
        return decorator(fn)

    return decorator


# region Tasks


@dataclass(frozen=True)
class Task(Generic[P, R]):
    """
    Wraps a Python function as a composable task in a DAG.

    Tasks provide methods for binding parameters (`.bind`), creating fan-out operations
    (`.extend`, `.zip`), and partial application (`.partial`). Tasks maintain type information
    through generic parameters `P` (input signature) and `R` (return type).

    Note that users should create tasks using the `@task` decorator, not by direct instantiation.

    Attributes:
        fn: The wrapped Python function
        name: Human-readable name for the task
        description: Detailed description (usually from docstring)
        backend: Optional backend override for this task's execution
    """

    fn: Callable[P, R]
    """Task function to be wrapped into a Task."""

    name: str
    """Name of the task."""

    description: str
    """Description of the task."""

    backend: Backend | None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    # NOTE: We should no define `__call__` in order to avoid confusing type checkers. We want
    # them to view this object as a `Task[P, R]` and not as a `Callable[P, R]` (which some type
    # checkers would do if we defined `__call__`).

    def bind(self, *, backend: str | Backend | None = None, **kwargs: Any) -> TaskFuture[R]:
        """
        Creates a `TaskFuture` future by binding parameters to this task.

        This is the primary way to instantiate a task with its inputs. Parameters can be
        concrete values or other TaskFutures, enabling composition of complex DAGs.

        Args:
            backend (str | daglite.backends.Backend, optinal):
                Backend override for executing this specific task instance. If None, uses the
                task's default backend or the engine's default.
            **kwargs (Any):
                Keyword arguments matching the task function's parameters. Values can be concrete
                Python objects or TaskFutures from other tasks.

        Raises:
            TypeError: If required parameters are missing or types don't match.

        Returns:
            A `TaskFuture` representing the unevaluated task invocation.

        Examples:
            Binding concrete values:
                >>> @task
                >>> def add(x: int, y: int) -> int:
                >>>     return x + y
                >>> future = add.bind(x=1, y=2)

            Composing tasks:
                >>> @task
                >>> def multiply(a: int, b: int) -> int:
                >>>     return a * b
                >>> sum_future = add.bind(x=1, y=2)
                >>> prod_future = multiply.bind(a=sum_future, b=3)
        """

        backend = _select_backend(backend, self.backend)
        return TaskFuture(task=self, kwargs=dict(kwargs), backend=backend)

    def partial(self, **kwargs: Any) -> "PartialTask[R]":
        """
        Create a reusable `PartialTask` with some parameters pre-bound.

        Args:
            **kwargs (Any):
                Keyword arguments to be pre-bound to this task. Can be a combination of concrete
                values and TaskFutures.

        Examples:
        >>> base = score.partial(y=seed)
        >>> branch1 = base.bind(x=lazy_x)  # TaskFuture[int]
        >>> branch2 = base.extend(x=[1, 2, 3, 4])  # MapTaskFuture[int]
        """
        return PartialTask(task=self, fixed_kwargs=dict(kwargs), backend=self.backend)

    def extend(self, *, backend: str | Backend | None = None, **kwargs: Any) -> MapTaskFuture[R]:
        """
        Create a fan-out operation by applying this task over all combinations of sequences.

        This creates a Cartesian product of all provided sequences, calling the task once
        for each combination. Useful for parameter sweeps and batch operations.

        Args:
            backend (str | daglit.backends.Backend, optional):
                Backend override for executing the map operation.
            **kwargs (Iterable[Any] | TaskFuture[Iterable[Any]]):
                Keyword arguments where values are sequences. Each sequence element will be
                combined with elements from other sequences in a Cartesian product. Can include
                TaskFutures that resolve to sequences.

        Raises:
            ParameterError: If no sequences provided or any sequence is empty.

        Returns:
            A `MapTaskFuture` representing the collection of results.

        Examples:
            Single sequence
            >>> @task
            >>> def square(x: int) -> int:
            >>>     return x * x
            >>> results = square.extend(x=[1, 2, 3, 4])
            >>> # Produces [1, 4, 9, 16]

            Multiple sequences (Cartesian product):
            >>> @task
            >>> def add(x: int, y: int) -> int:
            >>>     return x + y
            >>> results = add.extend(x=[1, 2], y=[10, 20])
            >>> # Produces [11, 21, 12, 22] from (1,10), (1,20), (2,10), (2,20)
        """

        if not kwargs:
            raise ParameterError(
                "extend() requires at least one sequence parameter. "
                "Use .bind() for scalar parameters."
            )
        return self.partial().extend(backend=backend, **kwargs)

    def zip(self, *, backend: str | Backend | None = None, **kwargs: Any) -> MapTaskFuture[R]:
        """
        Create a fan-out operation by applying this task to zipped sequences.

        Sequences are zipped element-wise (similar to Python's `zip(`) function), calling
        the task once for each aligned set of elements. All sequences must have the same length.

        Args:
            backend (str | daglite.backends.Backend, optional):
                Backend override for executing the map operation.
            **kwargs (Iterable[Any] | TaskFuture[Iterable[Any]]):
                Keyword arguments where values are equal-length sequences. Elements at the same
                index across sequences are combined in each call. Can include TaskFutures that
                resolve to sequences.

        Raises:
            ParameterError: If no sequences provided, sequences are empty, or sequences
                have different lengths.

        Returns:
            A `MapTaskFuture` representing the collection of results.

        Examples:
            Pairwise operations
            >>> @task
            >>> def add(x: int, y: int) -> int:
            >>>     return x + y
            >>> results = add.zip(x=[1, 2, 3], y=[10, 20, 30])
            >>> # Produces [11, 22, 33] from (1,10), (2,20), (3,30)

            Three sequences
            >>> @task
            >>> def formula(a: int, b: int, c: int) -> int:
            >>>     return a * b + c
            >>> results = formula.zip(a=[1, 2], b=[3, 4], c=[5, 6])
            >>> # Produces [8, 14] from (1,3,5), (2,4,6)
        """

        if not kwargs:
            raise ParameterError(
                "zip() requires at least one sequence parameter. Use .bind() for scalar parameters."
            )
        return self.partial().zip(backend=backend, **kwargs)


@dataclass(frozen=True)
class PartialTask(Generic[R]):
    """
    A task with some parameters pre-bound.

    NOTE: Users should **not** instantiate this class directly; instead, use the `.partial(...)`
    method on a `Task`.

    This class can be used to create:
      - scalar branches with `.bind(...)`
      - fan-out branches with `.extend(...)` or `.zip(...)`
    """

    task: Task[Any, R]
    """The underlying task to be called."""

    fixed_kwargs: Mapping[str, Any]
    """The parameters already bound in this PartialTask; can contain other TaskFutures."""

    backend: Backend | None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    def bind(self, *, backend: str | Backend | None = None, **kwargs: Any) -> TaskFuture[R]:
        """
        Complete parameter binding for this PartialTask, producing a `TaskFuture`.

        Combines the pre-bound parameters from the `PartialTask` with the newly provided
        parameters to create a fully-specified task invocation.

        Args:
            backend (str | Backend | None): Backend override for this specific invocation.
            **kwargs (Any): Additional keyword arguments to bind. Must not overlap with
                parameters already bound in the PartialTask.

        Returns:
            A TaskFuture with all parameters bound.

        Raises:
            ParameterError: If any parameter name is already bound in this PartialTask.
        """

        merged: dict[str, Any] = dict(self.fixed_kwargs)
        backend = _select_backend(backend, self.backend)

        for name, val in kwargs.items():
            if name in merged:
                raise ParameterError(
                    f"Parameter '{name}' is already bound in this PartialTask. "
                    f"Previously bound parameters: {list(self.fixed_kwargs.keys())}"
                )
            merged[name] = val
        return TaskFuture(self.task, merged, self.backend)

    def extend(
        self,
        backend: str | Backend | None = None,
        **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]],
    ) -> MapTaskFuture[R]:
        """
        Expands a given task so that it is evaluated multiple times in a Cartesian product fashion.

        Args:
            backend (str | daglite.backends.Backend, optional):
                Backend override for this partial task. If None, uses either the task's backend or
                the default engine backend.
            **kwargs (Iterable[Any] | TaskFuture[Iterable[Any]]):
                Keyword arguments mapping parameter names to sequences; can contain a combination
                of concrete values and TaskFutures. All combinations of the sequences will be
                called in a Cartesian product fashion. Passing scalar values (concrete or Lazy) is
                not allowed. Empty sequences are not allowed.

        Examples:
        >>> # The following creates 4 calls with y=seed, x=1,2,3,4
        >>> single = base.partial(y=seed).extend(x=[1, 2, 3, 4])
        >>>
        >>> # 4 calls with (x,z)=(1,10),(1,20),(2,10),(2,20)
        >>> multi = base.extend(x=[1, 2], z=[10, 20])

        Raises:
            ValueError: If no kwargs provided, if any sequence is empty, or if parameters
                are already bound.
        """

        if not kwargs:
            raise ParameterError(
                "extend() requires at least one sequence parameter. "
                "Use .bind() for scalar parameters."
            )

        invalid_params = self.fixed_kwargs.keys() & kwargs.keys()
        if invalid_params:
            raise ParameterError(
                f"Cannot use extend() with already-bound parameters: {sorted(invalid_params)}. "
                f"These parameters were bound in .partial(): {list(self.fixed_kwargs.keys())}"
            )

        backend = _select_backend(backend, self.task.backend)

        return MapTaskFuture(
            task=self.task,
            mode="extend",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=backend,
        )

    def zip(
        self,
        backend: str | Backend | None = None,
        **kwargs: Iterable[Any] | TaskFuture[Iterable[Any]],
    ) -> MapTaskFuture[R]:
        """
        Expands a given task so that it is evaluated multiple times in a pairwise zip fashion.

        Args:
            backend (str | daglite.backends.Backend | None):
                Backend override for this partial task. If None, uses either the task's backend or
                the default engine backend.
            **kwargs (Iterable[Any] | TaskFuture[Iterable[Any]]):
                Keyword arguments mapping parameter names to sequences; can contain a combination
                of concrete values and TaskFutures. All sequences will be called in a pairwise zip
                fashion. All sequences must have the same length. Passing scalar values (concrete
                or Lazy) is not allowed. Empty sequences are not allowed.

        Examples:
        >>> # The following creates 3 calls with y=seed, x=1,2,3
        >>> single = base.partial(y=seed).extend(x=[1, 2, 3])
        >>>
        >>> # The following creates 3 calls with (x,y)=(1,10), (2,20), (3,30)
        >>> pairs = base.zip(x=[1, 2, 3], y=[10, 20, 30])

        Raises:
            ValueError: If no kwargs provided, if any sequence is empty, if sequences have
                different lengths, or if parameters are already bound.
        """

        if not kwargs:
            raise ParameterError(
                "zip() requires at least one sequence parameter. Use .bind() for scalar parameters."
            )

        invalid_params = self.fixed_kwargs.keys() & kwargs.keys()
        if invalid_params:
            raise ParameterError(
                f"Cannot use zip() with already-bound parameters: {sorted(invalid_params)}. "
                f"These parameters were bound in .partial(): {list(self.fixed_kwargs.keys())}"
            )

        backend = _select_backend(backend, self.task.backend)

        return MapTaskFuture(
            task=self.task,
            mode="zip",
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=dict(kwargs),
            backend=backend,
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

    backend: Backend | None = None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    @override
    def to_graph(self, ctx: GraphBuildContext, visit: GraphBuildVisiter) -> TaskNode:
        from daglite.graph.nodes import TaskNode

        params: dict[str, ParamInput] = {}
        for name, value in self.kwargs.items():
            if isinstance(value, BaseTaskFuture):
                ref_id = visit(value)  # type: ignore[arg-type]
                params[name] = ParamInput.from_ref(ref_id)
            else:
                params[name] = ParamInput.from_value(value)
        return TaskNode(id=self.id, task=self.task, params=params, backend=self.backend)


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

    Note that fix parameters can be a combination of concrete values and TaskFutures.
    """

    mapped_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to sequences to be iterated over during calls.

    Note that sequence parameters can be a combination of concrete values and TaskFutures.
    """

    backend: Backend | None = None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    def map(
        self,
        mapped_task: Task[Any, S],
        *,
        param: str | None = None,
        backend: str | Backend | None = None,
        **fixed: Any,
    ) -> "MapTaskFuture[S]":
        """
        Apply a task to each element of this sequence, producing a new map task.

        Args:
            mapped_task (daglite.tasks.Task):
                Task to be applied to each element of this sequence. By default, the first
                parameter of `fn` is treated as the sequence parameter. You can override this with
                `param="name"`.
            param (str, optional):
                Name of the parameter in `fn` that should receive each element of this sequence.
                If `None`, the first parameter of `fn` is used.
            backend (str | daglite.backends.Backend, optional):
                Backend override for the mapped task. If `None`, uses the backend of this task or
                the default engine backend.
            **fixed (Any):
                Extra keyword arguments to be passed to `fn` as fixed scalar or lazy values. Can
                be a combination of concrete values and TaskFutures.

        Example:
        >>> @task
        >>> def scale_and_shift(x: int, factor: int, offset: int) -> int: ...
        >>>
        >>> xs: LazySequence[int] = ...
        >>> ys = xs.map(scale_and_shift, factor=2, offset=10)
        >>> # -> scale_and_shift(x=item, factor=2, offset=10) for each item
        """

        param_name = _get_param_name(mapped_task, param, fixed)
        mapped_kwargs: dict[str, Any] = {param_name: self}
        backend = _select_backend(backend, self.task.backend)

        return MapTaskFuture(
            task=mapped_task,
            mode="extend",
            fixed_kwargs=dict(fixed),
            mapped_kwargs=mapped_kwargs,
            backend=backend,
        )

    def join(
        self,
        reducer_task: Task[Any, S],
        *,
        param: str | None = None,
        backend: str | Backend | None = None,
        **fixed: Any,
    ) -> "TaskFuture[S]":
        """
        Collapse this sequence into a single value using a reducer task.

        Args:
            reducer_task (daglite.tasks.Task):
                Task to be applied to reduce this sequence to a single value. By default, the first
                parameter of `reducer` is treated as the sequence parameter. You can override this
                with `param="name"`.
            param (str, optional):
                Name of the parameter in `reducer` that should receive this sequence. If `None`,
                the first parameter of `reducer` is used.
            backend (str | daglite.backends.Backend, optional):
                Backend override for the mapped task. If `None`, uses the backend of this task or
                the default engine backend.
            **fixed (Any):
                Extra keyword arguments to be passed to `reducer` as fixed scalar or lazy values.
                Can be a combination of concrete values and TaskFutures.

        Example:
        >>> @task
        >>> def sum_list(xs: list[int]) -> int: ...
        >>> @task
        >>> def weighted_sum(xs: list[int], weight: float) -> float: ...
        >>>
        >>> total = xs.join(sum_list)  # xs -> first param
        >>> weighted_total = xs.join(weighted_sum, weight=0.5)  # xs -> first param, weight scalar
        """
        param_name = _get_param_name(reducer_task, param, fixed)
        kwargs: dict[str, Any] = dict(fixed)
        kwargs[param_name] = self

        backend = _select_backend(backend, self.task.backend)

        return TaskFuture(reducer_task, kwargs=kwargs, backend=backend)

    @override
    def to_graph(self, ctx: GraphBuildContext, visit: GraphBuildVisiter) -> MapTaskNode:
        from daglite.graph.nodes import MapTaskNode

        fixed_kwargs: dict[str, ParamInput] = {}
        mapped_kwargs: dict[str, ParamInput] = {}

        for name, value in self.fixed_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                ref_id = visit(value)  # type: ignore[arg-type]
                fixed_kwargs[name] = ParamInput.from_ref(ref_id)
            else:
                fixed_kwargs[name] = ParamInput.from_value(value)

        for name, seq in self.mapped_kwargs.items():
            if isinstance(seq, BaseTaskFuture):
                ref_id = visit(seq)  # type: ignore[arg-type]
                mapped_kwargs[name] = ParamInput.from_sequence_ref(ref_id)
            else:
                mapped_kwargs[name] = ParamInput.from_sequence(seq)

        return MapTaskNode(
            id=self.id,
            task=self.task,
            mode=self.mode,
            fixed_kwargs=fixed_kwargs,
            mapped_kwargs=mapped_kwargs,
            backend=self.backend,
        )


def _get_param_name(task: Task[Any, Any], param: str | None, fixed: dict[str, Any]) -> str:
    """
    Determine which parameter should receive the sequence in map/join operations.

    Args:
        task (Task[Any, Any]): The task being mapped/joined
        param (str | None): User-specified parameter name, or None to use first parameter
        fixed (dict[str, Any]): Fixed parameters already bound

    Returns:
        Name of the parameter that will receive the sequence

    Raises:
        ParameterError: If parameter determination fails or conflicts with fixed parameters
    """
    if param is None:
        try:
            signature = inspect.signature(task.fn)
            first_param_name = next(iter(signature.parameters))
        except StopIteration:
            raise ParameterError(
                f"Cannot use task '{task.name}' in map/join: function has no parameters. "
                f"The reducer task must accept at least one parameter for the sequence."
            )
        except Exception as e:
            raise ParameterError(
                f"Cannot inspect signature of task '{task.name}': {e}. "
                f"Try specifying the parameter name explicitly with param='name'."
            ) from e
        param = first_param_name

    if param in fixed:
        raise ParameterError(
            f"Parameter '{param}' cannot be used for both the sequence and fixed values. "
            f"The sequence parameter must be distinct from fixed parameters: {list(fixed.keys())}"
        )

    return param


def _select_backend(backend: Backend | str | None, task_backend: Backend | None) -> Backend | None:
    """Helper to select the appropriate backend for a task execution."""
    from daglite.backends import find_backend

    backend = find_backend(backend) if isinstance(backend, str) else backend

    if backend is not None:
        return backend
    return task_backend
