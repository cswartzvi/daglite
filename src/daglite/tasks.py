from __future__ import annotations

import abc
import inspect
import itertools
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, ParamSpec, Protocol, Sequence, TypeIs, TypeVar, overload, override
from uuid import UUID
from uuid import uuid4

from daglite.backends import find_backend
from daglite.engine import Backend
from daglite.engine import Engine

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)

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
    Decorator to turn a plain function into a daglite `Task`.

    Note: This is the recommended way to create tasks. Users are discouraged from instantiating
    the `Task` class directly.

    Args:
        fn (Callable[P, R]):
            Function to be wrapped into a Task.
        name (str, optional):
            Custom name for the task, defaults to the function's name. If the function has no name,
            for example if it is a lambda, "unnamed_task" is used.
        description (str, optional):
            Description for the task, defaults to the function's docstring.
        backend (str | daglite.backends.Backend, optional):
            Optional task executor backend override for this task. If None, the default backend
            configured in the evaluation engine is used.

    Examples:
    >>> @task
    >>> def my_func(): ...
    >>>
    >>> @task()
    >>> def my_func(): ...
    >>>
    >>> @task(name="custom_name")
    >>> def my_func(): ...
    """

    # Used as @task() or @task(name="...")
    def decorator(f: Callable[P, R]) -> Task[P, R]:
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
    A Task wraps a plain Python function with a concrete signature.

    Note: Users should **not** instantiate this class directly; instead, use the `@task` decorator.
    """

    fn: Callable[P, R]
    """Task function to be wrapped into a Task."""

    name: str
    """Name of the task."""

    description: str
    """Description of the task."""

    backend: Backend | None = None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    # NOTE: We should no define `__call__` in order to avoid confusing type checkers. We want
    # them to view this object as a `Task[P, R]` and not as a `Callable[P, R]` (which some type
    # checkers would do if we defined `__call__`).

    def bind(self, **kwargs: Any) -> TaskFuture[R]:
        """
        Build a Node[R] node representing this task with some parameters bound.

        Args:
            kwargs (Mapping[str, Any]):
                Keyword arguments to be passed to the task during graph execution. Can be a
                combination of concrete values and TaskFutures.

        Examples:
        >>> @task
        >>> def add(x: int, y: int) -> int: ...
        >>>
        >>> node = add.bind(x=1, y=2)  # TaskFuture[int]
        """

        return TaskFuture(task=self, kwargs=dict(kwargs), backend=self.backend)

    def partial(self, **kwargs: Any) -> "PartialTask[R]":
        """
        Create a reusable PartialTask with some parameters pre-bound.

        Args:
            kwargs (Mapping[str, Any]):
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
        Expands this task so that it is evaluated multiple times in a Cartesian product fashion.

        Args:
            backend (str | daglite.backends.Backend, optional):
                Backend override for this partial task. If `None`, uses either the task's backend
                or the default engine backend.
            kwargs (Mapping[str, Sequence | LazyValue[Sequence]]):
                Keyword arguments mapping parameter names to sequences; can contain a combination
                of concrete values and TaskFutures. All combinations of the sequences will be called
                in a Cartesian product fashion. Passing scalar values (concrete or Lazy) is not
                allowed and will raise a `ValueError`.
        """

        if not kwargs:
            raise ValueError("extend() requires at least one sequence argument.")
        return self.partial().extend(backend=backend, **kwargs)

    def zip(self, *, backend: str | Backend | None = None, **kwargs: Any) -> MapTaskFuture[R]:
        """
        Expands this task so that it is evaluated multiple times in a pairwise zip fashion.

        Args:
            backend (str | daglite.backends.Backend, optional):
                Backend override for this partial task. If `None`, uses either the task's backend
                or the default engine backend.
            kwargs (Mapping[str, Sequence | LazyValue[Sequence]]):
                Keyword arguments mapping parameter names to sequences; can contain a combination
                of concrete values and TaskFutures. All sequences will be called in a pairwise zip
                fashion. Passing scalar values (concrete or Lazy) is not allowed and will raise a
                `ValueError`.
        """

        if not kwargs:
            raise ValueError("zip() requires at least one sequence argument.")
        return self.partial().zip(backend=backend, **kwargs)


@dataclass(frozen=True)
class PartialTask(Generic[R]):
    """
    A task with some parameters pre-bound.

    NOTE: Users should **not** instantiate this class directly; instead, use the `.partial(...)`
    method on a `Task`.

    This class can be used to create:
      - scalar branches with .bind(...)
      - fan-out branches with .extend(...) or .zip(...)
    """

    task: Task[Any, R]
    """The underlying task to be called."""

    fixed_kwargs: Mapping[str, Any]
    """The parameters already bound in this PartialTask; can contain other TaskFutures."""

    backend: Backend | None = None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    def bind(self, **kwargs: Any) -> TaskFuture[R]:
        """
        Bind the remaining parameters to this `PartialTask`, producing a `TaskFuture`.

        Args:
            kwargs (Mapping[str, Any]):
                Keyword arguments to be passed to the task during execution. Can be a combination of
                concrete values and TaskFutures.
        """
        merged: dict[str, Any] = dict(self.fixed_kwargs)
        for name, val in kwargs.items():
            if name in merged:
                raise ValueError(f"Parameter {name!r} already bound in PartialTask.")
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
                Backend override for this partial task. If `None`, uses either the task's backend
                or the default engine backend.
            kwargs (Mapping[str, Sequence | LazyValue[Sequence]]):
                Keyword arguments mapping parameter names to sequences; can contain a combination
                of concrete values and TaskFutures. All combinations of the sequences will be called
                in a Cartesian product fashion. Passing scalar values (concrete or Lazy) is not
                allowed and will raise a `ValueError`.

        Examples:
        >>> # The following creates 4 calls with y=seed, x=1,2,3,4
        >>> single = base.partial(y=seed).extend(x=[1, 2, 3, 4])
        >>>
        >>> # 4 calls with (x,z)=(1,10),(1,20),(2,10),(2,20)
        >>> multi = base.extend(x=[1, 2], z=[10, 20])

        """
        if not kwargs:
            raise ValueError("extend() requires at least one sequence argument.")

        invalid_params = self.fixed_kwargs.keys() & kwargs.keys()
        if invalid_params:
            msg = f"Partially bound parameters cannot be used in `extend`: {invalid_params!r}"
            raise ValueError(msg)

        backend = find_backend(backend) if isinstance(backend, str) else backend
        backend = _select_backend(backend, self.task.backend)

        return MapTaskFuture(
            task=self.task,
            mode="extend",
            fixed_kwargs=self.fixed_kwargs,
            seq_kwargs=dict(kwargs),
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
            backend (str | daglite.backends.Backend, optional):
                Backend override for this partial task. If `None`, uses either the task's backend
                or the default engine backend.
            kwargs (Mapping[str, Sequence | LazyValue[Sequence]]):
                Keyword arguments mapping parameter names to sequences; can contain a combination
                of concrete values and TaskFutures. All sequences will be called in a pairwise zip
                fashion. Passing scalar values (concrete or Lazy) is not allowed and will raise a
                `ValueError`.

        Examples:
        >>> # The following creates 3 calls with y=seed, x=1,2,3
        >>> single = base.partial(y=seed).extend(x=[1, 2, 3])
        >>>
        >>> # The following creates 3 calls with (x,y)=(1,10), (2,20), (3,30)
        >>> pairs = base.zip(x=[1, 2, 3], y=[10, 20, 30])
        """
        if not kwargs:
            raise ValueError("zip() requires at least one sequence argument.")

        invalid_params = self.fixed_kwargs.keys() & kwargs.keys()
        if invalid_params:
            msg = f"Partially bound parameters cannot be used in `zip`: {invalid_params!r}"
            raise ValueError(msg)

        backend = find_backend(backend) if isinstance(backend, str) else backend
        backend = _select_backend(backend, self.task.backend)

        return MapTaskFuture(
            task=self.task,
            mode="zip",
            fixed_kwargs=self.fixed_kwargs,
            seq_kwargs=dict(kwargs),
            backend=backend,
        )


# region Task Futures


class EvaluatableTask(Protocol[T_co]):
    """Protocol for task futures that can be evaluated to a concrete value of type T."""

    @cached_property
    @abc.abstractmethod
    def id(self) -> UUID:
        """Unique identifier for this lazy value instance."""
        ...

    @abc.abstractmethod
    def _evaluate(self, engine: Engine) -> T_co:
        """
        Evaluate this task future within the given engine context.

        Note: This method is intended for internal use by the evaluation engine only. User code
        should should directly interact with either the the `evaluate` function or the `Engine`
        class.

        Args:
            engine (daglite.engine.Engine):
                Evaluation engine context to use for evaluating this task future.
        """
        ...

    @staticmethod
    def is_evaluatable(obj: Any) -> TypeIs[EvaluatableTask[Any]]:
        """Check if the given object conforms to the EvaluatableTask protocol."""
        return hasattr(obj, "id") and callable(getattr(obj, "_evaluate", None))


class BaseTaskFuture(abc.ABC):
    """Base class for all task futures, representing unevaluated task invocations."""

    @cached_property
    def id(self) -> UUID:
        """Unique identifier for this lazy value instance."""
        return uuid4()

    # NOTE: The following methods are to prevent accidental usage of unevaluated nodes.

    def __bool__(self) -> bool:
        raise TypeError("Task futures have no truthiness; evaluate it first.")

    def __len__(self) -> int:
        raise TypeError("Task futures have no length; evaluate it first.")

    def __repr__(self) -> str:  # pragma : no cover
        return f"<Lazy {id(self):#x}>"


@dataclass(frozen=True)
class TaskFuture(BaseTaskFuture, Generic[R], EvaluatableTask[R]):
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
    def _evaluate(self, engine: Engine) -> R:
        evaluated_params = _evaluate_parameters(engine, self.kwargs)
        return engine.backend.run_task(self.task.fn, evaluated_params)


@dataclass(frozen=True)
class MapTaskFuture(BaseTaskFuture, Generic[R], EvaluatableTask[Sequence[R]]):
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

    seq_kwargs: Mapping[str, Iterable[Any] | TaskFuture[Iterable[Any]]]
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
            param (str | None):
                Name of the parameter in `fn` that should receive each element of this sequence.
                If `None`, the first parameter of `fn` is used.
            backend (str | daglite.backends.Backend, optional):
                Backend override for the mapped task. If `None`, uses the backend of this task or
                the default engine backend.
            **fixed: Any:
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
        seq_kwargs: dict[str, Any] = {param_name: self}

        backend = find_backend(backend) if isinstance(backend, str) else backend
        backend = _select_backend(backend, self.task.backend)

        return MapTaskFuture(
            task=mapped_task,
            mode="extend",
            fixed_kwargs=dict(fixed),
            seq_kwargs=seq_kwargs,
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
            param (str | None):
                Name of the parameter in `reducer` that should receive this sequence. If `None`,
                the first parameter of `reducer` is used.
            backend (str | daglite.backends.Backend, optional):
                Backend override for the mapped task. If `None`, uses the backend of this task or
                the default engine backend.
            **fixed: Any:
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

        backend = find_backend(backend) if isinstance(backend, str) else backend
        backend = _select_backend(backend, self.task.backend)

        return TaskFuture(reducer_task, kwargs=kwargs, backend=backend)

    @override
    def _evaluate(self, engine: Engine) -> Sequence[R]:
        # Resolve parameters
        fixed_kwargs: dict[str, Any] = _evaluate_parameters(engine, self.fixed_kwargs)
        seq_values: dict[str, list[Any]] = _evaluate_parameters(engine, self.seq_kwargs)

        # Build per-call kwargs
        calls: list[dict[str, Any]] = []
        if self.mode == "extend":
            # Cartesian product over all sequences
            items = list(seq_values.items())
            names = [n for n, _ in items]
            lists = [vals for _, vals in items]

            for combo in itertools.product(*lists):
                kw = dict(fixed_kwargs)
                for param_name, val in zip(names, combo):
                    kw[param_name] = val
                calls.append(kw)

        elif self.mode == "zip":
            # Pairwise zip over all sequences
            lengths = {len(vals) for vals in seq_values.values()}
            if len(lengths) > 1:
                raise ValueError(f"All zip() sequences must have the same length; got {lengths}.")
            n = lengths.pop() if lengths else 0
            for i in range(n):
                kw = dict(fixed_kwargs)
                for param_name, vals in seq_values.items():
                    kw[param_name] = vals[i]
                calls.append(kw)

        else:
            raise ValueError(f"Unknown map mode: {self.mode!r}")

        # Delegate to backend for fan-out
        return engine.backend.run_many(self.task.fn, calls)


def _get_param_name(task: Task[Any, Any], param: str | None, fixed: dict[str, Any]) -> str:
    """Helper to find the name of the sequence parameter in a mapping task."""
    if param is None:
        try:
            signature = inspect.signature(task.fn)
            first_param_name = next(iter(signature.parameters))
        except StopIteration:
            raise ValueError("Cannot join with a reducer that has no parameters.")
        except Exception as e:
            raise ValueError(f"Cannot inspect reducer function signature: {e!r}") from e
        param = first_param_name

    if param in fixed:
        raise ValueError(f"Parameter {param!r} cannot be both sequence and fixed in join().")

    return param


def _select_backend(task_backend: Backend | None, map_backend: Backend | None) -> Backend | None:
    """Helper to select the appropriate backend for a task execution."""
    if map_backend is not None:
        return map_backend
    return task_backend


def _evaluate_parameters(engine: Engine, params: Mapping[str, Any]) -> Any:
    """Helper to evaluate the given parameters within the provided engine context."""
    evaluated_params = {
        key: engine.evaluate(val) if EvaluatableTask.is_evaluatable(val) else val
        for key, val in params.items()
    }
    return evaluated_params
