from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from typing_extensions import override

from daglite.backends import Backend
from daglite.graph.base import GraphBuilder
from daglite.graph.base import ParamInput
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode

# NOTE: Import types only for type checking to avoid circular imports, if you need
# to use them at runtime, import them within methods.
if TYPE_CHECKING:
    from daglite.tasks import FixedParamTask
    from daglite.tasks import Task
else:
    FixedParamTask = object
    Task = object

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")


@dataclass(frozen=True)
class BaseTaskFuture(abc.ABC, GraphBuilder, Generic[R]):
    """Base class for all task futures, representing unevaluated task invocations."""

    id: UUID = field(init=False, default_factory=uuid4)

    @abc.abstractmethod
    def then(
        self,
        next_task: Task[Any, S] | FixedParamTask[Any, S],
        **kwargs: Any,
    ) -> "TaskFuture[S]":
        """
        Chain this future as input to another task during evaluation.

        Args:
            next_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter.
            **kwargs: Additional parameters to pass to the next task.

        Returns:
            A `TaskFuture` representing the result of applying the task to this future's value.

        Examples:
            >>> @task
            >>> def prepare(n: int) -> int:
            >>>     return n * 2
            >>>
            >>> @task
            >>> def add(x: int, y: int) -> int:
            >>>     return x + y
            >>>
            >>> # NOTE: 'x' is unbound and will receive the value from 'prepare' during evaluation.
            >>> future = prepare.bind(n=5).then(add, y=10)
        """
        raise NotImplementedError()

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
    """Represents a single task invocation that will produce a value of type R."""

    task: Task[Any, R]
    """Underlying task to be called."""

    kwargs: Mapping[str, Any]
    """Parameters to be passed to the task during execution, can contain other task futures."""

    backend: Backend
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    @override
    def then(
        self,
        next_task: Task[Any, S] | FixedParamTask[Any, S],
        **kwargs: Any,
    ) -> "TaskFuture[S]":
        from daglite.tasks import FixedParamTask
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, FixedParamTask):
            check_overlap_params(next_task, kwargs)
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            actual_task = next_task

        unbound_param = get_unbound_param(actual_task, all_fixed)
        return actual_task.bind(**{unbound_param: self}, **all_fixed)

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []
        for value in self.kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)
        return deps

    @override
    def to_graph(self) -> TaskNode:
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

    Fan-out means applying a task multiple times over a set of input sequences.

    The following modes are supported:
    - Cartesian product: every combination of input parameters is used to invoke the task
    - Pairwise (zip): elements from each input sequence are paired by their index to invoke
        the task
    """

    task: Task[Any, R]
    """Underlying task to be called."""

    mode: str  # "product" or "zip"
    """Mode of operation ('product' for Cartesian product, 'zip' for pairwise)."""

    fixed_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to fixed values applied to every call.

    Note that fixed parameters can be a combination of concrete values and `TaskFuture`s.
    """

    mapped_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to sequences to be iterated over during calls.

    Note that sequence parameters can be a combination of concrete values and `TaskFuture`s.
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
            mapped_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter.
            **kwargs: Additional parameters to pass to the task. The sequence element
                will be bound to the single remaining unbound parameter.

        Returns:
            A new MapTaskFuture with the mapped task applied to each element.

        Examples:
            >>> # Single-parameter task
            >>> @task
            >>> def double(x: int) -> int:
            >>>     return x * 2
            >>>
            >>> numbers = identity.product(x=[1, 2, 3])
            >>> doubled = numbers.map(double)  # [2, 4, 6]
            >>>
            >>> # Multi-parameter task with inline kwargs
            >>> @task
            >>> def scale(x: int, factor: int) -> int:
            >>>     return x * factor
            >>>
            >>> scaled = numbers.map(scale, factor=10)  # [10, 20, 30]
            >>>
            >>> # Chaining map operations
            >>> result = (
            >>>     numbers
            >>>     .map(scale, factor=2)
            >>>     .map(add, offset=10)
            >>>     .map(square)
            >>> )
            >>>
            >>> # With pre-fixed tasks (still supported):
            >>> scaled = numbers.map(scale.fix(factor=10))  # [10, 20, 30]
        """
        from daglite.tasks import FixedParamTask
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(mapped_task, FixedParamTask):
            check_overlap_params(mapped_task, kwargs)
            all_fixed = {**mapped_task.fixed_kwargs, **kwargs}
            actual_task = mapped_task.task
        else:
            all_fixed = kwargs
            actual_task = mapped_task

        unbound_param = get_unbound_param(actual_task, all_fixed)
        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs={unbound_param: self},
            backend=self.backend,
        )

    @override
    def then(
        self, next_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> TaskFuture[S]:
        from daglite.tasks import FixedParamTask
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, FixedParamTask):
            check_overlap_params(next_task, kwargs)
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            actual_task = next_task

        # Add unbound param to merged kwargs
        unbound_param = get_unbound_param(actual_task, all_fixed)
        merged_kwargs = dict(all_fixed)
        merged_kwargs[unbound_param] = self

        return TaskFuture(
            task=actual_task,
            kwargs=merged_kwargs,
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
        Reduce this sequence to a single value (alias for `then()`).

        Args:
            reducer_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter.
            **kwargs: Additional parameters to pass to the reducer task.

        Returns:
            A TaskFuture representing the reduced single value.
        """
        return self.then(reducer_task, **kwargs)

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
