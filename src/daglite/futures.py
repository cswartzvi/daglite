from __future__ import annotations

import abc
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import UUID
from uuid import uuid4

from typing_extensions import override

from daglite.exceptions import ParameterError
from daglite.graph.base import GraphBuilder
from daglite.graph.base import ParamInput

# NOTE: Import types only for type checking to avoid circular imports, if you need
# to use them at runtime, import them within methods.
if TYPE_CHECKING:
    from daglite.engine import Backend
    from daglite.graph.nodes import MapTaskNode
    from daglite.graph.nodes import TaskNode
    from daglite.tasks import FixedParamTask
    from daglite.tasks import Task
else:
    Backend = object
    MapTaskNode = object
    TaskNode = object
    FixedParamTask = object
    Task = object

R = TypeVar("R")
S = TypeVar("S")


# region Task Futures


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

    def then(
        self,
        next_task: Task[Any, S] | FixedParamTask[Any, S],
        **kwargs: Any,
    ) -> "TaskFuture[S]":
        """
        Chain this task's output as input to another task.

        Additional kwargs are passed to the next task. The upstream output
        is automatically bound to the single remaining unbound parameter.
        """
        # NOTE: Import at runtime to avoid circular import issues
        from daglite.tasks import FixedParamTask

        if isinstance(next_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(next_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{next_task.name}' in `.then()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(next_task.task.func)
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(next_task.func)
            actual_task = next_task

        # Find the one unbound parameter
        unbound = [name for name in sig.parameters if name not in all_fixed]

        if len(unbound) == 0:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` has no unbound parameters for "
                f"upstream value. All parameters already provided: {list(all_fixed.keys())}"
            )
        if len(unbound) > 1:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` must have exactly one "
                f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
                f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
            )

        target_param = unbound[0]
        return actual_task.bind(**{target_param: self}, **all_fixed)

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

    def map(
        self, mapped_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> MapTaskFuture[S]:
        """
        Apply a task to each element of this sequence.

        Args:
            mapped_task (Task[Any, S] | FixedParamTask[Any, S]):
                `Task` with exactly ONE parameter, or a `FixedParamTask` where one parameter
                remains unbound.
            **kwargs: Additional parameters to pass to the task. The sequence element will be
                bound to the single remaining unbound parameter.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound parameter.

        Examples:
        Single-parameter task
        >>> @task
        >>> def double(x: int) -> int:
        >>>     return x * 2
        >>> results = numbers.map(double)

        Multi-parameter task with inline kwargs
        >>> @task
        >>> def scale(x: int, factor: int) -> int:
        >>>     return x * factor
        >>> results = numbers.map(scale, factor=10)

        Multi-parameter task with .fix() (still supported)
        >>> results = numbers.map(scale.fix(factor=10))
        """
        # NOTE: Import at runtime to avoid circular import issues
        from daglite.tasks import FixedParamTask

        if isinstance(mapped_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(mapped_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{mapped_task.name}' in `.map()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**mapped_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(mapped_task.task.func)
            actual_task = mapped_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(mapped_task.func)
            actual_task = mapped_task

        # Find the one unbound parameter
        unbound_params = [p for p in sig.parameters if p not in all_fixed]
        if len(unbound_params) != 1:
            raise ParameterError(
                f"Task '{mapped_task.name}' in `.map()` must have exactly one "
                f"unbound parameter, found {len(unbound_params)} "
                f"(use `.fix()` to set scalar parameters): {unbound_params}"
            )
        param_name = unbound_params[0]

        return MapTaskFuture(
            task=actual_task,
            mode="extend",
            fixed_kwargs=all_fixed,
            mapped_kwargs={param_name: self},
            backend=self.backend,
        )

    def join(
        self, reducer_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> TaskFuture[S]:
        """
        Reduce this sequence to a single value.

        Args:
            reducer_task:
                `Task` with exactly ONE parameter (the sequence), or a `FixedParamTask` with
                one unbound parameter.
            **kwargs: Additional parameters to pass to the reducer task. The sequence will be
                bound to the single remaining unbound parameter.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound parameter.

        Examples:
            # Simple reducer
            >>> @task
            >>> def sum_all(xs: list[int]) -> int:
            >>>    return sum(xs)
            >>> total = numbers.join(sum_all)

            # Reducer with inline kwargs
            >>> @task
            >>> def weighted_sum(xs: list[int], weight: float) -> float:
            >>>    return sum(xs) * weight
            >>> total = numbers.join(weighted_sum, weight=0.5)

            # Reducer with .fix() (still supported)
            >>> total = numbers.join(weighted_sum.fix(weight=0.5))
        """
        # NOTE: Import at runtime to avoid circular import issues
        from daglite.tasks import FixedParamTask

        if isinstance(reducer_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(reducer_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{reducer_task.name}' in `.join()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**reducer_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(reducer_task.task.func)
            actual_task = reducer_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(reducer_task.func)
            actual_task = reducer_task

        # Find the one unbound parameter
        unbound_params = [p for p in sig.parameters if p not in all_fixed]
        if len(unbound_params) != 1:
            raise ParameterError(
                f"Task '{reducer_task.name}' in `.join()` must have exactly one "
                f"unbound parameter, found {len(unbound_params)} "
                f"(use `.fix()` to set scalar parameters): {unbound_params}"
            )
        param_name = unbound_params[0]

        merged_kwargs = dict(all_fixed)
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
