"""Contains representations of mapped (fan-out) task futures (task invocations)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from typing_extensions import override

from daglite._validation import MapMode
from daglite._validation import check_overlap_params
from daglite._validation import get_unbound_param
from daglite.futures.base import BaseTaskFuture
from daglite.futures.graph_helpers import build_mapped_node_inputs
from daglite.futures.graph_helpers import build_output_configs
from daglite.futures.graph_helpers import build_node_inputs
from daglite.futures.graph_helpers import collect_dependencies
from daglite.graph.builder import GraphBuilder
from daglite.graph.nodes import MapTaskNode
from daglite.tasks import PartialTask
from daglite.tasks import Task
from daglite.utils import build_repr

# NOTE: To avoid circular imports, cross-referencing types should be imported within TYPE_CHECKING
# block. If runtime imports are needed, they should be done locally within methods. Be careful,
# forgetting to import at runtime will lead to hard to debug errors.
if TYPE_CHECKING:
    from daglite.futures.task_future import TaskFuture
else:
    TaskFuture = object

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


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

    mode: MapMode
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

    backend_name: str | None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    def __repr__(self) -> str:
        kwargs = {**self.fixed_kwargs, **self.mapped_kwargs}
        return build_repr("MapTaskFuture", self.task.name, f"mode={self.mode}", kwargs=kwargs)

    def then(
        self, mapped_task: Task[Any, T] | PartialTask[Any, T], **kwargs: Any
    ) -> MapTaskFuture[T]:
        """
        Chain this mapped future as input to another mapped task during evaluation.

        The mapped task is applied to each element of this future's sequence of values,
        continuing the chain.

        Args:
            mapped_task: Either a `Task` that accepts exactly ONE parameter, or a `PartialTask`
                with ONE unbound parameter.
            **kwargs: Additional fixed parameters to pass to the mapped task.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def generate_numbers(n: int) -> int:
            ...     return n
            >>> @task
            ... def square(x: int) -> int:
            ...     return x * x
            >>> @task
            ... def sum_values(values: list[int]) -> int:
            ...     return sum(values)

            Create a mapped future
            >>> numbers_future = generate_numbers.map(n=[0, 1, 2, 3, 4])

            Chain with another mapped task
            >>> squared_future = numbers_future.then(square).join(sum_values)

            Evaluate the final result
            >>> evaluate(squared_future)
            30

            Using the fluent API
            >>> result = generate_numbers.map(n=[0, 1, 2, 3, 4]).then(square).join(sum_values)
            >>> evaluate(result)
            30

        Returns:
            A `MapTaskFuture` representing the result of applying the mapped task to this
            future's sequence of values.
        """
        if isinstance(mapped_task, PartialTask):
            check_overlap_params(dict(mapped_task.fixed_kwargs), kwargs, mapped_task.name)
            all_fixed = {**mapped_task.fixed_kwargs, **kwargs}
            actual_task = mapped_task.task
        else:
            all_fixed = kwargs
            actual_task = mapped_task

        unbound_param = get_unbound_param(actual_task.signature, all_fixed, actual_task.name)
        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs={unbound_param: self},
            backend_name=self.backend_name,
            task_store=self.task_store,
        )

    @overload
    def join(self, reducer_task: Task[Any, T]) -> "TaskFuture[T]": ...

    @overload
    def join(
        self, reducer_task: Task[Any, T] | PartialTask[Any, T], **kwargs: Any
    ) -> "TaskFuture[T]": ...

    def join(
        self, reducer_task: Task[Any, T] | PartialTask[Any, T], **kwargs: Any
    ) -> TaskFuture[T]:
        """
        Reduce this sequence to a single value by applying a reducer task.

        Args:
            reducer_task: Either a `Task` that accepts exactly ONE parameter, or a `PartialTask`
                with ONE unbound parameter.
            **kwargs: Additional parameters to pass to the reducer task.

        Returns:
            A TaskFuture representing the reduced single value.
        """
        from daglite.futures.task_future import TaskFuture

        if isinstance(reducer_task, PartialTask):
            check_overlap_params(dict(reducer_task.fixed_kwargs), kwargs, reducer_task.name)
            all_fixed = {**reducer_task.fixed_kwargs, **kwargs}
            actual_task = reducer_task.task
        else:
            all_fixed = kwargs
            actual_task = reducer_task

        # Add unbound param to merged kwargs
        unbound_param = get_unbound_param(actual_task.signature, all_fixed, actual_task.name)
        merged_kwargs = dict(all_fixed)
        merged_kwargs[unbound_param] = self

        return TaskFuture(
            task=actual_task,
            kwargs=merged_kwargs,
            backend_name=self.backend_name,
            task_store=self.task_store,
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        kwargs = {**self.fixed_kwargs, **self.mapped_kwargs}
        return collect_dependencies(kwargs, self._output_futures)

    @override
    def to_graph(self) -> MapTaskNode:
        fixed_kwargs = build_node_inputs(self.fixed_kwargs)
        mapped_kwargs = build_mapped_node_inputs(self.mapped_kwargs)
        kwargs = {**fixed_kwargs, **mapped_kwargs}
        placeholders = set(kwargs.keys()) | {"iteration_index"}  # From map task nodes
        output_configs = build_output_configs(self._output_futures, placeholders)
        return MapTaskNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            mode=self.mode,
            fixed_kwargs=fixed_kwargs,
            mapped_kwargs=mapped_kwargs,
            backend_name=self.backend_name,
            retries=self.task.retries,
            timeout=self.task.timeout,
            output_configs=output_configs,
            cache=self.task.cache,
            cache_ttl=self.task.cache_ttl,
        )
