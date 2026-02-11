"""Contains representations of standard task futures (task invocations)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from typing_extensions import override

from daglite._validation import check_invalid_map_params
from daglite._validation import check_invalid_params
from daglite._validation import check_overlap_params
from daglite._validation import get_unbound_param
from daglite.exceptions import DagliteError
from daglite.exceptions import ParameterError
from daglite.futures.base import BaseTaskFuture
from daglite.futures.graph_helpers import build_output_configs
from daglite.futures.graph_helpers import build_parameters
from daglite.futures.graph_helpers import collect_dependencies
from daglite.graph.base import GraphBuilder
from daglite.graph.nodes import TaskNode
from daglite.tasks import PartialTask
from daglite.tasks import Task
from daglite.tasks import task
from daglite.utils import build_repr
from daglite.utils import infer_tuple_size

# NOTE: To avoid circular imports, cross-referencing types should be imported within TYPE_CHECKING
# block. If runtime imports are needed, they should be done locally within methods. Be careful,
# forgetting to import at runtime will lead to hard to debug errors.
if TYPE_CHECKING:
    from daglite.futures.map_future import MapTaskFuture
else:
    MapTaskFuture = object

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

S1 = TypeVar("S1")
S2 = TypeVar("S2")
S3 = TypeVar("S3")
S4 = TypeVar("S4")
S5 = TypeVar("S5")
S6 = TypeVar("S6")


@dataclass(frozen=True)
class TaskFuture(BaseTaskFuture[R]):
    """Represents a single task invocation that will produce a value of type R."""

    task: Task[Any, R]
    """Underlying task to be called."""

    kwargs: Mapping[str, Any]
    """Parameters to be passed to the task during execution, can contain other task futures."""

    backend_name: str | None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    def __repr__(self) -> str:
        return build_repr("TaskFuture", self.task.name, kwargs=self.kwargs)

    def then(
        self,
        next_task: Task[Any, T] | PartialTask[Any, T],
        **kwargs: Any,
    ) -> "TaskFuture[T]":
        """
        Chain this future as input to another task during evaluation.

        Args:
            next_task: Either a `Task` that accepts exactly ONE parameter, or a `PartialTask`
                with ONE unbound parameter.
            **kwargs: Additional parameters to pass to the next task.

        Returns:
            A `TaskFuture` representing the result of applying the task to this future's value.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def prepare(n: int) -> int:
            ...     return n * 2
            >>> @task
            ... def add(x: int, y: int) -> int:
            ...     return x + y

            Chain task calls via unbound parameter (`x`)
            >>> prepare(n=5).then(add, y=10)  # doctest: +ELLIPSIS
            TaskFuture(...)
            >>> evaluate(prepare(n=5).then(add, y=10))
            20
        """
        from daglite.tasks import PartialTask

        if isinstance(next_task, PartialTask):
            check_overlap_params(dict(next_task.fixed_kwargs), kwargs, next_task.name)
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            actual_task = next_task

        unbound_param = get_unbound_param(actual_task.signature, all_fixed, actual_task.name)
        return actual_task(**{unbound_param: self}, **all_fixed)

    def then_product(
        self,
        next_task: Task[Any, T] | PartialTask[Any, T],
        **mapped_kwargs: Any,
    ) -> MapTaskFuture[T]:
        """
        Fan out this future as input to another task by creating a Cartesian product.

        The current future's result is used as a fixed (scalar) argument to `next_task`, while a
        Cartesian product is formed over the provided mapped parameter sequences in `mapped_kwargs`.
        The next task is called once for each combination of the mapped parameters, with the same
        future value passed to every call.

        Args:
            next_task: Either a `Task` that accepts exactly ONE parameter, or a `PartialTask`
                with ONE unbound parameter; this unbound parameter will receive the current
                future's value for every combination of mapped parameters.
            **mapped_kwargs: Additional parameters to map over (sequences). Each sequence element
                will be combined with elements from other sequences in a Cartesian product.

        Returns:
            A `MapTaskFuture` representing the result of applying the task to all combinations.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def prepare(n: int) -> int:
            ...     return n * 2
            >>> @task
            ... def combine(x: int, y: int) -> int:
            ...     return x + y

            Chain task calls via unbound parameter (`x`) and product over mapped arg (`y`)
            >>> future = prepare(n=5).then_product(combine, y=[10, 20, 30])
            >>> evaluate(future)
            [20, 30, 40]
        """
        from daglite.futures.map_future import MapTaskFuture

        if isinstance(next_task, PartialTask):
            check_overlap_params(dict(next_task.fixed_kwargs), mapped_kwargs, next_task.name)
            all_fixed = next_task.fixed_kwargs
            actual_task = next_task.task
        else:
            all_fixed = {}
            actual_task = next_task

        check_invalid_params(actual_task.signature, mapped_kwargs, actual_task.name)
        check_invalid_map_params(actual_task.signature, mapped_kwargs, actual_task.name)

        if not mapped_kwargs:
            raise ParameterError(
                f"At least one mapped parameter required for task '{actual_task.name}' "
                f"with .then_product(). Use .then() for 1-to-1 chaining instead."
            )

        merged = {**all_fixed, **mapped_kwargs}
        unbound_param = get_unbound_param(actual_task.signature, merged, actual_task.name)

        # Scalar broadcasting: self goes in fixed_kwargs, not mapped_kwargs
        all_fixed = {**all_fixed, unbound_param: self}

        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs=mapped_kwargs,
            backend_name=self.backend_name,
        )

    def then_zip(
        self,
        next_task: Task[Any, T] | PartialTask[Any, T],
        **mapped_kwargs: Any,
    ) -> MapTaskFuture[T]:
        """
        Fan out this future as input to another task by zipping with other sequences.

        The current future's result is used as a fixed (scalar) argument to `next_task`, while
        elements from the provided mapped parameter sequences in `mapped_kwargs` are paired by
        their index. The next task is called once for each index, with the
        same future value passed to every call.

        Args:
            next_task: Either a `Task` that accepts exactly ONE parameter, or a `PartialTask`
                with ONE unbound parameter; this unbound parameter will receive the current
                future's value for every paired set of mapped parameters.
            **mapped_kwargs: Additional equal-length sequences to zip with. Elements at the
                same index across sequences are combined in each call.

        Returns:
            A `MapTaskFuture` representing the result of applying the task to zipped elements.

        Examples:
            >>> from daglite import task, evaluate
            >>> @task
            ... def prepare(n: int) -> int:
            ...     return n * 2
            >>> @task
            ... def combine(x: int, y: int, z: int) -> int:
            ...     return x + y + z

            Chain task calls via unbound parameter (`x`) and zip over mapped arg (`y`)
            >>> future = prepare(n=5).then_zip(combine, y=[10, 20, 30], z=[1, 2, 3])
            >>> evaluate(future)
            [21, 32, 43]
        """
        from daglite.futures.map_future import MapTaskFuture

        if isinstance(next_task, PartialTask):
            check_overlap_params(dict(next_task.fixed_kwargs), mapped_kwargs, next_task.name)
            all_fixed = next_task.fixed_kwargs
            actual_task = next_task.task
        else:
            all_fixed = {}
            actual_task = next_task

        check_invalid_params(actual_task.signature, mapped_kwargs, actual_task.name)
        check_invalid_map_params(actual_task.signature, mapped_kwargs, actual_task.name)

        if not mapped_kwargs:
            raise ParameterError(
                f"At least one mapped parameter required for task '{actual_task.name}' "
                f"with .then_zip(). Use .then() for 1-to-1 chaining instead."
            )

        # Check that all concrete sequences have the same length
        len_details = {
            len(val) for val in mapped_kwargs.values() if not isinstance(val, BaseTaskFuture)
        }
        if len(len_details) > 1:
            raise ParameterError(
                f"Mixed lengths for task '{actual_task.name}', pairwise fan-out with "
                f"`.then_zip()` requires all sequences to have the same length. "
                f"Found lengths: {sorted(len_details)}"
            )

        merged = {**all_fixed, **mapped_kwargs}
        unbound_param = get_unbound_param(actual_task.signature, merged, actual_task.name)

        # Scalar broadcasting: self goes in fixed_kwargs, not mapped_kwargs
        all_fixed = {**all_fixed, unbound_param: self}

        return MapTaskFuture(
            task=actual_task,
            mode="zip",
            fixed_kwargs=all_fixed,
            mapped_kwargs=mapped_kwargs,
            backend_name=self.backend_name,
        )

    @overload
    def split(self: TaskFuture[tuple[S1]]) -> tuple[TaskFuture[S1]]: ...

    @overload
    def split(
        self: TaskFuture[tuple[S1, S2]],
    ) -> tuple[TaskFuture[S1], TaskFuture[S2]]: ...

    @overload
    def split(
        self: TaskFuture[tuple[S1, S2, S3]],
    ) -> tuple[TaskFuture[S1], TaskFuture[S2], TaskFuture[S3]]: ...

    @overload
    def split(
        self: TaskFuture[tuple[S1, S2, S3, S4]],
    ) -> tuple[
        TaskFuture[S1],
        TaskFuture[S2],
        TaskFuture[S3],
        TaskFuture[S4],
    ]: ...

    @overload
    def split(
        self: TaskFuture[tuple[S1, S2, S3, S4, S5]],
    ) -> tuple[
        TaskFuture[S1],
        TaskFuture[S2],
        TaskFuture[S3],
        TaskFuture[S4],
        TaskFuture[S5],
    ]: ...

    @overload
    def split(
        self: TaskFuture[tuple[S1, S2, S3, S4, S5, S6]],
    ) -> tuple[
        TaskFuture[S1],
        TaskFuture[S2],
        TaskFuture[S3],
        TaskFuture[S4],
        TaskFuture[S5],
        TaskFuture[S6],
    ]: ...

    @overload
    def split(self, *, size: int | None = None) -> tuple[TaskFuture[Any], ...]: ...

    def split(self, *, size: int | None = None) -> tuple[TaskFuture[Any], ...]:
        """
        Split this tuple-producing TaskFuture into individual TaskFutures for each element.

        Creates independent accessor tasks for each tuple element, enabling parallel processing of
        tuple components. Type information is preserved when the tuple has explicit type
        annotations.

        Args:
            size: Optional explicit size. Required if type annotations don't specify tuple size.

        Returns:
            A tuple of TaskFutures, one for each element of this tuple-producing future.

        Raises:
            DagliteError: If size cannot be inferred from type hints and size parameter is not
                provided.

        Examples:
            >>> from daglite import task, evaluate

            With type annotations (size inferred)
            >>> @task
            ... def make_pair() -> tuple[int, str]:
            ...     return (42, "hello")
            >>> make_pair().split()  # doctest: +ELLIPSIS
            (TaskFuture(...), TaskFuture(...))

            With explicit size
            >>> @task
            ... def make_triple():
            ...     return (1, 2, 3)
            >>> make_triple().split(size=3)  # doctest: +ELLIPSIS
            (TaskFuture(...), TaskFuture(...), TaskFuture(...))

            Evaluation of split futures
            >>> @task
            ... def get_coords() -> tuple[int, int]:
            ...     return (10, 20)
            >>> @task
            ... def process(x: int, y: int) -> str:
            ...     return f"Coordinates: ({x}, {y})"
            >>> x, y = get_coords().split()
            >>> future = process(x=x, y=y)
            >>> evaluate(future)
            'Coordinates: (10, 20)'

        """
        final_size = infer_tuple_size(self.task.func) if size is None else size
        if final_size is None:
            raise DagliteError(
                f"Cannot infer tuple size from type annotations for future {self.task.name}. "
                f"Please provide an explicit size parameter to split()."
            )

        # Create index accessor task for each position
        @task
        def _get_index(tup: tuple[Any, ...], index: int) -> Any:
            return tup[index]

        # Bind the accessor task for each index
        return tuple(_get_index(tup=self, index=i) for i in range(final_size))

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        return collect_dependencies(self.kwargs, outputs=self._future_outputs)

    @override
    def to_graph(self) -> TaskNode:
        kwargs = build_parameters(self.kwargs)
        placeholders = set(self.kwargs.keys())
        output_configs = build_output_configs(self._future_outputs, placeholders)
        return TaskNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            kwargs=kwargs,
            backend_name=self.backend_name,
            retries=self.task.retries,
            timeout=self.task.timeout,
            output_configs=output_configs,
            cache=self.task.cache,
            cache_ttl=self.task.cache_ttl,
        )
