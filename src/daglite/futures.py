"""Contains classes representing unevaluated task invocations (futures) in Daglite."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from typing_extensions import Self, override

from daglite.graph.base import GraphBuilder
from daglite.graph.base import ParamInput
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode

# NOTE: Import types only for type checking to avoid circular imports, if you need
# to use them at runtime, import them within methods.
if TYPE_CHECKING:
    from daglite.tasks import PartialTask
    from daglite.tasks import Task
else:
    PartialTask = object
    Task = object

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
class BaseTaskFuture(abc.ABC, GraphBuilder, Generic[R]):
    """Base class for all task futures, representing unevaluated task invocations."""

    _id: UUID = field(init=False, repr=False)
    _output_key: str | None = field(init=False, repr=False, default=None)
    _is_checkpoint: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Generate unique ID at creation time."""
        object.__setattr__(self, "_id", uuid4())

    @property
    @override
    def id(self) -> UUID:
        return self._id

    def save(self, key: str) -> Self:
        """
        Save task output for inspection (non-blocking side effect).

        The output will be stored with the given key after task execution completes.
        This does NOT mark the task as a resumption point - use .checkpoint() for that.

        Args:
            key: Storage key for this output. Can use {param} format strings which
                will be auto-resolved from task parameters.

        Returns:
            The same TaskFuture instance with updated metadata for output storage.

        Examples:
            >>> from daglite import task
            >>> @task
            ... def process(data_id: str) -> Result: ...
            >>> process(data_id="abc123").save("processed_{data_id}")  # doctest: +ELLIPSIS
            TaskFuture(...)
        """
        new_future = replace(self)
        object.__setattr__(new_future, "_id", self._id)
        object.__setattr__(new_future, "_output_key", key)
        object.__setattr__(new_future, "_is_checkpoint", False)
        return new_future

    def checkpoint(self, key: str) -> Self:
        """
        Save task output and mark as a resumption point.

        Similar to .save() but additionally marks this as a checkpoint that can be
        used with evaluate(from_=key) to resume execution from this point.

        Args:
            key: Storage key for this checkpoint. Can use {param} format strings which
                will be auto-resolved from task parameters.

        Returns:
            The same TaskFuture instance with updated metadata for output storage.

        Examples:
            >>> from daglite import task
            >>> @task
            ... def train_model(model_type: str) -> Model: ...
            >>> train_model(model_type="linear").checkpoint(
            ...     "model_{model_type}"
            ... )  # doctest: +ELLIPSIS
            TaskFuture(...)
        """
        new_future = replace(self)
        object.__setattr__(new_future, "_id", self._id)
        object.__setattr__(new_future, "_output_key", key)
        object.__setattr__(new_future, "_is_checkpoint", True)
        return new_future

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

    backend_name: str | None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

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
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, PartialTask):
            check_overlap_params(next_task, kwargs)
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            actual_task = next_task

        unbound_param = get_unbound_param(actual_task, all_fixed)
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
        from daglite.exceptions import ParameterError
        from daglite.tasks import PartialTask
        from daglite.tasks import check_invalid_map_params
        from daglite.tasks import check_invalid_params
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, PartialTask):
            check_overlap_params(next_task, mapped_kwargs)
            all_fixed = next_task.fixed_kwargs
            actual_task = next_task.task
        else:
            all_fixed = {}
            actual_task = next_task

        check_invalid_params(actual_task, mapped_kwargs)
        check_invalid_map_params(actual_task, mapped_kwargs)

        if not mapped_kwargs:
            raise ParameterError(
                f"At least one mapped parameter required for task '{actual_task.name}' "
                f"with .then_product(). Use .then() for 1-to-1 chaining instead."
            )

        merged = {**all_fixed, **mapped_kwargs}
        unbound_param = get_unbound_param(actual_task, merged)

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
        from daglite.exceptions import ParameterError
        from daglite.tasks import PartialTask
        from daglite.tasks import check_invalid_map_params
        from daglite.tasks import check_invalid_params
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, PartialTask):
            check_overlap_params(next_task, mapped_kwargs)
            all_fixed = next_task.fixed_kwargs
            actual_task = next_task.task
        else:
            all_fixed = {}
            actual_task = next_task

        check_invalid_params(actual_task, mapped_kwargs)
        check_invalid_map_params(actual_task, mapped_kwargs)

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
        unbound_param = get_unbound_param(actual_task, merged)

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
        from daglite.exceptions import DagliteError
        from daglite.tasks import task

        final_size = _infer_tuple_size(self.task.func) if size is None else size
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
            backend_name=self.backend_name,
            retries=self.task.retries,
            timeout=self.task.timeout,
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

    backend_name: str | None
    """Engine backend override for this task, if `None`, uses the default engine backend."""

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
            >>> numbers_future = generate_numbers.zip(n=[0, 1, 2, 3, 4])

            Chain with another mapped task
            >>> squared_future = numbers_future.then(square).join(sum_values)

            Evaluate the final result
            >>> evaluate(squared_future)
            30

            Using the fluent API
            >>> result = generate_numbers.zip(n=[0, 1, 2, 3, 4]).then(square).join(sum_values)
            >>> evaluate(result)
            30

        Returns:
            A `MapTaskFuture` representing the result of applying the mapped task to this
            future's sequence of values.
        """
        from daglite.tasks import PartialTask
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(mapped_task, PartialTask):
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
            backend_name=self.backend_name,
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
        from daglite.tasks import PartialTask
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(reducer_task, PartialTask):
            check_overlap_params(reducer_task, kwargs)
            all_fixed = {**reducer_task.fixed_kwargs, **kwargs}
            actual_task = reducer_task.task
        else:
            all_fixed = kwargs
            actual_task = reducer_task

        # Add unbound param to merged kwargs
        unbound_param = get_unbound_param(actual_task, all_fixed)
        merged_kwargs = dict(all_fixed)
        merged_kwargs[unbound_param] = self

        return TaskFuture(
            task=actual_task,
            kwargs=merged_kwargs,
            backend_name=self.backend_name,
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
        fixed_kwargs: dict[str, ParamInput] = {}
        mapped_kwargs: dict[str, ParamInput] = {}

        for name, value in self.fixed_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                fixed_kwargs[name] = ParamInput.from_ref(value.id)
            else:
                fixed_kwargs[name] = ParamInput.from_value(value)

        for name, seq in self.mapped_kwargs.items():
            if isinstance(seq, MapTaskFuture):
                # MapTaskFuture produces a sequence - iterate over it
                mapped_kwargs[name] = ParamInput.from_sequence_ref(seq.id)
            elif isinstance(seq, TaskFuture):
                # TaskFuture produces a sequence (e.g., list) - iterate over it
                mapped_kwargs[name] = ParamInput.from_sequence_ref(seq.id)
            else:
                # Concrete sequence - iterate over it
                mapped_kwargs[name] = ParamInput.from_sequence(seq)

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
        )


# region Helpers


def _infer_tuple_size(task_func: Any) -> int | None:
    """Try to infer tuple size from type annotations of a task function."""
    # Import here to avoid issues with circular imports
    from typing import get_args, get_type_hints

    try:
        hints = get_type_hints(task_func)
    except Exception:  # pragma: no cover
        return None

    return_type = hints.get("return")
    if return_type is None:
        return None
    args = get_args(return_type)
    if args and (len(args) < 2 or args[-1] is not Ellipsis):  # Skip tuple[int, ...]
        return len(args)
    return None
