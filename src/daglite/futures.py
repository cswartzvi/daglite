from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from typing_extensions import override

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

    def __post_init__(self) -> None:
        """Generate unique ID at creation time."""
        object.__setattr__(self, "_id", uuid4())

    @property
    @override
    def id(self) -> UUID:
        return self._id

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
        next_task: Task[Any, T] | FixedParamTask[Any, T],
        **kwargs: Any,
    ) -> "TaskFuture[T]":
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

    def then_product(
        self,
        next_task: Task[Any, T] | FixedParamTask[Any, T],
        **mapped_kwargs: Any,
    ) -> MapTaskFuture[T]:
        """
        Fan out this future as input to another task by creating a Cartesian product.

        This creates a Cartesian product of this future's result (treated as a sequence)
        with other provided sequences, calling the next task once for each combination.

        Args:
            next_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter, this parameter will receive the current future's value.
            **mapped_kwargs: Additional parameters to map over (sequences). Each sequence element
                will be combined with elements from other sequences in a Cartesian product.

        Returns:
            A `MapTaskFuture` representing the result of applying the task to all combinations.

        Examples:
            >>> @task
            >>> def prepare(n: int) -> int:
            >>>     return n * 2
            >>>
            >>> @task
            >>> def combine(x: int, y: int) -> int:
            >>>     return x + y
            >>>
            >>> # Create [2, 4, 6] then combine with [10, 20] in Cartesian product
            >>> future = prepare.product(n=[1, 2, 3]).then_product(combine, y=[10, 20])
            >>> evaluate(future)
            [12, 22, 14, 24, 16, 26]
        """
        from daglite.tasks import FixedParamTask
        from daglite.tasks import check_invalid_map_params
        from daglite.tasks import check_invalid_params
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, FixedParamTask):
            check_overlap_params(next_task, mapped_kwargs)
            all_fixed = next_task.fixed_kwargs
            actual_task = next_task.task
        else:
            all_fixed = {}
            actual_task = next_task

        # Validate that provided mapped_kwargs are valid parameters
        check_invalid_params(actual_task, mapped_kwargs)
        check_invalid_map_params(actual_task, mapped_kwargs)

        # Get the unbound parameter that will receive values from self
        merged = {**all_fixed, **mapped_kwargs}
        unbound_param = get_unbound_param(actual_task, merged)
        all_mapped = {**mapped_kwargs, unbound_param: self}

        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs=all_mapped,
            backend_name=self.backend_name,
        )

    def then_zip(
        self,
        next_task: Task[Any, T] | FixedParamTask[Any, T],
        **mapped_kwargs: Any,
    ) -> MapTaskFuture[T]:
        """
        Fan out this future as input to another task by zipping with other sequences.

        Sequences are zipped element-wise (similar to Python's `zip()` function), calling
        the next task once for each aligned set of elements. All sequences must have the
        same length.

        Args:
            next_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter, this parameter will receive the current future's value.
            **mapped_kwargs: Additional equal-length sequences to zip with. Elements at the
                same index across sequences are combined in each call.

        Returns:
            A `MapTaskFuture` representing the result of applying the task to zipped elements.

        Examples:
            >>> @task
            >>> def prepare(n: int) -> int:
            >>>     return n * 2
            >>>
            >>> @task
            >>> def combine(x: int, y: int) -> int:
            >>>     return x + y
            >>>
            >>> # Create [2, 4, 6] then zip with [10, 20, 30]
            >>> future = prepare.product(n=[1, 2, 3]).then_zip(combine, y=[10, 20, 30])
            >>> evaluate(future)
            [12, 24, 36]
        """
        from daglite.exceptions import ParameterError
        from daglite.tasks import FixedParamTask
        from daglite.tasks import check_invalid_map_params
        from daglite.tasks import check_invalid_params
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(next_task, FixedParamTask):
            check_overlap_params(next_task, mapped_kwargs)
            all_fixed = next_task.fixed_kwargs
            actual_task = next_task.task
        else:
            all_fixed = {}
            actual_task = next_task

        # Validate that provided mapped_kwargs are valid parameters
        check_invalid_params(actual_task, mapped_kwargs)
        check_invalid_map_params(actual_task, mapped_kwargs)

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

        # Get the unbound parameter that will receive values from self
        merged = {**all_fixed, **mapped_kwargs}
        unbound_param = get_unbound_param(actual_task, merged)
        all_mapped = {**mapped_kwargs, unbound_param: self}

        return MapTaskFuture(
            task=actual_task,
            mode="zip",
            fixed_kwargs=all_fixed,
            mapped_kwargs=all_mapped,
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

        Creates independent accessor tasks for each tuple element, enabling parallel
        processing of tuple components. Type information is preserved when the tuple
        has explicit type annotations.

        Args:
            size: Optional explicit size. Required if type annotations don't specify tuple size.

        Returns:
            A tuple of TaskFutures, one for each element of this tuple-producing future.

        Raises:
            DagliteError: If size cannot be inferred from type hints and size parameter is not
            provided.

        Example:
            With type annotations (size inferred):
            >>> @task
            >>> def make_pair() -> tuple[int, str]:
            ...     return (42, "hello")

            >>> num, text = make_pair.bind().split()

            With explicit size:
            >>> @task
            >>> def make_triple():
            ...     return (1, 2, 3)

            >>> a, b, c = make_triple.bind().split(size=3)

            Chaining after split:
            >>> @task
            >>> def get_coords() -> tuple[int, int]:
            ...     return (10, 20)

            >>> x, y = get_coords.bind().split()
            >>> result = process.bind(x=x, y=y)
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
        return tuple(_get_index.bind(tup=self, index=i) for i in range(final_size))

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
        self, mapped_task: Task[Any, T] | FixedParamTask[Any, T], **kwargs: Any
    ) -> MapTaskFuture[T]:
        """
        Chain this mapped future as input to another mapped task during evaluation.

        Note that the mapped task must is applied to each element of this future's sequence of
        values continuing

        Args:
            mapped_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter.
            **kwargs: Additional fixed parameters to pass to the mapped task.

        Examples:
            >>> @task
            >>> def generate_numbers(n: int) -> int:
            ...     return n
            >>>
            >>> @task
            >>> def square(x: int) -> int:
            ...     return x * x
            >>>
            >>> @task
            >>> def sum_values(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> # Create a mapped future generating numbers 0 to 4
            >>> numbers_future = generate_numbers.bind(n=[0, 1, 2, 3, 4])
            >>>
            >>> # Chain to square each generated number and then sum the squares
            >>> squared_future = numbers_future.then(square).join(sum_values)
            >>>
            >>> # Evaluate the final result
            >>> evaluate(squared_future)
            30  # 0^2 + 1^2 + 2^2 + 3^2 + 4^2

        Returns:
            A `MapTaskFuture` representing the result of applying the mapped task to this
            future's sequence of values.
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
            backend_name=self.backend_name,
        )

    @overload
    def join(self, reducer_task: Task[Any, T]) -> "TaskFuture[T]": ...

    @overload
    def join(
        self, reducer_task: Task[Any, T] | FixedParamTask[Any, T], **kwargs: Any
    ) -> "TaskFuture[T]": ...

    def join(
        self, reducer_task: Task[Any, T] | FixedParamTask[Any, T], **kwargs: Any
    ) -> TaskFuture[T]:
        """
        Reduce this sequence to a single value by applying a reducer task.

        Args:
            reducer_task: Either a `Task` that accepts exactly ONE parameter, or a `FixedParamTask`
                with ONE unbound parameter.
            **kwargs: Additional parameters to pass to the reducer task.

        Returns:
            A TaskFuture representing the reduced single value.
        """
        from daglite.tasks import FixedParamTask
        from daglite.tasks import check_overlap_params
        from daglite.tasks import get_unbound_param

        if isinstance(reducer_task, FixedParamTask):
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
                # MapTaskFuture produces a sequence
                mapped_kwargs[name] = ParamInput.from_sequence_ref(seq.id)
            elif isinstance(seq, TaskFuture):
                # TaskFuture produces a scalar - will be broadcast if needed
                mapped_kwargs[name] = ParamInput.from_ref(seq.id)
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
            backend_name=self.backend_name,
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
