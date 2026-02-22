"""
Lazy iterator futures for streaming generator pipelines.

These futures allow a generator-producing task to be consumed lazily — one item at a time —
without materialising the full sequence in memory.  The entire pipeline (generator → mapping →
optional reduce) is submitted as a single backend job via :class:`IterNode`.

Typical usage::

    from daglite import task

    @task(backend_name="threading")
    def generate_numbers(n: int) -> Iterator[int]:
        for i in range(n):
            yield i

    @task(backend_name="threading")
    def double(x: int) -> int:
        return x * 2

    @task
    def add(acc: int, item: int) -> int:
        return acc + item

    # Lazy pipeline: generator → double → streaming reduce, all in one worker thread
    result = generate_numbers(n=10).iter().map(double).reduce(add, initial=0)
    print(result.run())   # 90
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import override

from daglite._typing import ReduceMode
from daglite._validation import get_unbound_params
from daglite.futures._shared import build_node_inputs
from daglite.futures._shared import build_output_configs
from daglite.futures._shared import collect_builders
from daglite.futures.base import BaseTaskFuture
from daglite.graph.builder import NodeBuilder
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.iter_node import IterNode
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.tasks import PartialTask
from daglite.tasks import Task
from daglite.utils import build_repr

if TYPE_CHECKING:
    from daglite.futures.task_future import TaskFuture

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# IterFuture — lazy marker, not a NodeBuilder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterFuture:
    """
    A lazy iterator over a generator task's output.

    Returned by :meth:`~daglite.futures.TaskFuture.iter`.  Not directly runnable — call
    :meth:`map` to apply a task to each yielded item and obtain a runnable future.
    """

    source: TaskFuture
    """The underlying generator task future."""

    def map(
        self,
        mapped_task: Task[Any, T] | PartialTask[Any, T],
        **fixed_kwargs: Any,
    ) -> IterMapFuture[T]:
        """
        Apply *mapped_task* to each item yielded by the generator.

        The unbound parameter of *mapped_task* is inferred automatically (same convention as
        :meth:`~daglite.futures.TaskFuture.then`) and will receive each yielded value in turn.

        Args:
            mapped_task: Task (or :class:`~daglite.tasks.PartialTask`) to apply to each item.
                Must have exactly one unbound parameter that will receive each yielded item.
            **fixed_kwargs: Additional fixed parameters for *mapped_task*.

        Returns:
            An :class:`IterMapFuture` that lazily processes each yielded item.

        Examples:
            >>> from typing import Iterator
            >>> from daglite import task
            >>> @task
            ... def numbers(n: int) -> Iterator[int]:
            ...     yield from range(n)
            >>> @task
            ... def double(x: int) -> int:
            ...     return x * 2
            >>> numbers(n=5).iter().map(double).run()
            [0, 2, 4, 6, 8]
        """
        if isinstance(mapped_task, PartialTask):
            all_fixed: dict[str, Any] = {**mapped_task.fixed_kwargs, **fixed_kwargs}
            actual_task = mapped_task.task
        else:
            all_fixed = dict(fixed_kwargs)
            actual_task = mapped_task

        iter_param = get_unbound_params(actual_task.signature, all_fixed, actual_task.name)

        return IterMapFuture(
            source=self.source,
            task=actual_task,
            iter_param=iter_param,
            fixed_kwargs=all_fixed,
            backend_name=self.source.backend_name,
        )


# ---------------------------------------------------------------------------
# IterMapFuture — generator → map, collects to list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterMapFuture(BaseTaskFuture[list[R]]):
    """
    Lazily applies a task to each item yielded by a generator, collecting results to a list.

    Executes in a **single** backend submission: the generator is never materialised and each
    item is immediately passed to the mapping task in the same worker thread.

    Use :meth:`reduce` to fold results into a scalar value instead of collecting them.
    """

    source: TaskFuture
    """The generator task future (not registered as a separate graph node)."""

    task: Task[Any, R]
    """Task applied to each yielded item."""

    iter_param: str
    """Parameter name in *task* that receives each yielded item."""

    fixed_kwargs: Mapping[str, Any]
    """Fixed parameters for *task* (other than the iterated item)."""

    backend_name: str | None
    """Backend override; ``None`` uses the engine default."""

    def __repr__(self) -> str:
        return build_repr("IterMapFuture", f"{self.source.task.name} → {self.task.name}")

    @override
    def run(self, *, plugins: list[Any] | None = None) -> list[R]:
        return super().run(plugins=plugins)

    @override
    async def run_async(self, *, plugins: list[Any] | None = None) -> list[R]:
        return await super().run_async(plugins=plugins)

    def reduce(
        self,
        reduce_task: Task[Any, T] | PartialTask[Any, T],
        *,
        initial: Any,
        ordered: bool = True,
    ) -> IterReduceFuture[T]:
        """
        Stream each mapped result into a reduce function, returning a single accumulated value.

        The entire pipeline (generate → map → reduce) executes in one backend submission.
        The reduce function runs in the worker thread — not on the coordinator's event-loop
        thread — avoiding contention with the submission loop.

        Args:
            reduce_task: A :class:`~daglite.tasks.Task` (or
                :class:`~daglite.tasks.PartialTask`) with exactly two unbound parameters
                ``(accumulator, item)`` that returns the new accumulator value.
            initial: Starting value for the accumulator.
            ordered: Reserved for API parity; iteration is always sequential within a single
                worker submission.

        Returns:
            An :class:`IterReduceFuture` representing the final accumulated value.

        Examples:
            >>> from typing import Iterator
            >>> from daglite import task
            >>> @task
            ... def numbers(n: int) -> Iterator[int]:
            ...     yield from range(n)
            >>> @task
            ... def double(x: int) -> int:
            ...     return x * 2
            >>> @task
            ... def add(acc: int, item: int) -> int:
            ...     return acc + item
            >>> numbers(n=5).iter().map(double).reduce(add, initial=0).run()
            20
        """
        if isinstance(reduce_task, PartialTask):
            actual_task = reduce_task.task
            all_fixed: dict[str, Any] = dict(reduce_task.fixed_kwargs)
        else:
            actual_task = reduce_task
            all_fixed = {}

        acc_param, item_param = get_unbound_params(
            actual_task.signature, all_fixed, actual_task.name, n=2
        )

        reduce_mode: ReduceMode = "ordered" if ordered else "unordered"

        return IterReduceFuture(
            source=self.source,
            task=self.task,
            iter_param=self.iter_param,
            fixed_kwargs=self.fixed_kwargs,
            reduce_task=actual_task,
            reduce_initial=initial,
            reduce_mode=reduce_mode,
            accumulator_param=acc_param,
            item_param=item_param,
            reduce_fixed_kwargs=all_fixed,
            backend_name=self.backend_name,
        )

    @override
    def get_upstream_builders(self) -> list[NodeBuilder]:
        # Collect builders from the *source's* kwargs (not the source node itself — it is
        # embedded directly inside the IterNode and must not appear as a separate graph node).
        upstream_kwargs: dict[str, Any] = {**self.source.kwargs, **self.fixed_kwargs}
        return collect_builders(upstream_kwargs, self._output_futures)

    @override
    def build_node(self) -> IterNode:
        source_kwargs = build_node_inputs(self.source.kwargs)
        map_fixed_kwargs = build_node_inputs(self.fixed_kwargs)
        placeholders = (
            set(source_kwargs.keys()) | set(map_fixed_kwargs.keys()) | {self.iter_param}
        )
        output_configs = build_output_configs(self._output_futures, placeholders)

        return IterNode(
            id=self.id,
            name=f"iter({self.source.task.name} → {self.task.name})",
            description=(
                f"Lazy iteration of '{self.source.task.name}' "
                f"mapped through '{self.task.name}'"
            ),
            backend_name=self.backend_name,
            source_func=self.source.task.func,
            source_kwargs=source_kwargs,
            map_func=self.task.func,
            iter_param=self.iter_param,
            map_fixed_kwargs=map_fixed_kwargs,
            terminal="collect",
            output_configs=output_configs,
        )


# ---------------------------------------------------------------------------
# IterReduceFuture — generator → map → reduce, returns scalar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterReduceFuture(BaseTaskFuture[R]):
    """
    Lazily applies a task to each generator item and folds results into a scalar value.

    The entire pipeline (generate → map → reduce) executes in a **single** backend submission
    with O(1) memory.  The reduce function runs in the worker thread, not on the main
    event-loop thread.
    """

    source: TaskFuture
    """The generator task future."""

    task: Task[Any, Any]
    """Task applied to each yielded item."""

    iter_param: str
    """Parameter name in *task* that receives each yielded item."""

    fixed_kwargs: Mapping[str, Any]
    """Fixed parameters for *task*."""

    reduce_task: Task[Any, R]
    """The accumulator task."""

    reduce_initial: Any
    """Initial accumulator value (may be a :class:`~daglite.futures.TaskFuture`)."""

    reduce_mode: ReduceMode
    """Ordered or unordered (reserved; always ordered in a single-thread submission)."""

    accumulator_param: str
    """Parameter name for the accumulator in *reduce_task*."""

    item_param: str
    """Parameter name for the current item in *reduce_task*."""

    reduce_fixed_kwargs: Mapping[str, Any]
    """Fixed keyword arguments from a :class:`~daglite.tasks.PartialTask` reduce task."""

    backend_name: str | None
    """Backend override; ``None`` uses the engine default."""

    def __repr__(self) -> str:
        return build_repr(
            "IterReduceFuture",
            f"{self.source.task.name} → {self.task.name} → reduce({self.reduce_task.name})",
        )

    @override
    def run(self, *, plugins: list[Any] | None = None) -> R:
        return super().run(plugins=plugins)

    @override
    async def run_async(self, *, plugins: list[Any] | None = None) -> R:
        return await super().run_async(plugins=plugins)

    @override
    def get_upstream_builders(self) -> list[NodeBuilder]:
        upstream_kwargs: dict[str, Any] = {**self.source.kwargs, **self.fixed_kwargs}
        upstream: list[NodeBuilder] = collect_builders(upstream_kwargs, self._output_futures)
        # The reduce initial value may itself be a future
        if isinstance(self.reduce_initial, BaseTaskFuture):
            upstream.append(self.reduce_initial)
        return upstream

    @override
    def build_node(self) -> IterNode:
        source_kwargs = build_node_inputs(self.source.kwargs)
        map_fixed_kwargs = build_node_inputs(self.fixed_kwargs)

        if isinstance(self.reduce_initial, BaseTaskFuture):
            initial_input = NodeInput.from_ref(self.reduce_initial.id)
        else:
            initial_input = NodeInput.from_value(self.reduce_initial)

        placeholders = (
            set(source_kwargs.keys()) | set(map_fixed_kwargs.keys()) | {self.iter_param}
        )
        output_configs = build_output_configs(self._output_futures, placeholders)

        # Bind any fixed kwargs from a PartialTask so they reach the reduce function.
        reduce_func = self.reduce_task.func
        if self.reduce_fixed_kwargs:
            reduce_func = functools.partial(reduce_func, **self.reduce_fixed_kwargs)

        reduce_config = ReduceConfig(
            func=reduce_func,
            mode=self.reduce_mode,
            accumulator_param=self.accumulator_param,
            item_param=self.item_param,
            name=self.reduce_task.name,
            description=self.reduce_task.description,
            retries=self.reduce_task.retries,
        )

        return IterNode(
            id=self.id,
            name=(
                f"iter({self.source.task.name} → {self.task.name}"
                f" → reduce({self.reduce_task.name}))"
            ),
            description="Lazy iter pipeline: generate → map → reduce",
            backend_name=self.backend_name,
            source_func=self.source.task.func,
            source_kwargs=source_kwargs,
            map_func=self.task.func,
            iter_param=self.iter_param,
            map_fixed_kwargs=map_fixed_kwargs,
            terminal="reduce",
            reduce_config=reduce_config,
            initial_input=initial_input,
            output_configs=output_configs,
        )
