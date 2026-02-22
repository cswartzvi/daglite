"""Contains the future representation for lazy iterator task invocations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import override

from daglite._validation import check_overlap_params
from daglite._validation import get_unbound_params
from daglite.futures._shared import build_node_inputs
from daglite.futures._shared import build_output_configs
from daglite.futures._shared import collect_builders
from daglite.futures.base import BaseTaskFuture
from daglite.graph.builder import NodeBuilder
from daglite.graph.nodes.iter_node import IterNode
from daglite.tasks import PartialTask
from daglite.tasks import Task
from daglite.utils import build_repr

if TYPE_CHECKING:
    from daglite.futures.map_future import MapTaskFuture
    from daglite.futures.reduce_future import ReduceFuture
    from daglite.futures.task_future import TaskFuture

R = TypeVar("R")
T = TypeVar("T")


@dataclass(frozen=True)
class IterTaskFuture(BaseTaskFuture[R]):
    """Represents a lazy iterator invocation."""

    task: Task[Any, R]
    """Underlying task whose function returns a generator/iterator."""

    kwargs: Mapping[str, Any]
    """Parameters to pass to the task function."""

    backend_name: str | None
    """Engine backend override, if ``None`` uses the default."""

    def __repr__(self) -> str:
        return build_repr("IterTaskFuture", self.task.name, kwargs=self.kwargs)

    @override
    def get_upstream_builders(self) -> list[NodeBuilder]:
        return collect_builders(self.kwargs, self._output_futures)

    @override
    def build_node(self) -> IterNode:
        node_inputs = build_node_inputs(self.kwargs)
        placeholders = set(self.kwargs.keys()) | {"iteration_index"}
        output_configs = build_output_configs(self._output_futures, placeholders)
        return IterNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            kwargs=node_inputs,
            backend_name=self.backend_name,
            retries=self.task.retries,
            timeout=self.task.timeout,
            output_configs=output_configs,
            cache=self.task.cache,
            cache_ttl=self.task.cache_ttl,
        )

    def then(
        self, mapped_task: Task[Any, T] | PartialTask[Any, T], **kwargs: Any
    ) -> MapTaskFuture[T]:
        """
        Apply mapped task to each item yielded by this iterator.

        Returns a `MapTaskFuture` whose upstream is this `IterTaskFuture`. When the graph optimizer
        is enabled, the resulting `IterNode → MapTaskNode` chain is folded into a single
        `CompositeMapTaskNode` so that items are dispatched lazily.

        Args:
            mapped_task: Either a `Task` or `PartialTask` with exactly **one** unbound parameter
                that will receive each item yielded by this iterator.
            **kwargs: Additional fixed parameters forwarded to *mapped_task*.

        Returns:
            A `MapTaskFuture` representing the per-item results.
        """
        from daglite.futures.map_future import MapTaskFuture

        if isinstance(mapped_task, PartialTask):
            check_overlap_params(dict(mapped_task.fixed_kwargs), kwargs, mapped_task.name)
            all_fixed = {**mapped_task.fixed_kwargs, **kwargs}
            actual_task = mapped_task.task
        else:
            all_fixed = kwargs
            actual_task = mapped_task

        unbound_param = get_unbound_params(actual_task.signature, all_fixed, actual_task.name)
        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs={unbound_param: self},
            backend_name=self.backend_name,
            task_store=self.task_store,
        )

    def join(self, join_task: Task[Any, T] | PartialTask[Any, T], **kwargs: Any) -> TaskFuture[T]:
        """
        Collect the iterator results and pass them to a single reducer task.

        This materializes the iterator into a list and feeds it to *join_task* as the single
        unbound parameter.

        Args:
            join_task: Either a `Task` or `PartialTask` with exactly **one** unbound parameter
                that will receive each item yielded by this iterator.
            **kwargs: Additional fixed parameters forwarded to *join_task*.

        Returns:
            A ``TaskFuture`` representing the single reduced value.
        """
        from daglite.futures.task_future import TaskFuture

        if isinstance(join_task, PartialTask):
            check_overlap_params(dict(join_task.fixed_kwargs), kwargs, join_task.name)
            all_fixed = {**join_task.fixed_kwargs, **kwargs}
            actual_task = join_task.task
        else:
            all_fixed = kwargs
            actual_task = join_task

        unbound_param = get_unbound_params(actual_task.signature, all_fixed, actual_task.name)
        merged_kwargs = dict(all_fixed)
        merged_kwargs[unbound_param] = self

        return TaskFuture(
            task=actual_task,
            kwargs=merged_kwargs,
            backend_name=self.backend_name,
            task_store=self.task_store,
        )

    def reduce(
        self,
        reduce_task: Task[Any, T] | PartialTask[Any, T],
        *,
        initial: Any,
        ordered: bool = True,
    ) -> ReduceFuture[T]:
        """
        Streaming fold over the items yielded by this iterator.

        Each yielded item is accumulated using *reduce_task(acc, item)*.

        For full lazy optimization (no intermediate list), chain via `.then()` first — the
        optimizer folds `IterNode → MapTaskNode → ReduceNode` into a single composite. Calling
        `.reduce()` directly on an `IterTaskFuture` works correctly but falls back to
        materializing the list when the optimizer cannot fold the `IterNode → ReduceNode` pair.

        Args:
            reduce_task: Either a `Task` or `PartialTask` with exactly **one** unbound parameter
                that will receive each item yielded by this iterator.
            initial: Starting value for the accumulator.
            ordered: If `True` (default), items are processed in iteration order.  If `False`,
            items are processed in completion order.

        Returns:
            A `ReduceFuture` representing the final accumulated value.
        """
        from daglite._typing import ReduceMode
        from daglite.futures.reduce_future import ReduceFuture

        if isinstance(reduce_task, PartialTask):
            actual_task = reduce_task.task
        else:
            actual_task = reduce_task

        all_fixed = dict(reduce_task.fixed_kwargs) if isinstance(reduce_task, PartialTask) else {}
        acc_param, item_param = get_unbound_params(
            actual_task.signature, all_fixed, actual_task.name, n=2
        )

        reduce_mode: ReduceMode = "ordered" if ordered else "unordered"

        return ReduceFuture(
            source=self,
            reduce_task=actual_task,
            initial=initial,
            reduce_mode=reduce_mode,
            accumulator_param=acc_param,
            item_param=item_param,
            backend_name=self.backend_name,
            task_store=self.task_store,
        )
