"""Contains the representation of a reduce future (streaming fold over a mapped future)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from typing_extensions import override

from daglite._typing import ReduceMode
from daglite.futures._shared import build_output_configs
from daglite.futures._shared import collect_builders
from daglite.futures.base import BaseTaskFuture
from daglite.graph.builder import NodeBuilder
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.reduce_node import ReduceNode
from daglite.tasks import Task
from daglite.utils import build_repr

R = TypeVar("R")


@dataclass(frozen=True)
class ReduceFuture(BaseTaskFuture[R]):
    """
    Represents a streaming reduce (fold) over a mapped future's output.

    When the graph is optimized, the optimizer folds this node and its upstream map chain into a
    `CompositeMapTaskNode` with `terminal='reduce'`, enabling true streaming accumulation with O(1)
    memory.

    When optimization is disabled, the `ReduceNode` falls back to `functools.reduce` over the
    materialized list.
    """

    source: BaseTaskFuture[Any]
    """The upstream map future whose sequence output will be reduced."""

    reduce_task: Task[Any, R]
    """
    Task with exactly two parameters (accumulator, item).

    The function signature is enforced: the two parameters receive the running accumulator and each
    item from the upstream sequence, respectively.
    """

    initial: Any
    """Initial value for the accumulator."""

    reduce_mode: ReduceMode
    """Whether to process items in iteration order or completion order."""

    accumulator_param: str
    """Name of the accumulator parameter in the reduce function."""

    item_param: str
    """Name of the item parameter in the reduce function."""

    backend_name: str | None
    """Engine backend override, if `None` uses the default engine backend."""

    def __repr__(self) -> str:
        return build_repr(
            "ReduceFuture",
            self.reduce_task.name,
            f"mode={self.reduce_mode}",
        )

    @override
    def run(self, *, plugins: list[Any] | None = None, cache_store: Any | None = None) -> R:
        return super().run(plugins=plugins, cache_store=cache_store)

    @override
    async def run_async(
        self, *, plugins: list[Any] | None = None, cache_store: Any | None = None
    ) -> R:
        return await super().run_async(plugins=plugins, cache_store=cache_store)

    @override
    def get_upstream_builders(self) -> list[NodeBuilder]:
        # Source is always an upstream builder; initial may be a future too
        upstream: dict[str, Any] = {"__source__": self.source}
        if isinstance(self.initial, BaseTaskFuture):
            upstream["__initial__"] = self.initial
        return collect_builders(upstream, outputs=self._output_futures)

    @override
    def build_node(self) -> ReduceNode:
        if isinstance(self.initial, BaseTaskFuture):
            initial_input = NodeInput.from_ref(self.initial.id)
        else:
            initial_input = NodeInput.from_value(self.initial)

        output_configs = build_output_configs(self._output_futures, set())

        config = ReduceConfig(
            func=self.reduce_task.func,
            mode=self.reduce_mode,
            accumulator_param=self.accumulator_param,
            item_param=self.item_param,
            name=self.reduce_task.name,
            description=self.reduce_task.description,
            retries=self.reduce_task.retries,
        )
        return ReduceNode(
            id=self.id,
            name=f"reduce({self.reduce_task.name})",
            description=self.reduce_task.description,
            backend_name=self.backend_name,
            source_id=self.source.id,
            reduce_config=config,
            initial_input=initial_input,
            timeout=self.reduce_task.timeout,
            output_configs=output_configs,
        )
