"""
IterNode: a generator pipeline that runs entirely in a single backend submission.

Unlike MapTaskNode — which materialises the source sequence and fans it out as N independent
submissions — IterNode executes the generator lazily, inside the same worker that applies the
mapping function.  This makes it suitable for large datasets or long-running generators that
cannot be held in memory.

Terminal modes
--------------
* **collect** — each mapped result is appended to a list returned to the coordinator.
* **reduce**  — mapped results are folded into an accumulator as they are produced.
  The reduce function runs in the worker thread, *not* on the main event-loop thread,
  which avoids contention with the coordinator's submission loop.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Callable
from uuid import UUID

from pluggy import HookRelay
from typing_extensions import override

from daglite._typing import NodeKind
from daglite.backends.base import Backend
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import remap_node_inputs
from daglite.graph.nodes._shared import remap_output_configs
from daglite.graph.nodes._shared import resolve_inputs
from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.reduce_node import ReduceConfig


@dataclass(frozen=True)
class IterNode(BaseGraphNode):
    """
    A node that lazily iterates a generator function and pipelines each item to a mapping task.

    The entire pipeline (source generator → mapping function → optional reduce) is submitted as
    a **single** backend job.  The generator is never materialised: each value is consumed and
    passed to the mapping function immediately, then either appended to a list (terminal=
    ``'collect'``) or folded into an accumulator (terminal=``'reduce'``).

    This design also moves the reduce function off the coordinator's event-loop thread — a
    limitation of the standard ``CompositeMapTaskNode`` streaming reduce path.
    """

    source_func: Callable[..., Any]
    """Generator (or async generator) function to iterate over."""

    source_kwargs: Mapping[str, NodeInput]
    """Resolved inputs for the source function (may contain refs to upstream nodes)."""

    map_func: Callable[..., Any]
    """Function applied to each value yielded by the source generator."""

    iter_param: str
    """Name of the parameter in ``map_func`` that receives each yielded item."""

    map_fixed_kwargs: Mapping[str, NodeInput]
    """Fixed parameters for ``map_func`` other than the iterated item."""

    terminal: str
    """Aggregation mode: ``'collect'`` (list) or ``'reduce'`` (fold)."""

    reduce_config: ReduceConfig | None = field(default=None)
    """Reduce configuration; required when ``terminal='reduce'``."""

    initial_input: NodeInput | None = field(default=None)
    """Initial accumulator; required when ``terminal='reduce'``, may reference another node."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "iter"

    @override
    def get_dependencies(self) -> set[UUID]:
        all_kwargs: dict[str, NodeInput] = {**self.source_kwargs, **self.map_fixed_kwargs}
        deps = collect_dependencies(all_kwargs, self.output_configs)
        if self.initial_input is not None and self.initial_input.reference is not None:
            deps.add(self.initial_input.reference)
        return deps

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> IterNode:
        changes: dict[str, Any] = {}

        new_source_kwargs = remap_node_inputs(self.source_kwargs, id_mapping)
        if new_source_kwargs is not self.source_kwargs:
            changes["source_kwargs"] = new_source_kwargs

        new_map_fixed = remap_node_inputs(self.map_fixed_kwargs, id_mapping)
        if new_map_fixed is not self.map_fixed_kwargs:
            changes["map_fixed_kwargs"] = new_map_fixed

        new_oc = remap_output_configs(self.output_configs, id_mapping)
        if new_oc is not None:
            changes["output_configs"] = new_oc

        if (
            self.initial_input is not None
            and self.initial_input.reference is not None
            and self.initial_input.reference in id_mapping
        ):
            changes["initial_input"] = NodeInput(
                _kind=self.initial_input._kind,
                value=self.initial_input.value,
                reference=id_mapping[self.initial_input.reference],
            )

        return replace(self, **changes) if changes else self

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        source_resolved = resolve_inputs(self.source_kwargs, completed_nodes)
        map_fixed_resolved = resolve_inputs(self.map_fixed_kwargs, completed_nodes)
        initial = self.initial_input.resolve(completed_nodes) if self.initial_input else None

        runner = functools.partial(
            _run_iter_node,
            source_func=self.source_func,
            source_inputs=source_resolved,
            map_func=self.map_func,
            iter_param=self.iter_param,
            map_fixed_inputs=map_fixed_resolved,
            terminal=self.terminal,
            reduce_config=self.reduce_config,
            initial=initial,
        )
        return await backend.submit(runner, timeout=self.timeout)


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


async def _run_iter_node(
    source_func: Callable[..., Any],
    source_inputs: dict[str, Any],
    map_func: Callable[..., Any],
    iter_param: str,
    map_fixed_inputs: dict[str, Any],
    terminal: str,
    reduce_config: ReduceConfig | None,
    initial: Any,
) -> Any:
    """
    Execute the full iterator pipeline inside a single worker submission.

    The generator is iterated lazily: each yielded value is passed directly to ``map_func``
    without accumulating the full sequence.  Supports both sync and async generators.

    Defined at module level so that ``functools.partial`` of this function is picklable by
    ``ProcessPoolExecutor``.
    """
    # Call the source to get the generator object
    if inspect.isasyncgenfunction(source_func):
        gen = source_func(**source_inputs)
    elif inspect.iscoroutinefunction(source_func):
        gen = await source_func(**source_inputs)
    else:
        gen = source_func(**source_inputs)

    # Route to async or sync iteration based on the generator type
    if isinstance(gen, (AsyncGenerator, AsyncIterator)):
        return await _run_pipeline_async(
            gen, map_func, iter_param, map_fixed_inputs, terminal, reduce_config, initial
        )
    else:
        return await _run_pipeline_sync(
            gen, map_func, iter_param, map_fixed_inputs, terminal, reduce_config, initial
        )


async def _run_pipeline_sync(
    gen: Any,
    map_func: Callable[..., Any],
    iter_param: str,
    map_fixed_inputs: dict[str, Any],
    terminal: str,
    reduce_config: ReduceConfig | None,
    initial: Any,
) -> Any:
    """Pipeline loop for sync generators/iterators."""
    if terminal == "collect":
        results: list[Any] = []
        for item in gen:
            mapped = await _call_func(map_func, {iter_param: item, **map_fixed_inputs})
            results.append(mapped)
        return results
    else:  # reduce
        assert reduce_config is not None
        acc = initial
        for item in gen:
            mapped = await _call_func(map_func, {iter_param: item, **map_fixed_inputs})
            acc = await _call_func(
                reduce_config.func,
                {reduce_config.accumulator_param: acc, reduce_config.item_param: mapped},
            )
        return acc


async def _run_pipeline_async(
    gen: Any,
    map_func: Callable[..., Any],
    iter_param: str,
    map_fixed_inputs: dict[str, Any],
    terminal: str,
    reduce_config: ReduceConfig | None,
    initial: Any,
) -> Any:
    """Pipeline loop for async generators/iterators."""
    if terminal == "collect":
        results: list[Any] = []
        async for item in gen:
            mapped = await _call_func(map_func, {iter_param: item, **map_fixed_inputs})
            results.append(mapped)
        return results
    else:  # reduce
        assert reduce_config is not None
        acc = initial
        async for item in gen:
            mapped = await _call_func(map_func, {iter_param: item, **map_fixed_inputs})
            acc = await _call_func(
                reduce_config.func,
                {reduce_config.accumulator_param: acc, reduce_config.item_param: mapped},
            )
        return acc


async def _call_func(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Invoke a function that may be either synchronous or a coroutine function."""
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    return func(**kwargs)
