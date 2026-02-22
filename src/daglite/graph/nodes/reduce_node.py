"""Reduce node and configuration for the graph IR."""

from __future__ import annotations

import functools as _functools
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Callable
from uuid import UUID

from pluggy import HookRelay
from typing_extensions import override

from daglite._typing import NodeKind
from daglite._typing import ReduceMode
from daglite.backends.base import Backend
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import remap_output_configs
from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput


@dataclass(frozen=True)
class ReduceConfig:
    """Configuration for a streaming reduce terminal."""

    func: Callable[..., Any]
    """Reduce function with signature ``(acc, item) -> new_acc``."""

    mode: ReduceMode = "ordered"
    """Whether to process results in iteration order or completion order."""

    accumulator_param: str = "acc"
    """Parameter name for the accumulator in the reduce function."""

    item_param: str = "item"
    """Parameter name for the current item in the reduce function."""

    name: str = "reduce"
    """Human-readable name for the reduce step (used in metadata)."""

    description: str | None = None
    """Optional description for the reduce step."""

    retries: int = 0
    """Number of times to retry the reduce step on failure."""


@dataclass(frozen=True)
class ReduceNode(BaseGraphNode):
    """
    Reduces a sequence (from an upstream map node) into a single value.

    When graph optimization is **enabled**, the optimizer folds this node together with its
    upstream map chain into a `CompositeMapTaskNode` with `terminal='reduce'`, enabling true
    streaming accumulation.

    When optimization is **disabled**, this node executes as a standalone fallback: it materializes
    the upstream sequence from `completed_nodes` and applies `functools.reduce` over it.
    """

    source_id: UUID
    """ID of the upstream map node whose list result will be reduced."""

    reduce_config: ReduceConfig
    """Configuration for the reduce operation."""

    initial_input: NodeInput = field(default_factory=lambda: NodeInput.from_value(None))
    """Initial accumulator value — may be a concrete value or a reference to another node."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "reduce"

    @override
    def get_dependencies(self) -> set[UUID]:
        deps = {self.source_id}
        if self.initial_input.reference is not None:
            deps.add(self.initial_input.reference)
        deps |= collect_dependencies({}, self.output_configs)
        return deps

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> ReduceNode:
        changed = False
        new_source = id_mapping.get(self.source_id, self.source_id)
        if new_source != self.source_id:
            changed = True
        new_initial = self.initial_input
        if self.initial_input.reference is not None and self.initial_input.reference in id_mapping:
            new_initial = NodeInput(
                _kind=self.initial_input._kind,
                value=self.initial_input.value,
                reference=id_mapping[self.initial_input.reference],
            )
            changed = True
        new_oc = remap_output_configs(self.output_configs, id_mapping)
        if new_oc is not None:
            changed = True
        if changed:
            return replace(
                self,
                source_id=new_source,
                initial_input=new_initial,
                **(dict(output_configs=new_oc) if new_oc is not None else {}),
            )
        return self

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        """
        Fallback execution: materialize the upstream list and apply reduce.

        This path is taken when graph optimization is disabled. When enabled,
        the optimizer folds this node into a ``CompositeMapTaskNode`` and this
        method is never called.
        """
        cfg = self.reduce_config
        initial = self.initial_input.resolve(completed_nodes)
        items = completed_nodes[self.source_id]

        if not isinstance(items, (list, tuple)):  # pragma: no cover — items always a list from map
            items = list(items)

        # Apply reduce — support async reduce functions with retries
        def _apply_with_retries_sync(acc: Any, item: Any) -> Any:
            kwargs = {cfg.accumulator_param: acc, cfg.item_param: item}
            last_error: Exception | None = None
            for attempt in range(1 + cfg.retries):
                try:
                    return cfg.func(**kwargs)
                except Exception as e:
                    last_error = e
                    if (
                        attempt < cfg.retries
                    ):  # pragma: no branch — coverage.py misses functools.reduce branches
                        continue
            raise last_error  # type: ignore[misc]  # pragma: no cover

        if inspect.iscoroutinefunction(cfg.func):
            accumulator = initial
            for item in items:
                kwargs = {cfg.accumulator_param: accumulator, cfg.item_param: item}
                last_error: Exception | None = None
                for attempt in range(1 + cfg.retries):  # pragma: no branch – arc tracked
                    try:
                        accumulator = await cfg.func(**kwargs)
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < cfg.retries:  # pragma: no branch
                            continue
                        raise last_error  # type: ignore[misc]  # pragma: no cover
            return accumulator
        else:
            return _functools.reduce(_apply_with_retries_sync, items, initial)
