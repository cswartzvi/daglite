"""Iter (iterator/generator) node representation within the graph IR."""

from __future__ import annotations

import functools
from collections.abc import Generator
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Callable
from uuid import UUID

from pluggy import HookRelay
from typing_extensions import override

from daglite._typing import NodeKind
from daglite.backends.base import Backend
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import remap_node_changes
from daglite.graph.nodes._shared import resolve_inputs
from daglite.graph.nodes._shared import resolve_output_parameters
from daglite.graph.nodes._workers import _save_outputs
from daglite.graph.nodes._workers import run_task_worker
from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput


@dataclass(frozen=True)
class IterNode(BaseGraphNode):
    """A lazy iterator task node — holds a generator function and its kwargs."""

    func: Callable[..., Any]
    """Generator/iterator-returning function to execute."""

    kwargs: Mapping[str, NodeInput]
    """Keyword parameters mapped to node inputs."""

    retries: int = 0
    """Number of times to retry on failure."""

    cache: bool = False
    """Whether hash-based caching is enabled."""

    cache_ttl: int | None = None
    """Time-to-live for cached results in seconds."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "iter"

    @override
    def get_dependencies(self) -> set[UUID]:
        return collect_dependencies(self.kwargs, self.output_configs)

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> IterNode:
        changes = remap_node_changes(id_mapping, self.output_configs, kwargs=self.kwargs)
        return replace(self, **changes) if changes else self

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        inputs = resolve_inputs(self.kwargs, completed_nodes)
        output_parameters = resolve_output_parameters(self.output_configs, completed_nodes)

        if self.output_configs:
            # Per-item save path: iterate on coordinator so each item is saved individually
            runner = functools.partial(
                run_task_worker,
                func=self.func,
                metadata=self.metadata,
                inputs=inputs,
                output_configs=(),  # suppress default save — we handle per-item below
                output_parameters=[],
                retries=self.retries,
                cache_enabled=self.cache,
                cache_ttl=self.cache_ttl,
            )
            result = await backend.submit(runner, timeout=self.timeout)

            # result is the materialized list (run_task_worker converts generators to lists)
            if isinstance(result, (Generator, Iterator)):
                result = list(result)  # pragma: no cover — defensive

            items = result if isinstance(result, list) else [result]
            for idx, item in enumerate(items):
                _save_outputs(
                    result=item,
                    resolved_inputs=inputs,
                    output_config=self.output_configs,
                    output_deps=output_parameters,
                    key_extras={"iteration_index": idx},
                )
            return result

        # No saves — cheap path: submit directly, let run_task_worker materialize + save
        runner = functools.partial(
            run_task_worker,
            func=self.func,
            metadata=self.metadata,
            inputs=inputs,
            output_configs=self.output_configs,
            output_parameters=output_parameters,
            retries=self.retries,
            cache_enabled=self.cache,
            cache_ttl=self.cache_ttl,
        )
        return await backend.submit(runner, timeout=self.timeout)
