"""Task node representation within the graph IR."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Callable
from uuid import UUID

from typing_extensions import override

from daglite._typing import NodeKind
from daglite._typing import Submission
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import remap_node_changes
from daglite.graph.nodes._shared import resolve_inputs
from daglite.graph.nodes._shared import resolve_output_parameters
from daglite.graph.nodes._workers import run_task_worker
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import PrepareCollectNode


@dataclass(frozen=True)
class TaskNode(PrepareCollectNode):
    """Basic function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for this task node."""

    kwargs: Mapping[str, NodeInput]
    """Keyword parameters from the task function mapped to node inputs."""

    retries: int = 0
    """Number of times to retry the task on failure."""

    cache: bool = False
    """Whether to enable hash-based caching for this task."""

    cache_ttl: int | None = None
    """Time-to-live for cached results in seconds. None means no expiration."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # This is unlikely to happen given retries is checked at task level, but just in case
        assert self.retries >= 0, "Retries must be non-negative"

    @property
    @override
    def kind(self) -> NodeKind:
        return "task"

    @override
    def get_dependencies(self) -> set[UUID]:
        return collect_dependencies(self.kwargs, self.output_configs)

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> TaskNode:
        changes = remap_node_changes(id_mapping, self.output_configs, kwargs=self.kwargs)
        return replace(self, **changes) if changes else self

    @override
    def _prepare(self, completed_nodes: Mapping[UUID, Any]) -> list[Submission]:
        inputs = resolve_inputs(self.kwargs, completed_nodes)
        output_parameters = resolve_output_parameters(self.output_configs, completed_nodes)
        func = functools.partial(
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
        return [func]

    @override
    def _collect(self, results: list[Any]) -> Any:
        return results[0] if results else None
