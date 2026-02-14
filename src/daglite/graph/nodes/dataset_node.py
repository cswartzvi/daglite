"""Dataset load node representation within the graph IR."""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from uuid import UUID

from typing_extensions import override

from daglite.datasets.store import DatasetStore
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import resolve_inputs
from daglite.graph.nodes._shared import resolve_output_parameters
from daglite.graph.nodes._workers import run_dataset_load
from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import NodeKind
from daglite.graph.nodes.base import Submission


@dataclass(frozen=True)
class DatasetNode(BaseGraphNode):
    """
    Dataset load node representation within the graph IR.

    Unlike task nodes, this node does not execute a user function.  Instead it
    loads a previously-saved dataset from a :class:`DatasetStore` and returns
    the deserialized value.

    The storage *key* may contain ``{placeholder}`` templates that are resolved
    from dependency values at runtime (exactly like output-save keys).
    """

    store: DatasetStore
    """The dataset store to load from."""

    load_key: str
    """Storage key template (may contain ``{param}`` placeholders)."""

    return_type: type | None = None
    """Expected Python type for deserialization dispatch."""

    load_format: str | None = None
    """Explicit serialization format hint (e.g. ``'pickle'``, ``'pandas/csv'``)."""

    load_options: dict[str, Any] = field(default_factory=dict)
    """Additional options forwarded to the ``Dataset`` constructor."""

    kwargs: Mapping[str, NodeInput] = field(default_factory=dict)
    """Keyword parameters used for key-template formatting."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "dataset"

    @override
    def get_dependencies(self) -> set[UUID]:
        return collect_dependencies(self.kwargs, self.output_configs)

    @override
    def _prepare(self, completed_nodes: Mapping[UUID, Any]) -> list[Submission]:
        resolved_inputs = resolve_inputs(self.kwargs, completed_nodes)
        output_parameters = resolve_output_parameters(self.output_configs, completed_nodes)
        func = functools.partial(
            run_dataset_load,
            store=self.store,
            load_key=self.load_key,
            return_type=self.return_type,
            load_format=self.load_format,
            load_options=self.load_options,
            metadata=self.metadata,
            resolved_inputs=resolved_inputs,
            output_configs=self.output_configs,
            output_parameters=output_parameters,
        )
        return [func]

    @override
    def _collect(self, results: list[Any]) -> Any:
        return results[0] if results else None
