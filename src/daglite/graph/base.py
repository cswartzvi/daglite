"""
Contains base classes and protocols for graph Intermediate Representation (IR).

Note that the graph IR is considered an internal implementation detail and is not part of the
public API. Therefore, the interfaces defined here use non-generic base classes and/or protocols
for maximum flexibility.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal, Protocol
from uuid import UUID

from daglite.datasets.store import DatasetStore
from daglite.exceptions import GraphError

ParamKind = Literal["value", "ref", "sequence", "sequence_ref"]
NodeKind = Literal["task", "map", "dataset"]


class GraphBuilder(Protocol):
    """Protocol for building graph Intermediate Representation (IR) components from tasks."""

    @property
    def id(self) -> UUID:
        """Returns the unique identifier for this builder's graph node."""
        ...

    @abc.abstractmethod
    def get_dependencies(self) -> list[GraphBuilder]:
        """
        Return the direct dependencies of this builder.

        Returns:
            list[GraphBuilder]: List of builders this node depends on.
        """
        ...

    @abc.abstractmethod
    def to_graph(self) -> BaseGraphNode:
        """
        Convert this builder into a GraphNode.

        All dependencies will have their IDs assigned before this is called,
        so implementations can safely access dependency.id.

        Returns:
            GraphNode: The constructed graph node.
        """
        ...


@dataclass(frozen=True)
class GraphMetadata:
    """Metadata for a compiled graph."""

    id: UUID
    """Unique identifier for this node."""

    name: str
    """Human-readable name for the graph."""

    kind: NodeKind
    """Kind of this graph node (e.g., 'task', 'map', etc.)."""

    description: str | None = field(default=None, kw_only=True)
    """Optional human-readable description for the graph."""

    backend_name: str | None = field(default=None, kw_only=True)
    """Default backend name for executing nodes in this graph."""

    key: str | None = field(default=None, kw_only=True)
    """Optional key identifying this specific node instance in the execution graph."""


@dataclass(frozen=True)
class BaseGraphNode(abc.ABC):
    """Represents a node in the compiled graph Intermediate Representation (IR)."""

    id: UUID
    """Unique identifier for this node."""

    name: str
    """Human-readable name for the graph."""

    description: str | None = field(default=None, kw_only=True)
    """Optional human-readable description for the graph."""

    backend_name: str | None = field(default=None, kw_only=True)
    """Default backend name for executing nodes in this graph."""

    key: str | None = field(default=None, kw_only=True)
    """Optional key identifying this specific node instance in the execution graph."""

    timeout: float | None = field(default=None, kw_only=True)
    """Maximum execution time in seconds (enforced by backend). If None, no timeout."""

    output_configs: tuple[OutputConfig, ...] = field(default=(), kw_only=True)
    """Output save/checkpoint configurations for this node."""

    def __post_init__(self) -> None:
        # This is unlikely to happen given timeout is checked at task level, but just in case
        assert self.timeout is None or self.timeout >= 0, "Timeout must be non-negative"

    @abc.abstractmethod
    def dependencies(self) -> set[UUID]:
        """
        IDs of nodes that the current node depends on (its direct predecessors).

        Each node implementation determines its own dependencies based on its
        internal structure (e.g., from InputParams, sub-graphs, etc.).
        """
        ...

    @abc.abstractmethod
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        """
        Resolve this node's inputs from completed predecessor nodes.

        Args:
            completed_nodes: Mapping from node IDs to their computed results.

        Returns:
            Dictionary of resolved parameter names to values, ready for execution.
        """
        ...

    def resolve_output_deps(self, completed_nodes: Mapping[UUID, Any]) -> list[dict[str, Any]]:
        """
        Resolve output dependencies to concrete values.

        Resolves InputParam extras in output_configs to actual values from completed nodes.
        Returns parallel list to output_configs containing only the resolved extras dicts.

        Args:
            completed_nodes: Mapping from node IDs to their computed results.

        Returns:
            List of resolved extras dictionaries, parallel to self.output_configs.
        """
        resolved_deps_list = []
        for config in self.output_configs:
            resolved_deps = {n: p.resolve(completed_nodes) for n, p in config.dependencies.items()}
            resolved_deps_list.append(resolved_deps)
        return resolved_deps_list

    @abc.abstractmethod
    async def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        """
        Execute this node asynchronously with resolved inputs.

        Similar to run() but for async execution contexts. This allows proper
        handling of async functions without forcing materialization.

        Args:
            resolved_inputs: Pre-resolved parameter inputs for this node.
            **kwargs: Additional runtime-specific execution parameters.

        Returns:
            Node execution result. May be an async generator or regular value.
        """
        ...

    @abc.abstractmethod
    def to_metadata(self) -> "GraphMetadata":
        """Returns a metadata object for this graph node."""
        ...


@dataclass(frozen=True)
class InputParam:
    """Input parameter representation for graph IR."""

    _kind: ParamKind
    """Kind of this parameter input, determining how it should be resolved."""

    value: Any | None = None
    """Concrete value for 'value' and 'sequence' kinds. Must be None for 'ref' kinds."""

    reference: UUID | None = None
    """Reference node ID for 'ref' and 'sequence_ref' kinds. Must be None for 'value' kinds."""

    def __post_init__(self) -> None:
        context = "This may indicate an internal error in graph construction."
        if self._kind in ("value", "sequence"):
            if self.reference is not None:
                raise GraphError(f"InputParam kind '{self._kind}' must not have a ref ID.")
        elif self._kind in ("ref", "sequence_ref"):
            if self.reference is None:
                raise GraphError(f"InputParam kind '{self._kind}' requires a ref ID.")
            if self.value is not None:
                raise GraphError(f"InputParam kind '{self._kind}' must not have a value.")
        else:  # pragma no cover
            raise GraphError(f"Unknown InputParam kind: '{self._kind}'. {context}")

    def resolve(self, completed_nodes: Mapping[UUID, Any]) -> Any:
        """
        Resolves this input to a concrete value using completed node outputs.

        Args:
            completed_nodes: Mapping from node IDs to their computed values.

        Returns:
            Resolved concrete value for this input.
        """
        match self._kind:
            case "value":
                return self.value
            case "ref":
                assert self.reference is not None  # Checked by post_init
                return completed_nodes[self.reference]
            case "sequence":
                assert self.value is not None  # Checked by post_init
                return list(self.value)
            case "sequence_ref":
                assert self.reference is not None  # Checked by post_init
                return list(completed_nodes[self.reference])
            case _:  # pragma no cover
                raise GraphError(f"Unknown InputParam kind: '{self._kind}'. ")

    @classmethod
    def from_value(cls, v: Any) -> InputParam:
        """Creates a InputParam from a concrete value."""
        return cls(_kind="value", value=v)

    @classmethod
    def from_ref(cls, node_id: UUID) -> InputParam:
        """Creates a InputParam that references another node's output."""
        return cls(_kind="ref", reference=node_id)

    @classmethod
    def from_sequence(cls, vals: Sequence[Any]) -> InputParam:
        """Creates a InputParam from a concrete sequence value."""
        return cls(_kind="sequence", value=list(vals))

    @classmethod
    def from_sequence_ref(cls, node_id: UUID) -> InputParam:
        """Creates a InputParam that references another node's sequence output."""
        return cls(_kind="sequence_ref", reference=node_id)


@dataclass(frozen=True)
class OutputConfig:
    """
    Configuration for saving or checkpointing a task output.

    Outputs can be saved with a storage key and optional checkpoint name for resumption.
    Extra parameters (as InputParams) can be included for formatting or metadata.
    """

    key: str
    """Storage key template with {param} placeholders for formatting."""

    store: DatasetStore | None = None
    """Dataset store where this output should be saved (None uses settings default)."""

    name: str | None = None
    """Optional checkpoint name for graph resumption via evaluate(from_={name: key})."""

    format: str | None = None
    """Optional serialization format hint (e.g., 'pickle', 'json', etc.)."""

    dependencies: Mapping[str, InputParam] = field(default_factory=dict)
    """Parameter dependencies for this output, used for key formatting"""

    options: dict[str, Any] = field(default_factory=dict)
    """Additional options passed to the Dataset's save method."""
