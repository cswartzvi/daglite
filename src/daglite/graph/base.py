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

from daglite.exceptions import ExecutionError
from daglite.outputs.base import OutputStore

ParamKind = Literal["value", "ref", "sequence", "sequence_ref"]
NodeKind = Literal["task", "map", "composite_task", "composite_map"]


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
        internal structure (e.g., from ParamInputs, sub-graphs, etc.).
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

    def resolve_output_extras(self, completed_nodes: Mapping[UUID, Any]) -> list[dict[str, Any]]:
        """
        Resolve output configuration extras to concrete values.

        Resolves ParamInput extras in output_configs to actual values from completed nodes.
        Returns parallel list to output_configs containing only the resolved extras dicts.

        Args:
            completed_nodes: Mapping from node IDs to their computed results.

        Returns:
            List of resolved extras dictionaries, parallel to self.output_configs.
        """
        resolved_extras_list = []
        for config in self.output_configs:
            resolved_extras = {
                name: param.resolve(completed_nodes) for name, param in config.extras.items()
            }
            resolved_extras_list.append(resolved_extras)
        return resolved_extras_list

    @abc.abstractmethod
    def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        """
        Execute this node synchronously with resolved inputs.

        This method runs in the worker context where plugin_manager and reporter
        are available via execution context (see daglite.backends.context).

        Args:
            resolved_inputs: Pre-resolved parameter inputs for this node.
            **kwargs: Additional backend-specific execution parameters.

        Returns:
            Node execution result. May be a coroutine, generator, or regular value.
        """
        ...

    @abc.abstractmethod
    async def run_async(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        """
        Execute this node asynchronously with resolved inputs.

        Similar to run() but for async execution contexts. This allows proper
        handling of async functions without forcing materialization.

        Args:
            resolved_inputs: Pre-resolved parameter inputs for this node.
            **kwargs: Additional backend-specific execution parameters.

        Returns:
            Node execution result. May be an async generator or regular value.
        """
        ...

    @abc.abstractmethod
    def to_metadata(self) -> "GraphMetadata":
        """Returns a metadata object for this graph node."""
        ...


@dataclass(frozen=True)
class ParamInput:
    """
    Parameter input representation for graph IR.

    Inputs can be one of four kinds:
    - value        : concrete Python value
    - ref          : scalar produced by another node
    - sequence     : concrete list/tuple
    - sequence_ref : sequence produced by another node
    """

    kind: ParamKind
    value: Any | None = None
    ref: UUID | None = None

    @property
    def is_ref(self) -> bool:
        """Returns `True` if this input is a reference to another node's output."""
        return self.kind in ("ref", "sequence_ref")

    def resolve(self, completed_nodes: Mapping[UUID, Any]) -> Any:
        """
        Resolves this input to a scalar value.

        Args:
            completed_nodes: Mapping from node IDs to their computed values.

        Returns:
           Resolved scalar value.
        """
        if self.kind == "value":
            return self.value
        if self.kind == "ref":
            assert self.ref is not None
            return completed_nodes[self.ref]

        raise ExecutionError(
            f"Cannot resolve parameter of kind '{self.kind}' as a scalar value. "
            f"Expected 'value' or 'ref', but got '{self.kind}'. "
            f"This may indicate an internal error in graph construction."
        )

    def resolve_sequence(self, completed_nodes: Mapping[UUID, Any]) -> Sequence[Any]:
        """
        Resolves this input to a sequence value.

        Args:
            completed_nodes: Mapping from node IDs to their computed values.

        Returns:
            Resolved sequence value.
        """
        if self.kind == "sequence":
            return list(self.value)  # type: ignore
        if self.kind == "sequence_ref":
            assert self.ref is not None
            return list(completed_nodes[self.ref])
        from daglite.exceptions import ExecutionError

        raise ExecutionError(
            f"Cannot resolve parameter of kind '{self.kind}' as a sequence. "
            f"Expected 'sequence' or 'sequence_ref', but got '{self.kind}'. "
            f"This may indicate an internal error in graph construction."
        )

    @classmethod
    def from_value(cls, v: Any) -> ParamInput:
        """Creates a ParamInput from a concrete value."""
        return cls(kind="value", value=v)

    @classmethod
    def from_ref(cls, node_id: UUID) -> ParamInput:
        """Creates a ParamInput that references another node's output."""
        return cls(kind="ref", ref=node_id)

    @classmethod
    def from_sequence(cls, vals: Sequence[Any]) -> ParamInput:
        """Creates a ParamInput from a concrete sequence value."""
        return cls(kind="sequence", value=list(vals))

    @classmethod
    def from_sequence_ref(cls, node_id: UUID) -> ParamInput:
        """Creates a ParamInput that references another node's sequence output."""
        return cls(kind="sequence_ref", ref=node_id)


@dataclass(frozen=True)
class OutputConfig:
    """
    Configuration for saving or checkpointing a task output.

    Outputs can be saved with a storage key and optional checkpoint name for resumption.
    Extra parameters (as ParamInputs) can be included for formatting or metadata.
    """

    key: str
    """Storage key template with {param} placeholders for formatting."""

    store: OutputStore | None = None
    """OutputStore instance where this output should be saved (None uses plugin default)."""

    name: str | None = None
    """Optional checkpoint name for graph resumption via evaluate(from_={name: key})."""

    extras: Mapping[str, ParamInput] = field(default_factory=dict)
    """Extra parameters for key formatting or storage metadata (values or refs)."""
