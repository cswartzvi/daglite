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
from typing import Any
from uuid import UUID

from pluggy import HookRelay
from typing_extensions import override

from daglite._typing import NodeKind
from daglite._typing import ParamKind
from daglite._typing import Submission
from daglite.backends.base import Backend
from daglite.datasets.store import DatasetStore
from daglite.exceptions import GraphError


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

    output_configs: tuple[NodeOutputConfig, ...] = field(default=(), kw_only=True)
    """Output save/checkpoint configurations for this node."""

    hidden: bool = field(default=False, kw_only=True)
    """If True, this node is an implementation detail and should be elided from
    user-facing visualizations and trace output."""

    def __post_init__(self) -> None:
        # This is unlikely to happen given timeout is checked at task level, but just in case
        assert self.timeout is None or self.timeout >= 0, "Timeout must be non-negative"

    @property
    @abc.abstractmethod
    def kind(self) -> NodeKind:
        """Returns the kind of this graph node (e.g., 'task', 'map', etc.)."""
        pass

    @property
    def metadata(self) -> NodeMetadata:
        """Converts this graph node to its metadata representation for external use."""
        return NodeMetadata(
            id=self.id,
            name=self.name,
            kind=self.kind,
            description=self.description,
            backend_name=self.backend_name,
            key=self.key,
            hidden=self.hidden,
        )

    @abc.abstractmethod
    def get_dependencies(self) -> set[UUID]:
        """
        Returns the IDs of nodes that the current node depends on (its direct predecessors).

        Each node implementation determines its own dependencies based on its internal structure.
        """
        ...

    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> BaseGraphNode:
        """
        Returns a copy of this node with all references remapped according to *id_mapping*.

        Subclasses that hold reference-bearing fields (`kwargs`, `fixed_kwargs`, `source_id`, etc.)
        **must** override this method to remap those fields.  The base implementation is a no-op
        that returns `self` unchanged.

        Args:
            id_mapping: Mapping from old node IDs to their replacement IDs.

        Returns:
            A new node with updated references, or `self` if nothing changed.
        """
        return self  # pragma: no cover – concrete subclasses always override

    @abc.abstractmethod
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        """
        Executes the node on the specified backend.

        Args:
            backend: Backend where node will be executed.
            completed_nodes: Mapping from node IDs to their computed results.
            hooks: Pluggy relay for preforming hook calls.

        Returns:
            Materialized result of the node execution.
        """
        ...


@dataclass(frozen=True)
class PrepareCollectNode(BaseGraphNode, abc.ABC):
    """
    Base class for nodes that follow the prepare → submit → gather → collect pattern.

    Subclasses implement `_prepare` to build backend submissions and `_collect` to post-process
    gathered results.  The default `execute` method orchestrates the full flow.  Subclasses may
    still override `execute` if they need coordinator-side hooks.
    """

    @abc.abstractmethod
    def _prepare(self, completed_nodes: Mapping[UUID, Any]) -> list[Submission]:
        """
        Returns parameterless async callables ready for backend submission.

        Args:
            completed_nodes: Mapping from node IDs to their computed results.
        """
        ...

    @abc.abstractmethod
    def _collect(self, results: list[Any]) -> Any:
        """
        Post-processed results gathered from execution.

        Args:
            results: List of raw results returned from the backend execution.

        Returns:
            Processed result for this node.
        """
        ...

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        import asyncio

        submissions = self._prepare(completed_nodes)
        futures = [backend.submit(fn, timeout=self.timeout) for fn in submissions]
        results = await asyncio.gather(*futures)
        return self._collect(results)


@dataclass(frozen=True)
class NodeMetadata:
    """Metadata for a compiled graph IR node."""

    id: UUID
    """Unique identifier of this graph IR node."""

    name: str
    """Human-readable name of this graph IR node."""

    kind: NodeKind
    """ (e.g., 'task', 'map', etc.)."""

    description: str | None = field(default=None, kw_only=True)
    """Optional human-readable description of this graph IR node."""

    backend_name: str | None = field(default=None, kw_only=True)
    """Name of backend used to execute this graph IR node."""

    key: str | None = field(default=None, kw_only=True)
    """Optional key identifying this specific node in the IR graph."""

    hidden: bool = field(default=False, kw_only=True)
    """If True, this node is an implementation detail and should be elided from
    user-facing visualizations and trace output."""


@dataclass(frozen=True)
class NodeInput:
    """Input parameter representation (scalar and referential) for nodes in the IR graph."""

    _kind: ParamKind
    """Kind of this parameter input, determining how it should be resolved."""

    value: Any | None = None
    """Concrete value for 'value' and 'sequence' kinds. Must be None for 'ref' kinds."""

    reference: UUID | None = None
    """Reference node ID for 'ref' and 'sequence_ref' kinds. Must be None for 'value' kinds."""

    def __post_init__(self) -> None:
        if self._kind in ("value", "sequence"):
            if self.reference is not None:
                raise GraphError(f"NodeInput kind '{self._kind}' must not have a ref ID.")
        elif self._kind in ("ref", "sequence_ref"):
            if self.reference is None:
                raise GraphError(f"NodeInput kind '{self._kind}' requires a ref ID.")
            if self.value is not None:
                raise GraphError(f"NodeInput kind '{self._kind}' must not have a value.")
        else:  # pragma no cover
            raise GraphError(f"Unknown NodeInput kind: '{self._kind}'.")

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
                raise GraphError(f"Unknown NodeInput kind: '{self._kind}'. ")

    @classmethod
    def from_value(cls, v: Any) -> NodeInput:
        """Creates a NodeInput from a concrete value."""
        return cls(_kind="value", value=v)

    @classmethod
    def from_ref(cls, node_id: UUID) -> NodeInput:
        """Creates a NodeInput that references another node's output."""
        return cls(_kind="ref", reference=node_id)

    @classmethod
    def from_sequence(cls, vals: Sequence[Any]) -> NodeInput:
        """Creates a NodeInput from a concrete sequence value."""
        return cls(_kind="sequence", value=list(vals))

    @classmethod
    def from_sequence_ref(cls, node_id: UUID) -> NodeInput:
        """Creates an NodeInput that references another node's sequence output."""
        return cls(_kind="sequence_ref", reference=node_id)


@dataclass(frozen=True)
class NodeOutputConfig:
    """
    Configuration for saving or checkpointing a task output.

    Outputs can be saved with a storage key and optional checkpoint name for resumption.
    Extra parameters (as NodeInputs) can be included for formatting or metadata.
    """

    key: str
    """Storage key template with {param} placeholders for formatting."""

    store: DatasetStore | None = None
    """Dataset store where this output should be saved (None uses settings default)."""

    name: str | None = None
    """Optional checkpoint name for graph resumption via evaluate(from_={name: key})."""

    format: str | None = None
    """Optional serialization format hint (e.g., 'pickle', 'json', etc.)."""

    dependencies: Mapping[str, NodeInput] = field(default_factory=dict)
    """Parameter dependencies for this output, used for key formatting"""

    options: dict[str, Any] = field(default_factory=dict)
    """Additional options passed to the Dataset's save method."""
