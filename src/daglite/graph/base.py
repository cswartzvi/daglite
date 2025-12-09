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
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import UUID

from typing_extensions import final

from daglite.exceptions import ExecutionError

if TYPE_CHECKING:
    from daglite.engine import Backend
else:
    Backend = object

ParamKind = Literal["value", "ref", "sequence", "sequence_ref"]
NodeKind = Literal["task", "map", "artifact"]


class GraphBuilder(Protocol):
    """Protocol for building graph Intermediate Representation (IR) components from tasks."""

    @cached_property
    @abc.abstractmethod
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
    def to_graph(self) -> GraphNode:
        """
        Convert this builder into a GraphNode.

        All dependencies will have their IDs assigned before this is called,
        so implementations can safely access dependency.id.

        Returns:
            GraphNode: The constructed graph node.
        """
        ...


@dataclass(frozen=True)
class GraphNode(abc.ABC):
    """Represents a node in the compiled graph Intermediate Representation (IR)."""

    id: UUID
    """Unique identifier for this node."""

    name: str
    """Human-readable name for this node."""

    description: str | None
    """Optional human-readable description for this node."""

    backend: str | Backend | None
    """Optional backend specified by the user for this node."""

    @cached_property
    @abc.abstractmethod
    def kind(self) -> NodeKind:
        """Describes the kind of this graph node."""
        ...

    @abc.abstractmethod
    def inputs(self) -> list[tuple[str, ParamInput]]:
        """
        Pairs of parameter names and their corresponding graph IR inputs.

        Note that this includes **both** value and reference inputs.
        """
        ...

    @final
    def dependencies(self) -> set[UUID]:
        """
        IDs of nodes that the current node depends on (its direct predecessors).

        Note that dependencies are derived from reference inputs only - value inputs are **not**
        considered dependencies.
        """
        return {p.ref for _, p in self.inputs() if p.is_ref and p.ref is not None}

    @abc.abstractmethod
    def submit(
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
    ) -> Future[Any] | list[Future[Any]]:
        """
        Submit this node's work to the backend.

        Args:
            resolved_backend: Backend instance resolved by the engine.
            resolved_inputs: Pre-resolved parameter inputs for this node.

        Returns:
            Single Future for TaskNode, list of Futures for MapTaskNode.
        """
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
