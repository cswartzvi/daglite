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

from daglite.exceptions import ExecutionError

if TYPE_CHECKING:
    from daglite.engine import Backend
else:
    Backend = object

ParamKind = Literal["value", "ref", "sequence", "sequence_ref"]
NodeKind = Literal["task", "map", "choose", "loop", "artifact"]


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
    """Optional backend name or instance for this node."""

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

    @abc.abstractmethod
    def dependencies(self) -> set[UUID]:
        """
        IDs of nodes that the current node depends on (its direct predecessors).

        Note that dependencies are derived from reference inputs only - value inputs are **not**
        considered dependencies.
        """
        ...

    @abc.abstractmethod
    def submit(
        self, backend: Backend, values: Mapping[UUID, Any]
    ) -> Future[Any] | list[Future[Any]]:
        """
        Submit this node's work to the backend.

        Args:
            backend (Backend): Backend instance to use for execution.
            values (Mapping[UUID, Any]): Mapping from node IDs to their computed values.

        Returns:
            - TaskNode: Single Future[T]
            - MapTaskNode: list[Future[T]]
        """
        ...


class GraphBuilder(Protocol):
    """Protocol for building graph Intermediate Representation (IR) components from tasks."""

    @cached_property
    @abc.abstractmethod
    def id(self) -> UUID:
        """Unique identifier for the graph node produced by this builder."""

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


@dataclass
class GraphBuildContext:
    """Context for building graph IR components."""

    nodes: dict[UUID, GraphNode]
"""Function that visits a GraphBuilder and returns its node ID."""


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

    def resolve(self, values: Mapping[UUID, Any]) -> Any:
        """
        Resolves this input to a scalar value.

        Args:
            values (Mapping[str, Any]): Mapping from node IDs to their computed values.

        Returns:
            Any: Resolved scalar value.
        """
        if self.kind == "value":
            return self.value
        if self.kind == "ref":
            assert self.ref is not None
            return values[self.ref]

        raise ExecutionError(
            f"Cannot resolve parameter of kind '{self.kind}' as a scalar value. "
            f"Expected 'value' or 'ref', but got '{self.kind}'. "
            f"This may indicate an internal error in graph construction."
        )

    def resolve_sequence(self, values: Mapping[UUID, Any]) -> Sequence[Any]:
        """
        Resolves this input to a sequence value.

        Args:
            values (Mapping[str, Any]): Mapping from node IDs to their computed values.

        Returns:
            Sequence[Any]: Resolved sequence value.
        """
        if self.kind == "sequence":
            return list(self.value)  # type: ignore
        if self.kind == "sequence_ref":
            assert self.ref is not None
            return list(values[self.ref])
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
