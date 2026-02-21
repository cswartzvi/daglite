"""Graph building utilities for daglite Intermediate Representation (IR)."""

from __future__ import annotations

import abc
from typing import Protocol
from uuid import UUID

from daglite.exceptions import GraphError
from daglite.graph.nodes.base import BaseGraphNode


class NodeBuilder(Protocol):
    """Protocol for building graph Intermediate Representation (IR) nodes from tasks."""

    @property
    def id(self) -> UUID:
        """Returns the unique identifier for this builder's graph node."""
        ...

    @abc.abstractmethod
    def get_upstream_builders(self) -> list[NodeBuilder]:
        """
        Returns all graph builders that this builder depends on.

        Returns:
            list[NodeBuilder]: List of builders this node depends on.
        """
        ...

    @abc.abstractmethod
    def build_node(self) -> BaseGraphNode:
        """
        Builds and configures a graph node.

        Returns:
            BaseGraphNode: A configured graph node instance.
        """
        ...


def build_graph_multi(roots: list[NodeBuilder]) -> dict[UUID, BaseGraphNode]:
    """
    Compile multiple `NodeBuilder` roots into a shared dict of GraphNodes keyed by node id.

    Uses an iterative post-order traversal to avoid stack overflow on deep chains.
    Common ancestors shared across multiple roots are visited exactly once.

    Raises:
        GraphError: If a circular dependency is detected.
    """
    nodes: dict[UUID, BaseGraphNode] = {}
    visiting: set[UUID] = set()
    stack: list[tuple[NodeBuilder, bool]] = []
    seen_root_ids: set[UUID] = set()

    for root in reversed(roots):
        if root.id not in seen_root_ids:
            seen_root_ids.add(root.id)
            stack.append((root, False))

    while stack:
        node_like, deps_collected = stack.pop()
        node_id = node_like.id

        # Skip if already processed via another root
        if node_id in nodes:
            continue

        if not deps_collected:
            # First visit: check for cycles and collect upstream builders
            if node_id in visiting:
                raise GraphError(
                    f"Circular dependency detected: node '{node_id}' references itself "
                    "through its dependencies"
                )

            visiting.add(node_id)
            upstream_builders = node_like.get_upstream_builders()
            stack.append((node_like, True))

            # Push upstream builders onto stack (in reverse so they process in order)
            for upstream_builder in reversed(upstream_builders):
                if upstream_builder.id not in nodes:
                    stack.append((upstream_builder, False))
        else:
            # Second visit: all upstream builders processed, now build this node
            nodes[node_id] = node_like.build_node()
            visiting.discard(node_id)

    return nodes


def build_graph(root: NodeBuilder) -> dict[UUID, BaseGraphNode]:
    """
    Compile a `NodeBuilder` tree into a dict of GraphNodes keyed by node id.

    Uses an iterative post-order traversal to avoid stack overflow on deep chains.

    Raises:
        GraphError: If a circular dependency is detected.
    """
    return build_graph_multi([root])
