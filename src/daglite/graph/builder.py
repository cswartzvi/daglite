"""Graph building utilities for daglite Intermediate Representation (IR)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from uuid import UUID

from daglite.exceptions import GraphConstructionError
from daglite.graph.base import ExecutionGuard
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode

if TYPE_CHECKING:
    from daglite.futures import ConditionalFuture


def build_graph(root: GraphBuilder) -> dict[UUID, GraphNode]:
    """
    Compile a GraphBuilder tree into a dict of GraphNodes keyed by node id.

    Uses an iterative post-order traversal to avoid stack overflow on deep chains.

    Raises:
        GraphConstructionError: If a circular dependency is detected.
    """
    nodes: dict[UUID, GraphNode] = {}
    visiting: set[UUID] = set()
    stack: list[tuple[GraphBuilder, bool]] = [(root, False)]

    while stack:
        node_like, deps_collected = stack.pop()
        node_id = node_like.id

        # Skip if already processed (defensive check)
        if node_id in nodes:  # pragma: no cover
            continue

        if not deps_collected:
            # First visit: check for cycles and collect dependencies
            if node_id in visiting:
                raise GraphConstructionError(
                    f"Circular dependency detected: node '{node_id}' references itself "
                    "through its dependencies"
                )

            visiting.add(node_id)
            deps = node_like.get_dependencies()
            stack.append((node_like, True))

            # Push dependencies onto stack (in reverse so they process in order)
            for dep in reversed(deps):
                if dep.id not in nodes:
                    stack.append((dep, False))
        else:
            # Second visit: all dependencies processed, now build this node
            node = node_like.to_graph()
            nodes[node_id] = node
            visiting.discard(node_id)

            # Special handling for ConditionalFuture: attach guards to branch nodes
            _attach_conditional_guards(node_like, nodes)

    return nodes


def _attach_conditional_guards(node_like: GraphBuilder, nodes: dict[UUID, GraphNode]) -> None:
    """
    Attach execution guards to conditional branch nodes.

    When we encounter a ConditionalFuture, we need to:
    1. Get the then_branch and else_branch node IDs
    2. Clone those nodes with ExecutionGuards attached
    3. Replace them in the nodes dict

    This ensures lazy evaluation: only the selected branch executes at runtime.
    """
    from daglite.futures import ConditionalFuture

    if not isinstance(node_like, ConditionalFuture):
        return

    condition_id = node_like.condition.id
    then_id = node_like.then_branch.id
    else_id = node_like.else_branch.id

    # Attach guard to then branch (execute if condition is True)
    if then_id in nodes:
        then_node = nodes[then_id]
        if then_node.execution_guard is None:  # Don't override existing guards
            nodes[then_id] = dataclasses.replace(
                then_node,
                execution_guard=ExecutionGuard(
                    condition_ref=condition_id,
                    expected_value=True,
                ),
            )

    # Attach guard to else branch (execute if condition is False)
    if else_id in nodes:
        else_node = nodes[else_id]
        if else_node.execution_guard is None:  # Don't override existing guards
            nodes[else_id] = dataclasses.replace(
                else_node,
                execution_guard=ExecutionGuard(
                    condition_ref=condition_id,
                    expected_value=False,
                ),
            )
