"""Graph building utilities for daglite Intermediate Representation (IR)."""

from uuid import UUID

from daglite.exceptions import GraphConstructionError
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode


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

    return nodes
