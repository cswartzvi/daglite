"""Graph building utilities for DAGLite Intermediate Representation (IR)."""

from uuid import UUID

from daglite.exceptions import GraphConstructionError
from daglite.graph.base import GraphBuildContext
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode


def build_graph(root: GraphBuilder) -> dict[UUID, GraphNode]:
    """
    Compile a GraphBuilder tree into a dict of GraphNodes keyed by node id.

    Raises:
        GraphConstructionError: If a circular dependency is detected.
    """
    ctx = GraphBuildContext(nodes={})
    visiting: set[UUID] = set()  # Track nodes in current recursion path

    def _visit(node_like: GraphBuilder) -> UUID:
        """Visit a GraphBuilder node and add its GraphNode to the context."""
        node_id = node_like.id
        if node_id in ctx.nodes:
            return node_id
        if node_id in visiting:
            raise GraphConstructionError(
                f"Circular dependency detected: node {node_id} references itself "
                "through its dependencies"
            )
        visiting.add(node_id)
        node = node_like.to_graph(ctx, _visit)
        ctx.nodes[node_id] = node
        visiting.remove(node_id)
        return node_id

    _visit(root)
    return ctx.nodes
