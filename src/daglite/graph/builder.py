from dataclasses import dataclass
from uuid import UUID

from daglite.graph.base import GraphBuildContext
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode


def build_graph(root: GraphBuilder) -> dict[UUID, GraphNode]:
    """Compile a GraphBuilder tree into a dict of GraphNodes keyed by node id."""
    ctx = GraphBuildContext(nodes={})

    def _visit(node_like: GraphBuilder) -> UUID:
        """Visit a GraphBuilder node and add its GraphNode to the context."""
        node_id = node_like.id
        if node_id in ctx.nodes:
            return node_id
        node = node_like.to_graph(ctx, _visit)
        ctx.nodes[node_id] = node
        return node_id

    _visit(root)
    return ctx.nodes
