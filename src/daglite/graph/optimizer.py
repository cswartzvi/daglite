"""Graph optimization for daglite - identifies and groups linear chains of nodes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from uuid import UUID
from uuid import uuid4

from daglite.graph.base import BaseGraphNode
from daglite.graph.nodes import CompositeMapTaskNode
from daglite.graph.nodes import CompositeTaskNode
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode


def optimize_graph(
    graph: dict[UUID, BaseGraphNode], root_id: UUID, enable_optimization: bool = True
) -> dict[UUID, BaseGraphNode]:
    """
    Optimize a task graph by identifying and grouping linear chains of nodes.

    Linear chains are sequences of nodes where each node has exactly one predecessor
    and one successor. These chains are replaced with composite nodes that execute
    the entire sequence in a single backend submission, reducing coordination overhead.

    Args:
        graph: Dictionary mapping node IDs to graph nodes.
        root_id: ID of the root node in the graph.
        enable_optimization: Whether to perform optimization. If False, returns the graph unchanged.

    Returns:
        Optimized graph with linear chains replaced by composite nodes.

    Examples:
        Simple linear chain:
            a -> b -> c  =>  composite(a, b, c)

        Branching prevents grouping:
            a -> b -> c
                 ↓
                 d
            In this case, b cannot be grouped because it has multiple successors.

        Different backends prevent grouping:
            a[backend=thread] -> b[backend=process]
            These cannot be grouped due to different backend requirements.
    """
    if not enable_optimization or len(graph) <= 1:
        return graph

    predecessors: dict[UUID, set[UUID]] = defaultdict(set)
    successors: dict[UUID, set[UUID]] = defaultdict(set)

    for node_id, node in graph.items():
        deps = node.dependencies()
        for dep_id in deps:
            successors[dep_id].add(node_id)
            predecessors[node_id].add(dep_id)

    # Find chain heads: root nodes and nodes after branch points
    chain_heads: set[UUID] = set()
    for node_id in graph:
        preds = predecessors.get(node_id, set())
        if len(preds) == 0:
            chain_heads.add(node_id)
        elif len(preds) == 1:
            pred_id = next(iter(preds))
            if len(successors.get(pred_id, set())) > 1:
                chain_heads.add(node_id)

    chains: list[list[UUID]] = []
    visited_in_chain: set[UUID] = set()

    for head_id in chain_heads:
        if head_id in visited_in_chain:
            continue

        chain = [head_id]
        current_id = head_id
        visited_in_chain.add(head_id)

        while True:
            succs = successors.get(current_id, set())

            if len(succs) != 1:
                break

            next_id = next(iter(succs))

            if len(predecessors.get(next_id, set())) != 1:
                break

            current_node = graph[current_id]
            next_node = graph[next_id]

            if not _are_nodes_compatible(current_node, next_node):
                break

            chain.append(next_id)
            visited_in_chain.add(next_id)
            current_id = next_id

        if len(chain) >= 2:
            chains.append(chain)

    # If no chains found, return original graph
    if not chains:
        return graph

    optimized_graph: dict[UUID, BaseGraphNode] = {}
    nodes_in_composite: set[UUID] = set()

    for chain in chains:
        nodes_in_composite.update(chain)

    # Map internal node IDs to their composite node ID for dependency remapping
    node_to_composite: dict[UUID, UUID] = {}
    for chain in chains:
        composite_node = _create_composite_node(graph, chain)
        optimized_graph[composite_node.id] = composite_node

        for node_id in chain:
            node_to_composite[node_id] = composite_node.id

    for node_id, node in graph.items():
        if node_id not in nodes_in_composite:
            node_deps = node.dependencies()
            needs_remapping = any(dep_id in node_to_composite for dep_id in node_deps)

            if needs_remapping:
                node = _remap_node_dependencies(node, node_to_composite)

            optimized_graph[node_id] = node

    return optimized_graph


def _are_nodes_compatible(node1: BaseGraphNode, node2: BaseGraphNode) -> bool:
    """
    Check if two nodes can be grouped into a composite node.

    Nodes are compatible if:
    1. They use the same backend
    2. They are both TaskNodes (or CompositeTaskNodes)
    3. They don't have output configs (for simplicity in this initial implementation)

    Note: Map nodes are currently not optimized in chains, as they require more
    complex handling of sequence dependencies.
    """
    # Must use the same backend
    if node1.backend_name != node2.backend_name:
        return False

    # For simplicity, don't group nodes with output configs
    # This could be relaxed in future versions
    if node1.output_configs or node2.output_configs:
        return False

    # Both must be TaskNode (or CompositeTaskNode)
    # Map nodes are not currently optimized in chains
    node1_is_task = isinstance(node1, (TaskNode, CompositeTaskNode))
    node2_is_task = isinstance(node2, (TaskNode, CompositeTaskNode))

    if node1_is_task and node2_is_task:
        return True
    else:
        return False


def _remap_node_dependencies(
    node: BaseGraphNode, node_to_composite: dict[UUID, UUID]
) -> BaseGraphNode:
    """
    Create a new version of the node with dependencies remapped to composites.

    Args:
        node: The node to remap.
        node_to_composite: Mapping from internal node IDs to their composite IDs.

    Returns:
        A new node with remapped dependencies.
    """
    from daglite.graph.base import ParamInput

    if isinstance(node, TaskNode):
        new_kwargs = {}
        for name, param in node.kwargs.items():
            if param.is_ref and param.ref in node_to_composite:
                new_kwargs[name] = ParamInput.from_ref(node_to_composite[param.ref])
            else:
                new_kwargs[name] = param
        return replace(node, kwargs=new_kwargs)

    elif isinstance(node, CompositeTaskNode):
        new_nodes = tuple(_remap_node_dependencies(n, node_to_composite) for n in node.nodes)
        return replace(node, nodes=new_nodes)

    elif isinstance(node, MapTaskNode):
        new_fixed_kwargs = {}
        for name, param in node.fixed_kwargs.items():
            if param.is_ref and param.ref in node_to_composite:
                new_fixed_kwargs[name] = ParamInput.from_ref(node_to_composite[param.ref])
            else:
                new_fixed_kwargs[name] = param

        new_mapped_kwargs = {}
        for name, param in node.mapped_kwargs.items():
            if param.is_ref and param.ref in node_to_composite:
                new_mapped_kwargs[name] = ParamInput.from_sequence_ref(node_to_composite[param.ref])
            elif param.kind == "sequence_ref" and param.ref in node_to_composite:
                new_mapped_kwargs[name] = ParamInput.from_sequence_ref(node_to_composite[param.ref])
            else:
                new_mapped_kwargs[name] = param
        return replace(node, fixed_kwargs=new_fixed_kwargs, mapped_kwargs=new_mapped_kwargs)

    elif isinstance(node, CompositeMapTaskNode):
        new_nodes = tuple(_remap_node_dependencies(n, node_to_composite) for n in node.nodes)
        return replace(node, nodes=new_nodes)

    return node


def _create_composite_node(
    graph: dict[UUID, BaseGraphNode], chain: list[UUID]
) -> CompositeTaskNode | CompositeMapTaskNode:
    """
    Create a composite node from a chain of node IDs.

    Args:
        graph: The original graph.
        chain: List of node IDs in the chain (in execution order).

    Returns:
        A CompositeTaskNode or CompositeMapTaskNode containing the chain.
    """
    nodes_in_chain = [graph[node_id] for node_id in chain]
    first_node = nodes_in_chain[0]

    # Determine the type of composite node to create
    is_map_chain = isinstance(first_node, (MapTaskNode, CompositeMapTaskNode))

    # Flatten any existing composite nodes in the chain
    flattened_nodes: list[BaseGraphNode] = []
    for node in nodes_in_chain:
        if isinstance(node, (CompositeTaskNode, CompositeMapTaskNode)):
            flattened_nodes.extend(node.nodes)
        else:
            flattened_nodes.append(node)

    # Create composite node with a new ID
    composite_id = uuid4()

    # Generate a descriptive name
    node_names = [node.name for node in flattened_nodes]
    composite_name = f"composite({', '.join(node_names)})"

    # Use the first node's backend and timeout
    backend_name = first_node.backend_name
    timeout = first_node.timeout

    if is_map_chain:
        return CompositeMapTaskNode(
            id=composite_id,
            name=composite_name,
            backend_name=backend_name,
            timeout=timeout,
            description=f"Composite of {len(flattened_nodes)} map tasks",
            key=composite_name,
            output_configs=(),
            nodes=tuple(flattened_nodes),  # type: ignore[arg-type]
        )
    else:
        return CompositeTaskNode(
            id=composite_id,
            name=composite_name,
            backend_name=backend_name,
            timeout=timeout,
            description=f"Composite of {len(flattened_nodes)} tasks",
            key=composite_name,
            output_configs=(),
            nodes=tuple(flattened_nodes),  # type: ignore[arg-type]
        )
