"""Graph optimization - detects and groups linear node chains."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID
from uuid import uuid4

from daglite.graph.base import GraphNode
from daglite.graph.base import ParamInput
from daglite.graph.composite import ChainLink
from daglite.graph.composite import CompositeMapTaskNode
from daglite.graph.composite import CompositeTaskNode
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode


def optimize_graph(nodes: dict[UUID, GraphNode], root_id: UUID) -> dict[UUID, GraphNode]:
    """
    Optimize the graph by creating composite nodes for chains.

    Transformations:
    - Linear TaskNode chains → CompositeTaskNode
    - Linear MapTaskNode chains → CompositeMapTaskNode

    Args:
        nodes: Original node dictionary
        root_id: ID of the root node

    Returns:
        Optimized node dictionary with composite nodes
    """
    successors = _build_successors(nodes)
    grouped_nodes: set[UUID] = set()
    task_composites = _detect_task_chains(nodes, successors, grouped_nodes)
    map_composites = _detect_map_chains(nodes, successors, grouped_nodes)

    optimized: dict[UUID, GraphNode] = {}
    optimized.update(task_composites)
    optimized.update(map_composites)

    for nid, node in nodes.items():
        if nid not in grouped_nodes:
            optimized[nid] = node

    return optimized


def _build_successors(nodes: dict[UUID, GraphNode]) -> dict[UUID, set[UUID]]:
    """Build successor mapping for the graph."""
    successors: dict[UUID, set[UUID]] = defaultdict(set)
    for nid, node in nodes.items():
        for dep_id in node.dependencies():
            successors[dep_id].add(nid)
    return dict(successors)


def _detect_task_chains(
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
) -> dict[UUID, CompositeTaskNode]:
    """Detect and create CompositeTaskNode for TaskNode chains."""
    composites: dict[UUID, CompositeTaskNode] = {}

    for nid in list(nodes.keys()):
        if nid in grouped_nodes:
            continue

        if not isinstance(nodes[nid], TaskNode):
            continue

        chain_ids = _detect_task_chain_starting_at(nid, nodes, successors, grouped_nodes)

        if chain_ids and len(chain_ids) >= 2:
            composite = _create_composite_task_node(chain_ids, nodes)
            composites[composite.id] = composite
            grouped_nodes.update(chain_ids)

    return composites


def _detect_task_chain_starting_at(
    start_id: UUID,
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
) -> list[UUID] | None:
    """
    Detect a linear TaskNode chain starting at the given node.

    Returns list of node IDs in the chain, or None if no chain found.
    """
    chain = [start_id]
    current = start_id
    current_node = nodes[start_id]

    while True:
        succ_ids = successors.get(current, set())

        if len(succ_ids) != 1:
            break # Only one successor allowed

        next_id = next(iter(succ_ids))

        if next_id in grouped_nodes:
            break

        next_node = nodes[next_id]

        if not isinstance(next_node, TaskNode):
            break

        if next_node.backend != current_node.backend:
            break

        next_deps = next_node.dependencies()
        if current not in next_deps:
            break

        chain_deps = [dep for dep in next_deps if dep in chain]
        if len(chain_deps) != 1:
            break  # Multiple dependencies within chain (diamond pattern)

        chain.append(next_id)
        current = next_id
        current_node = next_node

    return chain if len(chain) >= 2 else None


def _create_composite_task_node(
    chain_ids: list[UUID], nodes: dict[UUID, GraphNode]
) -> CompositeTaskNode:
    """Create a CompositeTaskNode from a chain of TaskNode IDs."""
    chain_links: list[ChainLink] = []

    for idx, nid in enumerate(chain_ids):
        node = nodes[nid]
        assert isinstance(node, TaskNode)

        if idx == 0:
            flow_param = None
        else:
            prev_id = chain_ids[idx - 1]
            flow_param = None
            for param_name, param_input in node.inputs():
                if param_input.is_ref and param_input.ref == prev_id:
                    flow_param = param_name
                    break

        external_params: dict[str, ParamInput] = {}
        for param_name, param_input in node.inputs():
            if idx == 0:
                continue  # First node inputs are initial inputs
            else:
                if param_name != flow_param:
                    external_params[param_name] = param_input

        chain_links.append(
            ChainLink(
                node=node,
                position=idx,
                flow_param=flow_param,
                external_params=external_params,
            )
        )

    first_node = nodes[chain_ids[0]]
    last_node = nodes[chain_ids[-1]]

    return CompositeTaskNode(
        id=uuid4(),
        name=f"{first_node.name}→…→{last_node.name}",
        description=f"Composite chain of {len(chain_links)} nodes",
        backend=first_node.backend,
        chain=tuple(chain_links),
    )


def _detect_map_chains(
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
) -> dict[UUID, CompositeMapTaskNode]:
    """Detect and create CompositeMapTaskNode for MapTaskNode chains."""
    composites: dict[UUID, CompositeMapTaskNode] = {}

    for nid in list(nodes.keys()):
        if nid in grouped_nodes:
            continue

        if not isinstance(nodes[nid], MapTaskNode):
            continue

        chain_ids = _detect_map_chain_starting_at(nid, nodes, successors, grouped_nodes)

        if chain_ids and len(chain_ids) >= 2:
            composite = _create_composite_map_node(chain_ids, nodes)
            composites[composite.id] = composite
            grouped_nodes.update(chain_ids)

    return composites


def _detect_map_chain_starting_at(
    start_id: UUID,
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
) -> list[UUID] | None:
    """
    Detect a linear MapTaskNode chain starting at the given node.

    Returns list of node IDs in the chain, or None if no chain found.
    """
    chain = [start_id]
    current = start_id
    current_node = nodes[start_id]

    while True:
        succ_ids = successors.get(current, set())

        if len(succ_ids) != 1:
            break

        next_id = next(iter(succ_ids))

        if next_id in grouped_nodes:
            break

        next_node = nodes[next_id]

        if not isinstance(next_node, MapTaskNode):
            break

        if next_node.backend != current_node.backend:
            break

        if len(next_node.mapped_kwargs) != 1:
            break  # Multiple mapped sources

        mapped_param = list(next_node.mapped_kwargs.values())[0]
        if not (mapped_param.is_ref and mapped_param.ref == current):
            break

        chain.append(next_id)
        current = next_id
        current_node = next_node

    return chain if len(chain) >= 2 else None


def _create_composite_map_node(
    chain_ids: list[UUID], nodes: dict[UUID, GraphNode]
) -> CompositeMapTaskNode:
    """Create a CompositeMapTaskNode from a chain of MapTaskNode IDs."""
    chain_links: list[ChainLink] = []
    source_map = nodes[chain_ids[0]]
    assert isinstance(source_map, MapTaskNode)

    for idx, nid in enumerate(chain_ids):
        node = nodes[nid]
        assert isinstance(node, MapTaskNode)

        if idx == 0:
            flow_param = None
        else:
            flow_param = list(node.mapped_kwargs.keys())[0]

        external_params: dict[str, ParamInput] = dict(node.fixed_kwargs)

        chain_links.append(
            ChainLink(
                node=node,
                position=idx,
                flow_param=flow_param,
                external_params=external_params,
            )
        )

    first_node = nodes[chain_ids[0]]
    last_node = nodes[chain_ids[-1]]

    return CompositeMapTaskNode(
        id=uuid4(),
        name=f"{first_node.name}→…→{last_node.name}",
        description=f"Composite map chain of {len(chain_links)} iterations",
        backend=first_node.backend,
        source_map=source_map,
        chain=tuple(chain_links),
    )
