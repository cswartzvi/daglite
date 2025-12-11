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


def optimize_graph(
    nodes: dict[UUID, GraphNode], root_id: UUID
) -> tuple[dict[UUID, GraphNode], dict[UUID, UUID]]:
    """
    Optimize the graph by creating composite nodes for chains.

    Transformations:
    - Linear TaskNode chains → CompositeTaskNode
    - Linear MapTaskNode chains → CompositeMapTaskNode

    Args:
        nodes: Original node dictionary
        root_id: ID of the root node

    Returns:
        Tuple of (optimized nodes dict, mapping from original IDs to composite IDs)
    """
    successors = _build_successors(nodes)
    grouped_nodes: set[UUID] = set()
    id_mapping: dict[UUID, UUID] = {}

    task_composites = _detect_task_chains(nodes, successors, grouped_nodes, id_mapping)
    map_composites = _detect_map_chains(nodes, successors, grouped_nodes, id_mapping)

    optimized: dict[UUID, GraphNode] = {}
    optimized.update(task_composites)
    optimized.update(map_composites)

    for nid, node in nodes.items():
        if nid not in grouped_nodes:
            optimized[nid] = node

    return optimized, id_mapping


def _build_successors(nodes: dict[UUID, GraphNode]) -> dict[UUID, set[UUID]]:
    """
    Build successor mapping for the graph.

    Creates a reverse dependency map: for each node, which nodes depend on it?
    This is the inverse of node.dependencies() and is essential for chain detection
    because we need to verify that nodes have exactly one successor to form a chain.

    Args:
        nodes: Dictionary mapping node IDs to GraphNode instances

    Returns:
        Dictionary mapping each node ID to the set of nodes that depend on it.
        Nodes with no successors are omitted from the result.

    Example:
        If node B depends on A, and node C depends on A:
        successors[A] = {B, C}
    """
    successors: dict[UUID, set[UUID]] = defaultdict(set)
    for nid, node in nodes.items():
        for dep_id in node.dependencies():
            successors[dep_id].add(nid)
    return dict(successors)


def _detect_task_chains(
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
    id_mapping: dict[UUID, UUID],
) -> dict[UUID, CompositeTaskNode]:
    """
    Detect and create CompositeTaskNode for TaskNode chains.

    Scans the graph for linear chains of TaskNode objects that can be grouped
    into composite nodes for more efficient execution. A valid chain must:
    - Contain only TaskNode instances (not MapTaskNode)
    - Have all nodes using the same backend
    - Be strictly linear (no diamonds, no multiple successors)
    - Contain at least 2 nodes

    Args:
        nodes: All nodes in the graph
        successors: Mapping from node ID to its successors
        grouped_nodes: Set tracking which nodes are already in composites
                      (mutated to include newly grouped nodes)
        id_mapping: Maps original node IDs to their composite IDs
                   (mutated to track new groupings)

    Returns:
        Dictionary of newly created composite nodes, keyed by composite ID

    Side Effects:
        - Mutates grouped_nodes to mark consumed nodes
        - Mutates id_mapping to track original_id → composite_id relationships

    Note:
        We iterate over list(nodes.keys()) rather than nodes.keys() directly
        because grouped_nodes changes during iteration, and we want a stable
        iteration order.
    """
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
            for chain_id in chain_ids:
                id_mapping[chain_id] = composite.id

    return composites


def _detect_task_chain_starting_at(
    start_id: UUID,
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
) -> list[UUID] | None:
    r"""
    Detect a linear TaskNode chain starting at the given node.

    Uses a greedy forward traversal algorithm to find the longest possible chain
    starting from start_id. The chain extends as long as:
    - Current node has exactly one successor
    - Successor is a TaskNode (not already grouped)
    - Successor uses the same backend
    - Successor depends on current node in the chain
    - No diamond patterns (successor must depend on only one chain node)

    Args:
        start_id: Node ID to start chain detection from
        nodes: All nodes in the graph
        successors: Mapping from node ID to its successors
        grouped_nodes: Set of nodes already consumed by other composites

    Returns:
        List of node IDs forming the chain (length >= 2), or None if no valid
        chain exists. The list is in execution order.

    Algorithm:
        1. Start with chain = [start_id]
        2. Check if current node has exactly one successor
        3. Verify successor is valid (TaskNode, same backend, not grouped)
        4. Check dependency structure to prevent diamonds:
           - Count how many nodes in chain the successor depends on
           - If > 1, we have a diamond → stop
        5. Add successor to chain and continue
        6. Return chain if len >= 2, else None

    Example Diamond (not allowed):
           A
          / \
         B   C
          \ /
           D
        If chain = [A, B] and we check D, we find D depends on both A and B,
        so we stop the chain at B.
    """
    chain = [start_id]
    current = start_id
    current_node = nodes[start_id]

    while True:
        succ_ids = successors.get(current, set())

        if len(succ_ids) != 1:
            break  # Only one successor allowed

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
    """
    Create a CompositeTaskNode from a chain of TaskNode IDs.

    Constructs the composite by building ChainLink metadata for each node in the
    chain. The links capture:
    - flow_param: Which parameter receives the previous node's output
    - external_params: Parameters from outside the chain (literals, fixed, or refs)

    Args:
        chain_ids: Ordered list of node IDs forming the chain
        nodes: All nodes in the graph (for lookup)

    Returns:
        A new CompositeTaskNode with generated ID and descriptive name

    Link Construction:
        - First node (idx=0):
          * flow_param = None (receives initial inputs)
          * external_params = {} (all inputs are exposed)
        - Later nodes (idx>0):
          * flow_param = name of parameter that refs previous node
          * external_params = all other parameters (literals, fixed, external refs)

    The resulting composite's inputs() will be:
        - All inputs from the first node
        - External ref parameters from later nodes (not literals)

    Example:
        Chain: [A, B, C] where:
        - A(x, y)
        - B(prev, z) where prev refs A
        - C(prev, w) where prev refs B, w refs external node D

        ChainLinks:
        - Link 0: node=A, flow_param=None, external_params={}
        - Link 1: node=B, flow_param="prev", external_params={"z": ...}
        - Link 2: node=C, flow_param="prev", external_params={"w": ref(D)}

        Composite.inputs() = [("x", ...), ("y", ...), ("w", ref(D))]
    """
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
    id_mapping: dict[UUID, UUID],
) -> dict[UUID, CompositeMapTaskNode]:
    """
    Detect and create CompositeMapTaskNode for MapTaskNode chains.

    Similar to _detect_task_chains but for map operations. A valid map chain must:
    - Contain only MapTaskNode instances
    - Have all nodes using the same backend
    - Each successor maps over exactly one parameter
    - That parameter must reference the previous map node's output
    - Contain at least 2 nodes

    This optimization transforms:
        for x in source: map1(x)
        for y in map1_results: map2(y)
        for z in map2_results: map3(z)

    Into:
        for x in source: map3(map2(map1(x)))

    Reducing backend submissions from (iterations × chain_length) to just (iterations).

    Args:
        nodes: All nodes in the graph
        successors: Mapping from node ID to its successors
        grouped_nodes: Set tracking which nodes are already in composites
                      (mutated to include newly grouped nodes)
        id_mapping: Maps original node IDs to their composite IDs
                   (mutated to track new groupings)

    Returns:
        Dictionary of newly created composite map nodes, keyed by composite ID

    Side Effects:
        - Mutates grouped_nodes to mark consumed nodes
        - Mutates id_mapping to track original_id → composite_id relationships
    """
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
            for chain_id in chain_ids:
                id_mapping[chain_id] = composite.id

    return composites


def _detect_map_chain_starting_at(
    start_id: UUID,
    nodes: dict[UUID, GraphNode],
    successors: dict[UUID, set[UUID]],
    grouped_nodes: set[UUID],
) -> list[UUID] | None:
    """
    Detect a linear MapTaskNode chain starting at the given node.

    Uses greedy forward traversal to find the longest map chain. The chain extends
    as long as:
    - Current node has exactly one successor
    - Successor is a MapTaskNode (not already grouped)
    - Successor uses the same backend
    - Successor maps over exactly one parameter (no fan-out/fan-in)
    - That parameter references the current map node's output

    Args:
        start_id: Node ID to start chain detection from
        nodes: All nodes in the graph
        successors: Mapping from node ID to its successors
        grouped_nodes: Set of nodes already consumed by other composites

    Returns:
        List of node IDs forming the chain (length >= 2), or None if no valid
        chain exists. The list is in execution order.

    Key Constraint:
        We require len(next_node.mapped_kwargs) == 1 to ensure the successor is
        a simple map over the previous result, not a complex fan-out/fan-in pattern.
        This keeps the optimization semantics simple and correct.

    Example Valid Chain:
        source.map(f).map(g).map(h)
        Each .map() has exactly one mapped parameter referencing the previous result.

    Example Invalid (multiple mapped sources):
        a = source1.map(f)
        b = source2.map(g)
        c = Pipeline.map_zip(h, a=a, b=b)  # Maps over both a and b
        Cannot chain c with either a or b.
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
    """
    Create a CompositeMapTaskNode from a chain of MapTaskNode IDs.

    Constructs the composite by building ChainLink metadata for each map node.
    The resulting composite executes the full chain per iteration rather than
    level-by-level.

    Args:
        chain_ids: Ordered list of node IDs forming the chain
        nodes: All nodes in the graph (for lookup)

    Returns:
        A new CompositeMapTaskNode with generated ID and descriptive name

    Link Construction:
        - First node (idx=0):
          * flow_param = None (receives mapped source value)
          * external_params = fixed_kwargs from source map
        - Later nodes (idx>0):
          * flow_param = name of the mapped parameter (refs previous map)
          * external_params = fixed_kwargs (parameters not being mapped)

    The resulting composite's inputs() will be:
        - All mapped_kwargs from source_map (defines iteration space)
        - All fixed_kwargs from source_map
        - All external fixed_kwargs from later nodes in chain

    Example:
        Chain: [map1, map2, map3] where:
        - map1 maps f(x, y=10) over source
        - map2 maps g(prev, z=20) over map1 results
        - map3 maps h(prev, w=external) over map2 results

        ChainLinks:
        - Link 0: node=map1, flow_param=None, external_params={"y": 10}
        - Link 1: node=map2, flow_param="prev", external_params={"z": 20}
        - Link 2: node=map3, flow_param="prev", external_params={"w": ref(external)}

        Execution per iteration:
        for x in source:
            result = f(x, y=10)
            result = g(result, z=20)
            result = h(result, w=external_value)
            yield result
    """
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
