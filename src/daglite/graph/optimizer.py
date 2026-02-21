"""
Graph optimization passes for the daglite IR.

The primary optimization is **composite node folding**: detecting linear chains
of nodes that share the same backend and collapsing them into a single
`CompositeTaskNode` or `CompositeMapTaskNode`. This reduces backend
submission overhead from O(chain_length) to O(1) for task chains and from
O(iterations × chain_length) to O(iterations) for map chains.

The optimizer runs between `build_graph()` and engine execution, producing
a rewritten graph and an `id_mapping` that the engine uses to alias
intermediate results for downstream dependency resolution.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID
from uuid import uuid4

from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.composite_node import ChainLink
from daglite.graph.nodes.composite_node import CompositeMapTaskNode
from daglite.graph.nodes.composite_node import CompositeTaskNode
from daglite.graph.nodes.composite_node import TerminalKind
from daglite.graph.nodes.map_node import MapTaskNode
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.reduce_node import ReduceNode
from daglite.graph.nodes.task_node import TaskNode


def optimize_graph(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Apply optimization passes to the graph IR.

    Currently performs composite node folding — collapsing linear chains of nodes sharing the same
    backend into `CompositeTaskNode` or `CompositeMapTaskNode` instances.

    Args:
        nodes: The original graph IR node dictionary.

    Returns:
        A tuple of `(optimized_nodes, id_mapping)` where:
        - `optimized_nodes` is the rewritten graph.
        - `id_mapping` maps original node IDs (that were folded into
          composites) to their composite node's ID. The **tail** node ID
          of each chain is stored as a key so that downstream nodes
          depending on the tail can find the composite's result.
    """
    nodes, id_mapping = _fold_task_chains(nodes)
    nodes, map_mapping = _fold_map_chains(nodes)
    id_mapping.update(map_mapping)
    return nodes, id_mapping


def _fold_task_chains(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Fold linear `TaskNode` chains into `CompositeTaskNode` instances.

    Returns the rewritten graph and an ID mapping from original tail IDs to composite IDs.
    """
    predecessors, successors = _build_adjacency(nodes)
    chains = _find_task_chains(nodes, predecessors, successors)

    if not chains:
        return nodes, {}

    result = dict(nodes)
    id_mapping: dict[UUID, UUID] = {}

    for chain_ids in chains:
        # Build chain links
        links: list[ChainLink] = []
        for i, nid in enumerate(chain_ids):
            node = nodes[nid]
            assert isinstance(node, TaskNode)

            if i == 0:
                # First link — no flow param, all params are external
                link = _build_chain_link(node, flow_param=None, predecessor_id=None)
            else:
                predecessor_id = chain_ids[i - 1]
                flow_param = _identify_flow_param(node.kwargs, predecessor_id)
                link = _build_chain_link(node, flow_param, predecessor_id)

            links.append(link)

        # Create composite node
        composite_id = uuid4()
        head = nodes[chain_ids[0]]
        tail_id = chain_ids[-1]

        composite = CompositeTaskNode(
            id=composite_id,
            name=f"composite({' → '.join(link.name for link in links)})",
            description=f"Composite of {len(links)} chained tasks",
            backend_name=_effective_backend(head),
            chain=tuple(links),
            timeout=_aggregate_timeout([link.timeout for link in links]),
        )

        # Remove chain nodes from result, add composite
        for nid in chain_ids:
            result.pop(nid, None)
        result[composite_id] = composite

        # Map tail ID → composite ID (downstream nodes depend on the tail)
        id_mapping[tail_id] = composite_id

        # Also map all interior IDs for completeness
        for nid in chain_ids[:-1]:
            id_mapping[nid] = composite_id

    # Remap dependencies in remaining nodes that referenced chain members
    result = _remap_dependencies(result, id_mapping)

    return result, id_mapping


def _fold_map_chains(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Fold `MapTaskNode → MapTaskNode → ...` chains (with optional join/reduce terminals) into
    `CompositeMapTaskNode` instances.
    """
    predecessors, successors = _build_adjacency(nodes)
    chains = _find_map_chains(nodes, predecessors, successors)

    if not chains:
        return nodes, {}

    result = dict(nodes)
    id_mapping: dict[UUID, UUID] = {}

    for mc in chains:
        map_node = nodes[mc.map_id]
        assert isinstance(map_node, MapTaskNode)

        # Build chain links for the .then() nodes
        links: list[ChainLink] = []
        prev_id = mc.map_id
        for nid in mc.then_ids:
            node = nodes[nid]
            if isinstance(node, MapTaskNode):
                flow_param = _identify_flow_param(node.mapped_kwargs, prev_id)
                link = _build_map_chain_link(node, flow_param)
            else:  # pragma: no cover — .then() on MapTaskFuture always creates MapTaskNode
                assert isinstance(node, TaskNode)
                flow_param = _identify_flow_param(node.kwargs, prev_id)
                link = _build_chain_link(node, flow_param, prev_id)
            links.append(link)
            prev_id = nid

        # Determine terminal
        terminal: TerminalKind = "collect"
        join_link: ChainLink | None = None
        reduce_config: ReduceConfig | None = None
        initial_input: NodeInput | None = None
        tail_id = mc.then_ids[-1] if mc.then_ids else mc.map_id

        if mc.join_id is not None:
            terminal = "join"
            join_node = nodes[mc.join_id]
            assert isinstance(join_node, TaskNode)
            flow_param = _identify_flow_param(join_node.kwargs, tail_id)
            join_link = _build_chain_link(join_node, flow_param, tail_id)
            tail_id = mc.join_id

        if mc.reduce_id is not None:
            terminal = "reduce"
            reduce_node = nodes[mc.reduce_id]
            assert isinstance(reduce_node, ReduceNode)
            reduce_config = reduce_node.reduce_config
            initial_input = reduce_node.initial_input
            tail_id = mc.reduce_id

        # Create composite
        composite_id = uuid4()
        link_names = [link.name for link in links]
        all_names = [map_node.name] + link_names
        if join_link:
            all_names.append(join_link.name)
        if reduce_config:
            all_names.append(f"reduce({reduce_config.name})")

        # Aggregate timeout across source map, .then() links, and the terminal (join)
        all_timeouts = [map_node.timeout] + [link.timeout for link in links]
        if join_link is not None:
            all_timeouts.append(join_link.timeout)
        composite_timeout = _aggregate_timeout(all_timeouts)

        composite = CompositeMapTaskNode(
            id=composite_id,
            name=f"composite_map({' → '.join(all_names)})",
            description=f"Composite map of {len(all_names)} chained steps",
            backend_name=_effective_backend(map_node),
            source_map=map_node,
            chain=tuple(links),
            terminal=terminal,
            join_link=join_link,
            reduce_config=reduce_config,
            initial_input=initial_input,
            timeout=composite_timeout,
        )

        # Remove folded nodes, add composite
        result.pop(mc.map_id, None)
        for nid in mc.then_ids:
            result.pop(nid, None)
        if mc.join_id is not None:
            result.pop(mc.join_id, None)
        if mc.reduce_id is not None:
            result.pop(mc.reduce_id, None)
        result[composite_id] = composite

        # Map all folded IDs → composite ID
        id_mapping[tail_id] = composite_id
        id_mapping[mc.map_id] = composite_id
        for nid in mc.then_ids:
            id_mapping[nid] = composite_id
        if (
            mc.join_id is not None and mc.join_id != tail_id
        ):  # pragma: no cover — tail_id is set to join_id above
            id_mapping[mc.join_id] = composite_id
        if (
            mc.reduce_id is not None and mc.reduce_id != tail_id
        ):  # pragma: no cover — tail_id is set to reduce_id above
            id_mapping[mc.reduce_id] = composite_id

    result = _remap_dependencies(result, id_mapping)

    return result, id_mapping


def _remap_dependencies(
    nodes: dict[UUID, BaseGraphNode],
    id_mapping: dict[UUID, UUID],
) -> dict[UUID, BaseGraphNode]:
    """
    Remap `NodeInput` references in remaining nodes so that references to folded node IDs point to
    the composite node that replaced them.

    This handles downstream nodes that depend on a node that was folded into a composite — their
    `NodeInput.reference` must be updated to point to the composite's ID.

    Delegates to each node's `remap_references` method so that every node type is responsible for
    knowing which of its own fields contain references.
    """
    if not id_mapping:  # pragma: no cover
        return nodes

    return {nid: node.remap_references(id_mapping) for nid, node in nodes.items()}


def _find_task_chains(
    nodes: dict[UUID, BaseGraphNode],
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
) -> list[list[UUID]]:
    """
    Detect maximal linear chains of `TaskNode`s in the graph.

    A chain is a sequence `[A, B, C, ...]` where:
    - Every node is a `TaskNode` (not `MapTaskNode`, `DatasetNode` etc.)
    - Each interior node has exactly **one predecessor** and **one successor**
      within the graph.
    - The head may have any number of predecessors.
    - The tail may have any number of successors.
    - All nodes share the same `backend_name`.
    - Chain length ≥ 2.
    """
    visited: set[UUID] = set()
    chains: list[list[UUID]] = []

    for nid, node in nodes.items():
        if not isinstance(node, TaskNode):
            continue
        if nid in visited:
            continue

        # Walk backward to find the true chain head — ensures maximal chains
        # regardless of dict iteration order.
        head = nid
        backend = _effective_backend(node)
        while True:
            preds = predecessors.get(head, set())
            if len(preds) != 1:
                break  # Multiple or no predecessors — head is head
            pred_id = next(iter(preds))
            pred_node = nodes.get(pred_id)
            if pred_id in visited:  # pragma: no cover
                break
            if not isinstance(pred_node, TaskNode):
                break
            if _effective_backend(pred_node) != backend:  # pragma: no cover
                break
            # Predecessor must have exactly one successor (head)
            pred_succs = successors.get(pred_id, set())
            if len(pred_succs) != 1:
                break
            head = pred_id

        # Walk forward from the true head
        chain = [head]
        visited.add(head)
        current = head
        while True:
            succs = successors.get(current, set())
            if len(succs) != 1:
                break  # Fan-out or terminal — end chain

            succ_id = next(iter(succs))
            succ_node = nodes.get(succ_id)

            if succ_id in visited:  # pragma: no cover
                break  # Already in another chain
            if not isinstance(succ_node, TaskNode):
                break  # Non-task node breaks the chain

            # The successor must have exactly one predecessor (current)
            preds = predecessors.get(succ_id, set())
            if len(preds) != 1:
                break  # Fan-in — end chain

            # Backend must match
            if _effective_backend(succ_node) != backend:
                break

            chain.append(succ_id)
            visited.add(succ_id)
            current = succ_id

        if len(chain) >= 2:
            chains.append(chain)

    return chains


def _build_chain_link(
    node: TaskNode,
    flow_param: str | None,
    predecessor_id: UUID | None,
) -> ChainLink:
    """
    Build a `ChainLink` from a `TaskNode`.

    External params are those that don't reference the immediate predecessor in the chain (the flow
    param is separated out).
    """
    external_params: dict[str, NodeInput] = {}

    for param_name, node_input in node.kwargs.items():
        if param_name == flow_param:
            continue  # Handled by the chain flow
        external_params[param_name] = node_input

    return ChainLink(
        id=node.id,
        name=node.name,
        description=node.description,
        func=node.func,
        flow_param=flow_param,
        external_params=external_params,
        output_configs=node.output_configs,
        retries=node.retries,
        cache=node.cache,
        cache_ttl=node.cache_ttl,
        timeout=node.timeout,
        link_kind=node.kind,
    )


@dataclass
class _MapChain:
    """Detected map chain ready for folding."""

    map_id: UUID
    """Head `MapTaskNode` ID."""

    then_ids: list[UUID]
    """IDs of `.then()` nodes (`MapTaskNode` or `TaskNode`) following the head."""

    join_id: UUID | None = None
    """If present, ID of the `.join()` `TaskNode` terminal."""

    reduce_id: UUID | None = None
    """If present, ID of the `.reduce()` `ReduceNode` terminal."""


def _find_map_chains(
    nodes: dict[UUID, BaseGraphNode],
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
) -> list[_MapChain]:
    """
    Detect chains starting with a `MapTaskNode` followed by `MapTaskNode`s
    (from `.then()`) and/or a terminal `ReduceNode` (from `.reduce()`).

    A `.then()` on a `MapTaskFuture` produces another `MapTaskNode` whose `mapped_kwargs` has a
    single `sequence_ref` to the prior map. A `.join()` produces a `TaskNode` that depends on the
    last map. A `.reduce()` produces a `ReduceNode` that depends on the last map.
    """
    chains: list[_MapChain] = []
    visited: set[UUID] = set()

    for nid, node in nodes.items():
        if not isinstance(node, MapTaskNode):
            continue
        if nid in visited:
            continue

        map_backend = _effective_backend(node)

        # Walk backward to find the true chain head — ensures maximal chains
        # regardless of dict iteration order.
        head = nid
        while True:
            preds = predecessors.get(head, set())
            if len(preds) != 1:
                break
            pred_id = next(iter(preds))
            pred_node = nodes.get(pred_id)
            if pred_id in visited:  # pragma: no cover
                break
            if not isinstance(pred_node, MapTaskNode):
                break
            if _effective_backend(pred_node) != map_backend:  # pragma: no cover
                break
            # Predecessor must have exactly one successor (head)
            if len(successors.get(pred_id, set())) != 1:
                break
            # head must be a .then() continuation of pred_id
            if not _is_then_map_node(nodes[head], pred_id):  # pragma: no cover
                break
            head = pred_id

        then_chain: list[UUID] = []
        current = head

        # Walk the .then() chain (MapTaskNode → MapTaskNode → ...)
        while True:
            succs = successors.get(current, set())
            if len(succs) != 1:
                break

            succ_id = next(iter(succs))
            succ_node = nodes.get(succ_id)

            if succ_id in visited:  # pragma: no cover
                break

            # .then() on MapTaskFuture creates another MapTaskNode
            if isinstance(succ_node, MapTaskNode):
                preds = predecessors.get(succ_id, set())
                if len(preds) != 1:  # pragma: no cover – .then() MapTaskNode always has one pred
                    break
                if not _is_then_map_node(succ_node, current):  # pragma: no cover
                    # Not a .then() continuation — end of the chain
                    break
                then_chain.append(succ_id)
                current = succ_id
                continue

            break  # Encountered a possible terminal node

        # Check for a terminal node (join or reduce)
        join_id: UUID | None = None
        reduce_id: UUID | None = None

        terminal_succs = successors.get(current, set())
        if len(terminal_succs) == 1:
            term_id = next(iter(terminal_succs))
            term_node = nodes.get(term_id)
            term_preds = predecessors.get(term_id, set())

            if (
                isinstance(term_node, ReduceNode)
                and len(term_preds) == 1
                and term_id not in visited
            ):
                reduce_id = term_id
            elif (
                isinstance(term_node, TaskNode)
                and not isinstance(term_node, MapTaskNode)
                and len(term_preds) == 1
                and term_id not in visited
                and _effective_backend(term_node) == map_backend
            ):
                # Potential .join() — the task takes the map's list output
                join_id = term_id

        # A chain is valid if there's at least one .then() node, or a terminal
        if then_chain or reduce_id is not None or join_id is not None:
            visited.add(head)
            visited.update(then_chain)
            if join_id is not None:
                visited.add(join_id)
            if reduce_id is not None:
                visited.add(reduce_id)
            chains.append(
                _MapChain(
                    map_id=head,
                    then_ids=then_chain,
                    join_id=join_id,
                    reduce_id=reduce_id,
                )
            )

    return chains


def _build_map_chain_link(node: MapTaskNode, flow_param: str | None) -> ChainLink:
    """
    Build a `ChainLink` from a `.then()`-style `MapTaskNode`.

    The `fixed_kwargs` become external params; the single `mapped_kwarg`
    (the flow) is separated out.
    """
    external_params: dict[str, NodeInput] = dict(node.fixed_kwargs)

    return ChainLink(
        id=node.id,
        name=node.name,
        description=node.description,
        func=node.func,
        flow_param=flow_param,
        external_params=external_params,
        output_configs=node.output_configs,
        retries=node.retries,
        cache=node.cache,
        cache_ttl=node.cache_ttl,
        timeout=node.timeout,
        link_kind=node.kind,
    )


def _is_then_map_node(node: BaseGraphNode, predecessor_id: UUID) -> bool:
    """
    Check whether a `MapTaskNode` is a `.then()` continuation of `predecessor_id`.

    A `.then()` pattern produces a `MapTaskNode` with exactly one `mapped_kwarg` whose `NodeInput`
    is a `sequence_ref` to the predecessor.
    """
    if not isinstance(node, MapTaskNode):  # pragma: no cover – callers only pass MapTaskNodes
        return False

    if len(node.mapped_kwargs) != 1:
        return False

    for _name, node_input in node.mapped_kwargs.items():
        if node_input.reference == predecessor_id:  # pragma: no branch – single matching kwarg
            return True
    return False  # pragma: no cover – only reachable if mapped_kwargs[0] has wrong reference


def _aggregate_timeout(timeouts: list[float | None]) -> float | None:
    """
    Compute an aggregate timeout from a list of individual timeouts.

    Returns the sum of all values. If any value is `None` (unbounded), the result is also `None`
    — since the total cannot be bounded.
    """
    total: float = 0.0
    for t in timeouts:
        if t is None:
            return None
        total += t
    return total if total > 0.0 else None


def _build_adjacency(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, set[UUID]], dict[UUID, set[UUID]]]:
    """
    Build predecessor and successor adjacency maps from the graph.

    Returns:
        `(predecessors, successors)` — both mapping node IDs to sets of neighbour IDs.
    """
    predecessors: dict[UUID, set[UUID]] = defaultdict(set)
    successors: dict[UUID, set[UUID]] = defaultdict(set)

    for nid, node in nodes.items():
        deps = node.get_dependencies()
        for dep_id in deps:
            if dep_id in nodes:  # only track deps within this graph  # pragma: no branch
                predecessors[nid].add(dep_id)
                successors[dep_id].add(nid)

    return dict(predecessors), dict(successors)


def _effective_backend(node: BaseGraphNode) -> str | None:
    """Return the effective backend name for a node."""
    return node.backend_name


def _identify_flow_param(params: Mapping[str, NodeInput], predecessor_id: UUID) -> str | None:
    """
    Identify the parameter that receives the result of `predecessor_id`.

    Works on any `NodeInput` mapping — pass `node.kwargs` for `TaskNode` or `node.mapped_kwargs`
    for `MapTaskNode`.
    """
    for param_name, node_input in params.items():
        if node_input.reference == predecessor_id:
            return param_name
    return None
