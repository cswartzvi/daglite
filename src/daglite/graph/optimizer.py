"""Graph optimization passes for the daglite IR."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID
from uuid import uuid4

from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.composite_node import CompositeMapTaskNode
from daglite.graph.nodes.composite_node import CompositeStep
from daglite.graph.nodes.composite_node import CompositeTaskNode
from daglite.graph.nodes.composite_node import TerminalKind
from daglite.graph.nodes.map_node import MapTaskNode
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.reduce_node import ReduceNode
from daglite.graph.nodes.task_node import TaskNode

# region API


def optimize_graph(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Apply optimization passes to the graph IR.

    Currently performs composite node folding — collapsing linear sequences of nodes sharing the
    same backend into `CompositeTaskNode` or `CompositeMapTaskNode` instances.

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
    nodes, id_mapping = _fold_task_paths(nodes)
    nodes, map_mapping = _fold_map_paths(nodes)
    id_mapping.update(map_mapping)
    return nodes, id_mapping


# region Find Composites


def _fold_task_paths(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Fold linear `TaskNode` paths into `CompositeTaskNode` instances.

    Returns the rewritten graph and an ID mapping from original tail IDs to composite IDs.
    """
    predecessors, successors = _build_adjacency(nodes)
    segments = _find_task_paths(nodes, predecessors, successors)

    if not segments:
        return nodes, {}

    result = dict(nodes)
    id_mapping: dict[UUID, UUID] = {}

    # Create composite nodes for each segment
    for segment_id in segments:
        steps: list[CompositeStep] = []

        for i, nid in enumerate(segment_id):
            node = nodes[nid]
            assert isinstance(node, TaskNode)

            if i == 0:
                # First step has all params as external
                step = _build_composite_step(node, flow_param=None, predecessor_id=None)
            else:
                predecessor_id = segment_id[i - 1]
                flow_param = _identify_flow_param(node.kwargs, predecessor_id)
                step = _build_composite_step(node, flow_param, predecessor_id)

            steps.append(step)

        composite_id = uuid4()
        head = nodes[segment_id[0]]
        tail_id = segment_id[-1]
        composite = CompositeTaskNode(
            id=composite_id,
            name=f"composite({' → '.join(step.name for step in steps)})",
            description=f"Composite of {len(steps)} chained tasks",
            backend_name=_effective_backend(head),
            steps=tuple(steps),
            timeout=_aggregate_timeout([step.timeout for step in steps]),
        )

        # Remove path nodes from result, add composite
        for nid in segment_id:
            result.pop(nid, None)
        result[composite_id] = composite

        # Map IDs to composite ID
        id_mapping[tail_id] = composite_id
        for nid in segment_id[:-1]:  # interior nodes
            id_mapping[nid] = composite_id

    result = _remap_dependencies(result, id_mapping)
    return result, id_mapping


def _find_task_paths(
    nodes: dict[UUID, BaseGraphNode],
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
) -> list[list[UUID]]:
    """Detects maximal linear paths of `TaskNode`s in the graph."""
    visited: set[UUID] = set()
    paths: list[list[UUID]] = []

    for nid, node in nodes.items():
        if not isinstance(node, TaskNode):
            continue
        if nid in visited:
            continue

        backend = _effective_backend(node)

        def _can_extend(pred_id: UUID, _head: UUID) -> bool:
            pred_node = nodes.get(pred_id)
            return isinstance(pred_node, TaskNode) and _effective_backend(pred_node) == backend

        head = _find_path_head(nid, predecessors, successors, visited, _can_extend)

        # Walk forward from the true head through .then() nodes
        path = [head]
        visited.add(head)
        current = head
        while True:
            succs = successors.get(current, set())
            if len(succs) != 1:
                break  # Fan-out or terminal — end path

            succ_id = next(iter(succs))
            succ_node = nodes.get(succ_id)

            if succ_id in visited:
                break  # pragma: no branch
            if not isinstance(succ_node, TaskNode):
                break  # Non-task node breaks the path

            if len(predecessors.get(succ_id, set())) != 1:
                break  # Fan-in — end path

            if _effective_backend(succ_node) != backend:
                break

            path.append(succ_id)
            visited.add(succ_id)
            current = succ_id

        if len(path) >= 2:
            paths.append(path)

    return paths


def _build_composite_step(
    node: TaskNode,
    flow_param: str | None,
    predecessor_id: UUID | None,
) -> CompositeStep:
    """
    Build a `CompositeStep` from a `TaskNode`.

    External params are those that don't reference the immediate predecessor in the path (the flow
    param is separated out).
    """
    external_params: dict[str, NodeInput] = {}

    for param_name, node_input in node.kwargs.items():
        if param_name == flow_param:
            continue  # Handled by the composite flow
        external_params[param_name] = node_input

    return CompositeStep(
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
        step_kind=node.kind,
    )


# region Find Composite Maps


@dataclass
class _MapPath:
    """Detected map path ready for folding."""

    map_id: UUID
    """Head `MapTaskNode` ID."""

    then_ids: list[UUID]
    """IDs of `.then()` nodes (`MapTaskNode` or `TaskNode`) following the head."""

    join_id: UUID | None = None
    """If present, ID of the `.join()` `TaskNode` terminal."""

    reduce_id: UUID | None = None
    """If present, ID of the `.reduce()` `ReduceNode` terminal."""


def _fold_map_paths(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Fold `MapTaskNode → MapTaskNode → ...` paths (with optional join/reduce terminals) into
    `CompositeMapTaskNode` instances.
    """
    predecessors, successors = _build_adjacency(nodes)
    map_paths = _find_map_task_paths(nodes, predecessors, successors)

    if not map_paths:
        return nodes, {}

    result = dict(nodes)
    id_mapping: dict[UUID, UUID] = {}

    for map_path in map_paths:
        map_node = nodes[map_path.map_id]
        assert isinstance(map_node, MapTaskNode)

        # Build composite steps for the .then() nodes
        steps: list[CompositeStep] = []
        prev_id = map_path.map_id
        for nid in map_path.then_ids:
            node = nodes[nid]
            if isinstance(node, MapTaskNode):
                flow_param = _identify_flow_param(node.mapped_kwargs, prev_id)
                step = _build_map_composite_step(node, flow_param)
            else:  # pragma: no cover — .then() on MapTaskFuture always creates MapTaskNode
                assert isinstance(node, TaskNode)
                flow_param = _identify_flow_param(node.kwargs, prev_id)
                step = _build_composite_step(node, flow_param, prev_id)
            steps.append(step)
            prev_id = nid

        # Determine terminal
        terminal: TerminalKind = "collect"
        join_step: CompositeStep | None = None
        reduce_config: ReduceConfig | None = None
        initial_input: NodeInput | None = None
        tail_id = map_path.then_ids[-1] if map_path.then_ids else map_path.map_id

        if map_path.join_id is not None:
            terminal = "join"
            join_node = nodes[map_path.join_id]
            assert isinstance(join_node, TaskNode)
            flow_param = _identify_flow_param(join_node.kwargs, tail_id)
            assert flow_param is not None, "join terminal must consume upstream mapped output"
            join_step = _build_composite_step(join_node, flow_param, tail_id)
            tail_id = map_path.join_id

        if map_path.reduce_id is not None:
            terminal = "reduce"
            reduce_node = nodes[map_path.reduce_id]
            assert isinstance(reduce_node, ReduceNode)
            reduce_config = reduce_node.reduce_config
            initial_input = reduce_node.initial_input
            tail_id = map_path.reduce_id

        # Create composite
        composite_id = uuid4()
        step_names = [step.name for step in steps]
        all_names = [map_node.name] + step_names
        if join_step:
            all_names.append(join_step.name)
        if reduce_config:
            all_names.append(f"reduce({reduce_config.name})")

        # Aggregate timeout across source map, .then() steps, and the terminal (join)
        all_timeouts = [map_node.timeout] + [step.timeout for step in steps]
        if join_step is not None:
            all_timeouts.append(join_step.timeout)
        composite_timeout = _aggregate_timeout(all_timeouts)

        composite = CompositeMapTaskNode(
            id=composite_id,
            name=f"composite_map({' → '.join(all_names)})",
            description=f"Composite map of {len(all_names)} chained steps",
            backend_name=_effective_backend(map_node),
            source_map=map_node,
            steps=tuple(steps),
            terminal=terminal,
            join_step=join_step,
            reduce_config=reduce_config,
            initial_input=initial_input,
            timeout=composite_timeout,
        )

        # Remove folded nodes, add composite
        result.pop(map_path.map_id, None)
        for nid in map_path.then_ids:
            result.pop(nid, None)
        if map_path.join_id is not None:
            result.pop(map_path.join_id, None)
        if map_path.reduce_id is not None:
            result.pop(map_path.reduce_id, None)
        result[composite_id] = composite

        # Map all folded IDs → composite ID
        id_mapping[tail_id] = composite_id
        id_mapping[map_path.map_id] = composite_id
        for nid in map_path.then_ids:
            id_mapping[nid] = composite_id

    result = _remap_dependencies(result, id_mapping)

    return result, id_mapping


def _find_map_task_paths(
    nodes: dict[UUID, BaseGraphNode],
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
) -> list[_MapPath]:
    """
    Detect paths starting with a `MapTaskNode` followed by `MapTaskNode`s.

    This is the pattern produced by chaining `.then()` calls on a `MapTaskFuture`, optionally
    ending with a `TaskNode` that joins the map output (from `.join()`) or a `ReduceNode` (from
    `.reduce()`).
    """
    map_paths: list[_MapPath] = []
    visited: set[UUID] = set()

    for nid, node in nodes.items():
        if not isinstance(node, MapTaskNode):
            continue
        if nid in visited:
            continue

        map_backend = _effective_backend(node)

        def _can_extend(pred_id: UUID, current_head: UUID) -> bool:
            pred_node = nodes.get(pred_id)
            return (
                isinstance(pred_node, MapTaskNode)
                and _effective_backend(pred_node) == map_backend
                and _is_then_map_node(nodes[current_head], pred_id)
            )

        head = _find_path_head(nid, predecessors, successors, visited, _can_extend)

        then_path: list[UUID] = []
        current = head

        # Walk forward from the true head through .then() nodes
        while True:
            succs = successors.get(current, set())
            if len(succs) != 1:
                break

            succ_id = next(iter(succs))
            succ_node = nodes.get(succ_id)

            if succ_id in visited:
                break  # pragma: no branch

            # .then() on MapTaskFuture creates another MapTaskNode
            if isinstance(succ_node, MapTaskNode):
                preds = predecessors.get(succ_id, set())
                if len(preds) != 1:  # pragma: no cover – .then() MapTaskNode always has one pred
                    break
                if not _is_then_map_node(succ_node, current):  # pragma: no cover
                    # Not a .then() continuation — end of the chain
                    break
                then_path.append(succ_id)
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
                flow_param = _identify_flow_param(term_node.kwargs, current)
                if flow_param is not None:
                    # Potential .join() — the task takes the map's list output
                    join_id = term_id

        # A path is valid if there's at least one .then() node, or a terminal
        if then_path or reduce_id is not None or join_id is not None:
            visited.add(head)
            visited.update(then_path)
            if join_id is not None:
                visited.add(join_id)
            if reduce_id is not None:
                visited.add(reduce_id)
            map_paths.append(
                _MapPath(
                    map_id=head,
                    then_ids=then_path,
                    join_id=join_id,
                    reduce_id=reduce_id,
                )
            )

    return map_paths


def _build_map_composite_step(node: MapTaskNode, flow_param: str | None) -> CompositeStep:
    """
    Build a `CompositeStep` from a `.then()`-style `MapTaskNode`.

    The `fixed_kwargs` become external params; the single `mapped_kwarg`
    (the flow) is separated out.
    """
    external_params: dict[str, NodeInput] = dict(node.fixed_kwargs)

    return CompositeStep(
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
        step_kind=node.kind,
    )


# region Helpers


def _build_adjacency(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, set[UUID]], dict[UUID, set[UUID]]]:
    """Builds predecessor and successor adjacency maps from the graph nodes."""
    predecessors: dict[UUID, set[UUID]] = defaultdict(set)
    successors: dict[UUID, set[UUID]] = defaultdict(set)

    for nid, node in nodes.items():
        deps = node.get_dependencies()
        for dep_id in deps:
            if dep_id in nodes:
                predecessors[nid].add(dep_id)
                successors[dep_id].add(nid)

    return dict(predecessors), dict(successors)


def _find_path_head(
    start: UUID,
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
    visited: set[UUID],
    can_extend: Callable[[UUID, UUID], bool],
) -> UUID:
    """
    Walk backward from `start` to find the true head of a maximal path.

    Keeps extending backward while all of the following hold for each candidate predecessor:
    - It has exactly one predecessor of its own (not a fan-in node).
    - It is not already part of a discovered path.
    - ``can_extend(pred_id, current_head)`` returns True.
    - It has exactly one successor (not a fan-out node).

    Ensures maximal paths are found regardless of graph iteration order.
    """
    head = start
    while True:
        preds = predecessors.get(head, set())
        if len(preds) != 1:
            break
        pred_id = next(iter(preds))
        if pred_id in visited:
            break
        if not can_extend(pred_id, head):
            break
        if len(successors.get(pred_id, set())) != 1:
            break
        head = pred_id
    return head


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


def _is_then_map_node(node: BaseGraphNode, predecessor_id: UUID) -> bool:
    """
    Check whether a `MapTaskNode` is a `.then()` continuation of `predecessor_id`.

    A `.then()` pattern produces a `MapTaskNode` with exactly one `mapped_kwarg` whose `NodeInput`
    is a `sequence_ref` to the predecessor.
    """
    if not isinstance(node, MapTaskNode):  # pragma: no cover – callers always pass MapTaskNodes
        return False
    if len(node.mapped_kwargs) != 1:
        return False
    ((_name, node_input),) = node.mapped_kwargs.items()
    return node_input.reference == predecessor_id


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


def _effective_backend(node: BaseGraphNode) -> str | None:
    """Return the effective backend name for a node."""
    return node.backend_name
