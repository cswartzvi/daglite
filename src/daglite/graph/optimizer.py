"""Graph optimization passes for the daglite IR."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from uuid import UUID
from uuid import uuid4

from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.composite_node import CompositeMapTaskNode
from daglite.graph.nodes.composite_node import CompositeStep
from daglite.graph.nodes.composite_node import CompositeTaskNode
from daglite.graph.nodes.composite_node import IterSourceConfig
from daglite.graph.nodes.composite_node import TerminalKind
from daglite.graph.nodes.iter_node import IterNode
from daglite.graph.nodes.map_node import MapTaskNode
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.reduce_node import ReduceNode
from daglite.graph.nodes.task_node import TaskNode
from daglite.utils import any_not_none

# region API


def optimize_graph(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """
    Apply optimization passes to the graph IR.

    Currently performs composite node folding — collapsing linear sequences of nodes sharing the
    same backend.

    Args:
        nodes: The original graph IR node dictionary.

    Returns:
        A tuple of `(optimized_nodes, id_mapping)` where
        - `optimized_nodes` is the rewritten graph.
        - `id_mapping` maps folded node IDs to their composite node's ID.

        Note: the **tail** node ID of each chain is stored as a key in `id_mapping` so that
        downstream nodes depending on the tail can find the composite's result.
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
    Folds detected linear task node paths into composite task instances.

    Returns the rewritten graph and an ID mapping from original tail IDs to composite IDs.
    """
    predecessors, successors = _build_adjacency(nodes)
    paths = _find_task_paths(nodes, predecessors, successors)

    if not paths:
        return nodes, {}

    result = dict(nodes)
    id_mapping: dict[UUID, UUID] = {}

    # Create composite nodes for each segment
    for path in paths:
        steps: list[CompositeStep] = []

        for i, nid in enumerate(path):
            node = nodes[nid]
            assert isinstance(node, TaskNode)

            if i == 0:  # First step in path has all params as external
                step = CompositeStep.from_node(node, flow_param=None)
            else:
                predecessor_id = path[i - 1]
                flow_param = _identify_flow_param(node.kwargs, predecessor_id)
                step = CompositeStep.from_node(node, flow_param=flow_param)

            steps.append(step)

        # Create composite node to replace the entire path
        composite_id = uuid4()
        head_node = nodes[path[0]]
        tail_id = path[-1]
        composite = CompositeTaskNode(
            id=composite_id,
            name=f"composite({' → '.join(step.name for step in steps)})",
            description=f"Composite of {len(steps)} chained tasks",
            backend_name=head_node.backend_name,
            steps=tuple(steps),
            timeout=_aggregate_timeout([step.timeout for step in steps]),
        )

        # Remove path nodes from result, add composite
        for nid in path:
            result.pop(nid, None)
        result[composite_id] = composite

        # Map IDs to composite ID
        id_mapping[tail_id] = composite_id
        for nid in path[:-1]:  # interior nodes
            id_mapping[nid] = composite_id

    result = _remap_dependencies(result, id_mapping)
    return result, id_mapping


def _find_task_paths(
    nodes: dict[UUID, BaseGraphNode],
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
) -> list[list[UUID]]:
    """Detects maximal linear paths of task nodes in the given graph."""
    visited: set[UUID] = set()
    paths: list[list[UUID]] = []

    for nid, node in nodes.items():
        if not isinstance(node, TaskNode):
            continue
        if nid in visited:
            continue

        backend = node.backend_name

        head = _find_path_head(
            nid,
            predecessors,
            successors,
            visited,
            partial(_can_extend_task, nodes, backend),
        )

        path = [head]
        visited.add(head)
        current_id = head

        # Walk forward while there's a single successor that can be folded into the path
        while True:
            successor_ids = successors.get(current_id, set())
            if len(successor_ids) != 1:
                break  # Fan-out or terminal — end path

            successor_id = next(iter(successor_ids))
            successor_node = nodes.get(successor_id)

            if successor_id in visited:  # pragma: no cover
                break

            if not isinstance(successor_node, TaskNode):
                break  # Non-task node breaks the path

            if len(predecessors.get(successor_id, set())) != 1:
                break  # Fan-in — end path

            if successor_node.backend_name != backend:
                break  # Different backend — can't fold

            path.append(successor_id)
            visited.add(successor_id)
            current_id = successor_id

        if len(path) >= 2:
            paths.append(path)

    return paths


# region Find Composite Maps


@dataclass
class _MapPath:
    """Detected map path ready for folding."""

    map_id: UUID
    """Head mapped node ID where the path starts."""

    step_ids: list[UUID]
    """Sequence of mapped task node IDs following the head map node."""

    join_id: UUID | None = None
    """If present, ID of the terminal join task node."""

    reduce_id: UUID | None = None
    """If present, ID of the terminal reduce node."""

    iter_node_id: UUID | None = None
    """If present, ID of the iteration node feeding into the head map node."""

    @property
    def all_node_ids(self) -> list[UUID]:
        """All node IDs involved in this path."""
        ids = [self.map_id, *self.step_ids]
        for optional in (self.join_id, self.reduce_id, self.iter_node_id):
            if optional is not None:
                ids.append(optional)
        return ids


def _fold_map_paths(
    nodes: dict[UUID, BaseGraphNode],
) -> tuple[dict[UUID, BaseGraphNode], dict[UUID, UUID]]:
    """Folds detected map paths into composite map task instances."""
    predecessors, successors = _build_adjacency(nodes)
    map_paths = _find_map_task_paths(nodes, predecessors, successors)

    if not map_paths:
        return nodes, {}

    result = dict(nodes)
    id_mapping: dict[UUID, UUID] = {}

    for map_path in map_paths:
        map_node = nodes[map_path.map_id]
        assert isinstance(map_node, MapTaskNode)

        # Build composite steps mapped
        steps: list[CompositeStep] = []
        prev_id = map_path.map_id
        for nid in map_path.step_ids:
            node = nodes[nid]
            if isinstance(node, MapTaskNode):
                flow_param = _identify_flow_param(node.mapped_kwargs, prev_id)
            else:  # pragma: no cover — .then() on MapTaskFuture always creates MapTaskNode
                assert isinstance(node, TaskNode)
                flow_param = _identify_flow_param(node.kwargs, prev_id)
            step = CompositeStep.from_node(node, flow_param=flow_param)
            steps.append(step)
            prev_id = nid

        # Determine terminal
        terminal: TerminalKind = "collect"
        join_step: CompositeStep | None = None
        reduce_config: ReduceConfig | None = None
        reduce_output_configs: tuple = ()
        initial_input: NodeInput | None = None
        tail_id = map_path.step_ids[-1] if map_path.step_ids else map_path.map_id

        if map_path.join_id is not None:
            terminal = "join"
            join_node = nodes[map_path.join_id]
            assert isinstance(join_node, TaskNode)
            flow_param = _identify_flow_param(join_node.kwargs, tail_id)
            assert flow_param is not None, "join terminal must consume upstream mapped output"
            join_step = CompositeStep.from_node(join_node, flow_param=flow_param)
            tail_id = map_path.join_id

        if map_path.reduce_id is not None:
            terminal = "reduce"
            reduce_node = nodes[map_path.reduce_id]
            assert isinstance(reduce_node, ReduceNode)
            reduce_config = reduce_node.reduce_config
            initial_input = reduce_node.initial_input
            reduce_output_configs = reduce_node.output_configs
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
            backend_name=map_node.backend_name,
            source_map=map_node,
            steps=tuple(steps),
            terminal=terminal,
            join_step=join_step,
            reduce_config=reduce_config,
            initial_accumulator=initial_input,
            timeout=composite_timeout,
            iter_source=_build_iter_source(nodes, map_path.iter_node_id),
            output_configs=reduce_output_configs,
        )

        # Remove folded nodes, add composite
        for nid in map_path.all_node_ids:
            result.pop(nid, None)
        result[composite_id] = composite

        # Map all folded IDs → composite ID
        for nid in map_path.all_node_ids:
            id_mapping[nid] = composite_id

    result = _remap_dependencies(result, id_mapping)

    return result, id_mapping


def _find_map_task_paths(
    nodes: dict[UUID, BaseGraphNode],
    predecessors: dict[UUID, set[UUID]],
    successors: dict[UUID, set[UUID]],
) -> list[_MapPath]:
    """
    Detects maximal linear paths of mapped task nodes in the graph.

    Results are returned as `_MapPath` objects that capture the path as well as the presence of
    terminal and iter source nodes.
    """
    map_paths: list[_MapPath] = []
    visited: set[UUID] = set()

    for nid, node in nodes.items():
        if not isinstance(node, MapTaskNode):
            continue
        if nid in visited:
            continue

        map_backend = node.backend_name

        head = _find_path_head(
            nid,
            predecessors,
            successors,
            visited,
            partial(_can_extend_map_task, nodes, map_backend),
        )

        then_path: list[UUID] = []
        current = head

        # Walk forward from the true head through .then() nodes
        while True:
            succs = successors.get(current, set())
            if len(succs) != 1:
                break

            succ_id = next(iter(succs))
            succ_node = nodes.get(succ_id)

            if succ_id in visited:  # pragma: no cover
                break

            # Check for mapped node continuation
            if isinstance(succ_node, MapTaskNode):
                preds = predecessors.get(succ_id, set())
                assert len(preds) == 1, "then node should have exactly one predecessor"
                if not _is_single_mapped_successor(succ_node, current):
                    break  # End of the chain
                then_path.append(succ_id)
                current = succ_id
                continue

            break  # Encountered a possible terminal node

        join_id: UUID | None = None
        reduce_id: UUID | None = None

        # Check for a terminal node (join or reduce) if available
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
                and term_node.backend_name == map_backend
            ):
                flow_param = _identify_flow_param(term_node.kwargs, current)
                if flow_param is not None:
                    join_id = term_id  # Join terminal

        iter_node_id: UUID | None = None
        head_preds = predecessors.get(head, set())

        # Check for an IterNode feeding into the head MapTaskNode (indicates a .iter() source)
        if len(head_preds) == 1:
            pred_id = next(iter(head_preds))
            pred_node = nodes.get(pred_id)
            if (
                isinstance(pred_node, IterNode)
                and pred_id not in visited
                and _is_single_mapped_successor(nodes[head], pred_id)
            ):
                iter_node_id = pred_id

        # Update visited and record the path if foldable component found
        if then_path or any_not_none(join_id, reduce_id, iter_node_id):
            path = _MapPath(
                map_id=head,
                step_ids=then_path,
                join_id=join_id,
                reduce_id=reduce_id,
                iter_node_id=iter_node_id,
            )
            visited.update(path.all_node_ids)
            map_paths.append(path)

    return map_paths


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
    Walks backward from `start` to find the true head of a maximal path.

    Ensures maximal paths are found regardless of graph iteration order.

    Keeps extending backward while all of the following hold for each candidate predecessor:
    - It has exactly one predecessor of its own (not a fan-in node).
    - It is not already part of a discovered path.
    - It has exactly one successor (not a fan-out node).
    - `can_extend(pred_id, current_head)` returns True.
    """
    head = start
    while True:
        preds = predecessors.get(head, set())
        if len(preds) != 1:
            break
        pred_id = next(iter(preds))
        if pred_id in visited:
            break
        if len(successors.get(pred_id, set())) != 1:
            break
        if not can_extend(pred_id, head):
            break
        head = pred_id
    return head


def _can_extend_task(
    nodes: dict[UUID, BaseGraphNode],
    backend: str | None,
    pred_id: UUID,
    head_id: UUID,
) -> bool:
    """
    Checks if `pred_id` and `head_id` are connected in a linear task node path.

    This is a strategy function for `_find_path_head`.
    """
    pred_node = nodes.get(pred_id)
    return isinstance(pred_node, TaskNode) and pred_node.backend_name == backend


def _can_extend_map_task(
    nodes: dict[UUID, BaseGraphNode], backend: str | None, pred_id: UUID, head_id: UUID
) -> bool:
    """
    Checks if `pred_id` and `head_id` are connected in a linear mapped task node path.

    This is a strategy function for `_find_path_head`.
    """
    pred_node = nodes.get(pred_id)
    return (
        isinstance(pred_node, MapTaskNode)
        and pred_node.backend_name == backend
        and _is_single_mapped_successor(nodes[head_id], pred_id)
    )


def _is_single_mapped_successor(node: BaseGraphNode, predecessor_id: UUID) -> bool:
    """Check whether a node is a single mapped successor of a given predecessor."""
    if not isinstance(node, MapTaskNode):  # pragma: no cover
        return False
    if len(node.mapped_kwargs) != 1:
        return False
    (node_input,) = node.mapped_kwargs.values()
    return node_input.reference == predecessor_id


def _identify_flow_param(params: Mapping[str, NodeInput], predecessor_id: UUID) -> str | None:
    """Identify the parameter that receives the result of predecessor node."""
    for param_name, node_input in params.items():
        if node_input.reference == predecessor_id:
            return param_name
    return None


def _remap_dependencies(
    nodes: dict[UUID, BaseGraphNode],
    id_mapping: dict[UUID, UUID],
) -> dict[UUID, BaseGraphNode]:
    """
    Remap node references to folded node references according to a given mapping.

    This handles downstream nodes that depend on a node that was folded into a composite — their
    reference must be updated to point to the composite's ID.
    """
    if not id_mapping:  # pragma: no cover
        return nodes

    return {nid: node.remap_references(id_mapping) for nid, node in nodes.items()}


def _build_iter_source(
    nodes: dict[UUID, BaseGraphNode], iter_node_id: UUID | None
) -> IterSourceConfig | None:
    """Builds an iteration source config from the given iter node ID, if present."""
    if iter_node_id is None:
        return None
    iter_node = nodes[iter_node_id]
    assert isinstance(iter_node, IterNode)
    return IterSourceConfig.from_iter_node(iter_node)


def _aggregate_timeout(timeouts: list[float | None]) -> float | None:
    """
    Compute an aggregate timeout from a list of individual timeouts.

    Returns the sum of all values. If any value is `None` (unbounded), the result is also `None`.
    """
    total: float = 0.0
    for t in timeouts:
        if t is None:
            return None
        total += t
    return total if total > 0.0 else None
