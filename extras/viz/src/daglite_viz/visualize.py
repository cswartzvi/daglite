"""Graph visualization utilities for daglite."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    from daglite.futures import BaseTaskFuture
    from daglite.graph.base import BaseGraphNode


def collect_nodes(futures: Collection[BaseTaskFuture]) -> list[BaseGraphNode]:
    """
    Collect all graph nodes from a collection of task futures.

    This function traverses the dependency graph starting from the given futures
    and collects all unique nodes in topological order.

    Args:
        futures: Collection of task futures to extract nodes from.

    Returns:
        List of graph nodes in topological order.
    """
    from daglite.futures import BaseTaskFuture

    visited: set[UUID] = set()
    nodes: list[BaseGraphNode] = []

    def visit(future: BaseTaskFuture) -> None:
        """Recursively visit dependencies and collect nodes."""
        if future.id in visited:
            return
        visited.add(future.id)

        # Visit dependencies first (topological order)
        for dep in future.get_dependencies():
            if isinstance(dep, BaseTaskFuture):
                visit(dep)

        # Convert to graph node and add to list
        node = future.to_graph()
        nodes.append(node)

    # Process all futures
    for future in futures:
        if isinstance(future, BaseTaskFuture):
            visit(future)

    return nodes


def to_mermaid(
    nodes: Collection[BaseGraphNode] | Collection[BaseTaskFuture],
    *,
    direction: Literal["TB", "BT", "LR", "RL"] = "TB",
) -> str:
    """
    Generate a Mermaid flowchart diagram from graph nodes or task futures.

    Args:
        nodes: Collection of graph nodes or task futures to visualize.
        direction: Flowchart direction (TB=top-bottom, BT=bottom-top,
                   LR=left-right, RL=right-left).

    Returns:
        Mermaid flowchart syntax as a string.

    Examples:
        >>> from daglite import task
        >>> from daglite_viz import to_mermaid
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> future = add(1, 2)
        >>> diagram = to_mermaid([future])
        >>> print(diagram)  # doctest: +ELLIPSIS
        flowchart TB
            ...add...
    """
    from daglite.futures import BaseTaskFuture

    # Convert futures to nodes if needed
    if nodes and isinstance(next(iter(nodes)), BaseTaskFuture):
        graph_nodes = collect_nodes(nodes)  # type: ignore
    else:
        graph_nodes = list(nodes)  # type: ignore

    if not graph_nodes:
        return f"flowchart {direction}\n"

    lines = [f"flowchart {direction}"]

    # Create node ID mapping (use short ID for readability)
    node_map: dict[UUID, str] = {}
    for i, node in enumerate(graph_nodes):
        node_map[node.id] = f"node{i}"

    # Generate node definitions
    for node_id, node_short_id in node_map.items():
        node_obj = next(n for n in graph_nodes if n.id == node_id)
        node_name = node_obj.name

        # Use different shapes for different node kinds
        if node_obj.to_metadata().kind == "map":
            # Use parallelogram for map nodes
            lines.append(f'    {node_short_id}[/"{node_name}"/]')
        else:
            # Use rectangle for task nodes
            lines.append(f'    {node_short_id}["{node_name}"]')

    # Generate edges
    for node in graph_nodes:
        node_short_id = node_map[node.id]
        for dep_id in node.dependencies():
            if dep_id in node_map:
                dep_short_id = node_map[dep_id]
                lines.append(f"    {dep_short_id} --> {node_short_id}")

    return "\n".join(lines)


def to_graphviz(
    nodes: Collection[BaseGraphNode] | Collection[BaseTaskFuture],
    *,
    rankdir: Literal["TB", "BT", "LR", "RL"] = "TB",
) -> str:
    """
    Generate Graphviz DOT format from graph nodes or task futures.

    Args:
        nodes: Collection of graph nodes or task futures to visualize.
        rankdir: Graph direction (TB=top-bottom, BT=bottom-top,
                 LR=left-right, RL=right-left).

    Returns:
        Graphviz DOT syntax as a string.

    Examples:
        >>> from daglite import task
        >>> from daglite_viz import to_graphviz
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> future = add(1, 2)
        >>> dot = to_graphviz([future])
        >>> print(dot)  # doctest: +ELLIPSIS
        digraph {
            rankdir=TB
            ...add...
        }
    """
    from daglite.futures import BaseTaskFuture

    # Convert futures to nodes if needed
    if nodes and isinstance(next(iter(nodes)), BaseTaskFuture):
        graph_nodes = collect_nodes(nodes)  # type: ignore
    else:
        graph_nodes = list(nodes)  # type: ignore

    if not graph_nodes:
        return f"digraph {{\n    rankdir={rankdir}\n}}"

    lines = ["digraph {", f"    rankdir={rankdir}"]

    # Create node ID mapping
    node_map: dict[UUID, str] = {}
    for i, node in enumerate(graph_nodes):
        node_map[node.id] = f"node{i}"

    # Generate node definitions
    for node_id, node_short_id in node_map.items():
        node_obj = next(n for n in graph_nodes if n.id == node_id)
        node_name = node_obj.name

        # Use different shapes for different node kinds
        if node_obj.to_metadata().kind == "map":
            # Use parallelogram for map nodes
            lines.append(f'    {node_short_id} [label="{node_name}" shape=parallelogram]')
        else:
            # Use box for task nodes
            lines.append(f'    {node_short_id} [label="{node_name}" shape=box]')

    # Generate edges
    for node in graph_nodes:
        node_short_id = node_map[node.id]
        for dep_id in node.dependencies():
            if dep_id in node_map:
                dep_short_id = node_map[dep_id]
                lines.append(f"    {dep_short_id} -> {node_short_id}")

    lines.append("}")
    return "\n".join(lines)


def visualize_future(
    future: BaseTaskFuture,
    *,
    format: Literal["mermaid", "graphviz"] = "mermaid",
    **kwargs,
) -> str:
    """
    Convenience function to visualize a single task future.

    This function automatically collects all nodes in the dependency graph
    and generates a visualization in the specified format.

    Args:
        future: Task future to visualize.
        format: Output format ("mermaid" or "graphviz").
        **kwargs: Additional arguments passed to the formatter (e.g., direction, rankdir).

    Returns:
        Visualization string in the specified format.

    Examples:
        >>> from daglite import task
        >>> from daglite_viz import visualize_future
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> @task
        ... def multiply(x: int, y: int) -> int:
        ...     return x * y
        >>> a = add(1, 2)
        >>> b = add(3, 4)
        >>> c = multiply(a, b)
        >>> diagram = visualize_future(c, format="mermaid")
        >>> print(diagram)  # doctest: +ELLIPSIS
        flowchart TB
            ...
    """
    if format == "mermaid":
        return to_mermaid([future], **kwargs)
    elif format == "graphviz":
        return to_graphviz([future], **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}. Expected 'mermaid' or 'graphviz'.")
