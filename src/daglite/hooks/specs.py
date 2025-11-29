"""Hook specifications for daglite execution lifecycle events."""

from typing import Any
from uuid import UUID

from daglite.backends.base import Backend
from daglite.graph.base import GraphNode

from .markers import hook_spec


class NodeSpec:
    """Hook specifications for node-level execution events."""

    @hook_spec
    def before_node_execute(
        self, node_id: UUID, node: GraphNode, backend: Backend, inputs: dict[str, Any]
    ) -> None:
        """
        Called before a node begins execution.

        Args:
            node_id: Unique identifier for the node
            node: The GraphNode being executed
            backend: Backend instance that will execute the node
            inputs: Resolved input values for the node
        """

    @hook_spec
    def after_node_execute(
        self,
        node_id: UUID,
        node: GraphNode,
        backend: Backend,
        result: Any,
        duration: float,
    ) -> None:
        """
        Called after a node completes execution successfully.

        Args:
            node_id: Unique identifier for the node
            node: The GraphNode that was executed
            backend: Backend instance that executed the node
            result: The execution result
            duration: Time taken to execute in seconds
        """

    @hook_spec
    def on_node_error(
        self,
        node_id: UUID,
        node: GraphNode,
        backend: Backend,
        error: Exception,
        duration: float,
    ) -> None:
        """
        Called when a node execution fails.

        Args:
            node_id: Unique identifier for the node
            node: The GraphNode that failed
            backend: Backend instance that was executing the node
            error: The exception that was raised
            duration: Time taken before failure in seconds
        """


class GraphSpec:
    """Hook specifications for graph-level execution events."""

    @hook_spec
    def before_graph_execute(self, root_id: UUID, node_count: int, mode: str) -> None:
        """
        Called before graph execution begins.

        Args:
            root_id: UUID of the root node
            node_count: Total number of nodes in the graph
            mode: Execution mode ("sequential" or "async")
        """

    @hook_spec
    def after_graph_execute(self, root_id: UUID, result: Any, duration: float, mode: str) -> None:
        """
        Called after graph execution completes successfully.

        Args:
            root_id: UUID of the root node
            result: Final result of the graph execution
            duration: Total time taken to execute in seconds
            mode: Execution mode ("sequential" or "async")
        """

    @hook_spec
    def on_graph_error(self, root_id: UUID, error: Exception, duration: float, mode: str) -> None:
        """
        Called when graph execution fails.

        Args:
            root_id: UUID of the root node
            error: The exception that was raised
            duration: Time taken before failure in seconds
            mode: Execution mode ("sequential" or "async")
        """
