"""Hook specifications for daglite execution lifecycle events."""

from typing import Any
from uuid import UUID

from daglite.graph.base import GraphMetadata
from daglite.plugins.hooks.markers import hook_spec
from daglite.plugins.reporters import EventReporter


class NodeSpec:
    """Hook specifications for node-level execution events."""

    @hook_spec
    def before_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        reporter: EventReporter | None,
    ) -> None:
        """
        Called before a node begins execution.

        Args:
            metadata: Metadata for the node to be executed.
            inputs: Resolved inputs for the node execution.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called after a node completes execution successfully.

        Args:
            metadata: Metadata for the executed node.
            inputs: Resolved inputs for the node execution.
            result: Result produced by the node execution.
            duration: Time taken to execute in seconds.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def on_node_error(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        error: Exception,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called when a node execution fails.

        Args:
            metadata: Metadata for the executed node.
            inputs: Resolved inputs for the node execution.
            error: The exception that was raised.
            duration: Time taken before failure in seconds.
            reporter: Optional event reporter for this execution context.
        """


class GraphSpec:
    """Hook specifications for graph-level execution events."""

    @hook_spec
    def before_graph_execute(
        self,
        root_id: UUID,
        node_count: int,
        is_async: bool,
    ) -> None:
        """
        Called before graph execution begins.

        Args:
            root_id: UUID of the root node
            node_count: Total number of nodes in the graph
            is_async: True for async execution, False for sequential
        """

    @hook_spec
    def after_graph_execute(
        self,
        root_id: UUID,
        result: Any,
        duration: float,
        is_async: bool,
    ) -> None:
        """
        Called after graph execution completes successfully.

        Args:
            root_id: UUID of the root node
            result: Final result of the graph execution
            duration: Total time taken to execute in seconds
            is_async: True for async execution, False for sequential
        """

    @hook_spec
    def on_graph_error(
        self,
        root_id: UUID,
        error: Exception,
        duration: float,
        is_async: bool,
    ) -> None:
        """
        Called when graph execution fails.

        Args:
            root_id: UUID of the root node
            error: The exception that was raised
            duration: Time taken before failure in seconds
            is_async: True for async execution, False for sequential
        """
