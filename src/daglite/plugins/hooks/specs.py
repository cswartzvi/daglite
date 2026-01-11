"""Hook specifications for daglite execution lifecycle events."""

from typing import Any
from uuid import UUID

from daglite.graph.base import GraphMetadata
from daglite.graph.base import OutputConfig
from daglite.plugins.hooks.markers import hook_spec
from daglite.plugins.reporters import EventReporter


class WorkerSideNodeSpecs:
    """Hook specifications for node-level execution events on the **backend worker**."""

    @hook_spec
    def before_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        output_config: tuple[OutputConfig, ...],
        output_extras: list[dict[str, Any]],
        reporter: EventReporter | None,
    ) -> None:
        """
        Called before a node begins execution.

        Args:
            metadata: Metadata for the node to be executed.
            inputs: Resolved inputs for the node execution.
            output_config: Output configuration tuple for this node.
            output_extras: Resolved extras for each output config (parallel list).
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        output_config: tuple[OutputConfig, ...],
        output_extras: list[dict[str, Any]],
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called after a node completes execution successfully.

        Args:
            metadata: Metadata for the executed node.
            inputs: Resolved inputs for the node execution.
            result: Result produced by the node execution.
            output_config: Output configuration tuple for this node.
            output_extras: Resolved extras for each output config (parallel list).
            duration: Time taken to execute in seconds.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def on_node_error(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        error: Exception,
        output_config: tuple[OutputConfig, ...],
        output_extras: list[dict[str, Any]],
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called when a node execution fails.

        Args:
            metadata: Metadata for the executed node.
            inputs: Resolved inputs for the node execution.
            error: The exception that was raised.
            output_config: Output configuration tuple for this node.
            output_extras: Resolved extras for each output config (parallel list).
            duration: Time taken before failure in seconds.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def before_node_retry(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        output_config: tuple[OutputConfig, ...],
        output_extras: list[dict[str, Any]],
        attempt: int,
        last_error: Exception,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called before retrying a failed node execution.

        Args:
            metadata: Metadata for the node to be retried.
            inputs: Resolved inputs for the node execution.
            output_config: Output configuration tuple for this node.
            output_extras: Resolved extras for each output config (parallel list).
            attempt: Attempt number (1-indexed, so attempt=2 means first retry).
            last_error: The exception that caused the previous attempt to fail.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def after_node_retry(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        output_config: tuple[OutputConfig, ...],
        output_extras: list[dict[str, Any]],
        attempt: int,
        succeeded: bool,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called after a retry attempt completes.

        Args:
            metadata: Metadata for the retried node.
            inputs: Resolved inputs for the node execution.
            output_config: Output configuration tuple for this node.
            output_extras: Resolved extras for each output config (parallel list).
            attempt: Attempt number (1-indexed).
            succeeded: True if this retry attempt succeeded, False if it failed.
            reporter: Optional event reporter for this execution context.
        """


class CoordinatorSideNodeSpecs:
    """Hook specifications for node-level execution events on the **coordinator**."""

    @hook_spec
    def before_mapped_node_execute(
        self,
        metadata: GraphMetadata,
        inputs_list: list[dict[str, Any]],
    ) -> None:
        """
        Called before a mapped node begins execution.

        Args:
            metadata: Metadata for the mapped node to be executed.
            inputs_list: List of resolved inputs for each mapping.
        """

    @hook_spec
    def after_mapped_node_execute(
        self,
        metadata: GraphMetadata,
        inputs_list: list[dict[str, Any]],
        results: list[Any],
        duration: float,
    ) -> None:
        """
        Called after a mapped node completes execution successfully.

        Args:
            metadata: Metadata for the executed mapped node.
            inputs_list: List of resolved inputs for each mapping.
            duration: Execution time in seconds for all mappings.
            results: List of results produced by each mapping.
        """


class GraphSpec:
    """Hook specifications for graph-level execution events on the **coordinator**."""

    @hook_spec
    def before_graph_execute(
        self,
        graph_id: UUID,
        root_id: UUID,
        node_count: int,
        is_async: bool,
    ) -> None:
        """
        Called before graph execution begins.

        Args:
            graph_id: UUID of the entire graph execution
            root_id: UUID of the root node
            node_count: Total number of nodes in the graph
            is_async: True for async execution, False for sequential
        """

    @hook_spec
    def after_graph_execute(
        self,
        graph_id: UUID,
        root_id: UUID,
        result: Any,
        duration: float,
        is_async: bool,
    ) -> None:
        """
        Called after graph execution completes successfully.

        Args:
            graph_id: UUID of the entire graph execution
            root_id: UUID of the root node
            result: Final result of the graph execution
            duration: Total time taken to execute in seconds
            is_async: True for async execution, False for sequential
        """

    @hook_spec
    def on_graph_error(
        self,
        graph_id: UUID,
        root_id: UUID,
        error: Exception,
        duration: float,
        is_async: bool,
    ) -> None:
        """
        Called when graph execution fails.

        Args:
            graph_id: UUID of the entire graph execution
            root_id: UUID of the root node
            error: The exception that was raised
            duration: Time taken before failure in seconds
            is_async: True for async execution, False for sequential
        """
