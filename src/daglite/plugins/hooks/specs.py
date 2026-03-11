"""Hook specifications for daglite execution lifecycle events."""

from typing import Any
from uuid import UUID

from daglite.plugins.hooks.markers import hook_spec
from daglite.plugins.reporters import EventReporter
from daglite.tasks import TaskMetadata


class SessionLifecycleSpecs:
    """Hook specifications for session and workflow lifecycle events."""

    @hook_spec
    def before_session_start(
        self,
        session_id: UUID,
    ) -> None:
        """
        Called when a session or workflow begins.

        Args:
            session_id: Unique identifier for this session.
        """

    @hook_spec
    def after_session_end(
        self,
        session_id: UUID,
        duration: float,
    ) -> None:
        """
        Called when a session or workflow ends.

        Args:
            session_id: Unique identifier for this session.
            duration: Total wall-clock time in seconds.
        """


class WorkerSideNodeSpecs:
    """Hook specifications for node-level execution events on the **backend worker**."""

    @hook_spec
    def before_node_execute(self, metadata: TaskMetadata, reporter: EventReporter | None) -> None:
        """
        Called before a node begins execution.

        Args:
            metadata: Metadata for the node to be executed.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def after_node_execute(
        self,
        metadata: TaskMetadata,
        result: Any,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called after a node completes execution successfully.

        Args:
            metadata: Metadata for the executed node.
            result: Result produced by the node execution.
            duration: Time taken to execute in seconds.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def on_node_error(
        self,
        metadata: TaskMetadata,
        error: Exception,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called when a node execution fails.

        Args:
            metadata: Metadata for the executed node.
            error: The exception that was raised.
            duration: Time taken before failure in seconds.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def node_iteration(
        self,
        metadata: TaskMetadata,
        index: int,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called each time a generator task yields an item.

        Args:
            metadata: Metadata for the streaming task.
            index: Zero-based index of the yielded item.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def before_node_retry(
        self,
        metadata: TaskMetadata,
        attempt: int,
        error: Exception,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called before retrying a failed node execution.

        Args:
            metadata: Metadata for the node to be retried.
            attempt: Retry number (1-indexed, so attempt=1 is the first retry).
            error: The exception that caused the previous attempt to fail.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def after_node_retry(
        self,
        metadata: TaskMetadata,
        attempt: int,
        succeeded: bool,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called after a retry attempt completes.

        Args:
            metadata: Metadata for the retried node.
            attempt: Retry number (1-indexed).
            succeeded: True if this retry attempt succeeded, False if it failed.
            reporter: Optional event reporter for this execution context.
        """

    @hook_spec
    def on_cache_hit(
        self,
        metadata: TaskMetadata,
        key: str,
        value: Any,
        reporter: EventReporter | None,
    ) -> None:
        """
        Called when a cached result is used instead of executing the node.

        Args:
            metadata: Metadata for the node.
            key: Cache key that was looked up.
            value: Cached result that was returned.
            reporter: Optional event reporter.
        """

    @hook_spec
    def before_dataset_save(
        self,
        key: str,
        value: Any,
        format: str | None,
        options: dict[str, Any] | None,
        metadata: TaskMetadata | None = None,
    ) -> None:
        """
        Called immediately before a dataset output is written to storage.

        Args:
            key: Storage key/path for the output.
            value: The Python object about to be serialized and saved.
            format: Serialization format (e.g. `"pickle"`, `"text"`).
            options: Additional options passed to the Dataset constructor.
            metadata: Task metadata if the save originates from a task, else ``None``.
        """

    @hook_spec
    def after_dataset_save(
        self,
        key: str,
        value: Any,
        format: str | None,
        options: dict[str, Any] | None,
        metadata: TaskMetadata | None = None,
    ) -> None:
        """
        Called immediately after a dataset output has been written to storage.

        Args:
            key: Storage key/path for the output.
            value: The Python object that was serialized and saved.
            format: Serialization format (e.g. `"pickle"`, `"text"`).
            options: Additional options passed to the Dataset constructor.
            metadata: Task metadata if the save originates from a task, else ``None``.
        """

    @hook_spec
    def before_dataset_load(
        self,
        key: str,
        return_type: type | None,
        format: str | None,
        options: dict[str, Any] | None,
        metadata: TaskMetadata | None = None,
    ) -> None:
        """
        Called immediately before a dataset is loaded from storage.

        Args:
            key: Storage key/path being loaded.
            return_type: Expected Python type for deserialization dispatch.
            format: Serialization format hint (e.g. `"pickle"`, `"pandas/csv"`).
            options: Additional options passed to the Dataset constructor.
            metadata: Task metadata if the load originates from a task, else ``None``.
        """

    @hook_spec
    def after_dataset_load(
        self,
        key: str,
        return_type: type | None,
        format: str | None,
        options: dict[str, Any] | None,
        result: Any,
        duration: float,
        metadata: TaskMetadata | None = None,
    ) -> None:
        """
        Called immediately after a dataset has been loaded from storage.

        Args:
            key: Storage key/path that was loaded.
            return_type: Expected Python type for deserialization dispatch.
            format: Serialization format hint (e.g. `"pickle"`, `"pandas/csv"`).
            options: Additional options passed to the Dataset constructor.
            result: The deserialized Python object.
            duration: Time taken to load in seconds.
            metadata: Task metadata if the load originates from a task, else ``None``.
        """
