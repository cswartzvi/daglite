from dataclasses import asdict
from typing import Any
from uuid import UUID

from rich import get_console
from rich.console import Console
from rich.progress import Progress
from rich.progress import TaskID
from typing_extensions import override

from daglite.graph.base import GraphMetadata
from daglite.plugins.default.logging import BidirectionalPlugin
from daglite.plugins.events import EventRegistry
from daglite.plugins.hooks.markers import hook_impl
from daglite.plugins.reporters import EventReporter


class RichProgressPlugin(BidirectionalPlugin):
    """
    Plugin that adds rich progress bars and logging to daglite tasks.

    This plugin registers event handlers to display rich progress bars on the coordinator
    side, enhancing the visibility of task execution progress and log messages.

    Args:
        console: Optional rich Console instance for logging output. If not provided,
            the default console (`rich.get_console()`) will be used.
        progress: Optional rich Progress instance for displaying progress bars. If not provided,
            a default Progress instance will be created.
    """

    def __init__(
        self,
        console: Console | None = None,
        progress: Progress | None = None,
    ) -> None:
        self._console = console or get_console()
        self._progress = progress or Progress(console=self._console, transient=False)
        self._id_to_task = {}
        self._root_task_id: TaskID | None = None

    @hook_impl
    def before_graph_execute(self, root_id: UUID, node_count: int, is_async: bool) -> None:
        """Called before a graph begins execution."""
        self._progress.start()
        self._root_task_id = self._progress.add_task("Executing DAG...", total=node_count)

    @hook_impl
    def before_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        reporter: EventReporter | None = None,
    ) -> None:
        """Called before a node begins execution."""
        data = {"metadata": asdict(metadata)}
        if reporter:
            reporter.report("task_start", data=data)
        else:
            self._handle_task_start(data)

    @hook_impl
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        duration: float,
        reporter: EventReporter | None = None,
    ) -> None:
        """Called after a node completes execution successfully."""
        data = {"metadata": asdict(metadata)}
        if reporter:
            reporter.report("task_complete", data=data)
        else:
            self._handle_task_update(data)

    @hook_impl
    def on_node_error(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        error: Exception,
        duration: float,
        reporter: EventReporter | None = None,
    ) -> None:
        """Called when a node execution fails."""
        if reporter:
            reporter.report("task_complete", {"metadata": asdict(metadata)})
        else:
            pass

    @hook_impl
    def after_graph_execute(
        self, root_id: UUID, result: Any, duration: float, is_async: bool
    ) -> None:
        """Called after a graph completes execution."""
        assert self._root_task_id is not None
        self._progress.update(
            task_id=self._root_task_id, advance=1.0, description="Execution complete"
        )
        self._progress.refresh()
        self._progress.stop()

    @override
    def register_event_handlers(self, registry: EventRegistry) -> None:
        registry.register("task_start", self._handle_task_start)
        registry.register("task_update", self._handle_task_update)

    def _handle_task_start(self, event: dict) -> None:
        metadata = GraphMetadata(**event["metadata"])
        description = f"Executing: {metadata.key or metadata.name}"
        assert self._root_task_id is not None
        self._progress.update(task_id=self._root_task_id, description=description)

    def _handle_task_update(self, event: dict) -> None:
        assert self._root_task_id is not None
        self._progress.advance(task_id=self._root_task_id)
