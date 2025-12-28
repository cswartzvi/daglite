from typing import Any, ClassVar
from uuid import UUID

from rich import get_console
from rich.console import Console
from rich.progress import Progress
from rich.progress import TaskID
from typing_extensions import override

from daglite.graph.base import GraphMetadata
from daglite.plugins.base import BidirectionalPlugin
from daglite.plugins.base import SerializablePlugin
from daglite.plugins.events import EventRegistry
from daglite.plugins.hooks.markers import hook_impl
from daglite.plugins.reporters import EventReporter


class RichProgressPlugin(BidirectionalPlugin, SerializablePlugin):
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

    __config_attrs__: ClassVar[list[str]] = []  # No serializable config

    def __init__(
        self,
        console: Console | None = None,
        progress: Progress | None = None,
    ) -> None:
        self._console = console or get_console()
        self._progress = progress or Progress(console=self._console, transient=False)
        self._id_to_task = {}
        self._root_task_id: TaskID | None = None
        self._total_tasks = 0

    @override
    def to_config(self) -> dict[str, Any]:
        return {}  # No config to serialize

    @classmethod
    @override
    def from_config(cls, config: dict[str, Any]) -> "RichProgressPlugin":
        return cls()  # Create new instance with defaults

    @hook_impl
    def before_graph_execute(self, root_id: UUID, node_count: int, is_async: bool) -> None:
        self._progress.start()
        self._total_tasks = node_count
        self._root_task_id = self._progress.add_task("Initializing", total=self._total_tasks)

    @hook_impl
    def before_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        reporter: EventReporter | None,
    ) -> None:
        data = {"name": metadata.name, "key": metadata.key}
        if reporter:
            reporter.report("node_start", data=data)
        else:  # pragma: no cover
            # Fallback if no reporter is available
            self._handle_task_start(data)

    @hook_impl
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        if reporter:
            reporter.report("node_end", data={})
        else:  # pragma: no cover
            # Fallback if no reporter is available
            self._handle_task_update({})

    @hook_impl
    def on_node_error(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        error: Exception,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        if reporter:
            reporter.report("node_end", data={})
        else:  # pragma: no cover
            # Fallback if no reporter is available
            self._handle_task_update({})

    @hook_impl
    def after_graph_execute(
        self, root_id: UUID, result: Any, duration: float, is_async: bool
    ) -> None:
        assert self._root_task_id is not None
        self._progress.update(
            task_id=self._root_task_id,
            completed=self._total_tasks,
            description="Execution complete",
        )
        self._progress.refresh()
        self._progress.stop()

    @override
    def register_event_handlers(self, registry: EventRegistry) -> None:
        registry.register("node_start", self._handle_task_start)
        registry.register("node_end", self._handle_task_update)

    def _handle_task_start(self, event: dict) -> None:
        description = f"Executing: {event['key'] or event['name']}"
        assert self._root_task_id is not None
        self._progress.update(task_id=self._root_task_id, description=description)

    def _handle_task_update(self, event: dict) -> None:
        assert self._root_task_id is not None
        self._progress.advance(task_id=self._root_task_id)
