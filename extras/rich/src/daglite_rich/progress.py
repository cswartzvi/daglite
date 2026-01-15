from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

from rich import get_console
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TaskID
from typing_extensions import override

from daglite.graph.base import GraphMetadata
from daglite.plugins.base import BidirectionalPlugin
from daglite.plugins.base import SerializablePlugin
from daglite.plugins.events import EventRegistry
from daglite.plugins.hooks.markers import hook_impl
from daglite.plugins.reporters import EventReporter

if TYPE_CHECKING:
    from rich.progress import Task as RichTask
    from rich.progress_bar import ProgressBar as RichProgressBar
else:
    RichTask = Any
    RichProgressBar = Any


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

    __config_attrs__: ClassVar[list[str]] = []

    def __init__(
        self,
        console: Console | None = None,
        progress: Progress | None = None,
        secondary_style: str | None = "bold yellow",
    ) -> None:
        self._id_to_task: dict[UUID, TaskID] = {}
        self._root_task_id: TaskID | None = None
        self._total_tasks = 0

        self._console = console or get_console()
        self.secondary_style = secondary_style
        columns = progress.columns if progress else Progress.get_default_columns()

        # Use CustomBarColumn to support per-task styling when secondary_style is set
        if secondary_style is not None:
            updated_columns = []
            for column in columns:
                if isinstance(column, BarColumn):
                    # Replace with CustomBarColumn (don't set complete_style here)
                    col = CustomBarColumn(bar_width=column.bar_width)
                    updated_columns.append(col)
                else:
                    updated_columns.append(column)
            columns = updated_columns

        # Initialize Progress instance
        self._progress = Progress(*columns, console=self._console, transient=False)

    @override
    def to_config(self) -> dict[str, Any]:
        return {}  # No configurable attributes for now

    @classmethod
    @override
    def from_config(cls, config: dict[str, Any]) -> "RichProgressPlugin":
        return cls()  # Create new instance with defaults

    @hook_impl(trylast=True)
    def before_graph_execute(self, root_id: UUID, node_count: int, is_async: bool) -> None:
        self._progress.start()
        self._total_tasks = node_count
        self._root_task_id = self._progress.add_task("Evaluating", total=self._total_tasks)

    @hook_impl(trylast=True)
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        if reporter:
            reporter.report("daglite-node-end", data={"node_id": metadata.id})
        else:  # pragma: no cover
            # Fallback if no reporter is available
            self._handle_node_update({"node_id": metadata.id})

    @hook_impl(trylast=True)
    def on_cache_hit(
        self,
        func: Any,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        reporter: EventReporter | None,
    ) -> None:
        if reporter:
            reporter.report("daglite-node-end", data={"node_id": metadata.id})
        else:  # pragma: no cover
            # Fallback if no reporter is available
            self._handle_node_update({"node_id": metadata.id})

    @hook_impl(trylast=True)
    def on_node_error(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        error: Exception,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        if reporter:
            reporter.report("daglite-node-end", data={"node_id": metadata.id})
        else:  # pragma: no cover
            # Fallback if no reporter is available
            self._handle_node_update({"node_id": metadata.id})

    @hook_impl(trylast=True)
    def before_mapped_node_execute(
        self,
        metadata: GraphMetadata,
        inputs_list: list[dict[str, Any]],
    ) -> None:
        description = f"Mapping '{metadata.key or metadata.name}'"
        map_task_id = self._progress.add_task(
            description, total=len(inputs_list), bar_style=self.secondary_style
        )
        self._id_to_task[metadata.id] = map_task_id

    @hook_impl(trylast=True)
    def after_mapped_node_execute(
        self,
        metadata: GraphMetadata,
        inputs_list: list[dict[str, Any]],
        results: list[Any],
        duration: float,
    ) -> None:
        map_task_id = self._id_to_task.pop(metadata.id, None)
        if map_task_id is not None:  # pragma: no branch
            self._progress.update(map_task_id, visible=False)

        # Advance the main progress bar for the completed map node
        assert self._root_task_id is not None
        self._progress.advance(task_id=self._root_task_id)

    @hook_impl(trylast=True)
    def after_graph_execute(
        self, root_id: UUID, result: Any, duration: float, is_async: bool
    ) -> None:
        assert self._root_task_id is not None
        self._progress.update(task_id=self._root_task_id, completed=self._total_tasks)
        self._progress.refresh()
        self._progress.stop()

    @override
    def register_event_handlers(self, registry: EventRegistry) -> None:
        registry.register("daglite-node-end", self._handle_node_update)

    def _handle_node_update(self, event: dict) -> None:
        # Check if this is a map iteration completion
        node_id = event.get("node_id")
        if node_id and node_id in self._id_to_task:
            self._progress.advance(task_id=self._id_to_task[node_id])
        else:
            assert self._root_task_id is not None
            self._progress.advance(task_id=self._root_task_id)


class CustomBarColumn(BarColumn):
    """BarColumn that supports per-task styling via task.fields['bar_style']."""

    @override
    def render(self, task: RichTask) -> RichProgressBar:
        """Render the bar with optional per-task style."""
        bar_style = task.fields.get("bar_style")
        if bar_style:
            original_complete_style = self.complete_style
            self.complete_style = bar_style
            result = super().render(task)
            self.complete_style = original_complete_style
            return result
        return super().render(task)
