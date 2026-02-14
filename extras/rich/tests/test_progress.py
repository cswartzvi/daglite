"""Unit tests for RichProgressPlugin."""

from unittest.mock import Mock
from unittest.mock import patch
from uuid import uuid4

from daglite_rich.progress import CustomBarColumn
from daglite_rich.progress import RichProgressPlugin
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TaskID

from daglite.graph.nodes.base import NodeMetadata
from daglite.plugins.registry import EventRegistry


class TestRichProgressPlugin:
    """Unit tests for RichProgressPlugin."""

    def test_initialization_defaults(self):
        """Test plugin initialization with default parameters."""
        plugin = RichProgressPlugin()

        assert plugin._id_to_task == {}
        assert plugin._root_task_id is None
        assert plugin._total_tasks == 0
        assert plugin._console is not None
        assert plugin._progress is not None
        assert plugin.secondary_style == "bold yellow"

    def test_initialization_with_custom_console(self):
        """Test plugin initialization with custom Console."""
        custom_console = Console()
        plugin = RichProgressPlugin(console=custom_console)

        assert plugin._console is custom_console

    def test_initialization_with_custom_progress(self):
        """Test plugin initialization with custom Progress instance."""
        custom_progress = Progress()
        plugin = RichProgressPlugin(progress=custom_progress)

        # Should use columns from custom progress
        assert plugin._progress is not None

    def test_initialization_with_custom_secondary_style(self):
        """Test plugin initialization with custom secondary_style."""
        plugin = RichProgressPlugin(secondary_style="bold green")

        assert plugin.secondary_style == "bold green"

    def test_initialization_with_no_secondary_style(self):
        """Test plugin initialization with secondary_style=None."""
        plugin = RichProgressPlugin(secondary_style=None)

        assert plugin.secondary_style is None

    def test_custom_bar_column_replacement(self):
        """Test that BarColumn is replaced with CustomBarColumn when secondary_style is set."""
        # Create progress with BarColumn
        progress = Progress(BarColumn())
        plugin = RichProgressPlugin(progress=progress, secondary_style="bold yellow")

        # Check that CustomBarColumn was used
        has_custom_bar = any(isinstance(col, CustomBarColumn) for col in plugin._progress.columns)
        assert has_custom_bar

    def test_no_custom_bar_column_when_secondary_style_none(self):
        """Test that BarColumn is not replaced when secondary_style is None."""
        progress = Progress(BarColumn())
        plugin = RichProgressPlugin(progress=progress, secondary_style=None)

        # Should keep original BarColumn
        has_bar_column = any(
            isinstance(col, BarColumn) and not isinstance(col, CustomBarColumn)
            for col in plugin._progress.columns
        )
        assert has_bar_column

    def test_before_graph_execute(self):
        """Test before_graph_execute hook."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()

        root_id = uuid4()
        node_count = 5

        plugin.before_graph_execute(root_id, node_count)

        assert plugin._total_tasks == 5
        plugin._progress.start.assert_called_once()
        plugin._progress.add_task.assert_called_once_with("Evaluating futures", total=5)

    def test_after_node_execute_with_reporter(self):
        """Test after_node_execute with reporter present."""
        plugin = RichProgressPlugin()
        metadata = NodeMetadata(id=uuid4(), name="test_task", key="test_task", kind="task")
        inputs = {"x": 1}
        result = 2
        duration = 0.5

        mock_reporter = Mock()

        plugin.after_node_execute(metadata, inputs, result, duration, mock_reporter)

        mock_reporter.report.assert_called_once_with(
            "daglite-node-end", data={"node_id": metadata.id}
        )

    def test_after_node_execute_without_reporter(self):
        """Test after_node_execute without reporter (fallback)."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._root_task_id = TaskID(0)

        metadata = NodeMetadata(id=uuid4(), name="test_task", key="test_task", kind="task")
        inputs = {"x": 1}
        result = 2
        duration = 0.5

        plugin.after_node_execute(metadata, inputs, result, duration, reporter=None)

        # Should advance root task
        plugin._progress.advance.assert_called_once_with(task_id=plugin._root_task_id)

    def test_on_node_error_with_reporter(self):
        """Test on_node_error with reporter present."""
        plugin = RichProgressPlugin()
        metadata = NodeMetadata(id=uuid4(), name="test_task", key="test_task", kind="task")
        inputs = {"x": 1}
        error = ValueError("test error")
        duration = 0.5

        mock_reporter = Mock()

        plugin.on_node_error(metadata, inputs, error, duration, mock_reporter)

        mock_reporter.report.assert_called_once_with(
            "daglite-node-end", data={"node_id": metadata.id}
        )

    def test_on_node_error_without_reporter(self):
        """Test on_node_error without reporter (fallback)."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._root_task_id = TaskID(0)

        metadata = NodeMetadata(id=uuid4(), name="test_task", key="test_task", kind="task")
        inputs = {"x": 1}
        error = ValueError("test error")
        duration = 0.5

        plugin.on_node_error(metadata, inputs, error, duration, reporter=None)

        # Should advance root task
        plugin._progress.advance.assert_called_once_with(task_id=plugin._root_task_id)

    def test_before_mapped_node_execute(self):
        """Test before_mapped_node_execute hook."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._progress.add_task.return_value = TaskID(1)

        metadata = NodeMetadata(id=uuid4(), name="map_task", key="map_task", kind="map")

        plugin.before_mapped_node_execute(metadata, iteration_count=3)

        plugin._progress.add_task.assert_called_once_with(
            "Mapping 'map_task'", total=3, bar_style="bold yellow"
        )
        assert metadata.id in plugin._id_to_task

    def test_after_mapped_node_execute(self):
        """Test after_mapped_node_execute hook."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._root_task_id = TaskID(0)

        metadata = NodeMetadata(id=uuid4(), name="map_task", key="map_task", kind="map")
        map_task_id = TaskID(1)
        plugin._id_to_task[metadata.id] = map_task_id

        duration = 0.5

        plugin.after_mapped_node_execute(metadata, iteration_count=3, duration=duration)

        # Should hide the map task and advance root task
        plugin._progress.update.assert_called_once_with(map_task_id, visible=False)
        plugin._progress.advance.assert_called_once_with(task_id=plugin._root_task_id)
        assert metadata.id not in plugin._id_to_task

    def test_after_graph_execute(self):
        """Test after_graph_execute hook."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._root_task_id = TaskID(0)
        plugin._total_tasks = 5

        root_id = uuid4()
        result = 42
        duration = 1.5

        plugin.after_graph_execute(root_id, result, duration)

        plugin._progress.update.assert_called_once_with(task_id=plugin._root_task_id, completed=5)
        plugin._progress.refresh.assert_called_once()
        plugin._progress.stop.assert_called_once()

    def test_register_event_handlers(self):
        """Test that event handlers are registered correctly."""
        plugin = RichProgressPlugin()
        registry = Mock(spec=EventRegistry)

        plugin.register_event_handlers(registry)

        registry.register.assert_called_once_with("daglite-node-end", plugin._handle_node_update)

    def test_handle_node_update_for_mapped_task(self):
        """Test _handle_node_update for mapped task iteration."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()

        node_id = uuid4()
        map_task_id = TaskID(1)
        plugin._id_to_task[node_id] = map_task_id

        event = {"node_id": node_id}
        plugin._handle_node_update(event)

        # Should advance the mapped task
        plugin._progress.advance.assert_called_once_with(task_id=map_task_id)

    def test_handle_node_update_for_regular_task(self):
        """Test _handle_node_update for regular (non-mapped) task."""
        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._root_task_id = TaskID(0)

        node_id = uuid4()
        event = {"node_id": node_id}

        plugin._handle_node_update(event)

        # Should advance the root task
        plugin._progress.advance.assert_called_once_with(task_id=plugin._root_task_id)

    def test_serialization(self):
        """Test plugin serialization to/from config."""
        plugin = RichProgressPlugin(secondary_style="bold green")

        # Serialize
        config = plugin.to_config()

        # Deserialize
        restored_plugin = RichProgressPlugin.from_config(config)

        assert restored_plugin is not None
        assert isinstance(restored_plugin, RichProgressPlugin)


class TestCustomBarColumn:
    """Unit tests for CustomBarColumn."""

    def test_render_with_bar_style(self):
        """Test that render applies bar_style from task fields."""
        column = CustomBarColumn()

        # Create a mock task with bar_style field
        mock_task = Mock()
        mock_task.fields = {"bar_style": "bold red"}

        # Mock the parent render method
        with patch.object(BarColumn, "render") as mock_render:
            column.render(mock_task)

            # Should have temporarily set complete_style to bar_style
            mock_render.assert_called_once_with(mock_task)

    def test_render_without_bar_style(self):
        """Test that render works normally without bar_style field."""
        column = CustomBarColumn()

        # Create a mock task without bar_style field
        mock_task = Mock()
        mock_task.fields = {}

        # Mock the parent render method
        with patch.object(BarColumn, "render") as mock_render:
            column.render(mock_task)

            mock_render.assert_called_once_with(mock_task)

    def test_bar_style_restoration(self):
        """Test that original complete_style is restored after rendering."""
        column = CustomBarColumn()
        original_style = column.complete_style

        mock_task = Mock()
        mock_task.fields = {"bar_style": "bold red"}

        with patch.object(BarColumn, "render"):
            column.render(mock_task)

        # Original style should be restored
        assert column.complete_style == original_style


class TestRichProgressOnCacheHit:
    """Tests for RichProgressPlugin.on_cache_hit hook."""

    def test_on_cache_hit_with_reporter(self):
        """Test that on_cache_hit reports event when reporter is present."""
        from daglite.graph.nodes.base import NodeMetadata

        plugin = RichProgressPlugin()
        metadata = NodeMetadata(id=uuid4(), name="test_task", kind="task", key="test_task")

        mock_reporter = Mock()

        plugin.on_cache_hit(
            func=Mock(),
            metadata=metadata,
            inputs={"x": 5},
            result=10,
            reporter=mock_reporter,
        )

        mock_reporter.report.assert_called_once_with(
            "daglite-node-end", data={"node_id": metadata.id}
        )

    def test_on_cache_hit_advances_progress(self):
        """Test that on_cache_hit advances the progress bar."""
        from daglite.graph.nodes.base import NodeMetadata

        plugin = RichProgressPlugin()
        plugin._progress = Mock()

        node_id = uuid4()
        metadata = NodeMetadata(id=node_id, name="test_task", kind="task", key="test_task")

        # Register a task for this node
        task_id = TaskID(1)
        plugin._id_to_task[node_id] = task_id

        # Call on_cache_hit with no reporter (fallback path)
        plugin.on_cache_hit(
            func=Mock(),
            metadata=metadata,
            inputs={"x": 5},
            result=10,
            reporter=None,
        )

        # Progress should advance
        plugin._progress.advance.assert_called_once_with(task_id=task_id)

    def test_on_cache_hit_without_registered_task(self):
        """Test that on_cache_hit handles missing task gracefully."""
        from daglite.graph.nodes.base import NodeMetadata

        plugin = RichProgressPlugin()
        plugin._progress = Mock()
        plugin._root_task_id = TaskID(0)

        node_id = uuid4()
        metadata = NodeMetadata(id=node_id, name="test_task", kind="task", key="test_task")

        # No task registered - should fall back to root task
        plugin.on_cache_hit(
            func=Mock(),
            metadata=metadata,
            inputs={"x": 5},
            result=10,
            reporter=None,
        )

        # Should advance root task
        plugin._progress.advance.assert_called_once_with(task_id=plugin._root_task_id)
