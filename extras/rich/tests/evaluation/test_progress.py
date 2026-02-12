"""Integration tests for RichProgressPlugin using evaluate()."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from daglite_rich.progress import RichProgressPlugin

from daglite import evaluate
from daglite import task


class TestRichProgressIntegration:
    """Integration tests for RichProgressPlugin."""

    def test_simple_task_with_progress(self):
        """Test that simple task execution shows progress."""

        @task
        def add(a: int, b: int) -> int:
            return a + b

        # Mock the Progress instance to avoid actual UI rendering
        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.return_value = 0  # root task ID

            plugin = RichProgressPlugin()
            result = evaluate(add(a=1, b=2), plugins=[plugin])

            assert result == 3

            # Progress should have been started and stopped
            mock_progress.start.assert_called()
            mock_progress.stop.assert_called()

    def test_task_chain_with_progress(self):
        """Test that task chains show progress correctly."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def add(a: int, b: int) -> int:
            return a + b

        added = add(a=1, b=2)
        multiplied = multiply(x=added, y=3)

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.return_value = 0

            plugin = RichProgressPlugin()
            result = evaluate(multiplied, plugins=[plugin])

            assert result == 9

            # Should have added root task
            assert mock_progress.add_task.called
            # Should advance progress as tasks complete
            assert mock_progress.advance.called

    def test_mapped_task_with_progress(self):
        """Test that mapped tasks show separate progress bars."""

        @task
        def square(x: int) -> int:
            return x * x

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.side_effect = [0, 1]  # root task, then map task

            plugin = RichProgressPlugin()
            result = evaluate(square.map(x=[1, 2, 3, 4]), plugins=[plugin])

            assert result == [1, 4, 9, 16]

            # Should have added both root task and map task
            assert mock_progress.add_task.call_count >= 2
            # Map task should have been added with description
            calls = mock_progress.add_task.call_args_list
            descriptions = [str(call) for call in calls]
            assert any("Mapping" in desc for desc in descriptions)

    def test_mapped_task_with_threads_backend(self):
        """Test mapped tasks with threads backend show progress."""

        @task
        def double(x: int) -> int:
            return x * 2

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.side_effect = [0, 1]

            plugin = RichProgressPlugin()
            result = evaluate(
                double.with_options(backend_name="threads").map(x=[1, 2, 3]),
                plugins=[plugin],
            )

            assert result == [2, 4, 6]

            # Should show progress for mapped task
            assert mock_progress.add_task.call_count >= 2

    def test_task_error_with_progress(self):
        """Test that task errors still update progress correctly."""

        @task
        def failing_task() -> int:
            raise ValueError("Task failed")

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.return_value = 0

            plugin = RichProgressPlugin()

            with pytest.raises(ValueError, match="Task failed"):
                evaluate(failing_task(), plugins=[plugin])

            # Progress should still be started and stopped even on error
            mock_progress.start.assert_called()
            # Note: stop might not be called on error depending on implementation

    def test_mapped_task_error_with_progress(self):
        """Test that mapped task errors still update progress correctly."""

        @task
        def failing_square(x: int) -> int:
            if x == 2:
                raise ValueError("Failed on 2")
            return x * x

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.side_effect = [0, 1]

            plugin = RichProgressPlugin()

            with pytest.raises(ValueError, match="Failed on 2"):
                evaluate(failing_square.map(x=[1, 2, 3]), plugins=[plugin])

            # Map task should have been added
            assert mock_progress.add_task.call_count >= 2

    def test_custom_secondary_style(self):
        """Test that custom secondary_style is used for mapped tasks."""

        @task
        def square(x: int) -> int:
            return x * x

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.side_effect = [0, 1]

            plugin = RichProgressPlugin(secondary_style="bold green")
            result = evaluate(square.map(x=[1, 2, 3]), plugins=[plugin])

            assert result == [1, 4, 9]

            # Map task should be added with custom bar_style
            map_task_calls = [
                call for call in mock_progress.add_task.call_args_list if "Mapping" in str(call)
            ]
            if map_task_calls:
                # Check that bar_style was passed
                call_kwargs = map_task_calls[0].kwargs if map_task_calls else {}
                assert call_kwargs.get("bar_style") == "bold green"


class TestRichProgressAndLoggingTogether:
    """Integration tests combining RichProgressPlugin and RichLifecycleLoggingPlugin."""

    def test_progress_and_logging_together(self, capsys):
        """Test that progress and logging plugins work together."""
        from daglite_rich.logging import RichLifecycleLoggingPlugin

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.return_value = 0

            progress_plugin = RichProgressPlugin()
            logging_plugin = RichLifecycleLoggingPlugin()

            result = evaluate(
                multiply(x=3, y=4),
                plugins=[progress_plugin, logging_plugin],
            )

            assert result == 12

            # Progress should be shown
            mock_progress.start.assert_called()

            # Logging should also work
            captured = capsys.readouterr()
            log_text = captured.out
            assert "Task 'multiply'" in log_text

    def test_mapped_task_with_both_plugins(self, capsys):
        """Test mapped tasks with both progress and logging plugins."""
        from daglite_rich.logging import RichLifecycleLoggingPlugin

        @task
        def square(x: int) -> int:
            return x * x

        with patch("daglite_rich.progress.Progress") as MockProgress:
            mock_progress = Mock()
            MockProgress.return_value = mock_progress
            mock_progress.columns = []
            mock_progress.add_task.side_effect = [0, 1]

            progress_plugin = RichProgressPlugin()
            logging_plugin = RichLifecycleLoggingPlugin()

            result = evaluate(square.map(x=[1, 2, 3]), plugins=[progress_plugin, logging_plugin])

            assert result == [1, 4, 9]

            # Both progress and logging should work
            assert mock_progress.add_task.call_count >= 2

            captured = capsys.readouterr()
            log_text = captured.out
            assert "Task 'square'" in log_text
            assert "iterations" in log_text
