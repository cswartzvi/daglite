"""Integration tests for RichLifecycleLoggingPlugin using evaluate()."""

import pytest
from daglite_rich.logging import RichLifecycleLoggingPlugin

from daglite import evaluate
from daglite import task

# Module-level tasks required for multiprocessing backend (pickle compatibility)


@task
def failing_map_task(x: int) -> int:
    """Task that fails for specific values."""
    if x == 2:
        raise ValueError(f"Failed on {x}")
    return x * 2


@task
def triple(x: int) -> int:
    """Task that triples a value."""
    return x * 3


class TestRichLifecycleLoggingIntegration:
    """Integration tests for RichLifecycleLoggingPlugin."""

    def test_logs_simple_task_execution(self, capsys):
        """Test that simple task execution is logged with rich formatting."""

        @task
        def add(a: int, b: int) -> int:
            return a + b

        result = evaluate(
            add(a=1, b=2),
            plugins=[RichLifecycleLoggingPlugin()],
        )

        assert result == 3

        # Check that we got log output (rich formatting will be present)
        captured = capsys.readouterr()
        log_text = captured.out
        assert "Starting evaluation" in log_text
        assert "Task 'add'" in log_text
        assert "Completed" in log_text

    def test_logs_task_chain(self, capsys):
        """Test that task chains are logged correctly."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def add(a: int, b: int) -> int:
            return a + b

        added = add(a=1, b=2)
        multiplied = multiply(x=added, y=3)

        result = evaluate(
            multiplied,
            plugins=[RichLifecycleLoggingPlugin()],
        )

        assert result == 9

        captured = capsys.readouterr()
        log_text = captured.out
        # Both tasks should be logged
        assert "Task 'add'" in log_text
        assert "Task 'multiply'" in log_text

    def test_logs_task_error(self, capsys):
        """Test that task errors are logged correctly with rich formatting."""

        @task
        def failing_task() -> int:
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError, match="Something went wrong"):
            evaluate(
                failing_task(),
                plugins=[RichLifecycleLoggingPlugin()],
            )

        captured = capsys.readouterr()
        log_text = captured.out
        # Check error logging
        assert "Task 'failing_task'" in log_text
        assert "Failed" in log_text
        assert "ValueError" in log_text

    def test_logs_mapped_task_execution(self, capsys):
        """Test that mapped task execution is logged correctly."""

        @task
        def square(x: int) -> int:
            return x * x

        result = evaluate(
            square.map(x=[1, 2, 3, 4]),
            plugins=[RichLifecycleLoggingPlugin()],
        )

        assert result == [1, 4, 9, 16]

        captured = capsys.readouterr()
        log_text = captured.out
        # Check mapped task logging
        assert "Task 'square'" in log_text
        assert "iterations" in log_text

    def test_logs_mapped_task_with_backend(self, capsys):
        """Test that mapped task with threads backend is logged correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        result = evaluate(
            double.with_options(backend_name="threads").map(x=[1, 2, 3]),
            plugins=[RichLifecycleLoggingPlugin()],
        )

        assert result == [2, 4, 6]

        captured = capsys.readouterr()
        log_text = captured.out
        # Should mention threads backend
        assert "threads backend" in log_text

    def test_logs_mapped_task_failure(self, capsys):
        """Test that mapped task failures are logged correctly."""

        @task
        def failing_square(x: int) -> int:
            if x == 2:
                raise ValueError("Failed on 2")
            return x * x

        with pytest.raises(ValueError, match="Failed on 2"):
            evaluate(
                failing_square.map(x=[1, 2, 3]),
                plugins=[RichLifecycleLoggingPlugin()],
            )

        captured = capsys.readouterr()
        log_text = captured.out
        # Check that failure is logged
        assert "Task 'failing_square" in log_text
        assert "iteration" in log_text.lower()
        assert "failed" in log_text.lower()
        assert "ValueError" in log_text
