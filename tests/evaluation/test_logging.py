"""Integration tests for centralized logging.

Tests verify that logs from worker tasks are properly routed to the coordinator
across different execution backends using the logging plugins.
"""

import logging

import pytest

from daglite import evaluate
from daglite import task
from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
from daglite.plugins.builtin.logging import get_logger


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests to prevent handler accumulation."""
    # Remove all handlers from all loggers before test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)

    logging.root.handlers.clear()

    yield

    # Clean up after test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)

    logging.root.handlers.clear()


# Module-level tasks required for multiprocessing backend (pickle compatibility)


@task
def worker_task(x: int) -> int:
    """Task that logs during execution."""
    logger = get_logger(__name__)
    logger.info(f"Worker processing {x}")
    return x * 2


@task
def failing_map_task(x: int) -> int:
    """Task that fails for specific values."""
    logger = get_logger()
    if x == 2:
        logger.error(f"Processing failed for {x}")
        raise ValueError(f"Failed on {x}")
    logger.info(f"Processing {x}")
    return x * 2


@task
def triple(x: int) -> int:
    """Task that triples a value."""
    return x * 3


@task
def metadata_task(x: int) -> int:
    """Task for testing metadata preservation."""
    logger = get_logger()
    logger.info(f"Metadata test {x}")
    return x


@task
def exception_task(x: int) -> int:
    """Task that logs an exception."""
    logger = get_logger(__name__)
    try:
        raise ValueError(f"Worker error for {x}")
    except ValueError:
        logger.exception("Exception in worker")
    return x


@task
def map_worker(x: int) -> int:
    """Task for testing mapped execution."""
    logger = get_logger()
    logger.info(f"Map processing {x}")
    return x * 2


@task
def multi_level_task(x: int) -> int:
    """Task that logs at different levels."""
    logger = get_logger(__name__)
    logger.debug(f"Debug: {x}")
    logger.info(f"Info: {x}")
    logger.warning(f"Warning: {x}")
    logger.error(f"Error: {x}")
    return x


@task
def multi_logger_task(x: int) -> int:
    """Task that uses multiple loggers."""
    logger1 = get_logger("custom.logger.one")
    logger2 = get_logger("custom.logger.two")
    logger1.info(f"Logger one: {x}")
    logger2.info(f"Logger two: {x}")
    return x


@task
def contextual_logging_task(x: int) -> int:
    """Task that uses contextual logging (no name argument)."""
    logger = get_logger()  # Should auto-derive name from task context
    logger.info(f"Processing {x}")
    return x * 2


@task
def dedupe_handler_task(x: int) -> int:
    """Task that calls get_logger multiple times to test handler deduplication."""
    from daglite.plugins.builtin.logging import _ReporterHandler

    logger1 = get_logger("test.dedupe.unique")
    logger2 = get_logger("test.dedupe.unique")
    logger3 = get_logger("test.dedupe.unique")

    # All should share the same underlying logger
    assert logger1.logger is logger2.logger is logger3.logger

    # Check handler count
    base_logger = logger1.logger
    reporter_handlers = [h for h in base_logger.handlers if isinstance(h, _ReporterHandler)]
    # Should have exactly 1 handler, not 3
    assert len(reporter_handlers) == 1

    # Verify all three adapters share the exact same handler instance
    assert all(h is reporter_handlers[0] for h in reporter_handlers)

    logger1.info(f"Test {x}")
    return x


class TestCentralizedLoggingIntegration:
    """Integration tests for centralized logging with different backends."""

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_logging_with_reporter_backends(self, backend_name, caplog):
        """Test that logging works correctly with backends that use reporters."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        # Configure task with backend
        task_with_backend = worker_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 84

        # Verify log was captured and routed through coordinator
        assert "Worker processing 42" in caplog.text
        assert len(caplog.records) > 0

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_task_metadata_preserved(self, backend_name, caplog):
        """Test that daglite_* metadata fields are preserved from workers."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = metadata_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=99), plugins=[plugin])
            assert result == 99

        # Verify task metadata fields exist in log records
        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert hasattr(record, "daglite_task_name")
        assert hasattr(record, "daglite_task_id")
        assert hasattr(record, "daglite_task_key")
        assert record.daglite_task_name == "metadata_task"

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_exception_logging(self, backend_name, caplog):
        """Test that exception tracebacks are transmitted from workers."""
        plugin = CentralizedLoggingPlugin(level=logging.ERROR)

        task_with_backend = exception_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.ERROR):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 42

        # Verify exception details are in logs
        assert "Exception in worker" in caplog.text
        assert "ValueError" in caplog.text
        assert "Worker error for 42" in caplog.text

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_mapped_task_logging(self, backend_name, caplog):
        """Test logging from mapped tasks with node keys."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = map_worker.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            future = task_with_backend.product(x=[1, 2, 3])
            results = evaluate(future, plugins=[plugin])
            assert results == [2, 4, 6]

        # Verify all mapped tasks logged
        log_text = caplog.text
        assert "Map processing 1" in log_text
        assert "Map processing 2" in log_text
        assert "Map processing 3" in log_text

        # Verify node keys are present (map_worker[0], map_worker[1], etc.)
        records = caplog.records
        assert len(records) >= 3
        for record in records:
            assert hasattr(record, "daglite_task_key")
            # Node keys should be like "map_worker[0]", "map_worker[1]", etc.
            assert record.daglite_task_key.startswith("map_worker")

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_log_level_filtering(self, backend_name, caplog):
        """Test that log level filtering works correctly."""
        plugin = CentralizedLoggingPlugin(level=logging.WARNING)

        task_with_backend = multi_level_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.WARNING):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 42

        # Only WARNING and ERROR should be present
        assert "Debug: 42" not in caplog.text
        assert "Info: 42" not in caplog.text
        assert "Warning: 42" in caplog.text
        assert "Error: 42" in caplog.text

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_multiple_loggers(self, backend_name, caplog):
        """Test that multiple logger names work correctly."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = multi_logger_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 42

        # Both logger messages should appear
        assert "Logger one: 42" in caplog.text
        assert "Logger two: 42" in caplog.text

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_mapped_task_error_logging(self, backend_name, caplog):
        """Test that errors in mapped tasks are logged correctly."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = failing_map_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError, match="Failed on 2"):
                future = task_with_backend.product(x=[1, 2, 3])
                evaluate(future, plugins=[plugin])

        # Verify error was logged before the exception was raised
        assert "Processing failed for 2" in caplog.text


class TestCentralizedLoggingDirectReporter:
    """Test centralized logging with DirectReporter backends (sequential, threads)."""

    @pytest.mark.parametrize("backend_name", ["sequential", "threads"])
    def test_contextual_logging(self, backend_name, caplog):
        """Test that get_logger() without name uses daglite.tasks logger."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        # Use sequential backend when backend_name is "sequential" (default behavior)
        if backend_name == "sequential":
            task_with_backend = contextual_logging_task
        else:
            task_with_backend = contextual_logging_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 84  # 42 * 2

        # Logger should use daglite.tasks (not task-specific name)
        assert "daglite.tasks" in caplog.text
        assert "Processing 42" in caplog.text


class TestCentralizedLoggingProcessBackend:
    """Test centralized logging specific behaviors with process backend."""

    def test_get_logger_no_duplicate_handlers(self, caplog):
        """Test that multiple get_logger calls for same name don't duplicate handlers."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            result = evaluate(
                dedupe_handler_task.with_options(backend_name="processes")(x=42),
                plugins=[plugin],
            )
            assert result == 42

        # Verify the log was captured
        assert "Test 42" in caplog.text


class TestLifecycleLoggingWithSequentialEvaluation:
    """Test LifecycleLoggingPlugin with sequential backend evaluation."""

    def test_logs_simple_task_execution(self, capsys):
        """Test that simple task execution is logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def add(a: int, b: int) -> int:
            return a + b

        # Run with logging plugin
        result = evaluate(
            add(a=1, b=2),
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == 3

        # Check that we got the expected log messages
        captured = capsys.readouterr()
        log_text = captured.out
        assert "Starting evaluation" in log_text
        assert "Task 'add' - Starting task using sequential backend" in log_text
        assert "Task 'add' - Completed task successfully in" in log_text
        assert "Completed evaluation" in log_text and "successfully in" in log_text

    def test_logs_task_chain(self, capsys):
        """Test that task chains are logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def add(a: int, b: int) -> int:
            return a + b

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        added = add(a=1, b=2)
        multiplied = multiply(x=added, y=3)

        result = evaluate(
            multiplied,
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == 9

        captured = capsys.readouterr()
        log_text = captured.out
        # Check all tasks are logged
        assert "Task 'add' - Starting task" in log_text
        assert "Task 'add' - Completed task successfully" in log_text
        assert "Task 'multiply' - Starting task" in log_text
        assert "Task 'multiply' - Completed task successfully" in log_text

    def test_duration_formatting(self, capsys):
        """Test that durations are formatted correctly."""
        import time

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def slow_task() -> str:
            time.sleep(0.05)  # 50ms
            return "done"

        result = evaluate(
            slow_task(),
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == "done"

        captured = capsys.readouterr()
        log_text = captured.out
        # Should have duration in milliseconds since it's < 1 second
        assert "Completed task successfully in" in log_text
        # Check format (should be like "50ms" or similar)
        assert "ms" in log_text

    def test_logs_retry_success(self, capsys):
        """Test that successful retries are logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        attempt_count = 0

        @task(retries=2)
        def flaky_task() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError(f"Attempt {attempt_count} failed")
            return "success"

        result = evaluate(
            flaky_task(),
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == "success"
        assert attempt_count == 2

        captured = capsys.readouterr()
        log_text = captured.out
        # Check retry logging
        assert "Task 'flaky_task' - Retrying after failure (attempt 2)" in log_text
        assert "ValueError: Attempt 1 failed" in log_text
        assert "Task 'flaky_task' - Retry succeeded on attempt 2" in log_text
        assert "Task 'flaky_task' - Completed task successfully" in log_text

    def test_logs_retry_exhausted(self, capsys):
        """Test that exhausted retries are logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        attempt_count = 0

        @task(retries=2)
        def always_fails() -> str:
            nonlocal attempt_count
            attempt_count += 1
            raise RuntimeError(f"Attempt {attempt_count} failed")

        with pytest.raises(RuntimeError, match="Attempt 3 failed"):
            evaluate(
                always_fails(),
                plugins=[LifecycleLoggingPlugin()],
            )

        assert attempt_count == 3  # Original + 2 retries

        captured = capsys.readouterr()
        log_text = captured.out
        # Check that retries are logged
        assert "Task 'always_fails' - Retrying after failure (attempt 2)" in log_text
        assert "RuntimeError: Attempt 1 failed" in log_text
        assert "Task 'always_fails' - Retrying after failure (attempt 3)" in log_text
        assert "RuntimeError: Attempt 2 failed" in log_text
        # Final failure should be logged
        assert "Task 'always_fails' - Failed after" in log_text


class TestLifecycleLoggingWithErrors:
    """Test LifecycleLoggingPlugin with error scenarios."""

    def test_logs_task_error(self, capsys):
        """Test that task errors are logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def failing_task() -> int:
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError, match="Something went wrong"):
            evaluate(
                failing_task(),
                plugins=[LifecycleLoggingPlugin()],
            )

        captured = capsys.readouterr()
        log_text = captured.out
        # Check error logging
        assert "Task 'failing_task' - Starting task" in log_text
        assert "Task 'failing_task' - Failed after" in log_text
        assert "ValueError: Something went wrong" in log_text

    def test_logs_evaluation_error(self, capsys):
        """Test that evaluation-level errors are logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def error_task() -> int:
            raise RuntimeError("Evaluation failed")

        with pytest.raises(RuntimeError, match="Evaluation failed"):
            evaluate(
                error_task(),
                plugins=[LifecycleLoggingPlugin()],
            )

        captured = capsys.readouterr()
        log_text = captured.out
        # Check that evaluation error is logged
        assert "Starting evaluation" in log_text
        assert "Evaluation" in log_text and "failed after" in log_text


class TestLifecycleLoggingWithMappedTasks:
    """Test LifecycleLoggingPlugin with mapped tasks."""

    def test_logs_mapped_task_execution(self, capsys):
        """Test that mapped task execution is logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def square(x: int) -> int:
            return x * x

        result = evaluate(
            square.product(x=[1, 2, 3, 4]),
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == [1, 4, 9, 16]

        captured = capsys.readouterr()
        log_text = captured.out
        # Check mapped task logging
        assert "Task 'square' - Starting task with 4 iterations" in log_text
        assert "Task 'square' - Completed task successfully in" in log_text
        assert "sequential backend" in log_text

    def test_logs_mapped_task_with_threads_backend(self, capsys):
        """Test that mapped task backend is logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def double(x: int) -> int:
            return x * 2

        result = evaluate(
            double.with_options(backend_name="threads").product(x=[1, 2, 3]),
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == [2, 4, 6]

        captured = capsys.readouterr()
        log_text = captured.out
        # Should mention threads backend for mapped task
        assert "using threads backend" in log_text
        assert "Task 'double' - Starting task with 3 iterations using threads backend" in log_text

    def test_logs_mapped_task_with_processes_backend(self, capsys):
        """Test that mapped task with processes backend is logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        result = evaluate(
            triple.with_options(backend_name="processes").product(x=[1, 2, 3]),
            plugins=[LifecycleLoggingPlugin()],
        )

        assert result == [3, 6, 9]

        captured = capsys.readouterr()
        log_text = captured.out
        # Should mention processes backend for mapped task
        assert "using processes backend" in log_text
        assert "Task 'triple' - Starting task with 3 iterations using processes backend" in log_text

    def test_logs_mapped_task_failure(self, capsys):
        """Test that mapped task failures are logged correctly."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        @task
        def failing_square(x: int) -> int:
            if x == 2:
                raise ValueError("Failed on 2")
            return x * x

        with pytest.raises(ValueError, match="Failed on 2"):
            evaluate(
                failing_square.product(x=[1, 2, 3]),
                plugins=[LifecycleLoggingPlugin()],
            )

        captured = capsys.readouterr()
        log_text = captured.out
        # Check that failure is logged with "Mapped iteration failed"
        assert "Task 'failing_square' - Starting task with 3 iterations" in log_text
        assert "Mapped iteration failed after" in log_text
        assert "ValueError: Failed on 2" in log_text
