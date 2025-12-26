"""Tests for the logging plugin."""

import logging

from daglite import evaluate
from daglite import task
from daglite.plugins import CentralizedLoggingPlugin
from daglite.plugins import get_logger


@task
def logging_task(x: int, message: str) -> int:
    """Task that logs a message."""
    logger = get_logger(__name__)
    logger.info(message)
    return x


@task
def contextual_logging_task(x: int) -> int:
    """Task that uses contextual logging (no name argument)."""
    logger = get_logger()  # Should auto-derive name from task context
    logger.info(f"Processing {x}")
    return x * 2


@task
def logging_task_with_levels(x: int) -> int:
    """Task that logs at different levels."""
    logger = get_logger(__name__)
    logger.debug(f"Debug: {x}")
    logger.info(f"Info: {x}")
    logger.warning(f"Warning: {x}")
    logger.error(f"Error: {x}")
    return x


@task
def logging_task_with_exception(x: int) -> int:
    """Task that logs an exception."""
    logger = get_logger(__name__)
    try:
        raise ValueError(f"Test error for {x}")
    except ValueError:
        logger.exception("Caught an exception")
    return x


class TestCentralizedLoggingPlugin:
    """Test suite for CentralizedLoggingPlugin."""

    def test_logging_with_sequential_backend(self, caplog):
        """Test logging works with sequential backend (fallback mode)."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            result = evaluate(logging_task(x=42, message="Test message"), plugins=[plugin])
            assert result == 42

        # Check that log was captured
        assert "Test message" in caplog.text

    def test_different_log_levels(self, caplog):
        """Test different log levels are handled correctly."""
        plugin = CentralizedLoggingPlugin(level=logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            result = evaluate(logging_task_with_levels(x=42), plugins=[plugin])
            assert result == 42

        # All levels should be present
        assert "Debug: 42" in caplog.text
        assert "Info: 42" in caplog.text
        assert "Warning: 42" in caplog.text
        assert "Error: 42" in caplog.text

    def test_log_level_filtering(self, caplog):
        """Test that log level filtering works at the plugin level."""
        plugin = CentralizedLoggingPlugin(level=logging.WARNING)

        with caplog.at_level(logging.WARNING):  # Only capture WARNING and above
            result = evaluate(logging_task_with_levels(x=42), plugins=[plugin])
            assert result == 42

        # Only WARNING and ERROR should be present (DEBUG/INFO filtered by caplog level)
        assert "Debug: 42" not in caplog.text
        assert "Info: 42" not in caplog.text
        assert "Warning: 42" in caplog.text
        assert "Error: 42" in caplog.text

    def test_exception_logging(self, caplog):
        """Test that exception logging includes traceback."""
        plugin = CentralizedLoggingPlugin(level=logging.ERROR)

        with caplog.at_level(logging.ERROR):
            result = evaluate(logging_task_with_exception(x=42), plugins=[plugin])
            assert result == 42

        # Exception message should be in logs
        assert "Caught an exception" in caplog.text
        assert "ValueError" in caplog.text
        assert "Test error for 42" in caplog.text

    def test_multiple_tasks_logging(self, caplog):
        """Test logging from multiple tasks."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            for i in range(5):
                result = evaluate(logging_task(x=i, message=f"Message {i}"), plugins=[plugin])
                assert result == i

        for i in range(5):
            assert f"Message {i}" in caplog.text

    def test_get_logger_returns_logger_adapter(self):
        """Test that get_logger returns a LoggerAdapter instance."""
        logger = get_logger(__name__)

        assert isinstance(logger, logging.LoggerAdapter)

    def test_logger_uses_standard_logging_when_no_reporter(self):
        """Test that logger uses standard logging when no reporter."""
        # When no reporter is available, ReporterHandler won't be added
        # and logs go through standard logging
        logger = get_logger(__name__)

        # Should not raise, should use standard logging
        logger.info("Test message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_contextual_logging_auto_name(self, caplog):
        """Test that get_logger() without name uses daglite.tasks logger."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            result = evaluate(contextual_logging_task(x=42), plugins=[plugin])
            assert result == 84  # 42 * 2

        # Logger should use daglite.tasks (not task-specific name)
        assert "daglite.tasks" in caplog.text
        assert "Processing 42" in caplog.text

    def test_task_metadata_fields_in_log_records(self, caplog):
        """Test that daglite_task_name, daglite_task_id, daglite_node_key are in LogRecords."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            result = evaluate(contextual_logging_task(x=99), plugins=[plugin])
            assert result == 198

        # Verify task metadata fields are present in the log record
        assert len(caplog.records) > 0
        record = caplog.records[0]

        # Check all three daglite_* fields exist
        assert hasattr(record, "daglite_task_name")
        assert hasattr(record, "daglite_task_id")
        assert hasattr(record, "daglite_node_key")

        # Verify values are sensible
        assert record.daglite_task_name == "contextual_logging_task"
        assert record.daglite_node_key == "contextual_logging_task"
        assert isinstance(record.daglite_task_id, str)
        assert len(record.daglite_task_id) > 0  # UUID should be non-empty

    def test_reporter_handler_error_handling(self, caplog):
        """Test that ReporterHandler.emit handles errors gracefully."""
        from unittest.mock import Mock

        from daglite.plugins.default.logging import _ReporterHandler

        # Create a mock reporter that raises an error
        mock_reporter = Mock()
        mock_reporter.report.side_effect = RuntimeError("Simulated reporter error")

        handler = _ReporterHandler(mock_reporter)

        # Create a log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Emit should not raise, but handle the error internally
        # (logging.Handler.handleError will be called)
        handler.emit(record)

        # Verify reporter was called (and raised)
        assert mock_reporter.report.called

    def test_get_logger_same_name_returns_same_logger(self):
        """Test that get_logger with the same name returns the same underlying logger."""
        # Get logger with a unique name (may or may not have reporter in context)
        logger1 = get_logger("test.unique.name.456")
        logger2 = get_logger("test.unique.name.456")

        # Should be the same underlying logger
        assert logger1.logger is logger2.logger

    def test_logging_plugin_handles_no_handlers(self):
        """Test CentralizedLoggingPlugin._handle_log_event when logger has no handlers."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        # Create a logger with no handlers
        test_logger = logging.getLogger("test.no.handlers")
        test_logger.handlers.clear()

        # Create event
        event = {
            "name": "test.no.handlers",
            "level": "INFO",
            "message": "Test message",
        }

        # Should not raise even with no handlers
        plugin._handle_log_event(event)

    def test_get_logger_no_duplicate_handlers(self, caplog):
        """Test that multiple get_logger calls for same name don't duplicate handlers."""
        from daglite.plugins.default.logging import _ReporterHandler

        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        # Create a task that calls get_logger multiple times
        @task
        def task_with_multiple_get_logger(x: int) -> int:
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

        with caplog.at_level(logging.INFO):
            result = evaluate(
                task_with_multiple_get_logger.with_options(backend_name="processes")(x=42),
                plugins=[plugin],
            )
            assert result == 42

        # Verify the log was captured
        assert "Test 42" in caplog.text
