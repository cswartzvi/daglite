"""Tests for the logging plugin."""

import logging

from daglite import evaluate
from daglite import task
from daglite.plugins import LoggingPlugin
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


class TestLoggingPlugin:
    """Test suite for LoggingPlugin."""

    def test_logging_with_sequential_backend(self, caplog):
        """Test logging works with sequential backend (fallback mode)."""
        plugin = LoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            result = evaluate(logging_task(x=42, message="Test message"), plugins=[plugin])
            assert result == 42

        # Check that log was captured
        assert "Test message" in caplog.text

    def test_different_log_levels(self, caplog):
        """Test different log levels are handled correctly."""
        plugin = LoggingPlugin(level=logging.DEBUG)

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
        plugin = LoggingPlugin(level=logging.WARNING)

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
        plugin = LoggingPlugin(level=logging.ERROR)

        with caplog.at_level(logging.ERROR):
            result = evaluate(logging_task_with_exception(x=42), plugins=[plugin])
            assert result == 42

        # Exception message should be in logs
        assert "Caught an exception" in caplog.text
        assert "ValueError" in caplog.text
        assert "Test error for 42" in caplog.text

    def test_multiple_tasks_logging(self, caplog):
        """Test logging from multiple tasks."""
        plugin = LoggingPlugin(level=logging.INFO)

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
        plugin = LoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            result = evaluate(contextual_logging_task(x=42), plugins=[plugin])
            assert result == 84  # 42 * 2

        # Logger should use daglite.tasks (not task-specific name)
        assert "daglite.tasks" in caplog.text
        assert "Processing 42" in caplog.text

    def test_contextual_logging_includes_task_metadata(self, caplog):
        """Test that logs include task context in the logger name."""
        plugin = LoggingPlugin(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            # Use sequential backend for simpler testing
            result = evaluate(contextual_logging_task(x=99), plugins=[plugin])
            assert result == 198

        # Check that log appeared with daglite.tasks logger name
        assert "Processing 99" in caplog.text
        assert "daglite.tasks" in caplog.text

    def test_reporter_handler_error_handling(self, caplog):
        """Test that ReporterHandler.emit handles errors gracefully."""
        from unittest.mock import Mock

        from daglite.plugins.default.logging import ReporterHandler

        # Create a mock reporter that raises an error
        mock_reporter = Mock()
        mock_reporter.report.side_effect = RuntimeError("Simulated reporter error")

        handler = ReporterHandler(mock_reporter)

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

    def test_get_logger_without_reporter(self):
        """Test that get_logger doesn't add ReporterHandler twice to same logger."""
        from daglite.plugins.default.logging import ReporterHandler
        from daglite.plugins.default.logging import get_logger

        # Get logger with a unique name
        logger1 = get_logger("test.unique.name.123")
        logger2 = get_logger("test.unique.name.123")

        # Should be the same underlying logger
        assert logger1.logger is logger2.logger

        # Should only have at most one ReporterHandler (not duplicated)
        base_logger = logger1.logger
        reporter_handlers = [h for h in base_logger.handlers if isinstance(h, ReporterHandler)]
        assert len(reporter_handlers) <= 1  # 0 or 1, but not 2+

    def test_logging_plugin_handles_no_handlers(self):
        """Test LoggingPlugin._handle_log_event when logger has no handlers."""
        plugin = LoggingPlugin(level=logging.INFO)

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
