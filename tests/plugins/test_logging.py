"""Unit tests for logging plugin.

This module contains pure unit tests that do not use evaluate().
Integration tests that use evaluate() are in tests/evaluation/test_logging.py.
"""

import logging
from multiprocessing import Queue as MultiprocessingQueue
from unittest.mock import Mock
from unittest.mock import patch

from daglite.plugins.default.logging import DEFAULT_LOGGER_NAME_COORD
from daglite.plugins.default.logging import LOGGER_EVENT
from daglite.plugins.default.logging import CentralizedLoggingPlugin
from daglite.plugins.default.logging import _ReporterHandler
from daglite.plugins.default.logging import _TaskLoggerAdapter
from daglite.plugins.default.logging import get_logger
from daglite.plugins.reporters import DirectReporter
from daglite.plugins.reporters import ProcessReporter


class TestGetLoggerUnit:
    """Unit tests for get_logger function."""

    def test_get_logger_default_name(self):
        """Test get_logger with default name."""
        logger = get_logger()
        assert isinstance(logger, logging.LoggerAdapter)
        assert logger.logger.name == DEFAULT_LOGGER_NAME_COORD

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger("custom.logger")
        assert isinstance(logger, logging.LoggerAdapter)
        assert logger.logger.name == "custom.logger"

    def test_get_logger_no_reporter(self):
        """Test get_logger when no reporter is available."""
        with patch("daglite.plugins.default.logging.get_reporter", return_value=None):
            logger = get_logger("test.no.reporter")
            base_logger = logger.logger

            # Should not have ReporterHandler
            from daglite.plugins.default.logging import _ReporterHandler

            assert not any(isinstance(h, _ReporterHandler) for h in base_logger.handlers)

    def test_get_logger_with_non_remote_reporter(self):
        """Test get_logger with DirectReporter (non-remote)."""
        mock_reporter = Mock()
        mock_reporter.is_remote = False

        with patch("daglite.plugins.default.logging.get_reporter", return_value=mock_reporter):
            logger = get_logger("test.direct.reporter")
            base_logger = logger.logger

            # Should NOT have ReporterHandler (DirectReporter uses normal logging)
            assert not any(isinstance(h, _ReporterHandler) for h in base_logger.handlers)

    def test_get_logger_with_remote_reporter(self):
        """Test get_logger with ProcessReporter (remote)."""
        mock_reporter = Mock()
        mock_reporter.is_remote = True

        with patch("daglite.plugins.default.logging.get_reporter", return_value=mock_reporter):
            logger = get_logger("test.process.reporter")
            base_logger = logger.logger

            # Should have ReporterHandler
            assert any(isinstance(h, _ReporterHandler) for h in base_logger.handlers)
            # Should set level to DEBUG
            assert base_logger.level == logging.DEBUG
            # Should disable propagation
            assert base_logger.propagate is False

    def test_get_logger_with_remote_reporter_already_debug(self):
        """Test get_logger doesn't change level if already DEBUG or lower."""
        mock_reporter = Mock()
        mock_reporter.is_remote = True

        # Pre-configure logger to DEBUG
        test_logger = logging.getLogger("test.already.debug")
        test_logger.setLevel(logging.DEBUG)
        original_level = test_logger.level

        with patch("daglite.plugins.default.logging.get_reporter", return_value=mock_reporter):
            logger = get_logger("test.already.debug")
            base_logger = logger.logger

            # Should not change level since it's already DEBUG
            assert base_logger.level == original_level

    def test_get_logger_no_duplicate_handlers(self):
        """Test that calling get_logger twice doesn't add duplicate handlers."""
        mock_reporter = Mock()
        mock_reporter.is_remote = True

        with patch("daglite.plugins.default.logging.get_reporter", return_value=mock_reporter):
            logger1 = get_logger("test.dedupe")
            _ = get_logger("test.dedupe")

            base_logger = logger1.logger
            reporter_handlers = [h for h in base_logger.handlers if isinstance(h, _ReporterHandler)]

            # Should only have ONE handler, not two
            assert len(reporter_handlers) == 1


class TestTaskLoggerAdapter:
    """Unit tests for _TaskLoggerAdapter."""

    def test_process_without_task_context(self):
        """Test process method when no task is executing."""
        base_logger = logging.getLogger("test.adapter")
        adapter = _TaskLoggerAdapter(base_logger, {})

        with patch("daglite.backends.context.get_current_task", return_value=None):
            msg, kwargs = adapter.process("test message", {})

            assert msg == "test message"
            assert "extra" in kwargs
            assert "daglite_task_id" not in kwargs["extra"]

    def test_process_with_task_context(self):
        """Test process method when task is executing."""
        from uuid import uuid4

        from daglite.graph.base import GraphMetadata

        base_logger = logging.getLogger("test.adapter")
        adapter = _TaskLoggerAdapter(base_logger, {})

        # Mock task metadata
        task_metadata = GraphMetadata(
            id=uuid4(),
            name="test_task",
            kind="task",
            description="Test",
            backend_name="processes",
            key="test_task[0]",
        )

        with patch("daglite.backends.context.get_current_task", return_value=task_metadata):
            msg, kwargs = adapter.process("test message", {})

            assert msg == "test message"
            assert "extra" in kwargs
            assert kwargs["extra"]["daglite_task_name"] == "test_task"
            assert kwargs["extra"]["daglite_node_key"] == "test_task[0]"
            assert "daglite_task_id" in kwargs["extra"]


class TestReporterHandler:
    """Unit tests for _ReporterHandler."""

    def test_emit_basic_log(self):
        """Test emitting a basic log record."""
        mock_reporter = Mock()
        handler = _ReporterHandler(mock_reporter)

        logger = logging.getLogger("test.emit")
        record = logger.makeRecord(
            name="test.emit",
            level=logging.INFO,
            fn="test.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Verify reporter.report was called
        assert mock_reporter.report.called
        call_args = mock_reporter.report.call_args
        assert call_args[0][0] == LOGGER_EVENT
        payload = call_args[0][1]
        assert payload["name"] == "test.emit"
        assert payload["level"] == "INFO"
        assert payload["message"] == "Test message"

    def test_emit_skips_already_forwarded(self):
        """Test that emit skips records already forwarded."""
        mock_reporter = Mock()
        handler = _ReporterHandler(mock_reporter)

        logger = logging.getLogger("test.skip")
        record = logger.makeRecord(
            name="test.skip",
            level=logging.INFO,
            fn="test.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Mark as already forwarded
        record._daglite_already_forwarded = True

        handler.emit(record)

        # Should NOT have called reporter
        assert not mock_reporter.report.called

    def test_emit_with_exception(self):
        """Test emitting a log record with exception info."""
        import sys

        mock_reporter = Mock()
        handler = _ReporterHandler(mock_reporter)

        logger = logging.getLogger("test.exception")

        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()
            record = logger.makeRecord(
                name="test.exception",
                level=logging.ERROR,
                fn="test.py",
                lno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

        handler.emit(record)

        call_args = mock_reporter.report.call_args
        payload = call_args[0][1]
        assert "exc_info" in payload
        assert "ValueError" in payload["exc_info"]
        assert "Test error" in payload["exc_info"]


class TestCentralizedLoggingPluginUnit:
    """Unit tests for CentralizedLoggingPlugin."""

    def test_plugin_initialization(self):
        """Test plugin initializes with correct level."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)
        assert plugin._level == logging.INFO

        plugin_default = CentralizedLoggingPlugin()
        assert plugin_default._level == logging.WARNING

    def test_handle_log_event_basic(self, caplog):
        """Test handling a basic log event."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        event = {
            "name": "test.logger",
            "level": "INFO",
            "message": "Test message from worker",
            "extra": {},
        }

        with caplog.at_level(logging.INFO):
            plugin._handle_log_event(event)

        assert "Test message from worker" in caplog.text

    def test_handle_log_event_level_filtering(self, caplog):
        """Test that plugin filters events below minimum level."""
        plugin = CentralizedLoggingPlugin(level=logging.WARNING)

        # INFO event should be filtered
        info_event = {
            "name": "test.logger",
            "level": "INFO",
            "message": "Info message",
            "extra": {},
        }

        with caplog.at_level(logging.DEBUG):
            plugin._handle_log_event(info_event)

        assert "Info message" not in caplog.text

        # WARNING event should pass
        warning_event = {
            "name": "test.logger",
            "level": "WARNING",
            "message": "Warning message",
            "extra": {},
        }

        with caplog.at_level(logging.DEBUG):
            plugin._handle_log_event(warning_event)

        assert "Warning message" in caplog.text

    def test_handle_log_event_with_exception(self, caplog):
        """Test handling log event with exception info."""
        plugin = CentralizedLoggingPlugin(level=logging.ERROR)

        event = {
            "name": "test.logger",
            "level": "ERROR",
            "message": "Error occurred",
            "exc_info": "Traceback (most recent call last):\n  ValueError: Test error",
            "extra": {},
        }

        with caplog.at_level(logging.ERROR):
            plugin._handle_log_event(event)

        assert "Error occurred" in caplog.text
        assert "ValueError: Test error" in caplog.text

    def test_handle_log_event_with_extra_fields(self, caplog):
        """Test handling log event with extra fields."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        event = {
            "name": "test.logger",
            "level": "INFO",
            "message": "Test message",
            "extra": {
                "filename": "worker.py",
                "lineno": 123,
                "daglite_task_name": "test_task",
                "daglite_node_key": "test_task[0]",
            },
        }

        with caplog.at_level(logging.INFO):
            plugin._handle_log_event(event)

        # Verify extra fields are preserved in record
        assert len(caplog.records) > 0
        record = caplog.records[-1]
        assert record.filename == "worker.py"
        assert record.lineno == 123
        assert record.daglite_task_name == "test_task"
        assert record.daglite_node_key == "test_task[0]"


class TestReporterImplementations:
    """Unit tests for reporter implementations."""

    def test_direct_reporter_report(self):
        """Test DirectReporter.report calls callback."""
        callback = Mock()
        reporter = DirectReporter(callback)

        # Verify is_remote is False
        assert reporter.is_remote is False

        # Report an event
        reporter.report("test_event", {"key": "value"})

        # Verify callback was called with correct event
        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event["type"] == "test_event"
        assert event["key"] == "value"

    def test_direct_reporter_thread_safety(self):
        """Test DirectReporter is thread-safe."""
        import threading

        callback = Mock()
        reporter = DirectReporter(callback)
        results = []

        def report_event(i):
            reporter.report(f"event_{i}", {"value": i})
            results.append(i)

        # Create multiple threads
        threads = [threading.Thread(target=report_event, args=(i,)) for i in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify all events were reported
        assert len(results) == 10
        assert callback.call_count == 10

    def test_direct_reporter_error_handling(self):
        """Test DirectReporter handles callback errors gracefully."""
        callback = Mock(side_effect=RuntimeError("Callback error"))
        reporter = DirectReporter(callback)

        # Should not raise, should log error instead
        reporter.report("test_event", {"key": "value"})

        # Callback was called despite error
        assert callback.called

    def test_process_reporter_report(self):
        """Test ProcessReporter.report puts event on queue."""
        queue = MultiprocessingQueue()
        reporter = ProcessReporter(queue)

        # Verify is_remote is True
        assert reporter.is_remote is True

        # Report an event
        reporter.report("test_event", {"key": "value"})

        # Verify event was put on queue
        event = queue.get(timeout=1)
        assert event["type"] == "test_event"
        assert event["key"] == "value"

        queue.close()

    def test_process_reporter_queue_property(self):
        """Test ProcessReporter.queue property."""
        queue = MultiprocessingQueue()
        reporter = ProcessReporter(queue)

        assert reporter.queue is queue

        queue.close()

    def test_process_reporter_close(self):
        """Test ProcessReporter.close() closes the queue."""
        queue = MultiprocessingQueue()
        reporter = ProcessReporter(queue)

        # Close should not raise
        reporter.close()

        # Queue should be closed (subsequent operations will fail)
        # Note: We can't easily test this without causing errors

    def test_process_reporter_error_handling(self):
        """Test ProcessReporter handles queue errors gracefully."""
        queue = Mock()
        queue.put.side_effect = RuntimeError("Queue error")

        reporter = ProcessReporter(queue)

        # Should not raise, should log error instead
        reporter.report("test_event", {"key": "value"})

        # put was called despite error
        assert queue.put.called
