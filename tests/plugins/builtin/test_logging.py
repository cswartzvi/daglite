"""Unit tests for logging plugin.

This module contains pure unit tests that do not use evaluate().
Integration tests that use evaluate() are in tests/evaluation/test_logging.py.
"""

import logging
from multiprocessing import Queue as MultiprocessingQueue
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from daglite.plugins.builtin.logging import DEFAULT_LOGGER_NAME_COORD
from daglite.plugins.builtin.logging import LOGGER_EVENT
from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
from daglite.plugins.builtin.logging import _ReporterHandler
from daglite.plugins.builtin.logging import _TaskLoggerAdapter
from daglite.plugins.builtin.logging import get_logger
from daglite.plugins.reporters import DirectEventReporter
from daglite.plugins.reporters import ProcessEventReporter


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration after each test to prevent interference with caplog.

    This is needed because LifecycleLoggingPlugin configures logging globally via YAML,
    which can interfere with pytest's caplog fixture.
    """
    yield

    # Clean up all daglite loggers after each test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("daglite"):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True


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
        with patch("daglite.plugins.builtin.logging.get_event_reporter", return_value=None):
            logger = get_logger("test.no.reporter")
            base_logger = logger.logger

            # Should not have ReporterHandler
            from daglite.plugins.builtin.logging import _ReporterHandler

            assert not any(isinstance(h, _ReporterHandler) for h in base_logger.handlers)

    def test_get_logger_with_direct_reporter(self):
        """Test get_logger with DirectReporter."""
        mock_reporter = Mock()
        mock_reporter.is_direct = True

        with patch(
            "daglite.plugins.builtin.logging.get_event_reporter", return_value=mock_reporter
        ):
            logger = get_logger("test.direct.reporter")
            base_logger = logger.logger

            # Should NOT have ReporterHandler (DirectReporter uses normal logging)
            assert not any(isinstance(h, _ReporterHandler) for h in base_logger.handlers)

    def test_get_logger_with_process_reporter(self):
        """Test get_logger with ProcessReporter."""
        mock_reporter = Mock()
        mock_reporter.is_direct = False

        with patch(
            "daglite.plugins.builtin.logging.get_event_reporter", return_value=mock_reporter
        ):
            logger = get_logger("test.process.reporter")
            base_logger = logger.logger

            # Should have ReporterHandler
            assert any(isinstance(h, _ReporterHandler) for h in base_logger.handlers)
            # Should set level to DEBUG
            assert base_logger.level == logging.DEBUG
            # Should disable propagation
            assert base_logger.propagate is False

    def test_get_logger_with_reporter_already_debug(self):
        """Test get_logger doesn't change level if already DEBUG or lower."""
        mock_reporter = Mock()
        mock_reporter.is_direct = False

        # Pre-configure logger to DEBUG
        test_logger = logging.getLogger("test.already.debug")
        test_logger.setLevel(logging.DEBUG)
        original_level = test_logger.level

        with patch(
            "daglite.plugins.builtin.logging.get_event_reporter", return_value=mock_reporter
        ):
            logger = get_logger("test.already.debug")
            base_logger = logger.logger

            # Should not change level since it's already DEBUG
            assert base_logger.level == original_level

    def test_get_logger_no_duplicate_handlers(self):
        """Test that calling get_logger twice doesn't add duplicate handlers."""
        mock_reporter = Mock()
        mock_reporter.is_direct = False

        with patch(
            "daglite.plugins.builtin.logging.get_event_reporter", return_value=mock_reporter
        ):
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

        from daglite.graph.base import NodeMetadata

        base_logger = logging.getLogger("test.adapter")
        adapter = _TaskLoggerAdapter(base_logger, {})

        # Mock task metadata
        task_metadata = NodeMetadata(
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
            assert kwargs["extra"]["daglite_task_key"] == "test_task[0]"
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
                "daglite_task_key": "test_task[0]",
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
        assert record.daglite_task_key == "test_task[0]"


class TestReporterImplementations:
    """Unit tests for reporter implementations."""

    def test_direct_reporter_report(self):
        """Test DirectReporter.report calls callback."""
        callback = Mock()
        reporter = DirectEventReporter(callback)

        assert reporter.is_direct is True

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
        reporter = DirectEventReporter(callback)
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
        reporter = DirectEventReporter(callback)

        # Should not raise, should log error instead
        reporter.report("test_event", {"key": "value"})

        # Callback was called despite error
        assert callback.called

    def test_process_reporter_report(self):
        """Test ProcessReporter.report puts event on queue."""
        queue = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)

        assert reporter.is_direct is False

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
        reporter = ProcessEventReporter(queue)

        assert reporter.queue is queue

        queue.close()

    def test_process_reporter_close(self):
        """Test ProcessReporter.close() closes the queue."""
        queue = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)

        # Close should not raise
        reporter.close()

        # Queue should be closed (subsequent operations will fail)
        # Note: We can't easily test this without causing errors

    def test_process_reporter_error_handling(self):
        """Test ProcessReporter handles queue errors gracefully."""
        queue = Mock()
        queue.put.side_effect = RuntimeError("Queue error")

        reporter = ProcessEventReporter(queue)

        # Should not raise, should log error instead
        reporter.report("test_event", {"key": "value"})

        # put was called despite error
        assert queue.put.called


class TestFormatDuration:
    """Unit tests for _format_duration helper function."""

    def test_format_milliseconds(self):
        """Test formatting durations less than 1 second."""
        from daglite.plugins.builtin.logging import _format_duration

        assert _format_duration(0.001) == "1 ms"
        assert _format_duration(0.5) == "500 ms"
        assert _format_duration(0.999) == "999 ms"

    def test_format_seconds(self):
        """Test formatting durations in seconds."""
        from daglite.plugins.builtin.logging import _format_duration

        assert _format_duration(1.0) == "1.00 s"
        assert _format_duration(1.5) == "1.50 s"
        assert _format_duration(59.99) == "59.99 s"

    def test_format_minutes(self):
        """Test formatting durations in minutes."""
        from daglite.plugins.builtin.logging import _format_duration

        assert _format_duration(60.0) == "1 min 0.00 s"
        assert _format_duration(90.5) == "1 min 30.50 s"
        assert _format_duration(125.0) == "2 min 5.00 s"


class TestBuildTaskContext:
    """Unit tests for _build_task_context helper function."""

    def test_build_task_context_with_key(self):
        """Test building task context with all fields."""
        from uuid import UUID

        from daglite.plugins.builtin.logging import _build_task_context

        task_id = UUID("12345678-1234-5678-1234-567812345678")
        context = _build_task_context(task_id, "my_task", "task_key")

        assert context["daglite_task_id"] == "12345678-1234-5678-1234-567812345678"
        assert context["daglite_task_name"] == "my_task"
        assert context["daglite_task_key"] == "task_key"

    def test_build_task_context_without_key(self):
        """Test building task context when key is None."""
        from uuid import UUID

        from daglite.plugins.builtin.logging import _build_task_context

        task_id = UUID("12345678-1234-5678-1234-567812345678")
        context = _build_task_context(task_id, "my_task", None)

        assert context["daglite_task_id"] == "12345678-1234-5678-1234-567812345678"
        assert context["daglite_task_name"] == "my_task"
        assert context["daglite_task_key"] == "my_task"  # Falls back to name


class TestLifecycleLoggingPlugin:
    """Unit tests for LifecycleLoggingPlugin."""

    def test_initialization_default(self):
        """Test plugin initialization with defaults."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()

        assert plugin._mapped_nodes == set()
        assert plugin._logger is not None

    def test_initialization_with_custom_level(self):
        """Test plugin initialization with custom log level."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin(level=logging.DEBUG)

        # Logger level should be set
        assert plugin._logger.logger.level == logging.DEBUG

    def test_loads_json_config(self):
        """Test that plugin loads logging.json configuration."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        # Create plugin instance to trigger JSON config loading
        _ = LifecycleLoggingPlugin()

        # Config should be loaded (handler names should exist)
        logger = logging.getLogger("daglite.lifecycle")
        handler_classes = [h.__class__.__name__ for h in logger.handlers]

        # Should have either StreamHandler or FileHandler or RotatingFileHandler
        assert any(
            name in ["StreamHandler", "FileHandler", "RotatingFileHandler"]
            for name in handler_classes
        )

    def test_serialization(self):
        """Test plugin serialization to/from config."""
        from uuid import UUID

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        plugin._mapped_nodes.add(UUID("12345678-1234-5678-1234-567812345678"))
        plugin._mapped_nodes.add(UUID("87654321-4321-8765-4321-876543218765"))

        # Serialize
        config = plugin.to_config()
        assert "mapped_nodes" in config
        assert len(config["mapped_nodes"]) == 2

        # Deserialize
        new_plugin = LifecycleLoggingPlugin.from_config(config)
        assert len(new_plugin._mapped_nodes) == 2

    def test_tracks_mapped_nodes(self):
        """Test that plugin tracks mapped node IDs."""
        from uuid import uuid4

        from daglite.graph.base import NodeMetadata
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()

        # Simulate before_mapped_node_execute hook
        node_id = uuid4()
        metadata = NodeMetadata(id=node_id, name="test_task", kind="map", key="test_key")

        plugin.before_mapped_node_execute(metadata, inputs_list=[{}, {}])

        # Should track the node ID
        assert node_id in plugin._mapped_nodes


class TestLifecycleLoggingPluginOnCacheHit:
    """Tests for LifecycleLoggingPlugin.on_cache_hit hook."""

    def test_on_cache_hit_logs_message(self, caplog):
        """Test that on_cache_hit logs an info message."""
        from unittest.mock import Mock
        from uuid import uuid4

        from daglite.graph.base import NodeMetadata
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()

        node_id = uuid4()
        metadata = NodeMetadata(id=node_id, name="test_task", kind="task", key="test_task")

        # Call hook - it should log without raising
        plugin.on_cache_hit(
            func=Mock(),
            metadata=metadata,
            inputs={"x": 5},
            result=10,
            reporter=None,
        )

        # Since the logger uses get_logger() which creates its own handlers,
        # we can't easily capture with caplog. Just verify it doesn't raise.


class TestLifecycleLoggingPluginDatasetSaveHooks:
    """Tests for LifecycleLoggingPlugin before/after_dataset_save hooks."""

    def test_before_dataset_save_with_format(self):
        """before_dataset_save includes format in message when provided."""
        from unittest.mock import patch

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        with patch.object(plugin._logger, "debug") as mock_debug:
            plugin.before_dataset_save(
                key="output.pkl", value={"data": 1}, format="pickle", options=None
            )
        mock_debug.assert_called_once()
        msg = mock_debug.call_args[0][0]
        assert "output.pkl" in msg
        assert "(format=pickle)" in msg

    def test_before_dataset_save_without_format(self):
        """before_dataset_save omits format portion when None."""
        from unittest.mock import patch

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        with patch.object(plugin._logger, "debug") as mock_debug:
            plugin.before_dataset_save(
                key="output.pkl", value={"data": 1}, format=None, options=None
            )
        mock_debug.assert_called_once()
        msg = mock_debug.call_args[0][0]
        assert "output.pkl" in msg
        assert "format=" not in msg

    def test_after_dataset_save_with_format(self, capsys):
        """after_dataset_save includes format in message when provided."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        plugin.after_dataset_save(
            key="result.csv", value="hello", format="pandas/csv", options=None
        )
        out = capsys.readouterr().out
        assert "result.csv" in out
        assert "(format=pandas/csv)" in out

    def test_after_dataset_save_without_format(self, capsys):
        """after_dataset_save omits format portion when None."""
        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        plugin.after_dataset_save(key="result.pkl", value="hello", format=None, options=None)
        out = capsys.readouterr().out
        assert "result.pkl" in out
        assert "format=" not in out
