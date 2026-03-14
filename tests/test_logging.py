"""Unit tests for the logging subsystem."""

from __future__ import annotations

import logging
import os
from unittest.mock import Mock
from unittest.mock import patch
from uuid import uuid4

import pytest

from daglite.logging.core import DEFAULT_LOGGER_NAME_COORD
from daglite.logging.core import LOGGER_EVENT
from daglite.logging.core import _ReporterHandler
from daglite.logging.core import _TaskLoggerAdapter
from daglite.logging.core import build_task_extra
from daglite.logging.core import format_duration
from daglite.logging.core import get_logger
from daglite.logging.plugin import CentralizedLoggingPlugin
from daglite.logging.plugin import LifecycleLoggingPlugin


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging after each test so one test's handlers don't leak."""
    yield
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("daglite"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()
            lgr.setLevel(logging.NOTSET)
            lgr.propagate = True


class TestFormatDuration:
    def test_milliseconds(self) -> None:
        assert format_duration(0.001) == "1 ms"
        assert format_duration(0.5) == "500 ms"
        assert format_duration(0.999) == "999 ms"

    def test_seconds(self) -> None:
        assert format_duration(1.0) == "1.00 s"
        assert format_duration(1.5) == "1.50 s"

    def test_minutes(self) -> None:
        assert format_duration(60.0) == "1 min 0.00 s"
        assert format_duration(90.5) == "1 min 30.50 s"


class TestBuildTaskExtra:
    def test_fields(self) -> None:
        tid = uuid4()
        extra = build_task_extra(tid, "my_task")
        assert extra["daglite_task_id"] == str(tid)
        assert extra["daglite_task_name"] == "my_task"
        assert extra["daglite_task_key"] == "my_task"


class TestGetLogger:
    def test_default_name(self) -> None:
        lgr = get_logger()
        assert isinstance(lgr, logging.LoggerAdapter)
        assert lgr.logger.name == DEFAULT_LOGGER_NAME_COORD

    def test_custom_name(self) -> None:
        lgr = get_logger("custom.logger")
        assert lgr.logger.name == "custom.logger"

    def test_no_reporter_no_reporter_handler(self) -> None:
        lgr = get_logger("test.no_reporter")
        assert not any(isinstance(h, _ReporterHandler) for h in lgr.logger.handlers)

    def test_direct_reporter_no_handler(self) -> None:
        mock_reporter = Mock(is_direct=True)

        with patch("daglite.logging.core.resolve_event_reporter", return_value=mock_reporter):
            lgr = get_logger("test.direct")
            assert not any(isinstance(h, _ReporterHandler) for h in lgr.logger.handlers)

    def test_process_reporter_adds_handler(self) -> None:
        mock_reporter = Mock(is_direct=False)

        with patch("daglite.logging.core.resolve_event_reporter", return_value=mock_reporter):
            lgr = get_logger("test.process_reporter")
            base = lgr.logger
            assert any(isinstance(h, _ReporterHandler) for h in base.handlers)
            assert base.level == logging.DEBUG
            assert base.propagate is False

    def test_no_duplicate_handlers(self) -> None:
        mock_reporter = Mock(is_direct=False)

        with patch("daglite.logging.core.resolve_event_reporter", return_value=mock_reporter):
            get_logger("test.dedupe")
            get_logger("test.dedupe")
            base = logging.getLogger("test.dedupe")
            reporter_h = [h for h in base.handlers if isinstance(h, _ReporterHandler)]
            assert len(reporter_h) == 1


class TestTaskLoggerAdapter:
    def test_process_injects_extra(self) -> None:
        base = logging.getLogger("test.adapter")
        adapter = _TaskLoggerAdapter(base, {})
        msg, kwargs = adapter.process("hello", {})
        assert msg == "hello"
        assert "extra" in kwargs


class TestReporterHandler:
    def test_emit_basic_log(self) -> None:
        mock_reporter = Mock()
        handler = _ReporterHandler(mock_reporter)

        base = logging.getLogger("test.emit")
        record = base.makeRecord("test.emit", logging.INFO, "test.py", 42, "Test message", (), None)

        handler.emit(record)

        mock_reporter.report.assert_called_once()
        call_args = mock_reporter.report.call_args
        assert call_args[0][0] == LOGGER_EVENT
        payload = call_args[0][1]
        assert payload["name"] == "test.emit"
        assert payload["level"] == "INFO"
        assert payload["message"] == "Test message"

    def test_emit_skips_already_forwarded(self) -> None:
        mock_reporter = Mock()
        handler = _ReporterHandler(mock_reporter)

        base = logging.getLogger("test.skip")
        record = base.makeRecord("test.skip", logging.INFO, "t.py", 1, "msg", (), None)
        record._daglite_already_forwarded = True  # type: ignore[attr-defined]

        handler.emit(record)
        assert not mock_reporter.report.called

    def test_emit_with_exception(self) -> None:
        import sys

        mock_reporter = Mock()
        handler = _ReporterHandler(mock_reporter)
        base = logging.getLogger("test.exception")

        try:
            raise ValueError("Test error")
        except ValueError:
            exc = sys.exc_info()
            record = base.makeRecord("test.exception", logging.ERROR, "t.py", 1, "Error", (), exc)

        handler.emit(record)
        payload = mock_reporter.report.call_args[0][1]
        assert "exc_info" in payload
        assert "ValueError" in payload["exc_info"]


class TestCentralizedLoggingPlugin:
    def test_default_level(self) -> None:
        plugin = CentralizedLoggingPlugin()
        assert plugin._level == logging.WARNING

    def test_custom_level(self) -> None:
        plugin = CentralizedLoggingPlugin(level=logging.INFO)
        assert plugin._level == logging.INFO

    def test_filtered_by_level(self, caplog) -> None:
        plugin = CentralizedLoggingPlugin(level=logging.WARNING)
        data = {
            "name": "test.logger",
            "level": "INFO",
            "message": "Should be filtered",
            "extra": {},
        }
        with caplog.at_level(logging.DEBUG):
            plugin._handle_log_event(LOGGER_EVENT, data)
        assert "Should be filtered" not in caplog.text

    def test_passes_above_threshold(self, caplog) -> None:
        plugin = CentralizedLoggingPlugin(level=logging.WARNING)
        data = {
            "name": "test.logger",
            "level": "WARNING",
            "message": "Warning message",
            "extra": {},
        }
        with caplog.at_level(logging.DEBUG):
            plugin._handle_log_event(LOGGER_EVENT, data)
        assert "Warning message" in caplog.text

    def test_exception_info_reconstructed(self, caplog) -> None:
        plugin = CentralizedLoggingPlugin(level=logging.ERROR)
        data = {
            "name": "test.logger",
            "level": "ERROR",
            "message": "Error occurred",
            "exc_info": "Traceback:\n  ValueError: Test error",
            "extra": {},
        }
        with caplog.at_level(logging.ERROR):
            plugin._handle_log_event(LOGGER_EVENT, data)
        assert "Error occurred" in caplog.text
        assert "ValueError: Test error" in caplog.text

    def test_extra_fields_restored_on_record(self, caplog) -> None:
        plugin = CentralizedLoggingPlugin(level=logging.INFO)
        data = {
            "name": "test.logger",
            "level": "INFO",
            "message": "Test message",
            "extra": {
                "filename": "worker.py",
                "lineno": 123,
                "daglite_task_name": "test_task",
            },
        }
        with caplog.at_level(logging.INFO):
            plugin._handle_log_event(LOGGER_EVENT, data)
        record = caplog.records[-1]
        assert record.filename == "worker.py"
        assert record.lineno == 123


class TestLifecycleInit:
    def test_custom_level(self) -> None:
        plugin = LifecycleLoggingPlugin(level=logging.DEBUG)
        assert plugin._logger.logger.level == logging.DEBUG

    def test_serialization_roundtrip(self) -> None:
        plugin = LifecycleLoggingPlugin()
        config = plugin.to_config()
        restored = LifecycleLoggingPlugin.from_config(config)
        assert isinstance(restored, LifecycleLoggingPlugin)

    def test_daglite_debug_assigns_handlers(self) -> None:
        with patch.dict(os.environ, {"DAGLITE_DEBUG": "1"}):
            LifecycleLoggingPlugin()
        base = logging.getLogger("daglite")
        lifecycle = logging.getLogger("daglite.lifecycle")
        assert len(base.handlers) > 0
        assert base.handlers == lifecycle.handlers
