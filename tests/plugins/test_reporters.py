"""Unit tests for event reporter implementations (DirectEventReporter, ProcessEventReporter)."""

from __future__ import annotations

import threading
from multiprocessing import Queue as MultiprocessingQueue
from unittest.mock import Mock

from daglite.plugins.reporters import DirectEventReporter
from daglite.plugins.reporters import ProcessEventReporter


class TestDirectEventReporter:
    """DirectEventReporter dispatches via in-process callback under a lock."""

    def test_report_and_is_direct(self) -> None:
        callback = Mock()
        reporter = DirectEventReporter(callback)
        assert reporter.is_direct is True

        reporter.report("test_event", {"key": "value"})

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.type == "test_event"
        assert event.data["key"] == "value"

    def test_thread_safety(self) -> None:
        callback = Mock()
        reporter = DirectEventReporter(callback)

        threads = [
            threading.Thread(target=reporter.report, args=(f"event_{i}", {"v": i}))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert callback.call_count == 10

    def test_error_handling(self, caplog) -> None:
        callback = Mock(side_effect=ValueError("Callback error"))
        reporter = DirectEventReporter(callback)

        reporter.report("test_event", {"key": "value"})

        assert "Error reporting event test_event" in caplog.text
        assert "Callback error" in caplog.text


class TestProcessEventReporter:
    """ProcessEventReporter puts events onto a multiprocessing Queue."""

    def test_report_and_is_direct(self) -> None:
        queue: MultiprocessingQueue = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)
        assert reporter.is_direct is False

        reporter.report("test_event", {"key": "value"})

        event = queue.get(timeout=1)
        assert event.type == "test_event"
        assert event.data["key"] == "value"
        queue.close()

    def test_error_handling(self, caplog) -> None:
        queue = Mock()
        queue.put.side_effect = RuntimeError("Queue error")
        reporter = ProcessEventReporter(queue)

        reporter.report("test_event", {"key": "value"})

        assert "Error reporting event test_event" in caplog.text
        assert "Queue error" in caplog.text

    def test_close(self) -> None:
        queue: MultiprocessingQueue = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)
        reporter.close()  # Should not raise
