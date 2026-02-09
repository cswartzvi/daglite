"""
Unit tests for event reporters.

These tests verify reporter implementations without requiring full evaluation.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/.
"""

from multiprocessing import Queue as MultiprocessingQueue
from typing import Any
from unittest.mock import Mock

import pytest

from daglite.plugins.reporters import DirectEventReporter
from daglite.plugins.reporters import ProcessEventReporter


class TestDirectReporter:
    """Test DirectReporter for direct callback-based event reporting."""

    def test_initialization(self) -> None:
        """DirectReporter can be initialized with a callback."""
        callback = Mock()
        reporter = DirectEventReporter(callback)
        assert reporter._callback is callback

    def test_report_calls_callback(self) -> None:
        """report() sends event via callback."""
        callback = Mock()
        reporter = DirectEventReporter(callback)

        reporter.report("test_event", {"key": "value"})

        callback.assert_called_once_with({"type": "test_event", "key": "value"})

    def test_report_with_multiple_events(self) -> None:
        """report() can be called multiple times."""
        callback = Mock()
        reporter = DirectEventReporter(callback)

        reporter.report("event1", {"data": 1})
        reporter.report("event2", {"data": 2})

        assert callback.call_count == 2
        callback.assert_any_call({"type": "event1", "data": 1})
        callback.assert_any_call({"type": "event2", "data": 2})

    def test_report_handles_callback_exception(self, caplog) -> None:
        """report() handles exceptions from callback gracefully."""
        callback = Mock(side_effect=ValueError("Callback error"))
        reporter = DirectEventReporter(callback)

        # Should not raise, but log the error
        reporter.report("test_event", {"key": "value"})

        # Verify error was logged
        assert "Error reporting event test_event" in caplog.text
        assert "Callback error" in caplog.text

    def test_thread_safety(self) -> None:
        """DirectReporter is thread-safe for concurrent calls."""
        import threading

        callback = Mock()
        reporter = DirectEventReporter(callback)

        # Call report from multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=reporter.report, args=(f"event_{i}", {"value": i}))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All events should have been reported
        assert callback.call_count == 10


class TestProcessReporter:
    """Test ProcessReporter for multiprocessing queue-based event reporting."""

    def test_initialization(self) -> None:
        """ProcessReporter can be initialized with a multiprocessing queue."""
        queue: MultiprocessingQueue[Any] = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)
        assert reporter.queue is queue
        queue.close()

    def test_report_puts_event_in_queue(self) -> None:
        """report() sends event via multiprocessing queue."""
        # Use a mock queue to avoid multiprocessing complexities in tests
        queue = Mock()
        reporter = ProcessEventReporter(queue)

        reporter.report("test_event", {"key": "value"})

        queue.put.assert_called_once_with({"type": "test_event", "key": "value"})

    def test_report_multiple_events(self) -> None:
        """report() can send multiple events."""
        queue = Mock()
        reporter = ProcessEventReporter(queue)

        reporter.report("event1", {"data": 1})
        reporter.report("event2", {"data": 2})

        assert queue.put.call_count == 2
        queue.put.assert_any_call({"type": "event1", "data": 1})
        queue.put.assert_any_call({"type": "event2", "data": 2})

    def test_report_handles_queue_exception(self, caplog) -> None:
        """report() handles exceptions when putting to queue."""
        queue = Mock()
        queue.put.side_effect = RuntimeError("Queue error")
        reporter = ProcessEventReporter(queue)

        # Should not raise, but log the error
        reporter.report("test_event", {"key": "value"})

        # Verify error was logged
        assert "Error reporting event test_event" in caplog.text
        assert "Queue error" in caplog.text

    def test_close_closes_queue(self) -> None:
        """close() closes the underlying multiprocessing queue."""
        queue: MultiprocessingQueue[Any] = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)

        reporter.close()

        # Queue should be closed - further operations should fail
        with pytest.raises((ValueError, AssertionError)):
            queue.put("test")

    def test_queue_property(self) -> None:
        """queue property returns the underlying multiprocessing queue."""
        queue: MultiprocessingQueue[Any] = MultiprocessingQueue()
        reporter = ProcessEventReporter(queue)
        assert reporter.queue is queue
        queue.close()
