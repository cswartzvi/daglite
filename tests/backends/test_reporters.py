"""
Unit tests for event reporters.

These tests verify reporter implementations without requiring full evaluation.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/.
"""

from multiprocessing import Queue as MultiprocessingQueue
from queue import Queue
from typing import Any
from unittest.mock import Mock

import pytest

from daglite.plugins.reporters import DirectReporter
from daglite.plugins.reporters import ProcessReporter
from daglite.plugins.reporters import ThreadReporter


class TestDirectReporter:
    """Test DirectReporter for direct callback-based event reporting."""

    def test_initialization(self) -> None:
        """DirectReporter can be initialized with a callback."""
        callback = Mock()
        reporter = DirectReporter(callback)
        assert reporter._callback is callback

    def test_report_calls_callback(self) -> None:
        """report() sends event via callback."""
        callback = Mock()
        reporter = DirectReporter(callback)

        reporter.report("test_event", {"key": "value"})

        callback.assert_called_once_with({"type": "test_event", "key": "value"})

    def test_report_with_multiple_events(self) -> None:
        """report() can be called multiple times."""
        callback = Mock()
        reporter = DirectReporter(callback)

        reporter.report("event1", {"data": 1})
        reporter.report("event2", {"data": 2})

        assert callback.call_count == 2
        callback.assert_any_call({"type": "event1", "data": 1})
        callback.assert_any_call({"type": "event2", "data": 2})

    def test_report_handles_callback_exception(self, caplog) -> None:
        """report() handles exceptions from callback gracefully."""
        callback = Mock(side_effect=ValueError("Callback error"))
        reporter = DirectReporter(callback)

        # Should not raise, but log the error
        reporter.report("test_event", {"key": "value"})

        # Verify error was logged
        assert "Error reporting event test_event" in caplog.text
        assert "Callback error" in caplog.text


class TestThreadReporter:
    """Test ThreadReporter for thread-safe queue-based event reporting."""

    def test_initialization(self) -> None:
        """ThreadReporter can be initialized with a queue."""
        queue: Queue = Queue()
        reporter = ThreadReporter(queue)
        assert reporter.queue is queue

    def test_report_puts_event_in_queue(self) -> None:
        """report() sends event via queue."""
        queue: Queue = Queue()
        reporter = ThreadReporter(queue)

        reporter.report("test_event", {"key": "value"})

        event = queue.get_nowait()
        assert event == {"type": "test_event", "key": "value"}

    def test_report_multiple_events(self) -> None:
        """report() can send multiple events."""
        queue: Queue = Queue()
        reporter = ThreadReporter(queue)

        reporter.report("event1", {"data": 1})
        reporter.report("event2", {"data": 2})

        event1 = queue.get_nowait()
        event2 = queue.get_nowait()

        assert event1 == {"type": "event1", "data": 1}
        assert event2 == {"type": "event2", "data": 2}

    def test_report_handles_queue_exception(self, caplog) -> None:
        """report() handles exceptions when putting to queue."""
        queue = Mock(spec=Queue)
        queue.put.side_effect = RuntimeError("Queue error")
        reporter = ThreadReporter(queue)

        # Should not raise, but log the error
        reporter.report("test_event", {"key": "value"})

        # Verify error was logged
        assert "Error reporting event test_event" in caplog.text
        assert "Queue error" in caplog.text

    def test_queue_property(self) -> None:
        """queue property returns the underlying queue."""
        queue: Queue = Queue()
        reporter = ThreadReporter(queue)
        assert reporter.queue is queue


class TestProcessReporter:
    """Test ProcessReporter for multiprocessing queue-based event reporting."""

    def test_initialization(self) -> None:
        """ProcessReporter can be initialized with a multiprocessing queue."""
        queue: MultiprocessingQueue[Any] = MultiprocessingQueue()
        reporter = ProcessReporter(queue)
        assert reporter.queue is queue
        queue.close()

    def test_report_puts_event_in_queue(self) -> None:
        """report() sends event via multiprocessing queue."""
        # Use a mock queue to avoid multiprocessing complexities in tests
        queue = Mock()
        reporter = ProcessReporter(queue)

        reporter.report("test_event", {"key": "value"})

        queue.put.assert_called_once_with({"type": "test_event", "key": "value"})

    def test_report_multiple_events(self) -> None:
        """report() can send multiple events."""
        queue = Mock()
        reporter = ProcessReporter(queue)

        reporter.report("event1", {"data": 1})
        reporter.report("event2", {"data": 2})

        assert queue.put.call_count == 2
        queue.put.assert_any_call({"type": "event1", "data": 1})
        queue.put.assert_any_call({"type": "event2", "data": 2})

    def test_report_handles_queue_exception(self, caplog) -> None:
        """report() handles exceptions when putting to queue."""
        queue = Mock()
        queue.put.side_effect = RuntimeError("Queue error")
        reporter = ProcessReporter(queue)

        # Should not raise, but log the error
        reporter.report("test_event", {"key": "value"})

        # Verify error was logged
        assert "Error reporting event test_event" in caplog.text
        assert "Queue error" in caplog.text

    def test_close_closes_queue(self) -> None:
        """close() closes the underlying multiprocessing queue."""
        queue: MultiprocessingQueue[Any] = MultiprocessingQueue()
        reporter = ProcessReporter(queue)

        reporter.close()

        # Queue should be closed - further operations should fail
        with pytest.raises((ValueError, AssertionError)):
            queue.put("test")

    def test_queue_property(self) -> None:
        """queue property returns the underlying multiprocessing queue."""
        queue: MultiprocessingQueue[Any] = MultiprocessingQueue()
        reporter = ProcessReporter(queue)
        assert reporter.queue is queue
        queue.close()
