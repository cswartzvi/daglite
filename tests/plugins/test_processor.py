"""Unit tests for EventProcessor."""

import time
from queue import Queue
from typing import Any

from daglite.plugins.processor import EventProcessor
from daglite.plugins.registry import EventRegistry


class TestEventProcessor:
    """Tests for EventProcessor."""

    def test_add_source(self) -> None:
        """Sources can be added to the processor."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        queue: Queue[Any] = Queue()
        source_id = processor.add_source(queue)

        assert source_id in processor._sources
        assert processor._sources[source_id] is queue

    def test_remove_source(self) -> None:
        """Sources can be removed from the processor."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        queue: Queue[Any] = Queue()
        source_id = processor.add_source(queue)

        processor.remove_source(source_id)

        assert source_id not in processor._sources

    def test_remove_unknown_source_logs_warning(self, caplog) -> None:
        """Removing an unknown source logs a warning."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        from uuid import uuid4

        unknown_id = uuid4()
        processor.remove_source(unknown_id)

        assert "Tried to remove unknown EventProcessor source" in caplog.text

    def test_start_and_stop(self) -> None:
        """Processor can be started and stopped."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        processor.start()
        assert processor._running is True
        assert processor._thread is not None
        assert processor._thread.is_alive()

        processor.stop()
        assert processor._running is False
        assert processor._thread is None  # type: ignore

    def test_start_when_already_started_logs_warning(self, caplog) -> None:
        """Starting an already-started processor logs a warning."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        processor.start()
        processor.start()  # Second start

        assert "EventProcessor already started" in caplog.text

        processor.stop()

    def test_stop_when_not_started_does_nothing(self) -> None:
        """Stopping a non-started processor doesn't raise errors."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        processor.stop()  # Should not raise

    def test_direct_dispatch(self) -> None:
        """Events can be dispatched directly without background processing."""
        events_received = []

        def handler(event: dict) -> None:
            events_received.append(event)

        registry = EventRegistry()
        registry.register("test_event", handler)

        processor = EventProcessor(registry)
        processor.dispatch({"type": "test_event", "data": "direct"})

        assert len(events_received) == 1
        assert events_received[0]["data"] == "direct"

    def test_background_processing_from_queue(self) -> None:
        """Background processor consumes events from queue sources."""
        events_received = []

        def handler(event: dict) -> None:
            events_received.append(event)

        registry = EventRegistry()
        registry.register("test_event", handler)

        processor = EventProcessor(registry)
        queue: Queue[Any] = Queue()
        processor.add_source(queue)

        processor.start()

        # Put events in queue
        queue.put({"type": "test_event", "data": "event1"})
        queue.put({"type": "test_event", "data": "event2"})

        # Wait for processing
        time.sleep(0.1)

        processor.stop()

        assert len(events_received) == 2
        assert events_received[0]["data"] == "event1"
        assert events_received[1]["data"] == "event2"

    def test_get_item_with_unknown_source_type_logs_warning(self, caplog) -> None:
        """_get_item with unknown source type logs a warning."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        # Use a source without a 'get' method
        unknown_source = "not a queue"
        result = processor._get_item(unknown_source)

        assert result is None
        assert "Unknown source type" in caplog.text

    def test_get_item_returns_none_when_queue_empty(self) -> None:
        """_get_item returns None when queue is empty."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        queue: Queue[Any] = Queue()
        result = processor._get_item(queue)

        assert result is None
