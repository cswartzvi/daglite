"""Unit tests for event registry and processor."""

import time
from queue import Queue
from typing import Any

from daglite.plugins.processor import EventProcessor
from daglite.plugins.registry import EventRegistry


class TestEventRegistry:
    """Tests for EventRegistry."""

    def test_register_handler(self) -> None:
        """Handlers can be registered for event types."""
        registry = EventRegistry()
        events_received = []

        def handler(event: dict) -> None:
            events_received.append(event)

        registry.register("test_event", handler)

        # Dispatch event
        registry.dispatch({"type": "test_event", "data": "hello"})

        assert len(events_received) == 1
        assert events_received[0]["type"] == "test_event"
        assert events_received[0]["data"] == "hello"

    def test_multiple_handlers_for_same_event(self) -> None:
        """Multiple handlers can be registered for the same event type."""
        registry = EventRegistry()
        handler1_events = []
        handler2_events = []

        def handler1(event: dict) -> None:
            handler1_events.append(event)

        def handler2(event: dict) -> None:
            handler2_events.append(event)

        registry.register("test_event", handler1)
        registry.register("test_event", handler2)

        registry.dispatch({"type": "test_event", "data": "test"})

        assert len(handler1_events) == 1
        assert len(handler2_events) == 1

    def test_dispatch_missing_type_field_logs_warning(self, caplog) -> None:
        """Events without 'type' field log a warning."""
        registry = EventRegistry()

        registry.dispatch({"data": "no type field"})

        assert "Event missing 'type' field" in caplog.text

    def test_dispatch_unknown_event_type_does_nothing(self) -> None:
        """Dispatching unknown event types doesn't raise errors."""
        registry = EventRegistry()
        events_received = []

        def handler(event: dict) -> None:
            events_received.append(event)

        registry.register("known_event", handler)

        # Dispatch unknown event type - should not call handler
        registry.dispatch({"type": "unknown_event", "data": "test"})

        assert len(events_received) == 0

    def test_handler_exception_logged_but_doesnt_stop_other_handlers(self, caplog) -> None:
        """Exceptions in handlers are logged but don't prevent other handlers from running."""
        registry = EventRegistry()
        handler1_called = []
        handler2_called = []

        def failing_handler(event: dict) -> None:
            handler1_called.append(True)
            raise ValueError("Handler failed!")

        def working_handler(event: dict) -> None:
            handler2_called.append(True)

        registry.register("test_event", failing_handler)
        registry.register("test_event", working_handler)

        registry.dispatch({"type": "test_event", "data": "test"})

        # Both handlers should have been called
        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

        # Error should be logged
        assert "Error in event handler" in caplog.text


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

        assert "Tried to remove unknown event source" in caplog.text

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

    def test_get_event_with_unknown_source_type_logs_warning(self, caplog) -> None:
        """_get_event with unknown source type logs a warning."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        # Use a source without a 'get' method
        unknown_source = "not a queue"
        result = processor._get_event(unknown_source)

        assert result is None
        assert "Unknown event source type" in caplog.text

    def test_get_event_returns_none_when_queue_empty(self) -> None:
        """_get_event returns None when queue is empty."""
        registry = EventRegistry()
        processor = EventProcessor(registry)

        queue: Queue[Any] = Queue()
        result = processor._get_event(queue)

        assert result is None
