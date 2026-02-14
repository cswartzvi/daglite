"""Unit tests for EventRegistry."""

from daglite.plugins.events import Event
from daglite.plugins.registry import EventRegistry


class TestEventRegistry:
    """Tests for EventRegistry."""

    def test_register_handler(self) -> None:
        """Handlers can be registered for event types."""
        registry = EventRegistry()
        events_received = []

        def handler(event: Event) -> None:
            events_received.append(event)

        registry.register("test_event", handler)

        # Dispatch event
        registry.dispatch(Event(type="test_event", data={"data": "hello"}))

        assert len(events_received) == 1
        assert events_received[0].type == "test_event"
        assert events_received[0].data["data"] == "hello"

    def test_multiple_handlers_for_same_event(self) -> None:
        """Multiple handlers can be registered for the same event type."""
        registry = EventRegistry()
        handler1_events = []
        handler2_events = []

        def handler1(event: Event) -> None:
            handler1_events.append(event)

        def handler2(event: Event) -> None:
            handler2_events.append(event)

        registry.register("test_event", handler1)
        registry.register("test_event", handler2)

        registry.dispatch(Event(type="test_event", data={"data": "test"}))

        assert len(handler1_events) == 1
        assert len(handler2_events) == 1

    def test_dispatch_unknown_event_type_does_nothing(self) -> None:
        """Dispatching unknown event types doesn't raise errors."""
        registry = EventRegistry()
        events_received = []

        def handler(event: Event) -> None:
            events_received.append(event)

        registry.register("known_event", handler)

        # Dispatch unknown event type - should not call handler
        registry.dispatch(Event(type="unknown_event", data={"data": "test"}))

        assert len(events_received) == 0

    def test_handler_exception_logged_but_doesnt_stop_other_handlers(self, caplog) -> None:
        """Exceptions in handlers are logged but don't prevent other handlers from running."""
        registry = EventRegistry()
        handler1_called = []
        handler2_called = []

        def failing_handler(event: Event) -> None:
            handler1_called.append(True)
            raise ValueError("Handler failed!")

        def working_handler(event: Event) -> None:
            handler2_called.append(True)

        registry.register("test_event", failing_handler)
        registry.register("test_event", working_handler)

        registry.dispatch(Event(type="test_event", data={"data": "test"}))

        # Both handlers should have been called
        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

        # Error should be logged
        assert "Error in event handler" in caplog.text
