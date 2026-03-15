"""Unit tests for EventRegistry."""

from typing import Any, Hashable

from daglite.plugins.events import EventRegistry


class TestEventRegistry:
    """Tests for EventRegistry."""

    def test_register_handler(self) -> None:
        """Handlers can be registered for event types."""
        registry = EventRegistry()
        events_received: list[tuple[Hashable, dict[str, Any]]] = []

        def handler(event_type: Hashable, event_data: dict[str, Any]) -> None:
            events_received.append((event_type, event_data))

        registry.register("test_event", handler)

        # Dispatch event
        registry.dispatch("test_event", {"data": "hello"})

        assert len(events_received) == 1
        assert events_received[0][0] == "test_event"
        assert events_received[0][1]["data"] == "hello"

    def test_multiple_handlers_for_same_event(self) -> None:
        """Multiple handlers can be registered for the same event type."""
        registry = EventRegistry()
        handler1_events: list[tuple[Hashable, dict[str, Any]]] = []
        handler2_events: list[tuple[Hashable, dict[str, Any]]] = []

        def handler1(event_type: Hashable, event_data: dict[str, Any]) -> None:
            handler1_events.append((event_type, event_data))

        def handler2(event_type: Hashable, event_data: dict[str, Any]) -> None:
            handler2_events.append((event_type, event_data))

        registry.register("test_event", handler1)
        registry.register("test_event", handler2)

        registry.dispatch("test_event", {"data": "test"})

        assert len(handler1_events) == 1
        assert len(handler2_events) == 1

    def test_dispatch_unknown_event_type_does_nothing(self) -> None:
        """Dispatching unknown event types doesn't raise errors."""
        registry = EventRegistry()
        events_received: list[tuple[Hashable, dict[str, Any]]] = []

        def handler(event_type: Hashable, event_data: dict[str, Any]) -> None:
            events_received.append((event_type, event_data))

        registry.register("known_event", handler)

        # Dispatch unknown event type - should not call handler
        registry.dispatch("unknown_event", {"data": "test"})

        assert len(events_received) == 0

    def test_handler_exception_logged_but_doesnt_stop_other_handlers(self, caplog) -> None:
        """Exceptions in handlers are logged but don't prevent other handlers from running."""
        registry = EventRegistry()
        handler1_called: list[bool] = []
        handler2_called: list[bool] = []

        def failing_handler(event_type: Hashable, event_data: dict[str, Any]) -> None:
            handler1_called.append(True)
            raise ValueError("Handler failed!")

        def working_handler(event_type: Hashable, event_data: dict[str, Any]) -> None:
            handler2_called.append(True)

        registry.register("test_event", failing_handler)
        registry.register("test_event", working_handler)

        registry.dispatch("test_event", {"data": "test"})

        # Both handlers should have been called
        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

        # Error should be logged
        assert "Error in event handler" in caplog.text
