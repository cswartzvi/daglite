"""Unit tests for EventProcessor (subclass-specific behaviour only).

Base class lifecycle tests live in tests/test_processors.py.
"""

from __future__ import annotations

import time
from queue import Queue
from typing import Any

from daglite.plugins.events import EventProcessor
from daglite.plugins.events import EventRegistry
from daglite.plugins.events import PluginEvent


class TestEventProcessor:
    def test_direct_dispatch(self) -> None:
        """Events can be dispatched directly without background processing."""
        events_received: list[tuple[Any, Any]] = []

        def handler(event_type, event_data):
            events_received.append((event_type, event_data))

        registry = EventRegistry()
        registry.register("test_event", handler)

        processor = EventProcessor(registry)
        processor.dispatch(PluginEvent(type="test_event", data={"data": "direct"}))

        assert len(events_received) == 1
        assert events_received[0][1]["data"] == "direct"

    def test_background_processing_from_queue(self) -> None:
        """Background processor consumes events from queue sources."""
        events_received: list[tuple[Any, Any]] = []

        def handler(event_type, event_data):
            events_received.append((event_type, event_data))

        registry = EventRegistry()
        registry.register("test_event", handler)

        processor = EventProcessor(registry)
        queue: Queue[Any] = Queue()
        processor.add_source(queue)

        processor.start()

        queue.put(PluginEvent(type="test_event", data={"data": "event1"}))
        queue.put(PluginEvent(type="test_event", data={"data": "event2"}))

        time.sleep(0.1)

        processor.stop()

        assert len(events_received) == 2
        assert events_received[0][1]["data"] == "event1"
        assert events_received[1][1]["data"] == "event2"
