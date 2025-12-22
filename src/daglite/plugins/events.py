"""Event registry and processing for coordinator-side event handling."""

import logging
import time
from threading import Thread
from typing import Any, Callable
from uuid import UUID
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventRegistry:
    """
    Registry for coordinator-side event handlers.

    Plugins register handlers for specific event types, which are called when events are dispatched.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}

    def register(self, event_type: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """
        Register handler for event type.

        Multiple handlers can be registered for the same event type.

        Args:
            event_type: Type of event to handle
            handler: Callable that takes event dict
        """
        self._handlers.setdefault(event_type, []).append(handler)

    def dispatch(self, event: dict[str, Any]) -> None:
        """
        Dispatch event to all registered handlers.

        Handlers are called synchronously. Errors are logged but don't prevent other handlers from
        running.

        Args:
            event: Event dict with "type" key and additional data
        """
        event_type = event.get("type")
        if not event_type:
            logger.warning(f"Event missing 'type' field: {event}")
            return

        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Error in event handler for '{event_type}': {e}")


class EventProcessor:
    """Background processor for dispatching worker → coordinator events."""

    def __init__(self, registry: EventRegistry):
        """
        Initialize processor with event registry.

        Args:
            registry: Registry containing event handlers
        """
        self._registry = registry
        self._sources: dict[UUID, Any] = {}
        self._running = False
        self._thread: Thread | None = None

    def add_source(self, source: Any) -> UUID:
        """
        Add an event source to process.

        Can be called before or after start(). Thread-safe for adding sources
        while processor is running.

        Args:
            source: Event source (e.g., multiprocessing.Queue).

        Returns:
            Unique ID for the added source.
        """
        id = uuid4()
        self._sources[id] = source
        logger.debug(f"Added event source (id: {id}): {type(source).__name__}")
        return id

    def remove_source(self, source_id: UUID) -> None:
        """
        Remove an event source by ID.

        Can be called before or after start(). Thread-safe for removing sources
        while processor is running.

        Args:
            source_id: Unique ID of the source to remove.
        """
        if source_id in self._sources:
            del self._sources[source_id]
            logger.debug(f"Removed event source (id: {source_id})")
        else:
            logger.warning(f"Tried to remove unknown event source (id: {source_id})")

    def start(self) -> None:
        """Start background processing of all registered sources."""
        if self._thread is not None:
            logger.warning("EventProcessor already started")
            return

        self._running = True
        self._thread = Thread(target=self._process_loop, daemon=True, name="EventProcessor")
        self._thread.start()
        logger.debug("EventProcessor background thread started")

    def dispatch(self, event: dict[str, Any]) -> None:
        """
        Dispatch event immediately (bypass background processing).

        Can be used for direct event dispatching from same process.

        Args:
            event: Event dict to dispatch
        """
        self._registry.dispatch(event)

    def stop(self) -> None:
        """Stop background processing and join thread."""
        if self._thread is None:
            return

        logger.debug("Stopping EventProcessor...")
        self._running = False
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():  # pragma: no cover
            logger.warning("EventProcessor thread did not stop cleanly")
        self._thread = None

    def _process_loop(self) -> None:
        """Background loop consuming events from all sources."""
        while self._running:
            has_events = False

            # Poll all sources for events
            for source in self._sources.values():
                event = self._get_event(source)
                if event:
                    self._registry.dispatch(event)
                    has_events = True

            # Sleep if no events to avoid busy-wait
            if not has_events:
                time.sleep(0.001)

        logger.debug("EventProcessor loop exited")

    def _get_event(self, source: Any) -> dict[str, Any] | None:
        """
        Get event from source (polymorphic based on source type).

        Args:
            source: Event source

        Returns:
            Event dict or None if no event available
        """
        # Check if source has a get method (duck typing for Queue-like objects)
        if hasattr(source, "get"):
            try:
                return source.get(timeout=0.001)
            except Exception:
                return None

        # Add other source types as needed (sockets, etc.)
        logger.warning(f"Unknown event source type: {type(source)}")
        return None
