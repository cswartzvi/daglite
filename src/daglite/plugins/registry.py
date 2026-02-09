"""Event registry and processing for coordinator-side event handling."""

import logging
from typing import Any, Callable

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
