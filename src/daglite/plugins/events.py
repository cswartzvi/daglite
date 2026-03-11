"""Event system for coordinator-side plugin handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Hashable, override

from daglite._processor import BackgroundQueueProcessor

logger = logging.getLogger(__name__)


class EventProcessor(BackgroundQueueProcessor):
    """Background processor for dispatching events from workers to coordinator."""

    def __init__(self, registry: EventRegistry):
        """
        Initialize processor with event registry.

        Args:
            registry: Registry containing event handlers
        """
        super().__init__(name="EventProcessor")
        self._registry = registry

    def dispatch(self, event: PluginEvent) -> None:
        """
        Dispatch event immediately (bypass background processing).

        Can be used for direct event dispatching from same process.

        Args:
            event: `Event` to dispatch
        """
        self._registry.dispatch(event.type, event.data)

    @override
    def _handle_item(self, item: PluginEvent) -> None:
        self._registry.dispatch(item.type, item.data)


class EventRegistry:
    """
    Registry for coordinator-side event handlers.

    Plugins register handlers for specific event types, which are called when events are dispatched.
    """

    def __init__(self) -> None:
        self._handlers: dict[Hashable, list[Callable[[Hashable, dict[str, Any]], None]]] = {}

    # NOTE: The methods should use `event_type` and `event_data` instead of `PluginEvent` to avoid
    # requiring plugins, plugin developers, from knowing about the `PluginEvent` class.

    def register(
        self, event_type: Hashable, handler: Callable[[Hashable, dict[str, Any]], None]
    ) -> None:
        """
        Registers a handler for event type.

        Multiple handlers can be registered for the same event type and will be called in the order
        they were registered.

        Args:
            event_type: Event type used to identify the event; must be hashable.
            handler: Callable that takes event type and data dict as arguments.
        """
        self._handlers.setdefault(event_type, []).append(handler)

    def dispatch(self, event_type: Hashable, event_data: dict[str, Any]) -> None:
        """
        Dispatch event to all registered handlers.

        Handlers are called synchronously. Errors are logged but don't prevent other handlers from
        running.

        Args:
            event_type: Event type to dispatch; must be hashable.
            event_data: Optional dict of event data to pass to handlers.
        """
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event_type, event_data or {})
            except Exception as e:
                logger.exception(f"Error in event handler for '{event_type}': {e}")


@dataclass(frozen=True)
class PluginEvent:
    """
    Represents an event sent from a worker to the coordinator.

    This class is used to encapsulate events sent from workers to the coordinator, and should be
    considered an **internal implementation detail**. Plugins should use the `EventRegistry` to
    register handlers and dispatch events, rather than creating `PluginEvent` instances directly.

    Attributes:
        type: Event type identifier; must be hashable.
        data: Arbitrary payload data.

    Examples:
        >>> event = PluginEvent(type="progress", data={"percent": 100})
        >>> event.type
        'progress'
        >>> event.data["percent"]
        100
    """

    type: Hashable
    data: dict[str, Any] = field(default_factory=dict)
