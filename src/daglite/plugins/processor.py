from typing import Any

from daglite._processor import BackgroundQueueProcessor
from daglite.plugins.events import EventRegistry
from daglite.plugins.events import PluginEvent


class EventProcessor(BackgroundQueueProcessor):
    """Background processor for dispatching worker → coordinator events."""

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
            event: `PluginEvent` to dispatch
        """
        self._registry.dispatch(event)

    def _handle_item(self, item: Any) -> None:
        """Dispatch an event through the registry."""
        self._registry.dispatch(item)
