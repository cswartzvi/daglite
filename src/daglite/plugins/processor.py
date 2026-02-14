from typing import Any

from daglite._processor import BackgroundQueueProcessor
from daglite.plugins.registry import EventRegistry


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

    def dispatch(self, event: dict[str, Any]) -> None:
        """
        Dispatch event immediately (bypass background processing).

        Can be used for direct event dispatching from same process.

        Args:
            event: Event dict to dispatch
        """
        self._registry.dispatch(event)

    def _handle_item(self, item: Any) -> None:
        """Dispatch an event through the registry."""
        self._registry.dispatch(item)
