"""Event reporter implementations for different backend types."""

import logging
import multiprocessing
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class EventReporter(Protocol):
    """Protocol for worker → coordinator communication."""

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Send event from worker to coordinator.

        Args:
            event_type: Type of event (e.g., "cache_hit", "progress")
            data: Event payload data
        """
        ...


class DirectReporter:
    """
    Direct function call reporter for ThreadPoolBackend.

    No serialization needed since everything runs in the same process.
    Events are dispatched immediately via callback.
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        """
        Initialize reporter with callback.

        Args:
            callback: Function to call with events (typically EventProcessor.dispatch)
        """
        self._callback = callback

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event via direct callback."""
        event = {"type": event_type, **data}
        try:
            self._callback(event)
        except Exception as e:
            logger.exception(f"Error reporting event {event_type}: {e}")


class QueueReporter:
    """
    Queue-based reporter for ProcessPoolBackend.

    Uses multiprocessing.Queue for IPC. Background thread on coordinator consumes queue and
    dispatches events.
    """

    def __init__(self, queue: multiprocessing.Queue):
        """
        Initialize reporter with queue.

        Args:
            queue: Multiprocessing queue for sending events
        """
        self._queue = queue

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event via queue."""
        event = {"type": event_type, **data}
        try:
            self._queue.put(event)
        except Exception as e:
            logger.exception(f"Error reporting event {event_type}: {e}")


class RemoteReporter:  # pragma: no cover
    """
    Network-based reporter for distributed backends.

    Sends events via HTTP/gRPC to coordinator.
    """

    def __init__(self, endpoint: str):
        """
        Initialize reporter with coordinator endpoint.

        Args:
            endpoint: URL or address of coordinator
        """
        self._endpoint = endpoint
        # TODO: Initialize HTTP session or gRPC stub

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event via network."""
        # TODO: Implement network transport
        raise NotImplementedError("RemoteReporter not yet implemented")
