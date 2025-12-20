"""Event reporter implementations for different backend types."""

import logging
from multiprocessing import Queue as MultiprocessingQueue
from queue import Queue
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


class ThreadReporter:
    """
    Thread-based reporter for ThreadBackend.

    Uses a thread-safe queue to send events from worker threads to coordinator.
    """

    def __init__(self, queue: Queue[Any]):
        """
        Initialize reporter with queue.

        Args:
            queue: Thread-safe queue for sending events
        """
        self._queue = queue

    @property
    def queue(self) -> Queue[Any]:
        """Get the underlying queue."""
        return self._queue

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event via queue."""
        event = {"type": event_type, **data}
        try:
            self._queue.put(event)
        except Exception as e:
            logger.exception(f"Error reporting event {event_type}: {e}")


class ProcessReporter:
    """
    Queue-based reporter for ProcessPoolBackend.

    Uses multiprocessing.Queue for IPC. Background thread on coordinator consumes queue and
    dispatches events.
    """

    def __init__(self, queue: MultiprocessingQueue):
        """
        Initialize reporter with queue.

        Args:
            queue: Multiprocessing queue for sending events
        """
        self._queue = queue

    @property
    def queue(self) -> MultiprocessingQueue:
        """Get the underlying multiprocessing queue."""
        return self._queue

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event via queue."""
        event = {"type": event_type, **data}
        try:
            self._queue.put(event)
        except Exception as e:
            logger.exception(f"Error reporting event {event_type}: {e}")

    def close(self) -> None:
        """Close the underlying queue."""
        self._queue.close()


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
