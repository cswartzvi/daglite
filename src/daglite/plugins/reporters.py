"""Event reporter implementations for different backend types."""

import abc
import logging
import threading
from multiprocessing import Queue as MultiprocessingQueue
from typing import Any, Callable

from typing_extensions import override

logger = logging.getLogger(__name__)


class EventReporter(abc.ABC):
    """Protocol for worker → coordinator communication."""

    @property
    @abc.abstractmethod
    def is_direct(self) -> bool:
        """
        Indicates whether this reporter sends events directly via function calls.

        Non-direct reporters require serialization and IPC, while direct reporters can call
        coordinator callbacks directly (e.g., in the same process or thread).

        Returns:
            True if this reporter sends events directly, False if it uses IPC/serialization.
        """
        ...

    @abc.abstractmethod
    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Send event from worker to coordinator.

        Args:
            event_type: Type of event (e.g., "cache_hit", "progress")
            data: Event payload data
        """
        ...


class DirectEventReporter(EventReporter):
    """
    Direct function call reporter for Inline and threaded execution.

    No serialization needed since everything runs in the same process. Events are dispatched
    immediately via callback. Thread-safe for use in ThreadPoolExecutor.

    Args:
        callback: Function to call with event dict when reporting an event.
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        self._callback = callback
        self._lock = threading.Lock()

    @property
    @override
    def is_direct(self) -> bool:
        return True

    @override
    def report(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, **data}
        try:
            with self._lock:
                self._callback(event)
        except Exception as e:
            logger.exception(f"Error reporting event {event_type}: {e}")


class ProcessEventReporter(EventReporter):
    """
    Queue-based reporter for ProcessPoolBackend.

    Uses multiprocessing.Queue for IPC. Background thread on coordinator consumes queue and
    dispatches events.

    Args:
        queue: MultiprocessingQueue for sending events to coordinator

    """

    def __init__(self, queue: MultiprocessingQueue):
        self._queue = queue

    @property
    @override
    def is_direct(self) -> bool:
        return False

    @property
    def queue(self) -> MultiprocessingQueue:
        """Get the underlying multiprocessing queue."""
        return self._queue

    @override
    def report(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, **data}
        try:
            self._queue.put(event)
        except Exception as e:
            logger.exception(f"Error reporting event {event_type}: {e}")

    def close(self) -> None:
        """Close the underlying queue."""
        self._queue.close()


class RemoteEventReporter(EventReporter):  # pragma: no cover
    """
    Network-based reporter for distributed backends.

    Sends events via HTTP/gRPC to coordinator.

    Args:
        endpoint: Network endpoint (e.g., URL) of coordinator's event receiver
    """

    # TODO: Initialize HTTP session or gRPC stub

    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    @property
    @override
    def is_direct(self) -> bool:
        return False

    @override
    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event via network."""
        raise NotImplementedError("RemoteReporter not yet implemented")
