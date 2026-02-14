"""Shared base class for background queue processors."""

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod
from threading import Thread
from typing import Any
from uuid import UUID
from uuid import uuid4

logger = logging.getLogger(__name__)


class BackgroundQueueProcessor(ABC):
    """
    Base class for processors that drain items from queue-like sources on a
    background daemon thread.

    Subclasses must implement :meth:`_handle_item` to define what happens when
    an item is dequeued.  The rest of the lifecycle — source management,
    thread start/stop, flushing — is handled here.

    Args:
        name: Human-readable name used in log messages and as the thread name.
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._sources: dict[UUID, Any] = {}
        self._running = False
        self._thread: Thread | None = None

    def add_source(self, source: Any) -> UUID:
        """
        Add a queue-like source to process.

        Can be called before or after ``start()``.  Thread-safe for adding
        sources while the processor is running.

        Args:
            source: A queue-like object with a ``get(timeout=...)`` method.

        Returns:
            Unique ID for the added source.
        """
        id = uuid4()
        self._sources[id] = source
        logger.debug(f"Added {self._name} source (id: {id}): {type(source).__name__}")
        return id

    def remove_source(self, source_id: UUID) -> None:
        """
        Remove a source by ID.

        Args:
            source_id: Unique ID of the source to remove.
        """
        if source_id in self._sources:
            del self._sources[source_id]
            logger.debug(f"Removed {self._name} source (id: {source_id})")
        else:
            logger.warning(f"Tried to remove unknown {self._name} source (id: {source_id})")

    def start(self) -> None:
        """Start background processing of all registered sources."""
        if self._thread is not None:
            logger.warning(f"{self._name} already started")
            return

        self._running = True
        self._thread = Thread(target=self._process_loop, daemon=True, name=self._name)
        self._thread.start()
        logger.debug(f"{self._name} background thread started")

    def flush(self, timeout: float = 2.0) -> None:
        """
        Drain all pending items from sources.

        Continues processing until all sources are empty or *timeout* is
        reached.

        Args:
            timeout: Maximum time to wait for sources to drain (seconds).
        """
        start_time = time.time()
        # Loop exits via break when queues drain (normal case) or timeout
        # (pathological case where items continuously arrive).
        # Timeout branch excluded from coverage.
        while time.time() - start_time < timeout:  # pragma: no branch
            has_items = False

            for source in list(self._sources.values()):
                item = self._get_item(source)
                if item is not None:
                    self._handle_item(item)
                    has_items = True

            if not has_items:
                break

            time.sleep(0.001)

    def stop(self) -> None:
        """Stop background processing and join thread."""
        if self._thread is None:
            return

        logger.debug(f"Stopping {self._name}...")

        # Drain remaining items before stopping
        self.flush()

        self._running = False
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():  # pragma: no cover
            logger.warning(f"{self._name} thread did not stop cleanly")
        self._thread = None

    @abstractmethod
    def _handle_item(self, item: Any) -> None:
        """
        Process a single item dequeued from a source.

        Subclasses **must** override this method.

        Args:
            item: The item retrieved from the queue.
        """

    def _process_loop(self) -> None:
        """Background loop consuming items from all sources."""
        while self._running:
            has_items = False

            for source in list(self._sources.values()):
                item = self._get_item(source)
                if item is not None:
                    self._handle_item(item)
                    has_items = True

            if not has_items:
                time.sleep(0.001)

        logger.debug(f"{self._name} loop exited")

    @staticmethod
    def _get_item(source: Any) -> Any | None:
        """
        Get an item from a source (duck-typed for Queue-like objects).

        Args:
            source: A queue-like object with a ``get`` method.

        Returns:
            The dequeued item, or *None* if nothing is available.
        """
        if hasattr(source, "get"):
            try:
                return source.get(timeout=0.001)
            except Exception:
                return None

        logger.warning(f"Unknown source type: {type(source)}")
        return None
