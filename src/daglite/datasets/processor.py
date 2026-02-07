"""
Background processor for dataset save requests from workers.

The ``DatasetProcessor`` mirrors the ``EventProcessor`` pattern: it runs a
background thread that drains save requests from IPC sources (e.g.
``multiprocessing.Queue`` instances used by ``ProcessDatasetReporter``) and
writes them using the ``DatasetStore`` carried within each request.

For direct reporters the save happens inline, so the processor only needs to
handle queue-based sources from process/remote backends.
"""

from __future__ import annotations

import logging
import time
from threading import Thread
from typing import Any
from uuid import UUID
from uuid import uuid4

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Background processor for draining worker → coordinator dataset save requests.

    Unlike the initial implementation, the processor does **not** own a single
    ``DatasetStore``.  Each save request arriving through the queue carries its
    own store reference, so different output configs can target different stores
    without ambiguity.
    """

    def __init__(self) -> None:
        self._sources: dict[UUID, Any] = {}
        self._running = False
        self._thread: Thread | None = None

    def add_source(self, source: Any) -> UUID:
        """
        Add a save-request source to process.

        Can be called before or after ``start()``.  Thread-safe for adding
        sources while the processor is running.

        Args:
            source: A queue-like object (``multiprocessing.Queue``) from which
                save requests are consumed.

        Returns:
            Unique ID for the added source.
        """
        id = uuid4()
        self._sources[id] = source
        logger.debug(f"Added dataset source (id: {id}): {type(source).__name__}")
        return id

    def remove_source(self, source_id: UUID) -> None:
        """
        Remove a save-request source by ID.

        Args:
            source_id: Unique ID of the source to remove.
        """
        if source_id in self._sources:
            del self._sources[source_id]
            logger.debug(f"Removed dataset source (id: {source_id})")
        else:
            logger.warning(f"Tried to remove unknown dataset source (id: {source_id})")

    def start(self) -> None:
        """Start background processing of all registered sources."""
        if self._thread is not None:
            logger.warning("DatasetProcessor already started")
            return

        self._running = True
        self._thread = Thread(target=self._process_loop, daemon=True, name="DatasetProcessor")
        self._thread.start()
        logger.debug("DatasetProcessor background thread started")

    def flush(self, timeout: float = 2.0) -> None:
        """
        Drain all pending save requests from sources.

        Continues processing until all sources are empty or *timeout* is
        reached.

        Args:
            timeout: Maximum time to wait for sources to drain (seconds).
        """
        start_time = time.time()
        while time.time() - start_time < timeout:  # pragma: no branch
            has_requests = False

            for source in self._sources.values():
                request = self._get_request(source)
                if request is not None:
                    self._handle_request(request)
                    has_requests = True

            if not has_requests:
                break

            time.sleep(0.001)

    def stop(self) -> None:
        """Stop background processing and join thread."""
        if self._thread is None:
            return

        logger.debug("Stopping DatasetProcessor...")

        # Drain remaining requests before stopping
        self.flush()

        self._running = False
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():  # pragma: no cover
            logger.warning("DatasetProcessor thread did not stop cleanly")
        self._thread = None

    # -- internal helpers --------------------------------------------------

    def _process_loop(self) -> None:
        """Background loop consuming save requests from all sources."""
        while self._running:
            has_requests = False

            for source in self._sources.values():
                request = self._get_request(source)
                if request is not None:
                    self._handle_request(request)
                    has_requests = True

            if not has_requests:
                time.sleep(0.001)

        logger.debug("DatasetProcessor loop exited")

    def _handle_request(self, request: dict[str, Any]) -> None:
        """
        Process a single save request received from a worker.

        The request dict is expected to have:
        - `key`: storage key/path
        - `value`: the Python object (already unpickled by the queue)
        - `store`: the ``DatasetStore`` to write to
        - `format`: optional serialization format hint

        Args:
            request: Save request dictionary.
        """
        try:
            store = request["store"]
            store.save(
                request["key"],
                request["value"],
                format=request.get("format"),
                options=request.get("options"),
            )
        except Exception as e:
            logger.exception(f"Error processing dataset save request: {e}")

    @staticmethod
    def _get_request(source: Any) -> dict[str, Any] | None:
        """
        Get a save request from a source (duck-typed for Queue-like objects).

        Args:
            source: A queue-like object with a ``get`` method.

        Returns:
            A save request dict, or *None* if no request is available.
        """
        if hasattr(source, "get"):
            try:
                return source.get(timeout=0.001)
            except Exception:
                return None

        logger.warning(f"Unknown dataset source type: {type(source)}")
        return None
