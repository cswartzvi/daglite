"""
Dataset reporter implementations for different backend types.

Dataset reporters handle persisting task outputs (via DatasetStore) back to the
coordinator.  The design mirrors the event reporter hierarchy:

* **DirectDatasetReporter** – for inline / threaded backends where the store
  is directly accessible in the same process.
* **ProcessDatasetReporter** – for multiprocessing backends that need to push
  save requests across a ``multiprocessing.Queue`` so the coordinator can
  write them.
* **RemoteDatasetReporter** – placeholder for future distributed backends.

The *store* is passed per-save rather than held by the reporter, because
different output configs may target different stores.  For local drivers the
reporter routes saves to the coordinator; for remote drivers the worker saves
directly (bypassing the reporter altogether).
"""

from __future__ import annotations

import abc
import logging
import threading
from multiprocessing import Queue as MultiprocessingQueue
from typing import TYPE_CHECKING, Any

from typing_extensions import override

if TYPE_CHECKING:
    from pluggy import HookRelay

    from daglite.datasets.store import DatasetStore

from daglite.datasets.events import DatasetSaveRequest

logger = logging.getLogger(__name__)


class DatasetReporter(abc.ABC):
    """Abstract base class for worker → coordinator dataset communication."""

    @property
    @abc.abstractmethod
    def is_direct(self) -> bool:
        """
        Indicates whether this reporter saves datasets directly via the store.

        Non-direct reporters require serialization and IPC, while direct reporters can
        access the ``DatasetStore`` directly (e.g., in the same process or thread).

        Returns:
            True if this reporter saves directly, False if it uses IPC/serialization.
        """
        ...

    @abc.abstractmethod
    def save(
        self,
        key: str,
        value: Any,
        store: DatasetStore,
        *,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Save a dataset output from a worker.

        Args:
            key: Storage key/path for the output.
            value: The Python object to persist.
            store: The ``DatasetStore`` to write to.
            format: Optional serialization format hint (e.g. ``"pickle"``).
            options: Additional options passed to the Dataset's save method.
        """
        ...


class DirectDatasetReporter(DatasetReporter):
    """
    Direct reporter for inline and threaded backends.

    Saves are performed immediately through the supplied ``DatasetStore`` in
    the current process.  Thread-safe for use in ``ThreadPoolExecutor``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    @override
    def is_direct(self) -> bool:
        return True

    @override
    def save(
        self,
        key: str,
        value: Any,
        store: DatasetStore,
        *,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        hook = self._get_hook()
        with self._lock:
            if hook:
                hook.before_dataset_save(key=key, value=value, format=format, options=options)
            store.save(key, value, format=format, options=options)
            if hook:
                hook.after_dataset_save(key=key, value=value, format=format, options=options)

    @staticmethod
    def _get_hook() -> HookRelay | None:
        """Attempt to retrieve the plugin hook from the execution context."""
        try:
            from daglite.backends.context import get_plugin_manager

            return get_plugin_manager().hook
        except RuntimeError:
            return None


class ProcessDatasetReporter(DatasetReporter):
    """
    Queue-based reporter for ``ProcessPoolBackend``.

    Workers push save requests onto a ``multiprocessing.Queue``.  A background
    thread on the coordinator (managed by ``DatasetProcessor``) consumes the
    queue and performs the actual writes.

    The raw *value* and *store* are placed on the queue as-is; the
    ``multiprocessing.Queue`` handles pickling transparently.  Avoid explicit
    ``pickle.dumps`` here to prevent double-serialization.

    Args:
        queue: ``multiprocessing.Queue`` for sending save requests to the coordinator.
    """

    def __init__(self, queue: MultiprocessingQueue) -> None:
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
    def save(
        self,
        key: str,
        value: Any,
        store: DatasetStore,
        *,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        request = DatasetSaveRequest(
            key=key,
            value=value,
            store=store,
            format=format,
            options=options,
        )
        self._queue.put(request)

    def close(self) -> None:
        """Close the underlying queue."""
        self._queue.close()


class RemoteDatasetReporter(DatasetReporter):  # pragma: no cover
    """
    Network-based reporter for distributed backends.

    Sends dataset save requests via HTTP/gRPC to the coordinator.

    Args:
        endpoint: Network endpoint of the coordinator's dataset receiver.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    @property
    @override
    def is_direct(self) -> bool:
        return False

    @override
    def save(
        self,
        key: str,
        value: Any,
        store: DatasetStore,
        *,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send dataset save request via network."""
        raise NotImplementedError("RemoteDatasetReporter not yet implemented")
