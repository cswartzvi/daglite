from collections.abc import Iterator
from contextlib import contextmanager

from pluggy import PluginManager

from daglite.backends.base import Backend
from daglite.plugins.events import EventProcessor


class BackendManager:
    """Manages global backend instance."""

    def __init__(self, plugin_manager: PluginManager, event_processor: EventProcessor) -> None:
        from daglite.backends.local import ProcessBackend
        from daglite.backends.local import SequentialBackend
        from daglite.backends.local import ThreadBackend

        self._started = False
        self._plugin_manager = plugin_manager
        self._event_processor = event_processor

        self._cached_backends: dict[str, Backend] = {}
        self._backend_types: dict[str, type[Backend]] = {
            "sequential": SequentialBackend,
            "synchronous": SequentialBackend,  # alias
            "threading": ThreadBackend,
            "threads": ThreadBackend,  # alias
            "multiprocessing": ProcessBackend,
            "processes": ProcessBackend,  # alias
        }

        # TODO : dynamic discovery of backends from entry points

    def get(self, backend: str = "") -> Backend:
        """
        Get or create backend instance by name.

        Args:
            backend: Name of the backend to get. If not provided, uses the default settings backend.

        Returns:
            An instance of the requested backend class (or the default).

        Raises:
            BackendError: If the requested backend is unknown.
        """
        from daglite.exceptions import BackendError

        if not self._started:
            raise RuntimeError("BackendManager has not been started yet.")

        if not backend:
            from daglite.settings import get_global_settings

            settings = get_global_settings()
            backend = settings.default_backend

        if backend not in self._cached_backends:
            try:
                backend_class = self._backend_types[backend]
            except KeyError:
                raise BackendError(
                    f"Unknown backend '{backend}'; available: {list(self._backend_types.keys())}"
                ) from None

            backend_instance = backend_class()
            backend_instance.start(self._plugin_manager, self._event_processor)
            self._cached_backends[backend] = backend_instance

        return self._cached_backends[backend]

    @contextmanager
    def start(self) -> Iterator[None]:
        """
        Context manager to start and stop the backend manager.

        Yields:
            None
        """
        if self._started:
            raise RuntimeError("BackendManager is already started.")

        self._started = True
        try:
            yield
        finally:
            for backend in self._cached_backends.values():
                backend.end()
            self._cached_backends.clear()
            self._started = False

    def _clear_backends(self) -> None:
        """Clear all cached backends without stopping them (for testing purposes)."""
        self._cached_backends.clear()
