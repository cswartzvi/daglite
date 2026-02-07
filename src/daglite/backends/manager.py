from pluggy import PluginManager

from daglite.backends.base import Backend
from daglite.plugins.processor import EventProcessor


class BackendManager:
    """Manages global backend instance."""

    def __init__(self, plugin_manager: PluginManager, event_processor: EventProcessor) -> None:
        from daglite.backends.impl.local import InlineBackend
        from daglite.backends.impl.local import ProcessBackend
        from daglite.backends.impl.local import ThreadBackend

        self._started = False
        self._plugin_manager = plugin_manager
        self._event_processor = event_processor

        self._cached_backends: dict[str, Backend] = {}
        self._backend_types: dict[str, type[Backend]] = {
            "inline": InlineBackend,
            "sequential": InlineBackend,  # alias
            "synchronous": InlineBackend,  # alias
            "threading": ThreadBackend,
            "thread": ThreadBackend,  # alias
            "threads": ThreadBackend,  # alias
            "multiprocessing": ProcessBackend,
            "processes": ProcessBackend,  # alias
            "process": ProcessBackend,  # alias
        }

        # TODO : dynamic discovery of backends from entry points

    def get(self, backend_name: str | None = None) -> Backend:
        """
        Get or create backend instance by name.

        Args:
            backend_name: Name of the backend to get. If not provided, uses the default settings
                backend.

        Returns:
            An instance of the requested backend class (or the default).

        Raises:
            BackendError: If the requested backend is unknown.
        """
        from daglite.exceptions import BackendError

        if not self._started:  # pragma: no cover
            raise RuntimeError("BackendManager has not been started yet.")

        if not backend_name:
            from daglite.settings import get_global_settings

            settings = get_global_settings()
            backend_name = settings.default_backend

        if backend_name not in self._cached_backends:
            try:
                backend_class = self._backend_types[backend_name]
            except KeyError:  # pragma: no cover
                raise BackendError(
                    f"Unknown backend '{backend_name}'; "
                    f"available: {list(self._backend_types.keys())}"
                ) from None

            backend_instance = backend_class()
            backend_instance.start(self._plugin_manager, self._event_processor)
            self._cached_backends[backend_name] = backend_instance

        return self._cached_backends[backend_name]

    def start(self) -> None:
        """Start all backends as needed."""
        if self._started:  # pragma: no cover
            raise RuntimeError("BackendManager is already started.")

        self._started = True

    def stop(self) -> None:
        """Stop all backends and clear cached instances."""
        for backend in self._cached_backends.values():
            backend.stop()
        self._cached_backends.clear()
        self._started = False
