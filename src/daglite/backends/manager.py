"""Backend manager — resolves backend names to `Backend` instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from daglite.backends.base import Backend

if TYPE_CHECKING:
    from daglite.session import RunContext
    from daglite.settings import DagliteSettings


class BackendManager:
    """
    Resolves backend names to `Backend` instances and manages their lifecycle.

    Backends are created lazily on the first ``get()`` call for a given name and
    reused for subsequent calls within the same session.

    Args:
        ctx: The active `RunContext` — passed to each backend on first use.
        settings: Global `DagliteSettings` snapshot.
    """

    def __init__(self, ctx: RunContext, settings: DagliteSettings) -> None:
        from daglite.backends.impl.local import InlineBackend
        from daglite.backends.impl.local import ProcessBackend
        from daglite.backends.impl.local import ThreadBackend

        self._ctx = ctx
        self._settings = settings
        self._backends: dict[str, Backend] = {}
        self._backend_types: dict[str, type[Backend]] = {
            "inline": InlineBackend,
            "sequential": InlineBackend,
            "synchronous": InlineBackend,
            "threading": ThreadBackend,
            "thread": ThreadBackend,
            "threads": ThreadBackend,
            "multiprocessing": ProcessBackend,
            "processes": ProcessBackend,
            "process": ProcessBackend,
        }

    def get(self, backend_name: str | None = None) -> Backend:
        """
        Get or create a backend instance by name.

        Args:
            backend_name: Name of the backend. Falls back to the settings default
                when *None*.

        Returns:
            A started `Backend` instance.

        Raises:
            BackendError: If the name is not recognized.
        """
        from daglite.exceptions import BackendError

        if not backend_name:
            backend_name = self._settings.default_backend

        if backend_name not in self._backends:
            try:
                backend_class = self._backend_types[backend_name]
            except KeyError:
                raise BackendError(
                    f"Unknown backend '{backend_name}'; "
                    f"available: {sorted(set(self._backend_types.keys()))}"
                ) from None

            backend_instance = backend_class()
            backend_instance.start(self._ctx, self._settings)
            self._backends[backend_name] = backend_instance

        return self._backends[backend_name]

    def register(self, name: str, backend_class: type[Backend]) -> None:
        """
        Register a custom backend class under *name*.

        This allows third-party backends (Dask, Ray, etc.) to be used via
        ``task_map(task, items, backend="my_custom_backend")``.

        Args:
            name: Backend name (used in ``task_map(backend=...)``.
            backend_class: A ``Backend`` subclass.
        """
        self._backend_types[name] = backend_class

    def stop(self) -> None:
        """Stop all started backends and clear the cache."""
        for backend in self._backends.values():
            backend.stop()
        self._backends.clear()
