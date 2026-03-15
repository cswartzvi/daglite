"""Backend manager — resolves backend names to `Backend` instances."""

from __future__ import annotations

import atexit
import logging
import threading
from contextvars import ContextVar
from contextvars import Token
from typing import TYPE_CHECKING, ClassVar

from daglite.backends.base import Backend

if TYPE_CHECKING:
    from daglite._context import SessionContext
    from daglite.settings import DagliteSettings

logger = logging.getLogger(__name__)


class BackendManager:
    """
    Resolves backend names to `Backend` instances and manages their lifecycle.

    Backends are created lazily on the first `get()` call for a given name and
    reused for subsequent calls within the same manager.

    Two modes of operation:

    1. **Session-scoped** — created by `session()` and activated via `activate()`.
       Stopped and deactivated when the session exits.
    2. **Global singleton** — created on first use outside a session, lives until interpreter
       shutdown (cleaned up via `atexit`).
    """

    _global: ClassVar[BackendManager | None] = None
    _global_lock: ClassVar[threading.Lock] = threading.Lock()
    _active: ClassVar[ContextVar[BackendManager | None]] = ContextVar(
        "_active_backend_manager", default=None
    )

    def __init__(self, session: SessionContext | None, settings: DagliteSettings) -> None:
        from daglite.backends.local import InlineBackend
        from daglite.backends.local import ProcessBackend
        from daglite.backends.local import ThreadBackend

        self._session = session
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

    # region Singleton

    @classmethod
    def get_active(cls) -> BackendManager:
        """
        Return the session-scoped manager if active, else the global singleton.

        The session-scoped manager is set via `activate()` and cleared via
        `deactivate()`.  The global singleton is created on first use and
        cleaned up at interpreter shutdown.
        """
        mgr = cls._active.get()
        if mgr is not None:
            return mgr
        return cls._get_global()

    def activate(self) -> Token[BackendManager | None]:
        """Push this manager as the active one for the current context."""
        return self._active.set(self)

    def deactivate(self, token: Token[BackendManager | None]) -> None:
        """Stop backends and restore the previous manager."""
        self.stop()
        self._active.reset(token)

    @classmethod
    def _get_global(cls) -> BackendManager:
        """Lazily create the global singleton with `atexit` cleanup."""
        if cls._global is None:
            with cls._global_lock:
                if cls._global is None:  # pragma: no branch - race guard
                    from daglite.settings import get_global_settings

                    cls._global = BackendManager(session=None, settings=get_global_settings())
                    atexit.register(cls._shutdown_global)
        return cls._global

    @classmethod
    def _shutdown_global(cls) -> None:  # pragma: no cover - atexit handler
        """Called at interpreter exit to stop global backends."""
        if cls._global is not None:
            try:
                cls._global.stop()
            except Exception:
                logger.exception("Failed to stop global backend manager at shutdown")
            cls._global = None

    # region Public API

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
            backend_name = self._session.backend if self._session else self._settings.backend

        # At this point a backend name must be resolved from session or settings.
        assert backend_name is not None

        if backend_name not in self._backends:
            try:
                backend_class = self._backend_types[backend_name]
            except KeyError:
                raise BackendError(
                    f"Unknown backend '{backend_name}'; "
                    f"available: {sorted(set(self._backend_types.keys()))}"
                ) from None

            backend_instance = backend_class()
            backend_instance.start(self._session)
            self._backends[backend_name] = backend_instance

        return self._backends[backend_name]

    def register(self, name: str, backend_class: type[Backend]) -> None:
        """
        Register a custom backend class under *name*.

        This allows third-party backends (Dask, Ray, etc.) to be used via
        `map_tasks(task, items, backend="my_custom_backend")`.

        Args:
            name: Backend name (used in `map_tasks(backend=...)`.
            backend_class: A `Backend` subclass.
        """
        self._backend_types[name] = backend_class

    def stop(self) -> None:
        """Stop all started backends and clear the cache."""
        for backend in self._backends.values():
            backend.stop()
        self._backends.clear()
