"""Abstract base class for task execution backends."""

from __future__ import annotations

import abc
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, ClassVar

from typing_extensions import final

from daglite._context import BackendContext
from daglite._context import SessionContext
from daglite.settings import get_global_settings


class Backend(abc.ABC):
    """
    Abstract base class for task execution backends.

    A backend defines *how* task calls are executed — sequentially, across threads, across
    processes, or on a remote cluster. Custom backends can be registered with the `BackendManager`
    to extend daglite.

    Subclasses **must** implement `_submit`.

    Lifecycle:

    1. `start(session)` — called once when the backend is first requested inside a session.
    2. `submit(func, args, kwargs)` — called to dispatch work.
    3. `stop()` — called when the session exits.
    """

    name: ClassVar[str]

    _started: bool = False

    # region Lifecycle

    @final
    def start(self, session: SessionContext | None) -> None:
        """
        Initialise backend resources (thread pools, process pools, queues, etc.).

        Args:
            session: Active session context carrying event/plugin infrastructure and settings.
        """
        if self._started:  # pragma: no cover
            raise RuntimeError("Backend is already started.")

        self._session = session
        self._settings = session.settings if session else get_global_settings()
        self._start()
        self._started = True

    def _start(self) -> None:
        """Subclass hook for creating per-backend resources."""

    @final
    def stop(self) -> None:
        """Releases backend resources."""
        if not self._started:  # pragma: no cover
            return

        self._stop()
        self._started = False

    def _stop(self) -> None:
        """Subclass hook for cleaning up per-backend resources."""

    # region Execution

    @final
    def submit(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        *,
        map_index: int | None = None,
    ) -> Future[Any]:
        """
        Submit a single task call and return a `Future` for its result.

        This is the fundamental execution primitive. Backends that wrap thread pools,
        process pools, or remote clusters implement this to dispatch one unit of work.

        Args:
            func: A callable to be executed on the backend.
            args: Positional arguments for the callable.
            kwargs: Optional keyword arguments for the callable.
            map_index: Optional index of the item in a fan-out map operation.

        Returns:
            A `concurrent.futures.Future` whose `.result()` yields the task return value.
        """
        context = BackendContext.from_session(self.name, map_index=map_index)
        return self._submit(func, args, kwargs or {}, context=context)

    @abc.abstractmethod
    def _submit(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        context: BackendContext,
    ) -> Future[Any]:
        """
        Subclass hook for dispatching a single task call.

        Args:
            func: A callable to be executed on the backend.
            args: Positional arguments for the callable.
            kwargs: Keyword arguments for the callable.
            context: Context containing session-level and task-level information.

        Returns:
            A `concurrent.futures.Future` whose `.result()` yields the task return value.
        """
        ...
