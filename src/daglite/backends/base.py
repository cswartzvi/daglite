"""Abstract base class for task execution backends."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daglite.session import RunContext
    from daglite.settings import DagliteSettings


class Backend(abc.ABC):
    """
    Abstract base class for task execution backends.

    A backend defines *how* a batch of task calls is executed — sequentially, across threads, across
    processes, or on a remote cluster. Custom backends can be registered with the `BackendManager`
    to extend daglite.

    Lifecycle:

    1. `start(ctx, settings)` — called once when the backend is first requested inside a session.
    2. `map(task, items)` — called zero or more times to fan out work.
    3. `stop()` — called when the session exits.
    """

    _started: bool = False

    # region Lifecycle

    def start(self, ctx: RunContext, settings: DagliteSettings) -> None:
        """
        Initialise backend resources (thread pools, process pools, queues, etc.).

        Args:
            ctx: The active `RunContext` carrying event/plugin infrastructure.
            settings: Global `DagliteSettings` snapshot.
        """
        if self._started:  # pragma: no cover
            raise RuntimeError("Backend is already started.")

        self._ctx = ctx
        self._settings = settings
        self._start()
        self._started = True

    def _start(self) -> None:
        """Subclass hook for creating per-backend resources."""

    def stop(self) -> None:
        """Releases backend resources."""
        if not self._started:  # pragma: no cover
            return

        self._stop()
        self._started = False

    def _stop(self) -> None:
        """Subclass hook for cleaning up per-backend resources."""

    # region Execution

    @abc.abstractmethod
    def map(
        self,
        task: Callable[..., Any],
        items: list[tuple[Any, ...]],
    ) -> list[Any]:
        """
        Execute *task* once per entry in *items* and return an ordered list of results.

        Each element of *items* is a tuple of positional arguments unpacked into the
        task call: `task(*args)`.

        Args:
            task: A `@task`-decorated callable.
            items: Zipped argument tuples.

        Returns:
            Ordered list of results, one per item.
        """
        ...
