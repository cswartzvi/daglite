"""Abstract base class for task execution backends."""

from __future__ import annotations

import abc
import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daglite._context import RunContext
    from daglite.settings import DagliteSettings


class Backend(abc.ABC):
    """
    Abstract base class for task execution backends.

    A backend defines *how* task calls are executed — sequentially, across threads, across
    processes, or on a remote cluster. Custom backends can be registered with the `BackendManager`
    to extend daglite.

    Subclasses **must** implement `submit`. An optional `map` override can replace the default
    fan-out-and-collect when the backend supports native batch operations.

    Lifecycle:

    1. `start(ctx, settings)` — called once when the backend is first requested inside a session.
    2. `submit(task, args)` / `map(task, items)` — called to dispatch work.
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
    def submit(
        self,
        task: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> Future[Any]:
        """
        Submit a single task call and return a `Future` for its result.

        This is the fundamental execution primitive. Backends that wrap thread pools,
        process pools, or remote clusters implement this to dispatch one unit of work.

        Args:
            task: A `@task`-decorated callable.
            args: Positional arguments unpacked into the task call: `task(*args)`.

        Returns:
            A `concurrent.futures.Future` whose `.result()` yields the task return value.
        """
        ...

    def map(
        self,
        task: Callable[..., Any],
        items: list[tuple[Any, ...]],
    ) -> list[Any]:
        """
        Execute *task* once per entry in *items* and return an ordered list of results.

        The default implementation calls `submit` for each item and collects results.
        Backends with native batch support (e.g. Dask, Ray) can override this for
        better performance.

        Args:
            task: A `@task`-decorated callable.
            items: Zipped argument tuples.

        Returns:
            Ordered list of results, one per item.
        """
        futures = [self.submit(task, args) for args in items]
        return [f.result() for f in futures]

    async def async_map(
        self,
        task: Callable[..., Any],
        items: list[tuple[Any, ...]],
    ) -> list[Any]:
        """
        Async counterpart of `map` — execute *task* for each item and return ordered results.

        The default implementation submits every item through `submit` and converts the resulting
        `concurrent.futures.Future` objects into asyncio futures via `asyncio.wrap_future` so the
        event loop is not blocked while workers execute. Async tasks are wrapped so each worker
        runs the coroutine in its own event loop via `asyncio.run`.

        Backends with native async support can override this.

        Args:
            task: A `@task`-decorated callable (sync or async).
            items: Zipped argument tuples.

        Returns:
            Ordered list of results, one per item.
        """
        is_async = getattr(
            task, "is_async", inspect.iscoroutinefunction(getattr(task, "func", task))
        )

        if is_async:
            futures = [self.submit(_run_async_task, (task, *args)) for args in items]
        else:
            futures = [self.submit(task, args) for args in items]

        async_futures = [asyncio.wrap_future(f) for f in futures]
        return list(await asyncio.gather(*async_futures))


def _run_async_task(task: Callable[..., Any], *args: Any) -> Any:
    """Run an async task in a new event loop."""
    return asyncio.run(task(*args))
