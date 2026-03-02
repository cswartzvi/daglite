"""
Task mapping utilities for eager tasks.

`task_map` fans out a sync task across iterables using the active session's backend (or an explicit
override). `async_task_map` does the same for async tasks, using `asyncio.gather` for concurrency.

Both functions emit per-item `TaskStarted` / `TaskCompleted` / `TaskFailed` events automatically —
the decorated task handles its own event lifecycle, so no extra wiring is needed here.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterable
from typing import Any, TypeVar, overload

from daglite.session import RunContext
from daglite.session import get_run_context

logger = logging.getLogger(__name__)

R = TypeVar("R")


# region Public API


def _check_task(task: Any, func_name: str) -> None:
    """
    Validate that *task* is a daglite ``@task``-decorated callable.

    Raises ``TypeError`` with a helpful message when a plain function is passed instead.
    """
    from daglite.eager import _BaseEagerTask

    if not isinstance(task, _BaseEagerTask):
        raise TypeError(
            f"`{func_name}` expects a `@task`-decorated callable, "
            f"got {type(task).__name__!r}. Wrap your function with `@task` first."
        )


def task_map(
    task: Callable[..., R], *iterables: Iterable[Any], backend: str | None = None
) -> list[R]:
    """
    Map a sync task across iterables using the active backend.

    Each item is executed as a separate task call with full event emission
    and hook dispatch. The backend determines the concurrency strategy:

    * `"inline"` — sequential loop (default outside a session).
    * `"thread"` — `ThreadPoolExecutor` with context propagation.
    * `"process"` — `ProcessPoolExecutor` with cross-process event wiring.

    Args:
        task: A sync eager task (decorated with ``@task``).
        *iterables: One or more iterables whose elements are zipped and unpacked as positional
            arguments to *task*.
        backend: Backend name override. `None` inherits from the active session, falling back to
            `"inline"`.

    Returns:
        Ordered list of results, one per zipped item tuple.

    Raises:
        TypeError: If *task* is not a ``@task``-decorated callable.
    """
    _check_task(task, "task_map")

    ctx = get_run_context()
    items = list(zip(*iterables))

    if not items:
        return []

    # With a session — delegate to the backend.
    if ctx is not None and ctx.backend_manager is not None:
        backend_name = _resolve_backend(backend, ctx)
        return ctx.backend_manager.get(backend_name).map(task, items)

    # No session — inline fallback.
    return [task(*args) for args in items]


@overload
async def async_task_map(
    task: Callable[..., Coroutine[Any, Any, R]],
    *iterables: Iterable[Any],
    backend: str | None = None,
) -> list[R]: ...


@overload
async def async_task_map(
    task: Callable[..., R], *iterables: Iterable[Any], backend: str | None = None
) -> list[R]: ...


async def async_task_map(
    task: Callable[..., Any], *iterables: Iterable[Any], backend: str | None = None
) -> list[Any]:
    """
    Async map of a task across iterables.

    For async tasks the returned coroutines are gathered concurrently via `asyncio.gather`. For sync
    tasks each call is dispatched to the default executor via `run_in_executor` so the event loop is
    not blocked.

    When the resolved backend is `"inline"` items are processed sequentially (one `await` at a time)
    to match inline semantics.

    Args:
        task: An async or sync eager task (decorated with ``@task``).
        *iterables: One or more iterables whose elements are zipped and unpacked as positional
            arguments to *task*.
        backend: Backend name override. `None` inherits from the active session, falling back to
            `"inline"`.

    Returns:
        Ordered list of results, one per zipped item tuple.

    Raises:
        TypeError: If *task* is not a ``@task``-decorated callable.
    """
    _check_task(task, "async_task_map")
    ctx = get_run_context()
    backend_name = _resolve_backend(backend, ctx)
    items = list(zip(*iterables))

    if not items:
        return []

    is_async = getattr(task, "is_async", inspect.iscoroutinefunction(getattr(task, "func", task)))

    # Inline backend: sequential execution to match inline semantics.
    if backend_name in _INLINE_NAMES:
        if is_async:
            return [await task(*args) for args in items]
        return [task(*args) for args in items]

    # Concurrent path — async tasks use gather directly.
    if is_async:
        coros = [task(*args) for args in items]
        return list(await asyncio.gather(*coros))

    # Sync task in an async context — dispatch to the default thread executor
    # so the event loop stays responsive.
    loop = asyncio.get_running_loop()
    futs = [loop.run_in_executor(None, task, *args) for args in items]
    return list(await asyncio.gather(*futs))


# region Internals

_INLINE_NAMES = frozenset({"inline", "sequential", "synchronous"})


def _resolve_backend(backend: str | None, ctx: RunContext | None) -> str:
    """
    Determines the effective backend name from the explicit override or the
    active session context, falling back to `"inline"`.
    """
    if backend is not None:
        return backend
    if ctx is not None:
        return ctx.backend_name
    return "inline"
