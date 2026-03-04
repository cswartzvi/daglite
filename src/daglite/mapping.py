"""
Task mapping utilities for eager tasks.

`task_map` fans out a sync task across iterables using the active session's backend (or an explicit
override). `async_task_map` does the same for async tasks, delegating to the backend's `async_map`
method for concurrency.

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

from daglite._context import RunContext
from daglite._context import _map_iteration_index
from daglite._context import get_run_context
from daglite._context import set_map_iteration_index

logger = logging.getLogger(__name__)

R = TypeVar("R")


# region Public API


def _check_task(task: Any, func_name: str) -> None:
    """
    Validate that *task* is a daglite ``@task``-decorated callable.

    Raises ``TypeError`` with a helpful message when a plain function is passed instead.
    """
    from daglite.tasks import _BaseTask

    if not isinstance(task, _BaseTask):
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

    # With a session — submit each item individually so the map iteration
    # index context var is captured per-item (ThreadBackend copies context
    # at submit time; InlineBackend runs synchronously).
    if ctx is not None and ctx.backend_manager is not None:
        backend_name = _resolve_backend(backend, ctx)
        be = ctx.backend_manager.get(backend_name)
        futures = []
        for i, args in enumerate(items):
            token = set_map_iteration_index(i)
            try:
                futures.append(be.submit(task, args))
            finally:
                _map_iteration_index.reset(token)
        return [f.result() for f in futures]

    # No session — inline fallback.
    results: list[R] = []
    for i, args in enumerate(items):
        token = set_map_iteration_index(i)
        try:
            results.append(task(*args))
        finally:
            _map_iteration_index.reset(token)
    return results


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

    When an active session provides a `BackendManager`, the resolved backend's `async_map` method
    handles dispatch — async tasks are gathered concurrently and sync tasks are offloaded to the
    executor so the event loop stays responsive. Without a session, items are processed sequentially
    as a simple inline fallback.

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
    items = list(zip(*iterables))

    if not items:
        return []

    is_async = getattr(task, "is_async", inspect.iscoroutinefunction(getattr(task, "func", task)))

    # With a session — use the backend for sync tasks, asyncio.gather for async tasks.
    if ctx is not None and ctx.backend_manager is not None:
        backend_name = _resolve_backend(backend, ctx)
        be = ctx.backend_manager.get(backend_name)

        if is_async:
            # Async tasks run concurrently on the event loop with per-item index.
            # Each asyncio.Task gets its own context copy so index isolation is automatic.
            return list(
                await asyncio.gather(
                    *[_async_indexed_call(task, i, args) for i, args in enumerate(items)]
                )
            )

        # Sync tasks — submit to backend with per-item index.
        futures = []
        for i, args in enumerate(items):
            token = set_map_iteration_index(i)
            try:
                futures.append(be.submit(task, args))
            finally:
                _map_iteration_index.reset(token)
        return list(await asyncio.gather(*[asyncio.wrap_future(f) for f in futures]))

    # No session — concurrent fallback via gather.
    if is_async:
        return list(
            await asyncio.gather(
                *[_async_indexed_call(task, i, args) for i, args in enumerate(items)]
            )
        )

    results: list[Any] = []
    for i, args in enumerate(items):
        token = set_map_iteration_index(i)
        try:
            results.append(task(*args))
        finally:
            _map_iteration_index.reset(token)
    return results


# region Internals


async def _async_indexed_call(task: Callable[..., Any], index: int, args: tuple[Any, ...]) -> Any:
    """
    Await *task* with the map iteration index set for this item.

    Each ``asyncio.Task`` created by ``asyncio.gather`` receives its own
    context copy, so the set/reset here is isolated per-coroutine.
    """
    token = set_map_iteration_index(index)
    try:
        return await task(*args)
    finally:
        _map_iteration_index.reset(token)


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
