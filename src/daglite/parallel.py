"""
Parallel mapping utilities for eager tasks.

`parallel_map` fans out a sync task across iterables using the active session's backend (or an
explicit override). `async_map` does the same for async tasks, using `asyncio.gather` for
concurrency.

Both functions emit per-item `TaskStarted` / `TaskCompleted` / `TaskFailed` events automatically —
the decorated task handles its own event lifecycle, so no extra wiring is needed here.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from daglite.session import RunContext
from daglite.session import get_run_context
from daglite.session import set_run_context

logger = logging.getLogger(__name__)

R = TypeVar("R")

_INLINE_NAMES = frozenset({"inline", "sequential", "synchronous"})
_THREAD_NAMES = frozenset({"thread", "threading", "threads"})
_PROCESS_NAMES = frozenset({"process", "multiprocessing", "processes"})


# region Public API


def parallel_map(
    task: Callable[..., R], *iterables: Iterable[Any], backend: str | None = None
) -> list[R]:
    """
    Fan-out a sync task across iterables using the active backend.

    Each item is executed as a separate task call with full event emission
    and hook dispatch. The backend determines the concurrency strategy:

    * `"inline"` — sequential loop (default outside a session).
    * `"thread"` — `ThreadPoolExecutor` with context propagation.
    * `"process"` — `ProcessPoolExecutor` with cross-process event wiring.

    Args:
        task: A sync eager task (or any callable).
        *iterables: One or more iterables whose elements are zipped and unpacked as positional
            arguments to *task*.
        backend: Backend name override. `None` inherits from the active session, falling back to
            `"inline"`.

    Returns:
        Ordered list of results, one per zipped item tuple.

    Raises:
        ValueError: If the resolved backend name is not recognized.
    """
    ctx = get_run_context()
    backend_name = _resolve_backend(backend, ctx)
    items = list(zip(*iterables))

    if not items:
        return []

    if backend_name in _INLINE_NAMES:
        return [task(*args) for args in items]

    if backend_name in _THREAD_NAMES:
        return _thread_map(task, items, ctx)

    if backend_name in _PROCESS_NAMES:
        return _process_map(task, items, ctx)

    raise ValueError(
        f"Unknown backend {backend_name!r}. "
        f"Expected one of: inline, thread, process (and their aliases)."
    )


async def async_map(
    task: Callable[..., Any], *iterables: Iterable[Any], backend: str | None = None
) -> list[Any]:
    """
    Async fan-out of a task across iterables using `asyncio.gather`.

    For async tasks the returned coroutines are gathered concurrently. For sync tasks each call is
    dispatched to the default executor via `run_in_executor` so the event loop is not blocked.

    When the resolved backend is `"inline"` items are processed sequentially (one `await` at a time)
    to match inline semantics.

    Args:
        task: An async or sync eager task (or any callable).
        *iterables: One or more iterables whose elements are zipped and unpacked as positional
            arguments to *task*.
        backend: Backend name override. `None` inherits from the active session, falling back to
            `"inline"`.

    Returns:
        Ordered list of results, one per zipped item tuple.
    """
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


def _thread_map(
    task: Callable[..., Any],
    items: list[tuple[Any, ...]],
    ctx: RunContext | None,
) -> list[Any]:
    """
    Executes task calls across *items* in a `ThreadPoolExecutor`.

    Each submission is wrapped with `contextvars.copy_context().run()` so
    that the active `RunContext` is visible to every worker thread.
    """
    from daglite.settings import get_global_settings

    settings = ctx.settings if ctx and ctx.settings else get_global_settings()
    max_workers = min(len(items), settings.max_backend_threads)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # ThreadPoolExecutor.submit() does NOT propagate contextvars; only
        # asyncio's run_in_executor does.  Wrap each call explicitly so the
        # active RunContext is visible inside every worker thread.
        futures = [executor.submit(contextvars.copy_context().run, task, *args) for args in items]
        return [f.result() for f in futures]


def _process_map(
    task: Callable[..., Any],
    items: list[tuple[Any, ...]],
    ctx: RunContext | None,
) -> list[Any]:
    """
    Executes task calls across *items* in a `ProcessPoolExecutor`.

    Each worker process receives a serialised plugin manager and a
    `ProcessEventReporter` backed by a shared multiprocessing queue so
    that events flow back to the main process's event processor.
    """
    from daglite.backends.impl.local import ProcessBackend
    from daglite.plugins.manager import serialize_plugin_manager
    from daglite.settings import get_global_settings

    settings = ctx.settings if ctx and ctx.settings else get_global_settings()
    max_workers = min(len(items), settings.max_parallel_processes)
    backend_name = ctx.backend_name if ctx else "process"

    # Serialize the plugin manager for cross-process transfer.
    serialized_pm = None
    if ctx and ctx.plugin_manager:
        serialized_pm = serialize_plugin_manager(ctx.plugin_manager)

    # Validate that the cache store can be pickled into workers.
    cache_store = ctx.cache_store if ctx else None
    if cache_store is not None:
        import pickle

        try:
            pickle.dumps(cache_store)
        except Exception as exc:
            raise TypeError(
                "cache_store must be picklable for the process backend. "
                "Use the thread or inline backend instead."
            ) from exc

    mp_ctx = ProcessBackend._determine_mp_context()
    event_queue = mp_ctx.Queue()

    source_id = None
    if ctx and ctx.event_processor:
        source_id = ctx.event_processor.add_source(event_queue)

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_ctx,
            initializer=_map_process_init,
            initargs=(serialized_pm, event_queue, cache_store, backend_name),
        ) as executor:
            futures = [executor.submit(task, *args) for args in items]
            return [f.result() for f in futures]
    finally:
        if ctx and ctx.event_processor and source_id is not None:
            ctx.event_processor.flush()
            ctx.event_processor.remove_source(source_id)
        event_queue.close()


def _map_process_init(
    serialized_pm: dict[str, Any] | None,
    event_queue: Any,
    cache_store: Any,
    backend_name: str,
) -> None:
    """
    Initializer for `ProcessPoolExecutor` workers used by `parallel_map`.

    Deserialises the plugin manager, creates a `ProcessEventReporter`, and
    pushes a `RunContext` into the worker's context variable so that eager
    tasks emit events back to the main process.
    """
    from daglite.plugins.manager import deserialize_plugin_manager
    from daglite.plugins.reporters import ProcessEventReporter

    pm = deserialize_plugin_manager(serialized_pm) if serialized_pm else None
    reporter = ProcessEventReporter(event_queue)

    ctx = RunContext(
        backend_name=backend_name,
        cache_store=cache_store,
        event_reporter=reporter,
        plugin_manager=pm,
    )
    set_run_context(ctx)
