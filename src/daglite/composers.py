"""Functions for composing tasks for various execution patterns."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any, TypeVar, overload

from daglite._context import BackendContext
from daglite._resolvers import resolve_dataset_reporter
from daglite._resolvers import resolve_dataset_store
from daglite._resolvers import resolve_hook
from daglite._resolvers import resolve_task_metadata
from daglite.datasets.store import DatasetStore

logger = logging.getLogger(__name__)

R = TypeVar("R")


# region Public API


@overload
def map_tasks(  # type: ignore[overload-overlap]
    task: Callable[..., AsyncIterator[R]], *iterables: Iterable[Any], backend: str | None = None
) -> list[list[R]]: ...


@overload
def map_tasks(  # type: ignore[overload-overlap]
    task: Callable[..., Iterator[R]], *iterables: Iterable[Any], backend: str | None = None
) -> list[list[R]]: ...


@overload
def map_tasks(
    task: Callable[..., R], *iterables: Iterable[Any], backend: str | None = None
) -> list[R]: ...


def map_tasks(
    task: Callable[..., Any], *iterables: Iterable[Any], backend: str | None = None
) -> list[Any]:
    """
    Map a task across iterables using the active backend.

    Each item is executed as a separate task call with full event emission and hook dispatch (if
    a session is active). Async tasks are transparently wrapped with `asyncio.run` so that they can
    be dispatched to backends that execute callables in worker threads or processes.

    Args:
        task: Task to be mapped. Must be a `@task`-decorated callable, either sync or async.
        *iterables: Iterables whose elements are zipped and unpacked as arguments to `task`.
        backend: Backend override. `None` inherits from the active session or global settings.

    Returns:
        Ordered list of results, one per zipped item tuple.

    Raises:
        TypeError: If *task* is not a `@task`-decorated callable.
    """
    from daglite.backends.manager import BackendManager

    callable_task = _make_backend_callable(task, "map_tasks")

    # Avoid over-saturation: nested maps default to inline when no explicit backend is given.
    if backend is None and resolve_task_metadata():
        backend = "inline"

    items = list(zip(*iterables))
    if not items:
        return []

    instance = BackendManager.get_active().get(backend)
    futures = [instance.submit(callable_task, args, map_index=i) for i, args in enumerate(items)]

    return [f.result() for f in futures]


@overload
async def gather_tasks(  # type: ignore[overload-overlap]
    task: Callable[..., AsyncIterator[R]],
    *iterables: Iterable[Any],
    backend: str | None = None,
) -> list[list[R]]: ...


@overload
async def gather_tasks(
    task: Callable[..., Coroutine[Any, Any, R]],
    *iterables: Iterable[Any],
    backend: str | None = None,
) -> list[R]: ...


@overload
async def gather_tasks(
    task: Callable[..., R], *iterables: Iterable[Any], backend: str | None = None
) -> list[R]: ...


async def gather_tasks(
    task: Callable[..., Any], *iterables: Iterable[Any], backend: str | None = None
) -> list[Any]:
    """
    Gather async tasks concurrently across iterables using `asyncio.gather`.

    Args:
        task: An async `@task`-decorated callable.
        *iterables: Iterables whose elements are zipped and unpacked as arguments to `task`.
        backend: Backend override. `None` inherits from the active session or global settings.

    Returns:
        Ordered list of results, one per zipped item tuple.

    Raises:
        TypeError: If *task* is not a `@task`-decorated callable or is not async.
    """
    task = _ensure_task(task, "gather_tasks")

    if not getattr(task, "is_async", False):
        raise TypeError(
            "`gather_tasks` requires an async `@task`. Use `map_tasks` for sync tasks, or make "
            "your task async."
        )

    items = list(zip(*iterables))

    if not items:
        return []

    futures = [_async_indexed_call(task, i, args) for i, args in enumerate(items)]
    return list(await asyncio.gather(*futures))


@overload
def load_dataset(
    key: str,
    return_type: type[R],
    *,
    format: str | None = None,
    store: DatasetStore | str | None = None,
    options: dict[str, Any] | None = None,
) -> R: ...


@overload
def load_dataset(
    key: str,
    *,
    format: str | None = None,
    store: DatasetStore | str | None = None,
    options: dict[str, Any] | None = None,
) -> Any: ...


def load_dataset(
    key: str,
    return_type: type[R] | None = None,
    *,
    format: str | None = None,
    store: DatasetStore | str | None = None,
    options: dict[str, Any] | None = None,
) -> R | Any:
    """
    Load a dataset from the active (or explicitly provided) dataset store.

    Args:
        key: Storage key/path.  May contain ``{param}`` placeholders that are resolved from the
            current task's bound arguments.
        return_type: Expected Python type for deserialization dispatch.
        format: Serialization format hint (e.g. ``"pickle"``).
        store: Explicit ``DatasetStore`` or string path.  When ``None``, the
            store is resolved from the context chain.
        options: Additional options forwarded to the Dataset constructor.

    Returns:
        The deserialized Python object.
    """
    resolved_store = resolve_dataset_store(store)

    hook = resolve_hook()
    metadata = resolve_task_metadata()

    hook_kw = {
        "key": key,
        "return_type": return_type,
        "format": format,
        "options": options,
        "metadata": metadata,
    }

    hook.before_dataset_load(**hook_kw)

    t0 = time.perf_counter()
    result = resolved_store.load(key, return_type=return_type, format=format, options=options)
    duration = time.perf_counter() - t0

    hook.after_dataset_load(**hook_kw, result=result, duration=duration)

    return result


def save_dataset(
    key: str,
    value: Any,
    *,
    format: str | None = None,
    store: DatasetStore | str | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """
    Save a value to the active (or explicitly provided) dataset store.

    When a ``DatasetReporter`` is available (inside a session), the save is routed through
    the reporter so that process/remote backends push the write back to the coordinator.

    The store is resolved from the context chain: explicit argument -> task context
    -> session context -> global settings.

    Args:
        key: Storage key/path.  May contain ``{param}`` placeholders resolved from the current
            task's bound arguments.
        value: The Python object to serialize and persist.
        format: Serialization format hint (e.g. ``"pickle"``).
        store: Explicit ``DatasetStore`` or string path.  When ``None``, the
            store is resolved from the context chain.
        options: Additional options forwarded to the Dataset constructor.

    Returns:
        The actual path where data was stored.
    """
    resolved_store = resolve_dataset_store(store)

    metadata = resolve_task_metadata()
    reporter = resolve_dataset_reporter()

    # Route through the reporter when available (handles coordination + hooks).
    if reporter is not None:
        reporter.save(key, value, resolved_store, format=format, options=options, metadata=metadata)
        return key

    # No reporter (bare call outside session) — save directly with hook dispatch.
    hook = resolve_hook()
    hook_kw: dict[str, Any] = {
        "key": key,
        "value": value,
        "format": format,
        "options": options,
        "metadata": metadata,
    }

    hook.before_dataset_save(**hook_kw)

    path = resolved_store.save(key, value, format=format, options=options)

    hook.after_dataset_save(**hook_kw)

    return path


# region Internals


def _ensure_task(func: Callable[..., Any], func_name: str) -> Any:
    """
    Ensure *func* is a daglite `@task`-decorated callable.

    If a plain callable is passed it is automatically wrapped with the default `@task` settings.
    Non-callable objects raise `TypeError`.
    """
    from daglite.tasks import _BaseTask

    if isinstance(func, _BaseTask):
        return func

    if callable(func):
        from daglite.tasks import task

        return task(func)

    raise TypeError(
        f"`{func_name}` expects a callable or `@task`-decorated function, "
        f"got {type(func).__name__!r}."
    )


class _AsyncWrapper:
    """Picklable wrapper that runs an async task synchronously via ``asyncio.run``."""

    def __init__(self, task: Any) -> None:
        self._task = task

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return asyncio.run(self._task(*args, **kwargs))


class _GeneratorWrapper:
    """Picklable wrapper that collects a generator task's yields into a list."""

    def __init__(self, task: Any) -> None:
        self._task = task

    def __call__(self, *args: Any, **kwargs: Any) -> list[Any]:
        return list(self._task(*args, **kwargs))


class _AsyncGeneratorWrapper:
    """Picklable wrapper that collects async generator yields into a list via ``asyncio.run``."""

    def __init__(self, task: Any) -> None:
        self._task = task

    def __call__(self, *args: Any, **kwargs: Any) -> list[Any]:
        async def _collect() -> list[Any]:
            return [item async for item in self._task(*args, **kwargs)]

        return asyncio.run(_collect())


def _make_backend_callable(func: Callable[..., Any], func_name: str) -> Callable[..., Any]:
    """
    Validate *func* as a task and return a sync callable suitable for backend pool submission.

    Plain callables are auto-wrapped with ``@task``.  Async tasks are wrapped with
    ``asyncio.run`` so they can run in worker threads/processes.  Generator tasks are
    wrapped to collect yields into a list.  Wrappers are picklable for process backends.
    """
    task = _ensure_task(func, func_name)

    is_async = getattr(task, "is_async", False)
    is_gen = getattr(task, "is_generator", False)

    if is_async and is_gen:
        return _AsyncGeneratorWrapper(task)
    if is_async:
        return _AsyncWrapper(task)
    if is_gen:
        return _GeneratorWrapper(task)
    return task


async def _async_indexed_call(task: Callable[..., Any], index: int, args: tuple[Any, ...]) -> Any:
    """Await *task* with a `BackendContext` carrying the `map_index`."""
    ctx = BackendContext.from_session(map_index=index)
    with ctx:
        if getattr(task, "is_generator", False):
            return [item async for item in task(*args)]
        return await task(*args)
