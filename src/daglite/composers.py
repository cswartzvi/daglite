"""Functions for composing tasks for various execution patterns."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterable
from functools import wraps
from typing import Any, TypeVar, overload

from daglite._context import BackendContext
from daglite._context import SessionContext
from daglite._context import TaskContext
from daglite._context import resolve_from_chain
from daglite.datasets.store import DatasetStore

logger = logging.getLogger(__name__)

R = TypeVar("R")


# region Public API


def map_tasks(
    task: Callable[..., R], *iterables: Iterable[Any], backend: str | None = None
) -> list[R]:
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

    task = _ensure_task(task, "map_tasks")

    # Avoid over-saturation: nested maps default to inline when no explicit backend is given.
    if backend is None and _inside_task():
        backend = "inline"

    callable_task: Callable[..., Any] = _resolve_callable(task)

    items = list(zip(*iterables))
    if not items:
        return []

    instance = BackendManager.get_active().get(backend)
    futures = [instance.submit(callable_task, args, map_index=i) for i, args in enumerate(items)]
    return [f.result() for f in futures]


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

    if not _is_async(task):
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
    resolved_store = _get_store(store)
    hook = _get_hook()
    metadata = _get_metadata()
    hook_kw = {
        "key": key,
        "return_type": return_type,
        "format": format,
        "options": options,
        "metadata": metadata,
    }

    if hook is not None:
        hook.before_dataset_load(**hook_kw)

    t0 = time.perf_counter()
    result = resolved_store.load(key, return_type=return_type, format=format, options=options)
    duration = time.perf_counter() - t0

    if hook is not None:
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
    resolved_store = _get_store(store)
    metadata = _get_metadata()
    reporter = _get_dataset_reporter()

    # Route through the reporter when available (handles coordination + hooks).
    if reporter is not None:
        reporter.save(key, value, resolved_store, format=format, options=options, metadata=metadata)
        return key

    # No reporter (bare call outside session) — save directly with hook dispatch.
    hook = _get_hook()
    hook_kw: dict[str, Any] = {
        "key": key,
        "value": value,
        "format": format,
        "options": options,
        "metadata": metadata,
    }

    if hook is not None:
        hook.before_dataset_save(**hook_kw)

    path = resolved_store.save(key, value, format=format, options=options)

    if hook is not None:
        hook.after_dataset_save(**hook_kw)

    return path


# region Internals


def _ensure_task(func: Callable[..., Any], func_name: str) -> Any:
    """
    Ensure *func* is a daglite `@task`-decorated callable.

    If a plain callable is passed it is automatically wrapped with the default ``@task`` settings.
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


def _is_async(task: Any) -> bool:
    """Return `True` when *task* is an async `@task`-decorated callable."""
    return getattr(task, "is_async", False)


def _is_generator(task: Any) -> bool:
    """Return `True` when *task* is a generator `@task`-decorated callable."""
    return getattr(task, "is_generator", False)


def _inside_task() -> bool:
    """Return `True` when called from inside an active task execution."""
    return TaskContext.get() is not None


def _resolve_callable(task: Callable[..., Any]) -> Callable[..., Any]:
    """
    Resolve a task to a plain callable suitable for backend submission.

    Generators are wrapped to collect yielded items into a list. Async tasks (including async
    generators) are wrapped to run via `asyncio.run` so they can be dispatched to backends that
    execute callables in worker threads or processes.
    """
    is_async = _is_async(task)
    is_gen = _is_generator(task)

    if is_async and is_gen:
        return _wrap_async_generator(task)
    if is_async:
        return _wrap_async(task)
    if is_gen:
        return _wrap_generator(task)
    return task


async def _async_indexed_call(task: Callable[..., Any], index: int, args: tuple[Any, ...]) -> Any:
    """Await *task* with a `BackendContext` carrying the `map_index`."""
    ctx = BackendContext.from_session(map_index=index)
    with ctx:
        if _is_generator(task):
            return [item async for item in task(*args)]
        return await task(*args)


def _wrap_generator(task: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync generator task to collect yielded items into a list."""

    @wraps(task)
    def _wrapper(*args: Any, **kwargs: Any) -> list[Any]:
        return list(task(*args, **kwargs))

    return _wrapper


def _wrap_async(task: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap an async `@task` so it can be called as a sync callable via `asyncio.run`.

    This is used by `map_tasks` so that async tasks can be dispatched to backends that execute
    callables in worker threads or processes.
    """

    @wraps(task)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(task(*args, **kwargs))

    return _wrapper


def _wrap_async_generator(task: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async generator task to collect yielded items into a list via `asyncio.run`."""

    @wraps(task)
    def _wrapper(*args: Any, **kwargs: Any) -> list[Any]:
        async def _collect() -> list[Any]:
            return [item async for item in task(*args, **kwargs)]

        return asyncio.run(_collect())

    return _wrapper


def _get_store(store: DatasetStore | str | None = None) -> DatasetStore:
    """Resolve *store* from the provided value, the active context chain, or raise."""
    if isinstance(store, DatasetStore):
        return store
    if isinstance(store, str):
        return DatasetStore(store)

    ds: DatasetStore | None = resolve_from_chain(
        "dataset_store",
        TaskContext.get(),
        SessionContext.get(),
        default=None,
    )
    if ds is not None:
        return ds

    from daglite.settings import get_global_settings

    settings = get_global_settings()
    ds_setting = getattr(settings, "dataset_store", None)
    if ds_setting is not None:
        return DatasetStore(ds_setting) if isinstance(ds_setting, str) else ds_setting

    raise RuntimeError(
        "No dataset store available. Pass a store explicitly, configure one on the "
        "session, or set the DAGLITE_DATASET_STORE environment variable."
    )


def _get_hook() -> Any:
    """Return the pluggy hook relay from the active session, or ``None``."""
    session_ctx = SessionContext.get()
    if session_ctx is not None:
        return session_ctx.plugin_manager.hook
    return None


def _get_metadata() -> Any:
    """Return the active ``TaskMetadata``, or ``None`` if outside a task."""
    task_ctx = TaskContext.get()
    return task_ctx.metadata if task_ctx is not None else None


def _get_dataset_reporter() -> Any:
    """Return the ``DatasetReporter`` from the active session, or ``None``."""
    session_ctx = SessionContext.get()
    return session_ctx.dataset_reporter if session_ctx is not None else None
