"""
Managed execution contexts for eager tasks.

The `session` context manager sets up backend, cache, plugin, and event
infrastructure so that eager tasks automatically participate in caching,
hook dispatch, and event reporting. Workflows use `session`
internally; notebooks and scripts can use it directly.

Three tiers of usage::

    # 1. Bare call — no context, inline defaults, no reporters
    result = add(x=1, y=2)

    # 2. session — managed context for notebooks / scripts
    with session(backend="thread", cache=True):
        a = add(x=1, y=2)


    # 3. @workflow — named, CLI-discoverable entry point
    @workflow
    def my_pipeline(x: int):
        return add(x=x, y=1)


    my_pipeline.run(x=5)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from typing import Any

from daglite._context import RunContext
from daglite._context import _map_iteration_index
from daglite._context import _parent_task_id
from daglite._context import _run_context
from daglite._context import _task_call_args
from daglite._context import get_event_reporter
from daglite._context import get_map_iteration_index
from daglite._context import get_parent_task_id
from daglite._context import get_plugin_manager
from daglite._context import get_run_context
from daglite._context import get_task_call_args
from daglite._context import reset_run_context
from daglite._context import set_map_iteration_index
from daglite._context import set_parent_task_id
from daglite._context import set_run_context
from daglite._context import set_task_call_args

logger = logging.getLogger(__name__)

# Re-exports for backward compatibility.
__all__ = [
    "RunContext",
    "_map_iteration_index",
    "_parent_task_id",
    "_run_context",
    "_task_call_args",
    "get_map_iteration_index",
    "get_parent_task_id",
    "get_run_context",
    "set_map_iteration_index",
    "set_parent_task_id",
    "set_run_context",
    "reset_run_context",
    "get_task_call_args",
    "set_task_call_args",
    "get_event_reporter",
    "get_plugin_manager",
    "session",
    "async_session",
]


# region session


@contextmanager
def session(
    *,
    backend: str | None = None,
    cache: bool | str | Any | None = None,
    plugins: list[Any] | None = None,
    settings: Any | None = None,
) -> Iterator[RunContext]:
    """
    Synchronous context manager that sets up an eager execution context.

    Within the block every `@task` call automatically uses the configured backend, cache, and
    plugins. On exit all processors are stopped and resources are cleaned up.

    Args:
        backend: Backend name (e.g. `"inline"`, `"thread"`, `"process"`). Defaults to
            `settings.default_backend`.
        cache: Cache configuration. `True` uses the settings default, a string is treated as a
            path, a `CacheStore` instance is used directly, `False` / `None` disables caching.
        plugins: Extra plugin instances for this session only.
        settings: A `DagliteSettings` override. Falls back to the global settings singleton.

    Yields:
        The active `RunContext` for the duration of the block.
    """
    ctx = _build_context(
        backend=backend,
        cache=cache,
        plugins=plugins,
        settings=settings,
    )

    token = _run_context.set(ctx)
    try:
        _start_processors(ctx)
        yield ctx
    finally:
        _stop_processors(ctx)
        _run_context.reset(token)


@asynccontextmanager
async def async_session(
    *,
    backend: str | None = None,
    cache: bool | str | Any | None = None,
    plugins: list[Any] | None = None,
    settings: Any | None = None,
) -> AsyncIterator[RunContext]:
    """
    Async context manager that sets up an eager execution context.

    Identical to `session` but usable in async code via ``async with``.

    Args:
        backend: Backend name. Defaults to `settings.default_backend`.
        cache: Cache configuration (see `session`).
        plugins: Extra plugin instances for this session only.
        settings: A `DagliteSettings` override.

    Yields:
        The active `RunContext` for the duration of the block.
    """
    ctx = _build_context(
        backend=backend,
        cache=cache,
        plugins=plugins,
        settings=settings,
    )

    token = _run_context.set(ctx)
    try:
        _start_processors(ctx)
        yield ctx
    finally:
        _stop_processors(ctx)
        _run_context.reset(token)


# region Internals


def _build_context(
    *,
    backend: str | None,
    cache: bool | str | Any | None,
    plugins: list[Any] | None,
    settings: Any | None,
) -> RunContext:
    """
    Assembles a `RunContext` from the provided arguments and global defaults.

    Mirrors the setup sequence in the existing engine but without the graph
    and dataset machinery.
    """
    from daglite.backends.manager import BackendManager
    from daglite.plugins.reporters import DirectEventReporter
    from daglite.settings import get_global_settings

    resolved_settings = settings if settings is not None else get_global_settings()
    backend_name = backend if backend is not None else resolved_settings.default_backend

    # Cache
    cache_store = _resolve_cache(cache, resolved_settings)

    # Plugins + events
    plugin_manager, event_processor = _setup_plugins(plugins or [])

    # Event reporter (direct / in-process)
    event_reporter = DirectEventReporter(callback=event_processor.dispatch)

    ctx = RunContext(
        backend_name=backend_name,
        cache_store=cache_store,
        event_reporter=event_reporter,
        event_processor=event_processor,
        plugin_manager=plugin_manager,
        backend_manager=None,
        settings=resolved_settings,
    )

    # Backend manager — created after RunContext so backends can read ctx
    ctx.backend_manager = BackendManager(ctx, resolved_settings)

    return ctx


def _resolve_cache(cache: bool | str | Any | None, settings: Any) -> Any | None:
    """
    Resolves the cache argument into a `CacheStore` or `None`.

    Resolution: explicit `CacheStore` > `True` (use settings default) >
    string path > `False` / `None` (disabled).
    """
    from daglite.cache.store import CacheStore

    if cache is None or cache is False:
        return None

    if isinstance(cache, CacheStore):
        return cache

    if cache is True:
        # Delegate to settings — may still be None
        settings_cache = settings.cache_store
        if settings_cache is None:
            return None
        if isinstance(settings_cache, str):
            return CacheStore(settings_cache)
        return settings_cache

    if isinstance(cache, str):
        return CacheStore(cache)

    raise ValueError(
        f"Invalid cache argument {cache!r}. Expected bool, str path, CacheStore instance, or None."
    )


def _setup_plugins(plugins: list[Any]) -> tuple[Any, Any]:
    """
    Creates a plugin manager and event processor for this session.

    Follows the same pattern as `engine._setup_plugin_system`.
    """
    from daglite.plugins.events import EventRegistry
    from daglite.plugins.manager import build_plugin_manager
    from daglite.plugins.processor import EventProcessor

    registry = EventRegistry()
    plugin_manager = build_plugin_manager(plugins, registry)
    event_processor = EventProcessor(registry)
    return plugin_manager, event_processor


def _start_processors(ctx: RunContext) -> None:
    """Starts background processors attached to the context."""
    if ctx.event_processor is not None:
        ctx.event_processor.start()


def _stop_processors(ctx: RunContext) -> None:
    """Stops background processors and backends, flushing pending work."""
    # Stop backends first — they may have queues that feed the event processor.
    if ctx.backend_manager is not None:
        try:
            ctx.backend_manager.stop()
        except Exception:
            logger.exception("Failed to stop backend manager")

    if ctx.event_processor is not None:
        try:
            ctx.event_processor.stop()
        except Exception:
            logger.exception("Failed to stop event processor")
