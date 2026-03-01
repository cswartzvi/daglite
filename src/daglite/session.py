"""
Managed execution contexts for eager tasks.

The `session` context manager sets up backend, cache, plugin, and event
infrastructure so that eager tasks automatically participate in caching,
hook dispatch, and event reporting. `Workflow.run()` uses `session`
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
import time
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import Any

logger = logging.getLogger(__name__)


# region RunContext


@dataclass
class RunContext:
    """
    Execution context carrying all infrastructure references for a session.

    Eager tasks read this from a `ContextVar` to access caching, event
    reporting, and plugin hooks. When no context is active the task runs
    inline with no overhead.

    Fields with a default of `None` are optional — a minimal context only
    needs `backend_name` and an `event_reporter` to be useful.
    """

    backend_name: str = "inline"
    """Name of the active backend (recorded in events)."""

    cache_store: Any | None = None
    """A `CacheStore` instance, or `None` if caching is disabled."""

    event_reporter: Any | None = None
    """An `EventReporter` for sending events to the event processor."""

    event_processor: Any | None = None
    """The `EventProcessor` background thread, or `None`."""

    plugin_manager: Any | None = None
    """A pluggy `PluginManager` instance, or `None`."""

    backend_manager: Any | None = None
    """A `BackendManager` instance, or `None`."""

    settings: Any | None = None
    """The `DagliteSettings` snapshot active for this session."""

    started_at: float = field(default_factory=time.time)
    """Unix timestamp when the context was created."""


_run_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def get_run_context() -> RunContext | None:
    """Returns the active run context, or `None` if outside a session."""
    return _run_context.get()


def set_run_context(ctx: RunContext) -> Any:
    """Pushes a run context. Returns a token for `reset_run_context`."""
    return _run_context.set(ctx)


def reset_run_context(token: Any) -> None:
    """Restores the previous run context using a token from `set_run_context`."""
    _run_context.reset(token)


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

    Within the block every `@eager_task` call automatically uses the configured backend, cache, and
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

    return RunContext(
        backend_name=backend_name,
        cache_store=cache_store,
        event_reporter=event_reporter,
        event_processor=event_processor,
        plugin_manager=plugin_manager,
        backend_manager=None,  # created lazily in Phase 3 (parallel_map)
        settings=resolved_settings,
    )


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
    from daglite.plugins.manager import build_plugin_manager
    from daglite.plugins.processor import EventProcessor
    from daglite.plugins.registry import EventRegistry

    registry = EventRegistry()
    plugin_manager = build_plugin_manager(plugins, registry)
    event_processor = EventProcessor(registry)
    return plugin_manager, event_processor


def _start_processors(ctx: RunContext) -> None:
    """Starts background processors attached to the context."""
    if ctx.event_processor is not None:
        ctx.event_processor.start()

    if ctx.backend_manager is not None:
        ctx.backend_manager.start()


def _stop_processors(ctx: RunContext) -> None:
    """Stops background processors, flushing pending work."""
    if ctx.event_processor is not None:
        try:
            ctx.event_processor.stop()
        except Exception:
            logger.exception("Failed to stop event processor")

    if ctx.backend_manager is not None:
        try:
            ctx.backend_manager.stop()
        except Exception:
            logger.exception("Failed to stop backend manager")
