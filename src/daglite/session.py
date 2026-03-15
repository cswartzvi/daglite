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
import time
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from typing import Any

from pluggy import PluginManager

from daglite._context import SessionContext
from daglite.cache.store import CacheStore
from daglite.datasets.store import DatasetStore
from daglite.plugins.base import SerializablePlugin
from daglite.plugins.events import EventProcessor
from daglite.settings import DagliteSettings

logger = logging.getLogger(__name__)


# region session


@contextmanager
def session(
    *,
    backend: str | None = None,
    cache_store: str | CacheStore | None = None,
    dataset_store: str | DatasetStore | None = None,
    plugins: SerializablePlugin | list[SerializablePlugin] | None = None,
    settings: DagliteSettings | None = None,
) -> Iterator[SessionContext]:
    """
    Synchronous context manager that sets up an execution context.

    Args:
        backend: Name of the backend to default. If `None` uses `settings.default_backend`.
        cache_store: Cache store configuration. Can be a `CacheStore` or a string path.
        dataset_store: Dataset store configuration. Can be a `DatasetStore` or a string path.
        plugins: Extra plugin instances for this session only.
        settings: A `DagliteSettings` override. Falls back to the global settings singleton.

    Yields:
        The active `SessionContext` for the duration of the block.
    """
    plugins = plugins if isinstance(plugins, list) else [plugins] if plugins is not None else []
    with _build_context(
        backend=backend,
        cache_store=cache_store,
        dataset_store=dataset_store,
        plugins=plugins,
        settings=settings,
    ) as ctx:
        with _processors_context(ctx):
            yield ctx


@asynccontextmanager
async def async_session(
    *,
    backend: str | None = None,
    cache_store: str | CacheStore | None = None,
    dataset_store: str | DatasetStore | None = None,
    plugins: SerializablePlugin | list[SerializablePlugin] | None = None,
    settings: DagliteSettings | None = None,
) -> AsyncIterator[SessionContext]:
    """
    Async context manager that sets up an eager execution context.

    Args:
        backend: Name of the backend to default. If `None` uses `settings.default_backend`.
        cache_store: Cache store configuration. Can be a `CacheStore` or a string path.
        dataset_store: Dataset store configuration. Can be a `DatasetStore` or a string path.
        plugins: Extra plugin instances for this session only.
        settings: A `DagliteSettings` override. Falls back to the global settings singleton.

    Yields:
        The active `SessionContext` for the duration of the block.
    """
    plugins = plugins if isinstance(plugins, list) else [plugins] if plugins is not None else []
    with _build_context(
        backend=backend,
        cache_store=cache_store,
        dataset_store=dataset_store,
        plugins=plugins,
        settings=settings,
    ) as ctx:
        with _processors_context(ctx):
            yield ctx


# region Internals


def _build_context(
    *,
    backend: str | None,
    cache_store: str | CacheStore | None,
    dataset_store: str | DatasetStore | None,
    plugins: list[SerializablePlugin] | None,
    settings: DagliteSettings | None,
) -> SessionContext:
    """
    Assembles a `SessionContext` from the provided arguments and global defaults.

    Mirrors the setup sequence in the existing engine but without the graph
    and dataset machinery.
    """
    from daglite.datasets.processor import DatasetProcessor
    from daglite.datasets.reporters import DirectDatasetReporter
    from daglite.plugins.reporters import DirectEventReporter
    from daglite.settings import get_global_settings

    timestamp = time.perf_counter()

    settings = settings if settings is not None else get_global_settings()
    backend = backend if backend is not None else settings.backend

    plugin_manager, event_processor = _setup_plugins(plugins or [])
    hook = plugin_manager.hook
    event_reporter = DirectEventReporter(callback=event_processor.dispatch)

    cache_store = _resolve_cache_store(cache_store, settings)

    dataset_store = _resolve_dataset_store(dataset_store, settings)
    dataset_reporter = DirectDatasetReporter() if dataset_store is not None else None
    dataset_processor = DatasetProcessor(hook=hook) if dataset_store is not None else None

    context = SessionContext(
        plugin_manager=plugin_manager,
        event_reporter=event_reporter,
        backend=backend,
        cache_store=cache_store,
        dataset_store=dataset_store,
        dataset_reporter=dataset_reporter,
        settings=settings,
        event_processor=event_processor,
        dataset_processor=dataset_processor,
        timestamp=timestamp,
    )

    return context


def _setup_plugins(plugins: list[Any]) -> tuple[PluginManager, EventProcessor]:
    """
    Creates a plugin manager and event processor for this session.

    Follows the same pattern as `engine._setup_plugin_system`.
    """
    from daglite.plugins.events import EventProcessor
    from daglite.plugins.events import EventRegistry
    from daglite.plugins.manager import build_plugin_manager

    registry = EventRegistry()
    plugin_manager = build_plugin_manager(plugins, registry)
    event_processor = EventProcessor(registry)
    return plugin_manager, event_processor


def _resolve_cache_store(cache: str | CacheStore | None, settings: Any) -> CacheStore | None:
    """Resolves the cache argument into a `CacheStore` or `None`."""
    from daglite.cache.store import CacheStore

    if cache is None:
        return None

    if isinstance(cache, CacheStore):
        return cache

    if isinstance(cache, str):  # pragma: no branch
        return CacheStore(cache)


def _resolve_dataset_store(
    dataset_store: str | DatasetStore | None, settings: Any
) -> DatasetStore | None:
    """Resolves the dataset_store argument into a `DatasetStore` or `None`."""
    if dataset_store is None:
        ds_path = getattr(settings, "dataset_store", None)
        if ds_path is None:
            return None
        return DatasetStore(ds_path) if isinstance(ds_path, str) else ds_path

    if isinstance(dataset_store, DatasetStore):
        return dataset_store

    if isinstance(dataset_store, str):  # pragma: no branch
        return DatasetStore(dataset_store)


@contextmanager
def _processors_context(ctx: SessionContext) -> Iterator[None]:
    """Context manager that starts and stops the event processor and backend manager."""
    from daglite.backends.manager import BackendManager

    mgr = BackendManager(session=ctx, settings=ctx.settings)
    token = mgr.activate()
    _start_processors(ctx)
    t0 = time.perf_counter()
    ctx.plugin_manager.hook.before_session_start(session_id=ctx.session_id)
    try:
        yield
    finally:
        duration = time.perf_counter() - t0
        ctx.plugin_manager.hook.after_session_end(session_id=ctx.session_id, duration=duration)
        _stop_processors(ctx, mgr, token)


def _start_processors(ctx: SessionContext) -> None:
    """Starts background processors attached to the context."""
    if ctx.event_processor is not None:  # pragma: no branch
        ctx.event_processor.start()
    if ctx.dataset_processor is not None:  # pragma: no branch
        ctx.dataset_processor.start()


def _stop_processors(ctx: SessionContext, mgr: Any, token: Any) -> None:
    """Stops background processors and backends, flushing pending work."""
    # Stop backends first — they may have queues that feed the processors.
    try:
        mgr.deactivate(token)
    except Exception:
        logger.exception("Failed to stop backend manager")

    if ctx.dataset_processor is not None:  # pragma: no branch
        try:
            ctx.dataset_processor.stop()
        except Exception:  # pragma: no cover - cleanup guard
            logger.exception("Failed to stop dataset processor")

    if ctx.event_processor is not None:  # pragma: no branch
        try:
            ctx.event_processor.stop()
        except Exception:  # pragma: no cover - cleanup guard
            logger.exception("Failed to stop event processor")
