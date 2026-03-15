"""Shared context-chain resolution helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pluggy import HookRelay
from pluggy import PluginManager

from daglite._context import BackendContext
from daglite._context import SessionContext
from daglite._context import SubmitContext
from daglite._context import TaskContext

if TYPE_CHECKING:
    from daglite.cache.store import CacheStore
    from daglite.plugins.reporters import EventReporter
    from daglite.tasks import TaskMetadata
else:
    CacheStore = Any
    EventReporter = Any
    TaskMetadata = Any

_MISSING = object()


def resolve_task_metadata() -> TaskMetadata | None:
    """
    Resolve the active task metadata from the context chain.

    Resolution order: ``TaskContext`` → ``BackendContext`` → ``SessionContext``.
    """
    task_ctx = TaskContext._get()
    return task_ctx.metadata if task_ctx is not None else None


def resolve_task_inputs() -> dict[str, object] | None:
    """Return the bound input arguments of the currently executing task, or ``None``."""
    meta = resolve_task_metadata()
    return meta.inputs if meta is not None else None


def resolve_parent_id() -> UUID | None:
    """
    Resolve the parent task ID from the active context chain.

    Resolution order: ``TaskContext`` → ``BackendContext`` → ``SessionContext``.
    """
    parent_ctx = TaskContext._get()
    parent_id = parent_ctx.metadata.id if parent_ctx is not None else None
    return parent_id


def resolve_map_index() -> int | None:
    """
    Resolve the map index from the active context chain.

    Resolution order: ``SubmitContext``.
    """
    submit_ctx = SubmitContext._get()
    if submit_ctx is None:
        return None
    return submit_ctx.map_index


def resolve_backend() -> str:
    """
    Resolve the active backend name.

    Resolution order: ``BackendContext`` → ``SessionContext`` → session settings →
    global settings.
    """
    from daglite.settings import get_global_settings

    session_ctx = SessionContext._get()
    return _resolve_from_chain(
        "backend",
        BackendContext._get(),
        session_ctx,
        session_ctx.settings if session_ctx else None,
        get_global_settings(),
    )


def resolve_plugin_manager() -> PluginManager:
    """
    Resolve the active ``PluginManager``.

    Resolution order: ``BackendContext`` → ``SessionContext`` → null plugin manager.
    """
    from daglite.plugins.manager import create_null_plugin_manager

    return _resolve_from_chain(
        "plugin_manager",
        BackendContext._get(),
        SessionContext._get(),
        default_factory=create_null_plugin_manager,
    )


def resolve_hook() -> HookRelay:
    """
    Resolve the active pluggy ``HookRelay``.

    Shorthand for ``resolve_plugin_manager().hook``.
    """
    return resolve_plugin_manager().hook


def resolve_event_reporter() -> EventReporter | None:
    """
    Return the active ``EventReporter``, or ``None``.

    Resolution order: ``BackendContext`` → ``SessionContext``.
    """
    return _resolve_from_chain(
        "event_reporter",
        BackendContext._get(),
        SessionContext._get(),
        default=None,
    )


def resolve_cache_store(
    cache: Any = None,
    cache_store: Any | str | None = None,
) -> CacheStore | None:
    """
    Resolve a ``CacheStore`` from the task's cache settings and the active context chain.

    Args:
        cache: Task-level cache flag/value.  ``False`` disables caching.  ``True``
            triggers the context-chain lookup.  A ``CacheStore`` or ``str`` path is used
            directly (as a shorthand for ``cache_store``).
        cache_store: Explicit per-task store override (string path or ``CacheStore``).
            Takes priority over the context chain when non-``None``.

    Returns:
        A ``CacheStore`` instance, or ``None`` if caching is disabled.
    """
    from daglite.cache.store import CacheStore
    from daglite.settings import get_global_settings

    # cache itself can be a str or CacheStore (legacy convenience).
    if isinstance(cache, CacheStore):
        return cache
    if isinstance(cache, str):
        return CacheStore(cache)
    if not cache:
        return None

    # Explicit per-task store takes priority.
    if isinstance(cache_store, CacheStore):
        return cache_store
    if isinstance(cache_store, str):
        return CacheStore(cache_store)

    # Fall back to context chain.
    session_ctx = SessionContext._get()
    store: CacheStore | str | None = _resolve_from_chain(
        "cache_store",
        BackendContext._get(),
        session_ctx,
        session_ctx.settings if session_ctx else None,
        get_global_settings(),
    )
    return CacheStore(store) if isinstance(store, str) else store


def resolve_dataset_store(
    store: Any | str | None = None,
) -> Any:
    """
    Resolve a ``DatasetStore`` from *store*, the active context chain, or global settings.

    Resolution order: explicit argument → ``TaskContext`` → ``BackendContext`` →
    ``SessionContext`` → global settings.

    Args:
        store: Explicit ``DatasetStore`` instance or string path.  ``None`` triggers
            the context-chain lookup.

    Returns:
        A ``DatasetStore`` instance.

    Raises:
        RuntimeError: If no store can be resolved from any source.
    """
    from daglite.datasets.store import DatasetStore

    if isinstance(store, DatasetStore):
        return store
    if isinstance(store, str):
        return DatasetStore(store)

    ds: DatasetStore | None = _resolve_from_chain(
        "dataset_store",
        TaskContext._get(),
        BackendContext._get(),
        SessionContext._get(),
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


def resolve_dataset_reporter() -> Any | None:
    """
    Return the active ``DatasetReporter``, or ``None``.

    Checks ``BackendContext`` first (worker processes carry a ``ProcessDatasetReporter``),
    then falls back to ``SessionContext`` (coordinator carries a ``DirectDatasetReporter``).
    """
    return _resolve_from_chain(
        "dataset_reporter",
        BackendContext._get(),
        SessionContext._get(),
        default=None,
    )


def _resolve_from_chain(
    field_name: str,
    *sources: Any,
    default: Any = _MISSING,
    default_factory: Callable[[], Any] | None = None,
) -> Any:
    """
    Return the first non-None value of *field_name* across an ordered chain of providers.

    Args:
        field_name: Attribute name to look up on each source.
        *sources: Objects to inspect in order. ``None`` entries are skipped.
        default: Value to return when no source has a non-None value for *field_name*.
            Mutually exclusive with *default_factory*.
        default_factory: Callable that produces the default value lazily.
            Mutually exclusive with *default*.

    Returns:
        The first non-None attribute value, or the computed default.
    """
    invalid_state = default is not _MISSING and default_factory is not None
    assert not invalid_state, "Must provide either a default or default_factory"

    for src in sources:
        if src is None:
            continue
        val = getattr(src, field_name, _MISSING)
        assert val is not _MISSING, f"{type(src).__name__} has no attribute {field_name!r}"
        if val is not None:
            return val

    if default_factory is not None:
        return default_factory()

    if default is not _MISSING:
        return default

    return None  # pragma: no cover - all callers provide a default
