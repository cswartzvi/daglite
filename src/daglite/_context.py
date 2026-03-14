"""Execution context and ContextVar management for eager tasks."""

from __future__ import annotations

import time
from contextvars import ContextVar
from contextvars import Token
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import TYPE_CHECKING, Any, ClassVar, override
from uuid import UUID
from uuid import uuid4

from typing_extensions import Self

if TYPE_CHECKING:
    from daglite.cache.store import CacheStore
    from daglite.datasets.reporters import DatasetReporter
    from daglite.datasets.store import DatasetStore
    from daglite.plugins.events import EventProcessor
    from daglite.plugins.manager import PluginManager
    from daglite.plugins.reporters import EventReporter
    from daglite.settings import DagliteSettings
    from daglite.tasks import TaskMetadata
else:
    CacheStore = Any
    DatasetReporter = Any
    DatasetStore = Any
    PluginManager = Any
    EventProcessor = Any
    EventReporter = Any
    DagliteSettings = Any
    TaskMetadata = Any


@dataclass
class BaseContext:
    """Base class for execution contexts."""

    __var__: ClassVar[ContextVar[Self | None]]
    _token: Token[Self | None] | None = field(default=None, init=False, repr=False)

    @classmethod
    def _get(cls) -> Self | None:
        """Returns the active context, or `None` if outside a session."""
        return cls.__var__.get()

    def __enter__(self) -> Self:
        if self._token is not None:
            raise RuntimeError("Context already entered. Context enter calls cannot be nested.")
        self._token = self.__var__.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._token:
            raise RuntimeError("Asymmetric use of context. Context exit called without an enter.")
        self.__var__.reset(self._token)
        self._token = None


class SerializableContext(BaseContext):
    """Base class for contexts that can be serialized to/from a dictionary."""

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the context."""
        # NOTE: The base implementation serializes all public fields. Subclasses can override this
        # to add extra fields or remove fields.
        data = {}
        for field_ in fields(self):
            field_name = field_.name
            if field_name.startswith("_"):
                continue
            field_value = getattr(self, field_name)
            data[field_name] = field_value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> Self:
        """Creates/restores the context from a dictionary representation."""
        # NOTE: The base implementation deserializes all provided fields. Subclasses can override
        # this to handle extra fields.
        combined_data = {**data, **kwargs}
        return cls(**combined_data)


@dataclass
class SessionContext(BaseContext):
    """Execution context for a session."""

    __var__ = ContextVar("session_context", default=None)

    plugin_manager: PluginManager
    """Pluggy plugin manager for the session."""

    event_reporter: EventReporter
    """A reporter for sending events to the event processor."""

    session_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this session."""

    backend: str | None = None
    """Default backend to use for task execution, or `None` when inheriting from settings."""

    cache_store: CacheStore | None = None
    """Default cache store to use for tasks, or `None` when inheriting from settings."""

    dataset_store: DatasetStore | None = None
    """Default dataset store to use fo tasks, or `None` when inheriting from settings."""

    dataset_reporter: DatasetReporter | None = None
    """A reporter for routing dataset saves to the coordinator."""

    settings: DagliteSettings = field(default_factory=lambda: DagliteSettings())
    """Settings snapshot for the session."""

    event_processor: EventProcessor | None = None
    """Processor for handling events; set to `None` on backend workers."""

    dataset_processor: Any | None = None
    """Processor for handling dataset save requests; set to `None` on backend workers."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the context was created."""


@dataclass
class TaskContext(BaseContext):
    """Execution context for a task."""

    __var__ = ContextVar("task_context", default=None)

    metadata: TaskMetadata
    """Metadata for the current task."""

    cache_store: CacheStore | None = None
    """A cache store override for this task, or `None` to inherit from the session."""

    dataset_store: DatasetStore | None = None
    """A dataset store override for this task, or `None` to inherit from the session."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the context was created."""


@dataclass
class BackendContext(SerializableContext):
    """
    Execution context for coordinating task submission to a backend.

    Note: This context is used to pass information to worker processes when a task is submitted to
    a backend. It contains both session-level and task-level information, so that the worker can
    reconstruct the necessary infrastructure to execute the task.
    """

    __var__ = ContextVar("submit_context", default=None)

    backend: str | None = None
    """Name of the active backend."""

    plugin_manager: PluginManager | None = None
    """Pluggy plugin manager for the session, or `None` outside a session."""

    event_reporter: EventReporter | None = None
    """A reporter for sending events to the event processor, or `None` outside a session."""

    map_index: int | None = None
    """Index of the current map iteration, or `None` if not inside a map."""

    cache_store: CacheStore | None = None
    """A cache store override for this task, or `None` to inherit from the session."""

    dataset_store: DatasetStore | None = None
    """A dataset store override for this task, or `None` to inherit from the session."""

    dataset_reporter: DatasetReporter | None = None
    """A reporter for saving datasets, or `None` outside a session."""

    @classmethod
    def from_session(cls, backend: str | None = None, map_index: int | None = None) -> Self:
        """Creates a `BackendContext` from the active session context."""
        session = SessionContext._get()
        return cls(
            backend=backend,
            map_index=map_index,
            cache_store=session.cache_store if session else None,
            dataset_store=session.dataset_store if session else None,
            dataset_reporter=session.dataset_reporter if session else None,
            event_reporter=session.event_reporter if session else None,
            plugin_manager=session.plugin_manager if session else None,
        )

    @override
    def to_dict(self) -> dict[str, Any]:
        from daglite.plugins.manager import serialize_plugin_manager

        data = super().to_dict()

        # Reporters depend on execution context and are never serialized.
        data.pop("event_reporter", None)
        data.pop("dataset_reporter", None)

        # Dehydrate the plugin manager to a serialized representation.
        if self.plugin_manager is not None:
            data["plugin_manager"] = serialize_plugin_manager(self.plugin_manager)

        # Dehydrate stores to path strings (JSON-safe, works for local and remote via fsspec).
        if self.cache_store is not None:
            data["cache_store"] = self.cache_store.base_path
        if self.dataset_store is not None:
            data["dataset_store"] = self.dataset_store.base_path

        return data

    @override
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        event_reporter: EventReporter | None = None,
        dataset_reporter: DatasetReporter | None = None,
        **kwargs: Any,
    ) -> Self:
        from daglite.cache.store import CacheStore as _CacheStore
        from daglite.datasets.store import DatasetStore as _DatasetStore
        from daglite.plugins.manager import deserialize_plugin_manager

        # Reporters depend on execution context and must be provided explicitly.
        data["event_reporter"] = event_reporter
        data["dataset_reporter"] = dataset_reporter

        # Rehydrate the plugin manager from serialized representation.
        if "plugin_manager" in data and isinstance(data["plugin_manager"], dict):
            data["plugin_manager"] = deserialize_plugin_manager(data["plugin_manager"])

        # Rehydrate stores from path strings.
        if isinstance(data.get("cache_store"), str):
            data["cache_store"] = _CacheStore(data["cache_store"])
        if isinstance(data.get("dataset_store"), str):
            data["dataset_store"] = _DatasetStore(data["dataset_store"])

        return super().from_dict(data)
