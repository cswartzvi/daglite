"""Typed events, task event dataclasses, and the event registry."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable
from uuid import UUID

logger = logging.getLogger(__name__)


# region PluginEvent


@dataclass(frozen=True)
class PluginEvent:
    """
    Generic string-typed envelope for the coordinator-side plugin event system.

    `PluginEvent` is **infrastructure plumbing** — it carries an opaque ``type`` string and a
    ``data`` dict through `EventRegistry.dispatch` so that coordinator-side handlers can react
    without importing concrete types.

    This is distinct from the **typed lifecycle dataclasses** (`TaskStarted`, `TaskCompleted`,
    etc.) which are emitted by hooks via `TaskMetadata` and carry strongly-typed fields.

    Attributes:
        type: Event type identifier (e.g. ``"daglite-logging-node-start"``).
        data: Arbitrary payload data.  Plugin-specific — the exact keys depend
            on the event type.

    Examples:
        >>> event = PluginEvent(type="progress", data={"percent": 100})
        >>> event.type
        'progress'
        >>> event.data["percent"]
        100
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)


# region Task events


@dataclass(frozen=True)
class TaskStarted:
    """Emitted immediately before a task function begins execution."""

    task_name: str
    """Human-readable name of the task (from `@task(name=...)` or `func.__name__`)."""

    task_id: UUID
    """Unique identifier for this specific invocation."""

    args: tuple[Any, ...]
    """Positional arguments passed to the task function."""

    kwargs: dict[str, Any]
    """Keyword arguments passed to the task function."""

    backend: str
    """Name of the backend executing this task (e.g. `"inline"`, `"thread"`)."""

    parent_task_id: UUID | None = None
    """Task ID of the parent task, or `None` if this is a top-level call."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the event was created."""


@dataclass(frozen=True)
class TaskCompleted:
    """Emitted after a task function returns successfully."""

    task_name: str
    """Human-readable name of the task."""

    task_id: UUID
    """Unique identifier for this specific invocation."""

    result: Any
    """The value returned by the task function."""

    elapsed: float
    """Wall-clock seconds the task took to execute."""

    cached: bool
    """`True` if the result came from the cache rather than execution."""

    parent_task_id: UUID | None = None
    """Task ID of the parent task, or `None` if this is a top-level call."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the event was created."""


@dataclass(frozen=True)
class TaskFailed:
    """Emitted when a task function raises an exception."""

    task_name: str
    """Human-readable name of the task."""

    task_id: UUID
    """Unique identifier for this specific invocation."""

    error: BaseException
    """The exception that was raised."""

    elapsed: float
    """Wall-clock seconds before the failure occurred."""

    parent_task_id: UUID | None = None
    """Task ID of the parent task, or `None` if this is a top-level call."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the event was created."""


@dataclass(frozen=True)
class WorkflowStarted:
    """Emitted when a workflow or session begins."""

    workflow_name: str
    """Name of the workflow (or `"session"` for ad-hoc contexts)."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the event was created."""


@dataclass(frozen=True)
class WorkflowFinished:
    """Emitted when a workflow or session completes."""

    workflow_name: str
    """Name of the workflow (or `"session"` for ad-hoc contexts)."""

    elapsed: float
    """Wall-clock seconds the workflow/session ran."""

    error: BaseException | None = None
    """If the workflow/session failed, the exception; otherwise `None`."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when the event was created."""


# region Type Aliases

TaskEvent = TaskStarted | TaskCompleted | TaskFailed
"""Union of all task-level events."""

SessionEvent = WorkflowStarted | WorkflowFinished
"""Union of all session/workflow-level events."""

EagerEvent = TaskEvent | SessionEvent
"""Union of all events emitted by the eager execution model."""


# region EventRegistry


class EventRegistry:
    """
    Registry for coordinator-side event handlers.

    Plugins register handlers for specific event types, which are called when events are dispatched.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[PluginEvent], None]]] = {}

    def register(self, event_type: str, handler: Callable[[PluginEvent], None]) -> None:
        """
        Register handler for event type.

        Multiple handlers can be registered for the same event type.

        Args:
            event_type: Type of event to handle
            handler: Callable that takes a `PluginEvent`
        """
        self._handlers.setdefault(event_type, []).append(handler)

    def dispatch(self, event: PluginEvent) -> None:
        """
        Dispatch event to all registered handlers.

        Handlers are called synchronously. Errors are logged but don't prevent other handlers from
        running.

        Args:
            event: `Event` to dispatch
        """
        for handler in self._handlers.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Error in event handler for '{event.type}': {e}")
