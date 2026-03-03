"""Typed event dataclasses for the eager execution model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from uuid import UUID


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
