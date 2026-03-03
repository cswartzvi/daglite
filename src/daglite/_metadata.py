"""Shared metadata type for task execution contexts."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from uuid import UUID


@dataclass(frozen=True)
class TaskMetadata:
    """
    Lightweight metadata describing the currently executing task.

    Used by hook specs, the logging plugin, and the task runner
    to communicate identity and context without coupling to the old
    graph IR.
    """

    id: UUID
    """Unique identifier for this task invocation."""

    name: str
    """Human-readable task name."""

    description: str | None = field(default=None, kw_only=True)
    """Optional human-readable description."""

    backend_name: str | None = field(default=None, kw_only=True)
    """Name of the backend used to execute this task."""

    hidden: bool = field(default=False, kw_only=True)
    """Whether this task should be hidden from progress/logging output."""
