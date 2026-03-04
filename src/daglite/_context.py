"""
Execution context and ContextVar management for eager tasks.

`RunContext` carries all infrastructure references (cache, events, plugins, backends) for a
session. The ContextVar-based accessors let eager tasks read the active context without explicit
parameter threading.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from uuid import UUID

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


# region ContextVars

_run_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)

_task_call_args: ContextVar[dict[str, Any] | None] = ContextVar("task_call_args", default=None)
"""Bound arguments of the currently executing task, used for ``{param}`` template resolution."""

_map_iteration_index: ContextVar[int | None] = ContextVar("map_iteration_index", default=None)
"""Index of the current map iteration, set by `task_map` / `async_task_map`."""

_parent_task_id: ContextVar[UUID | None] = ContextVar("parent_task_id", default=None)
"""Task ID of the currently executing parent task, for nested-task tracking."""


# region Accessors


def get_run_context() -> RunContext | None:
    """Returns the active run context, or `None` if outside a session."""
    return _run_context.get()


def set_run_context(ctx: RunContext) -> Any:
    """Pushes a run context. Returns a token for `reset_run_context`."""
    return _run_context.set(ctx)


def reset_run_context(token: Any) -> None:
    """Restores the previous run context using a token from `set_run_context`."""
    _run_context.reset(token)


def get_task_call_args() -> dict[str, Any] | None:
    """Returns the bound arguments of the currently executing task, or `None`."""
    return _task_call_args.get()


def set_task_call_args(args: dict[str, Any] | None) -> Any:
    """Sets the current task's bound arguments. Returns a token for reset."""
    return _task_call_args.set(args)


def get_event_reporter() -> Any | None:
    """
    Get the event reporter for the current execution context.

    Reads the active `RunContext` from the session `ContextVar`.

    Returns:
        The `EventReporter` if a session/workflow is active, *None* otherwise.
    """
    ctx = get_run_context()
    return ctx.event_reporter if ctx is not None else None


def get_plugin_manager() -> Any | None:
    """
    Get the plugin manager for the current execution context.

    Returns:
        The `PluginManager` if a session/workflow is active, *None* otherwise.
    """
    ctx = get_run_context()
    return ctx.plugin_manager if ctx is not None else None


def get_map_iteration_index() -> int | None:
    """Returns the current map iteration index, or `None` if not inside a map."""
    return _map_iteration_index.get()


def set_map_iteration_index(index: int) -> Any:
    """Sets the current map iteration index. Returns a token for reset."""
    return _map_iteration_index.set(index)


def get_parent_task_id() -> UUID | None:
    """Returns the task ID of the parent task, or `None` if top-level."""
    return _parent_task_id.get()


def set_parent_task_id(task_id: UUID | None) -> Any:
    """Sets the parent task ID. Returns a token for reset."""
    return _parent_task_id.set(task_id)
