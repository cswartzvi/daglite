"""Execution context utilities for task workers."""

from contextvars import ContextVar
from contextvars import Token

from pluggy import PluginManager

from daglite.graph.base import GraphMetadata
from daglite.plugins.reporters import EventReporter

# Context variables for worker execution environment
_plugin_manager: ContextVar[PluginManager | None] = ContextVar("plugin_manager", default=None)
_event_reporter: ContextVar[EventReporter | None] = ContextVar("event_reporter", default=None)
_current_task: ContextVar["GraphMetadata | None"] = ContextVar("current_task", default=None)


def set_execution_context(
    plugin_manager: PluginManager, reporter: EventReporter
) -> tuple[Token[PluginManager | None], Token[EventReporter | None]]:
    """
    Set execution context for current worker (thread/process/machine).

    This should be called once during worker initialization (e.g., in ThreadPoolExecutor
    initializer) to establish the plugin manager and event reporter for the worker.

    Args:
        plugin_manager: Plugin manager instance for hook execution.
        reporter: Event reporter for worker → coordinator communication.
    """
    return _plugin_manager.set(plugin_manager), _event_reporter.set(reporter)


def get_plugin_manager() -> PluginManager:
    """
    Get plugin manager for current execution context.

    Returns:
        PluginManager instance set via set_execution_context().

    Raises:
        RuntimeError: If called outside an execution context.
    """
    pm = _plugin_manager.get()
    if pm is None:  # pragma: no cover
        raise RuntimeError(
            "No plugin manager in execution context. "
            "Ensure set_execution_context() was called during worker initialization."
        )
    return pm


def get_reporter() -> EventReporter | None:
    """
    Get event reporter for current execution context.

    Returns:
        EventReporter instance if set, None otherwise (e.g., for InlineBackend).
    """
    return _event_reporter.get()


def reset_execution_context() -> None:  # pragma: no cover
    """
    Resets the execution context by clearing all context variables.

    Useful for testing or explicit cleanup. Not typically needed in production
    as context vars are thread/task-local.
    """
    _plugin_manager.set(None)
    _event_reporter.set(None)
    _current_task.set(None)


def set_current_task(metadata: "GraphMetadata") -> Token["GraphMetadata | None"]:
    """
    Set the currently executing task's metadata.

    This should be called before executing a task function to provide
    contextual information for logging and other plugins.

    Args:
        metadata: Task metadata (name, id, description, etc.)

    Returns:
        Token for resetting the context later
    """
    return _current_task.set(metadata)


def get_current_task() -> "GraphMetadata | None":
    """
    Get the currently executing task's metadata.

    Returns:
        GraphMetadata if a task is currently executing, None otherwise.
        Useful for contextual logging and debugging.
    """
    return _current_task.get()


def reset_current_task(token: Token["GraphMetadata | None"]) -> None:  # pragma: no cover
    """
    Reset current task context to previous state.

    Args:
        token: Token returned by set_current_task()
    """
    _current_task.reset(token)
