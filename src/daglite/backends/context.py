"""Execution context for worker threads/processes."""

from contextvars import ContextVar
from contextvars import Token

from pluggy import PluginManager

from daglite.plugins.reporters import EventReporter

# Context variables for worker execution environment
_plugin_manager: ContextVar[PluginManager | None] = ContextVar("plugin_manager", default=None)
_event_reporter: ContextVar[EventReporter | None] = ContextVar("event_reporter", default=None)


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
        EventReporter instance if set, None otherwise (e.g., for SequentialBackend).
    """
    return _event_reporter.get()


def reset_execution_context(  # pragma: no cover
    pm_token: Token[PluginManager | None],
    reporter_token: Token[EventReporter | None],
) -> None:
    """
    Reset execution context to previous state.

    Useful for cleaning up after a worker is done executing tasks. Typically called
    in a finally block to ensure context is restored.

    Args:
        pm_token: Token returned by set_execution_context() for plugin manager.
        reporter_token: Token returned by set_execution_context() for event reporter.
    """
    _plugin_manager.reset(pm_token)
    _event_reporter.reset(reporter_token)


def clear_execution_context() -> None:  # pragma: no cover
    """
    Clear execution context.

    Useful for testing or explicit cleanup. Not typically needed in production
    as context vars are thread/task-local.
    """
    _plugin_manager.set(None)
    _event_reporter.set(None)
