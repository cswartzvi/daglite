"""
Execution context bridge for plugins.

Plugins (especially the logging plugin) need access to the current event
reporter and task metadata. In the eager model this information lives in the
`RunContext` stored in a `ContextVar` by `session.py`.

The functions here read from the `RunContext` so that existing plugin code
continues to work without modification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pluggy import PluginManager

    from daglite.plugins.reporters import EventReporter


def get_event_reporter() -> EventReporter | None:
    """
    Get the event reporter for the current execution context.

    Reads the active `RunContext` from the session `ContextVar`.

    Returns:
        The `EventReporter` if a session/workflow is active, *None* otherwise.
    """
    from daglite.session import get_run_context

    ctx = get_run_context()
    return ctx.event_reporter if ctx is not None else None


def get_plugin_manager() -> PluginManager | None:
    """
    Get the plugin manager for the current execution context.

    Returns:
        The `PluginManager` if a session/workflow is active, *None* otherwise.
    """
    from daglite.session import get_run_context

    ctx = get_run_context()
    return ctx.plugin_manager if ctx is not None else None
