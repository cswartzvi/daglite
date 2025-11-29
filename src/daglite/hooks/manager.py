"""Utility functions to manage the project-wide hook configuration."""

import logging
from inspect import isclass
from typing import Any

from pluggy import PluginManager

from .markers import HOOK_NAMESPACE
from .specs import NodeSpec

logger = logging.getLogger(__name__)

_PLUGIN_HOOKS = "daglite.hooks"  # entry-point to load hooks from for installed plugins
_HOOK_MANAGER: PluginManager | None = None


def initialize_hooks() -> None:
    """Initializes hooks for the daglite library."""
    manager = _create_hook_manager()
    global _HOOK_MANAGER
    _HOOK_MANAGER = manager


def get_hook_manager() -> PluginManager:
    """Returns initialized hook plugin manager or raises an exception."""
    hook_manager = _HOOK_MANAGER
    if hook_manager is None:  # pragma: no cover
        # NOTE: This should not happen in normal practice since we initialize hooks at the
        # top-level daglite/__init__.py module. However, it could happen if a distributed
        # runner is used without proper initialization.
        raise RuntimeError(
            "Attempted access of Hook plugin manager without initialization. "
            "Normally this happens when `initialize_hooks()` has not been called. "
            "Please report this if you continue to see this error."
        )
    assert hook_manager is not None
    return hook_manager


def register_hooks(*hooks: Any) -> None:
    """Register specified daglite pluggy hooks."""
    hook_manager = get_hook_manager()
    for hooks_collection in hooks:
        if not hook_manager.is_registered(hooks_collection):
            if isclass(hooks_collection):
                raise TypeError(
                    "daglite expects hooks to be registered as instances. "
                    "Have you forgotten the `()` when registering a hook class?"
                )
            hook_manager.register(hooks_collection)


def register_hooks_entry_points() -> None:
    """Register daglite pluggy hooks from Python package entrypoints."""
    hook_manager = get_hook_manager()
    hook_manager.load_setuptools_entrypoints(_PLUGIN_HOOKS)  # Despite name setuptools not required


def create_hook_manager_with_plugins(plugins: list[Any]) -> PluginManager:
    """
    Create a new hook manager with both global and execution-specific plugins.

    This combines globally registered hooks with additional hooks for a specific execution.
    Used internally by Engine to support per-execution hooks.

    Args:
        plugins: Additional hook implementations to register.

    Returns:
        A new PluginManager with global + execution-specific hooks.
    """
    # Create new manager with hook specs
    manager = _create_hook_manager()

    # Copy global hooks
    global_manager = get_hook_manager()
    for plugin in global_manager.get_plugins():
        if not manager.is_registered(plugin):  # pragma: no branch
            manager.register(plugin)

    # Add execution-specific hooks
    for plugin in plugins:
        if not manager.is_registered(plugin):  # pragma: no branch
            if isclass(plugin):
                raise TypeError(
                    "daglite expects hooks to be registered as instances. "
                    "Have you forgotten the `()` when registering a hook class?"
                )
            manager.register(plugin)

    return manager


def _create_hook_manager() -> PluginManager:
    """Create a new PluginManager instance and register daglite's hook specs."""
    manager = PluginManager(HOOK_NAMESPACE)
    manager.trace.root.setwriter(
        logger.debug if logger.getEffectiveLevel() == logging.DEBUG else None
    )
    manager.enable_tracing()
    manager.add_hookspecs(NodeSpec)
    from .specs import GraphSpec

    manager.add_hookspecs(GraphSpec)
    return manager
