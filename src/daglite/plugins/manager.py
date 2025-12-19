"""Utility functions to manage the project-wide hook configuration."""

import importlib
import logging
from inspect import isclass
from typing import Any

from pluggy import PluginManager

from daglite.plugins.base import isinstance_serializable_plugin
from daglite.plugins.base import issubclass_serializable_plugin

from .hooks.markers import HOOK_NAMESPACE
from .hooks.specs import NodeSpec

logger = logging.getLogger(__name__)

_PLUGIN_ENTRY_POINT = "daglite.hooks"  # entry-point to load hooks from for installed plugins
_PLUGIN_MANAGER: PluginManager | None = None


# region API


def register_hooks(*hooks: Any) -> None:
    """Register specified daglite pluggy hooks."""
    hook_manager = _get_global_plugin_manager()
    for hooks_collection in hooks:
        if not hook_manager.is_registered(hooks_collection):
            if isclass(hooks_collection):
                raise TypeError(
                    "daglite expects hooks to be registered as instances. "
                    "Have you forgotten the `()` when registering a hook class?"
                )
            hook_manager.register(hooks_collection)


def register_plugins_entry_points(_plugin_manager: PluginManager | None = None) -> None:
    """Register daglite plugins from Python package entrypoints."""
    _plugin_manager = _plugin_manager if _plugin_manager else _get_global_plugin_manager()
    _plugin_manager.load_setuptools_entrypoints(_PLUGIN_ENTRY_POINT)  # Doesn't use setuptools


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
    manager = _create_plugin_manager()

    # Copy global hooks
    global_manager = _get_global_plugin_manager()
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


def serialize_plugin_manager(plugin_manager: PluginManager) -> dict[str, Any]:
    """
    Serialize the given PluginManager's serializable plugins to a config dict.

    Args:
        plugin_manager: The PluginManager to serialize.

    Returns:
        A dict mapping plugin class names to their serialized config.
    """
    plugin_configs: dict[str, Any] = {}
    for plugin in plugin_manager.get_plugins():
        if isinstance_serializable_plugin(plugin):
            cls = plugin.__class__
            fqcn = f"{cls.__module__}.{cls.__qualname__}"
            plugin_configs[fqcn] = plugin.to_config()
    return plugin_configs


def deserialize_plugin_manager(plugin_configs: dict[str, Any]) -> PluginManager:
    """
    Deserialize a PluginManager from a config dict of plugin class names to configs.

    Args:
        plugin_configs: A dict mapping plugin class names to their serialized config.

    Returns:
        A PluginManager with the deserialized plugins registered.
    """
    plugin_manager = _create_plugin_manager()

    for class_path, plugin_configs in plugin_configs.items():
        plugin_class = _resolve_class_from_path(class_path)

        if plugin_class is None:
            logger.warning(f"Could not resolve plugin class '{class_path}' for deserialization.")
            continue

        # Ensure plugin class supports from_config
        if not issubclass_serializable_plugin(plugin_class):
            logger.warning(f"Plugin class '{class_path}' is not serializable.")
            continue

        plugin_instance = plugin_class.from_config(plugin_configs)
        plugin_manager.register(plugin_instance)

    return plugin_manager


# region Helpers


def _initialize_plugin_system() -> PluginManager:
    """Initializes hooks for the daglite library."""
    manager = _create_plugin_manager()
    global _PLUGIN_MANAGER
    _PLUGIN_MANAGER = manager
    return manager


def _get_global_plugin_manager() -> PluginManager:
    """Returns initialized global plugin manager or raises an exception."""
    plugin_manager = _PLUGIN_MANAGER
    if plugin_manager is None:
        plugin_manager = _initialize_plugin_system()
    return plugin_manager


def _create_plugin_manager() -> PluginManager:
    """Create a new PluginManager instance and register daglite's hook specs."""
    manager = PluginManager(HOOK_NAMESPACE)
    manager.trace.root.setwriter(
        logger.debug if logger.getEffectiveLevel() == logging.DEBUG else None
    )
    manager.enable_tracing()
    manager.add_hookspecs(NodeSpec)
    from .hooks.specs import GraphSpec

    manager.add_hookspecs(GraphSpec)
    return manager


def _resolve_class_from_path(path: str) -> type[Any] | None:
    """Resolve a dotted import path to a class/type object."""
    parts = path.split(".")

    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        attr_parts = parts[i:]
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        obj: Any = module
        try:
            for attr in attr_parts:
                obj = getattr(obj, attr)
            if isinstance(obj, type):
                return obj
        except AttributeError:
            continue

    return None
