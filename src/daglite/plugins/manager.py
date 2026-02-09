"""Utility functions to manage the project-wide hook configuration."""

import importlib
import logging
from inspect import isclass
from typing import Any

from pluggy import PluginManager

from daglite.plugins.base import isinstance_bidirectional_plugin
from daglite.plugins.base import isinstance_serializable_plugin
from daglite.plugins.base import issubclass_serializable_plugin
from daglite.plugins.registry import EventRegistry

from .hooks.markers import HOOK_NAMESPACE

logger = logging.getLogger(__name__)

_PLUGIN_MANAGER: PluginManager | None = None


# region API


def register_plugins(*plugins: Any, _plugin_manager: PluginManager | None = None) -> None:
    """
    Registers daglite plugins with the global plugin manager.

    This function can be used by the user to register custom plugins globally. Note that the
    plugins registered here will be present in all evaluation contexts, however they will not
    run outside of the evaluation process (e.g., start immediately upon registration).

    Args:
        plugins: Plugin instances to register.
    """
    _plugin_manager = _plugin_manager if _plugin_manager else _get_global_plugin_manager()
    for plugin in plugins:
        if not _plugin_manager.is_registered(plugin):
            if isclass(plugin):
                raise TypeError(
                    "daglite expects plugins to be registered as instances. "
                    "Have you forgotten the `()` when registering a plugin class?"
                )
            _plugin_manager.register(plugin)


def build_plugin_manager(plugins: list[Any], registry: EventRegistry) -> PluginManager:
    """
    Creates a new plugin manager with both global and execution-specific plugins.

    This is the canonical way to create a PluginManager for a specific execution context,
    ensuring that both global and execution-specific plugins are registered, and that event

    Args:
        plugins: Additional plugin implementations to register.
        registry: Event registry for event handling.

    Returns:
        A new PluginManager with global + execution-specific plugins.
    """
    # Create new manager with hook specs
    new_manager = _create_plugin_manager()

    # Copy global hooks
    global_manager = _get_global_plugin_manager()
    for plugin in global_manager.get_plugins():
        if not new_manager.is_registered(plugin):  # pragma: no branch
            new_manager.register(plugin)

    # Add execution-specific plugins
    register_plugins(*plugins, _plugin_manager=new_manager)

    # Register event handlers from bidirectional plugins
    for plugin in new_manager.get_plugins():
        if isinstance_bidirectional_plugin(plugin):  # pragma: no cover
            plugin.register_event_handlers(registry)

    return new_manager


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
    plugin_manager = _initialize_plugin_system() if plugin_manager is None else plugin_manager
    return plugin_manager


def _create_plugin_manager() -> PluginManager:
    """Create a new PluginManager instance and register daglite's hook specs."""
    from .hooks.specs import CoordinatorSideNodeSpecs
    from .hooks.specs import GraphSpec
    from .hooks.specs import WorkerSideNodeSpecs

    manager = PluginManager(HOOK_NAMESPACE)
    manager.add_hookspecs(WorkerSideNodeSpecs)
    manager.add_hookspecs(CoordinatorSideNodeSpecs)
    manager.add_hookspecs(GraphSpec)

    # Enable plugin hook tracing if configured
    from daglite.settings import get_global_settings

    settings = get_global_settings()
    if settings.enable_plugin_tracing:
        manager.trace.root.setwriter(logger.debug)
        manager.enable_tracing()

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
            if isinstance(obj, type):  # pragma: no branch
                return obj
        except AttributeError:  # pragma: no cover
            continue

    return None
