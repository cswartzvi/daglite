"""Interfaces for Daglite plugins."""

from typing import Any, Protocol, runtime_checkable

from typing_extensions import TypeIs

from daglite.plugins.registry import EventRegistry


@runtime_checkable
class EventHandlerPlugin(Protocol):
    """
    Plugin that registers coordinator-side event handlers.

    Plugins implementing this protocol can:
    1. Register coordinator-side event handlers via register_event_handlers()
    2. Use reporter in worker-side hooks to send events back to coordinator

    Examples:
    >>> from daglite.plugins.hooks.markers import hook_impl
    >>> class ProgressPlugin:
    ...     def register_event_handlers(self, registry):
    ...         registry.on("progress", self._update_progress)
    ...
    ...     def _update_progress(self, event):
    ...         # Coordinator-side handler
    ...         print(f"Progress: {event['percent']}%")
    ...
    ...     @hook_impl
    ...     def after_node_execute(self, node_id, result, reporter=None):
    ...         # Worker-side hook
    ...         if reporter:
    ...             reporter.report("progress", {"percent": 100})
    """

    def register_event_handlers(self, registry: EventRegistry) -> None:
        """
        Register coordinator-side event handlers.

        Called during engine initialization to set up event processing.

        Args:
            registry: Event registry for registering handlers
        """
        ...


@runtime_checkable
class SerializablePlugin(Protocol):
    """
    Plugin that can be serialized for cross-process execution.

    ProcessPool and distributed backends require plugins to be serializable.
    Plugins implementing this protocol can reconstruct themselves on workers.

    Example:
    >>> class MyPlugin:
    ...     def __init__(self, threshold=0.5):
    ...         self.threshold = threshold
    ...         self.stats = {}  # Not serialized
    ...
    ...     def to_config(self):
    ...         return {"threshold": self.threshold}
    ...
    ...     @classmethod
    ...     def from_config(cls, config):
    ...         return cls(**config)
    """

    def to_config(self) -> dict[str, Any]:
        """
        Serialize plugin state to config dict.

        Should return only serializable constructor arguments, not runtime state.

        Returns:
            Serializable config dict
        """
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SerializablePlugin":
        """
        Reconstruct plugin from config dict.

        Args:
            config: Config dict from to_config()

        Returns:
            New plugin instance
        """
        ...


def isinstance_serializable_plugin(obj: Any) -> TypeIs[SerializablePlugin]:  # pragma: no cover
    """Type narrowing check for SerializablePlugin protocol."""
    return isinstance(obj, SerializablePlugin)


def issubclass_serializable_plugin(
    cls: Any,
) -> TypeIs[type[SerializablePlugin]]:  # pragma: no cover
    """Type narrowing check for SerializablePlugin protocol on classes."""
    return (
        isinstance(cls, type)
        and issubclass(cls, SerializablePlugin)
        and callable(getattr(cls, "from_config", None))
    )


def isinstance_event_handler_plugin(obj: Any) -> TypeIs[EventHandlerPlugin]:  # pragma: no cover
    """Type narrowing check for EventHandlerPlugin protocol."""
    return isinstance(obj, EventHandlerPlugin)
