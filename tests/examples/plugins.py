"""
Reusable test plugins for daglite tests.

This module provides simple, well-documented plugin implementations that can be
used across both unit and integration tests. These plugins are designed to be:

1. Easy to understand for both humans and AIs
2. Focused on testing specific hook behaviors
3. Reusable across multiple test files
4. Well-documented with clear purpose

All plugins here are SIMPLE examples for testing purposes only.
For production plugin examples, see the extras/ directory.
"""

from typing import Any

from daglite.plugins.hooks.markers import hook_impl


class CounterPlugin:
    """
    Simple plugin that counts hook invocations.

    This is the most commonly used test plugin. It tracks how many times
    specific hooks are called during execution.

    Usage:
        counter = CounterPlugin()
        with session(plugins=[counter]):
            my_task(1, 2)
        assert counter.before_node_count == 1
    """

    def __init__(self):
        """Initialize all counters to zero."""
        self.before_session_count = 0
        self.after_session_count = 0
        self.before_node_count = 0
        self.after_node_count = 0
        self.on_error_count = 0

    @hook_impl
    def before_session_start(self, session_id):
        """Count before_session_start invocations."""
        self.before_session_count += 1

    @hook_impl
    def after_session_end(self, session_id, duration):
        """Count after_session_end invocations."""
        self.after_session_count += 1

    @hook_impl
    def before_node_execute(self, metadata, reporter=None):
        """Count before_node_execute invocations."""
        self.before_node_count += 1

    @hook_impl
    def after_node_execute(self, metadata, result, duration, reporter=None):
        """Count after_node_execute invocations."""
        self.after_node_count += 1

    @hook_impl
    def on_node_error(self, metadata, error, duration, reporter=None):
        """Count on_node_error invocations."""
        self.on_error_count += 1


class ParameterCapturePlugin:
    """
    Plugin that captures all hook parameters for validation.

    Use this plugin when you need to verify that hooks receive correct
    parameters during execution.
    """

    def __init__(self):
        """Initialize empty capture lists."""
        self.session_starts = []
        self.session_ends = []
        self.node_starts = []
        self.node_results = []
        self.node_errors = []

    @hook_impl
    def before_session_start(self, session_id):
        """Capture session start parameters."""
        self.session_starts.append({"session_id": session_id})

    @hook_impl
    def after_session_end(self, session_id, duration):
        """Capture session end parameters."""
        self.session_ends.append({"session_id": session_id, "duration": duration})

    @hook_impl
    def before_node_execute(self, metadata, reporter=None):
        """Capture node execution parameters."""
        self.node_starts.append({"metadata": metadata})

    @hook_impl
    def after_node_execute(self, metadata, result, duration, reporter=None):
        """Capture node result parameters."""
        self.node_results.append({"metadata": metadata, "result": result, "duration": duration})

    @hook_impl
    def on_node_error(self, metadata, error, duration, reporter=None):
        """Capture node error parameters."""
        self.node_errors.append({"metadata": metadata, "error": error, "duration": duration})


class OrderTrackingPlugin:
    """
    Plugin that tracks the order of hook invocations.

    Usage:
        tracker = OrderTrackingPlugin()
        with session(plugins=[tracker]):
            my_task(1, 2)
        assert "before_node" in tracker.call_order[0]
    """

    def __init__(self):
        """Initialize empty call order list."""
        self.call_order = []

    @hook_impl
    def before_session_start(self, session_id):
        """Record before_session call."""
        self.call_order.append("before_session")

    @hook_impl
    def after_session_end(self, session_id, duration):
        """Record after_session call."""
        self.call_order.append("after_session")

    @hook_impl
    def before_node_execute(self, metadata, reporter=None):
        """Record before_node call."""
        self.call_order.append(f"before_node:{metadata.name}")

    @hook_impl
    def after_node_execute(self, metadata, result, duration, reporter=None):
        """Record after_node call."""
        self.call_order.append(f"after_node:{metadata.name}")

    @hook_impl
    def on_node_error(self, metadata, error, duration, reporter=None):
        """Record on_error call."""
        self.call_order.append(f"on_error:{metadata.name}")


class ErrorRaisingPlugin:
    """
    Plugin that raises errors in hooks for testing error handling.

    Usage:
        error_plugin = ErrorRaisingPlugin(raise_in="before_node")
    """

    def __init__(self, raise_in: str | None = None):
        """
        Initialize plugin with specified hook to raise errors in.

        Args:
            raise_in: Which hook should raise an error.
        """
        self.raise_in = raise_in

    @hook_impl
    def before_session_start(self, session_id):
        """Raise error if configured."""
        if self.raise_in == "before_session":
            raise RuntimeError("Test error in before_session_start")

    @hook_impl
    def after_session_end(self, session_id, duration):
        """Raise error if configured."""
        if self.raise_in == "after_session":
            raise RuntimeError("Test error in after_session_end")

    @hook_impl
    def before_node_execute(self, metadata, reporter=None):
        """Raise error if configured."""
        if self.raise_in == "before_node":
            raise RuntimeError("Test error in before_node_execute")

    @hook_impl
    def after_node_execute(self, metadata, result, duration, reporter=None):
        """Raise error if configured."""
        if self.raise_in == "after_node":
            raise RuntimeError("Test error in after_node_execute")

    @hook_impl
    def on_node_error(self, metadata, error, duration, reporter=None):
        """Raise error if configured."""
        if self.raise_in == "on_error":
            raise RuntimeError("Test error in on_node_error")


# --- Serialization Test Plugins ---


class SerializablePlugin:
    """
    Plugin that supports serialization via to_config/from_config.

    Usage:
        plugin = SerializablePlugin(threshold=0.7, name="test")
        config = plugin.to_config()
        restored = SerializablePlugin.from_config(config)
        assert restored.threshold == 0.7
    """

    def __init__(self, threshold: float = 0.5, name: str = "default"):
        self.threshold = threshold
        self.name = name
        self.call_count = 0  # Runtime state (not serialized)

    def to_config(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SerializablePlugin":
        return cls(**config)

    @hook_impl
    def before_node_execute(self, metadata, reporter=None):
        """Track hook calls (runtime state)."""
        self.call_count += 1


class AnotherSerializablePlugin:
    """Another serializable plugin for testing multiple plugins."""

    def __init__(self, multiplier: int = 2, enabled: bool = True):
        self.multiplier = multiplier
        self.enabled = enabled

    def to_config(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "enabled": self.enabled,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnotherSerializablePlugin":
        return cls(**config)

    @hook_impl
    def after_node_execute(self, metadata, result, duration, reporter=None):
        """Example hook implementation."""
        pass


class NonSerializablePlugin:
    """Plugin that does NOT support serialization."""

    @hook_impl
    def before_node_execute(self, metadata, reporter=None):
        """Example hook implementation."""
        pass
