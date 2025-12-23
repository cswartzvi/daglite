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

from daglite.plugins import hook_impl


class CounterPlugin:
    """
    Simple plugin that counts hook invocations.

    This is the most commonly used test plugin. It tracks how many times
    specific hooks are called during execution.

    Usage:
        counter = CounterPlugin()
        result = evaluate(task(...), plugins=[counter])
        assert counter.before_node_count == 1
    """

    def __init__(self):
        """Initialize all counters to zero."""
        self.before_graph_count = 0
        self.after_graph_count = 0
        self.before_node_count = 0
        self.after_node_count = 0
        self.on_error_count = 0

    @hook_impl
    def before_graph_execute(self, root_id, node_count, is_async):
        """Count before_graph_execute invocations."""
        self.before_graph_count += 1

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        """Count after_graph_execute invocations."""
        self.after_graph_count += 1

    @hook_impl
    def before_node_execute(self, key, metadata, inputs):
        """Count before_node_execute invocations."""
        self.before_node_count += 1

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        """Count after_node_execute invocations."""
        self.after_node_count += 1

    @hook_impl
    def on_node_error(self, key, metadata, inputs, error, duration, reporter=None):
        """Count on_node_error invocations."""
        self.on_error_count += 1


class ParameterCapturePlugin:
    """
    Plugin that captures all hook parameters for validation.

    Use this plugin when you need to verify that hooks receive correct
    parameters during execution. It stores all captured data in lists
    for post-execution assertions.

    Usage:
        capture = ParameterCapturePlugin()
        result = evaluate(task(...), plugins=[capture])
        assert capture.graph_starts[0]["node_count"] == 1
        assert capture.node_executions[0]["inputs"] == {"x": 2, "y": 3}
    """

    def __init__(self):
        """Initialize empty capture lists."""
        self.graph_starts = []
        self.graph_ends = []
        self.node_executions = []
        self.node_results = []
        self.node_errors = []

    @hook_impl
    def before_graph_execute(self, root_id, node_count, is_async):
        """Capture graph start parameters."""
        self.graph_starts.append(
            {
                "root_id": root_id,
                "node_count": node_count,
                "is_async": is_async,
            }
        )

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        """Capture graph end parameters."""
        self.graph_ends.append(
            {
                "root_id": root_id,
                "result": result,
                "duration": duration,
                "is_async": is_async,
            }
        )

    @hook_impl
    def before_node_execute(self, key, metadata, inputs):
        """Capture node execution parameters."""
        self.node_executions.append(
            {
                "key": key,
                "metadata": metadata,
                "inputs": inputs,
            }
        )

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        """Capture node result parameters."""
        self.node_results.append(
            {
                "key": key,
                "metadata": metadata,
                "inputs": inputs,
                "result": result,
                "duration": duration,
            }
        )

    @hook_impl
    def on_node_error(self, key, metadata, inputs, error, duration, reporter=None):
        """Capture node error parameters."""
        self.node_errors.append(
            {
                "key": key,
                "metadata": metadata,
                "inputs": inputs,
                "error": error,
                "duration": duration,
            }
        )


class OrderTrackingPlugin:
    """
    Plugin that tracks the order of hook invocations.

    Use this plugin when you need to verify that hooks are called in the
    correct order (e.g., before_graph -> before_node -> after_node -> after_graph).

    Usage:
        tracker = OrderTrackingPlugin()
        result = evaluate(task(...), plugins=[tracker])
        assert tracker.call_order == [
            "before_graph",
            "before_node",
            "after_node",
            "after_graph"
        ]
    """

    def __init__(self):
        """Initialize empty call order list."""
        self.call_order = []

    @hook_impl
    def before_graph_execute(self, root_id, node_count, is_async):
        """Record before_graph call."""
        self.call_order.append("before_graph")

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        """Record after_graph call."""
        self.call_order.append("after_graph")

    @hook_impl
    def before_node_execute(self, key, metadata, inputs):
        """Record before_node call."""
        self.call_order.append(f"before_node:{key}")

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        """Record after_node call."""
        self.call_order.append(f"after_node:{key}")

    @hook_impl
    def on_node_error(self, key, metadata, inputs, error, duration, reporter=None):
        """Record on_error call."""
        self.call_order.append(f"on_error:{key}")


class ErrorRaisingPlugin:
    """
    Plugin that raises errors in hooks for testing error handling.

    Use this plugin to test that the system correctly handles plugin errors
    without breaking execution.

    Usage:
        error_plugin = ErrorRaisingPlugin(raise_in="before_node")
        result = evaluate(task(...), plugins=[error_plugin])
        # Should handle error gracefully
    """

    def __init__(self, raise_in: str | None = None):
        """
        Initialize plugin with specified hook to raise errors in.

        Args:
            raise_in: Which hook should raise an error. One of:
                - "before_graph"
                - "after_graph"
                - "before_node"
                - "after_node"
                - "on_error"
                - None (no errors)
        """
        self.raise_in = raise_in

    @hook_impl
    def before_graph_execute(self, root_id, node_count, is_async):
        """Raise error if configured."""
        if self.raise_in == "before_graph":
            raise RuntimeError("Test error in before_graph_execute")

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        """Raise error if configured."""
        if self.raise_in == "after_graph":
            raise RuntimeError("Test error in after_graph_execute")

    @hook_impl
    def before_node_execute(self, key, metadata, inputs):
        """Raise error if configured."""
        if self.raise_in == "before_node":
            raise RuntimeError("Test error in before_node_execute")

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        """Raise error if configured."""
        if self.raise_in == "after_node":
            raise RuntimeError("Test error in after_node_execute")

    @hook_impl
    def on_node_error(self, key, metadata, inputs, error, duration, reporter=None):
        """Raise error if configured."""
        if self.raise_in == "on_error":
            raise RuntimeError("Test error in on_node_error")


# --- Serialization Test Plugins ---


class SerializablePlugin:
    """
    Plugin that supports serialization via to_config/from_config.

    Use this plugin to test plugin serialization and deserialization.
    This pattern follows the daglite serialization protocol.

    Usage:
        plugin = SerializablePlugin(threshold=0.7, name="test")
        config = plugin.to_config()
        restored = SerializablePlugin.from_config(config)
        assert restored.threshold == 0.7
    """

    def __init__(self, threshold: float = 0.5, name: str = "default"):
        """
        Initialize serializable plugin.

        Args:
            threshold: Example configuration parameter
            name: Example string parameter
        """
        self.threshold = threshold
        self.name = name
        self.call_count = 0  # Runtime state (not serialized)

    def to_config(self) -> dict[str, Any]:
        """
        Serialize plugin configuration.

        Returns:
            Dictionary with serializable configuration parameters.
            Note: call_count is NOT included (runtime state).
        """
        return {
            "threshold": self.threshold,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SerializablePlugin":
        """
        Deserialize plugin from configuration.

        Args:
            config: Configuration dictionary from to_config()

        Returns:
            New plugin instance with restored configuration.
        """
        return cls(**config)

    @hook_impl
    def before_node_execute(self, key, metadata, inputs):
        """Track hook calls (runtime state)."""
        self.call_count += 1


class AnotherSerializablePlugin:
    """
    Another serializable plugin for testing multiple plugins.

    Use this when you need to test systems with multiple different plugins
    registered simultaneously.
    """

    def __init__(self, multiplier: int = 2, enabled: bool = True):
        """
        Initialize plugin with configuration.

        Args:
            multiplier: Example integer parameter
            enabled: Example boolean parameter
        """
        self.multiplier = multiplier
        self.enabled = enabled

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration."""
        return {
            "multiplier": self.multiplier,
            "enabled": self.enabled,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnotherSerializablePlugin":
        """Deserialize from configuration."""
        return cls(**config)

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        """Example hook implementation."""
        pass


class NonSerializablePlugin:
    """
    Plugin that does NOT support serialization.

    Use this plugin to test that the system correctly handles plugins
    without serialization support (should skip them during serialization).

    This plugin intentionally does not implement to_config/from_config.
    """

    @hook_impl
    def before_node_execute(self, key, metadata, inputs):
        """Example hook implementation."""
        pass
