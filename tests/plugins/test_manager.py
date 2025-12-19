"""Tests for per-execution hooks and manager functionality."""

import pytest

from daglite import evaluate
from daglite import task
from daglite.engine import evaluate_async
from daglite.plugins import hooks
from daglite.plugins.manager import _get_global_plugin_manager
from daglite.plugins.manager import _initialize_plugin_system
from daglite.plugins.manager import deserialize_plugin_manager
from daglite.plugins.manager import register_plugins
from daglite.plugins.manager import register_plugins_entry_points
from daglite.plugins.manager import serialize_plugin_manager


# Module-level plugin classes for serialization tests
class SerializableTestPlugin:
    """Test plugin that supports serialization."""

    def __init__(self, threshold: float = 0.5, name: str = "default"):
        self.threshold = threshold
        self.name = name
        self.call_count = 0  # Runtime state, not serialized

    def to_config(self) -> dict:
        return {"threshold": self.threshold, "name": self.name}

    @classmethod
    def from_config(cls, config: dict) -> "SerializableTestPlugin":
        return cls(**config)

    @hooks.hook_impl
    def before_node_execute(self, node_id, node, backend, inputs):  # pragma: no cover
        self.call_count += 1


class AnotherSerializablePlugin:
    """Another test plugin for testing multiple plugins."""

    def __init__(self, multiplier: int = 2, enabled: bool = True):
        self.multiplier = multiplier
        self.enabled = enabled

    def to_config(self) -> dict:
        return {"multiplier": self.multiplier, "enabled": self.enabled}

    @classmethod
    def from_config(cls, config: dict) -> "AnotherSerializablePlugin":
        return cls(**config)

    @hooks.hook_impl
    def after_node_execute(self, node_id, node, backend, result, duration):  # pragma: no cover
        pass


class NonSerializableTestPlugin:
    """Test plugin that does NOT support serialization."""

    @hooks.hook_impl
    def before_node_execute(self, node_id, node, backend, inputs):  # pragma: no cover
        pass


class TestPerExecutionHooks:
    """Test per-execution hooks functionality."""

    def test_per_execution_hooks_basic(self) -> None:
        """Per-execution hooks work correctly."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3), hooks=[counter])
        assert result == 5
        assert counter.count == 1

    def test_per_execution_hooks_no_global_pollution(self) -> None:
        """Per-execution hooks don't affect global hook manager."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        @task
        def add(x: int, y: int) -> int:
            return x + y

        # First execution with counter1
        counter1 = Counter()
        result1 = evaluate(add.bind(x=2, y=3), hooks=[counter1])
        assert result1 == 5
        assert counter1.count == 1

        # Second execution with counter2
        counter2 = Counter()
        result2 = evaluate(add.bind(x=5, y=7), hooks=[counter2])
        assert result2 == 12
        assert counter1.count == 1  # Unchanged
        assert counter2.count == 1

    def test_per_execution_hooks_combine_with_global(self) -> None:
        """Per-execution hooks combine with globally registered hooks."""

        class GlobalCounter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        class LocalCounter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        global_counter = GlobalCounter()
        local_counter = LocalCounter()

        # Register global hook
        register_plugins(global_counter)

        @task
        def add(x: int, y: int) -> int:
            return x + y

        # Execute with local hook - both should fire
        result = evaluate(add.bind(x=2, y=3), hooks=[local_counter])
        assert result == 5
        assert global_counter.count == 1
        assert local_counter.count == 1

        # Execute without local hook - only global should fire
        result2 = evaluate(add.bind(x=5, y=7))
        assert result2 == 12
        assert global_counter.count == 2  # Incremented
        assert local_counter.count == 1  # Unchanged

        # Cleanup
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(global_counter)

    def test_per_execution_hooks_multiple(self) -> None:
        """Multiple per-execution hooks can be registered."""

        class Counter1:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        class Counter2:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter1 = Counter1()
        counter2 = Counter2()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3), hooks=[counter1, counter2])
        assert result == 5
        assert counter1.count == 1
        assert counter2.count == 1

    def test_per_execution_hooks_async(self) -> None:
        """Per-execution hooks work with evaluate_async."""
        import asyncio

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        async def run():
            return await evaluate_async(add.bind(x=2, y=3), hooks=[counter])

        result = asyncio.run(run())
        assert result == 5
        assert counter.count == 1

    def test_per_execution_hooks_error_propagates(self) -> None:
        """Errors in user code still propagate correctly with per-execution hooks."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        @task
        def failing_task() -> int:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            evaluate(failing_task.bind(), hooks=[counter])

        # Hook should have been called before the error
        assert counter.count == 1

    def test_per_execution_hooks_validates_instances(self) -> None:
        """Per-execution hooks must be instances, not classes."""

        class MyHook:
            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):  # pragma: no cover
                pass

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        # Passing class instead of instance should raise TypeError
        with pytest.raises(TypeError, match="daglite expects hooks to be registered as instances"):
            evaluate(add.bind(x=2, y=3), hooks=[MyHook])  # Missing ()


class TestRegisterHooksEntryPoints:
    """Test entry point loading functionality."""

    def test_register_hooks_entry_points_exists(self) -> None:
        """register_hooks_entry_points function exists and is callable."""
        # This just verifies the function exists and can be called
        # without error (even if no entry points are defined)
        register_plugins_entry_points()

    def test_register_hooks_entry_points_no_crash_on_missing(self) -> None:
        """Entry point loading doesn't crash when no plugins are installed."""
        # Should gracefully handle no entry points being defined
        register_plugins_entry_points()

        # Should still be able to evaluate normally
        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5


class TestHookManagerFunctions:
    """Test hook manager utility functions."""

    def test_initialize_hooks_idempotent(self) -> None:
        """initialize_hooks can be called multiple times safely."""
        # Already called on import, but should be safe to call again
        _initialize_plugin_system()
        _initialize_plugin_system()

        # Should still work
        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5

    def test_get_hook_manager_returns_same_instance(self) -> None:
        """get_hook_manager returns the same global instance."""
        manager1 = _get_global_plugin_manager()
        manager2 = _get_global_plugin_manager()
        assert manager1 is manager2

    def test_register_hooks_validation(self) -> None:
        """register_hooks validates that hooks are instances."""

        class MyHook:
            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):  # pragma: no cover
                pass

        # Should raise TypeError when passing class instead of instance
        with pytest.raises(TypeError, match="daglite expects hooks to be registered as instances"):
            register_plugins(MyHook)  # Missing ()

    def test_register_hooks_prevents_duplicates(self) -> None:
        """register_hooks doesn't register the same instance twice."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        # Register twice
        register_plugins(counter)
        register_plugins(counter)  # Should be no-op

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5
        # Should only increment once (not registered twice)
        assert counter.count == 1

        # Cleanup
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(counter)


class TestPluginSerialization:
    """Test plugin manager serialization and deserialization."""

    def test_serialize_empty_plugin_manager(self) -> None:
        """Serializing a plugin manager with no plugins returns empty dict."""
        from daglite.plugins.manager import _create_plugin_manager

        manager = _create_plugin_manager()
        config = serialize_plugin_manager(manager)
        assert config == {}

    def test_serialize_plugin_manager_with_serializable_plugin(self) -> None:
        """Serializable plugins are included in serialization."""
        from daglite.plugins.manager import _create_plugin_manager

        manager = _create_plugin_manager()
        plugin = SerializableTestPlugin(threshold=0.7, name="test")
        manager.register(plugin)

        config = serialize_plugin_manager(manager)

        # Should have one entry with fully qualified class name as key
        assert len(config) == 1
        key = f"{SerializableTestPlugin.__module__}.{SerializableTestPlugin.__qualname__}"
        assert key in config
        assert config[key] == {"threshold": 0.7, "name": "test"}

    def test_serialize_plugin_manager_skips_non_serializable(self) -> None:
        """Non-serializable plugins are not included in serialization."""
        from daglite.plugins.manager import _create_plugin_manager

        manager = _create_plugin_manager()
        manager.register(NonSerializableTestPlugin())
        manager.register(SerializableTestPlugin(threshold=0.8, name="included"))

        config = serialize_plugin_manager(manager)

        # Only serializable plugin should be in config
        assert len(config) == 1
        key = f"{SerializableTestPlugin.__module__}.{SerializableTestPlugin.__qualname__}"
        assert key in config
        assert config[key] == {"threshold": 0.8, "name": "included"}

    def test_deserialize_plugin_manager_empty(self) -> None:
        """Deserializing empty config returns plugin manager with no plugins."""
        manager = deserialize_plugin_manager({})
        # Should have no user plugins (only hook specs are registered)
        plugins = [
            p
            for p in manager.get_plugins()
            if isinstance(p, (SerializableTestPlugin, AnotherSerializablePlugin))
        ]
        assert len(plugins) == 0

    def test_deserialize_plugin_manager_with_valid_plugin(self) -> None:
        """Valid plugin configs are deserialized correctly."""
        # Serialize then deserialize
        fqcn = f"{SerializableTestPlugin.__module__}.{SerializableTestPlugin.__qualname__}"
        config = {fqcn: {"threshold": 0.95, "name": "deserialized"}}

        manager = deserialize_plugin_manager(config)

        # Find our plugin
        plugins = [p for p in manager.get_plugins() if isinstance(p, SerializableTestPlugin)]
        assert len(plugins) == 1

        plugin = plugins[0]
        assert plugin.threshold == 0.95
        assert plugin.name == "deserialized"
        # Runtime state should be initialized fresh
        assert plugin.call_count == 0

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Serialization followed by deserialization preserves plugin state."""
        from daglite.plugins.manager import _create_plugin_manager

        # Create original manager with plugin
        original_manager = _create_plugin_manager()
        original_plugin = SerializableTestPlugin(threshold=0.75, name="roundtrip")
        original_plugin.call_count = 42  # Runtime state
        original_manager.register(original_plugin)

        # Serialize
        config = serialize_plugin_manager(original_manager)

        # Deserialize
        new_manager = deserialize_plugin_manager(config)

        # Find deserialized plugin
        plugins = [p for p in new_manager.get_plugins() if isinstance(p, SerializableTestPlugin)]
        assert len(plugins) == 1

        new_plugin = plugins[0]
        # Config values should be preserved
        assert new_plugin.threshold == 0.75
        assert new_plugin.name == "roundtrip"
        # Runtime state should be reset
        assert new_plugin.call_count == 0

    def test_deserialize_with_invalid_class_path_logs_warning(self, caplog) -> None:
        """Invalid class paths are logged and skipped."""
        config = {"nonexistent.module.ClassName": {"value": 42}}

        manager = deserialize_plugin_manager(config)

        # Should log warning
        assert "Could not resolve plugin class" in caplog.text
        assert "nonexistent.module.ClassName" in caplog.text

        # Should return manager without the plugin
        plugins = [p for p in manager.get_plugins() if hasattr(p, "value")]
        assert len(plugins) == 0

    def test_deserialize_with_non_serializable_class_logs_warning(self, caplog) -> None:
        """Classes without from_config are logged and skipped."""
        # NonSerializableTestPlugin doesn't have to_config/from_config
        fqcn = f"{NonSerializableTestPlugin.__module__}.{NonSerializableTestPlugin.__qualname__}"
        config = {fqcn: {}}  # type: ignore

        manager = deserialize_plugin_manager(config)

        # Should log warning about not being serializable
        assert "is not serializable" in caplog.text

        # Should return manager without the plugin
        plugins = [p for p in manager.get_plugins() if isinstance(p, NonSerializableTestPlugin)]
        assert len(plugins) == 0

    def test_serialize_multiple_plugins(self) -> None:
        """Multiple serializable plugins are all serialized."""
        from daglite.plugins.manager import _create_plugin_manager

        manager = _create_plugin_manager()
        manager.register(SerializableTestPlugin(threshold=0.6, name="plugin1"))
        manager.register(AnotherSerializablePlugin(multiplier=5, enabled=False))

        config = serialize_plugin_manager(manager)

        assert len(config) == 2
        key_a = f"{SerializableTestPlugin.__module__}.{SerializableTestPlugin.__qualname__}"
        key_b = (
            f"{AnotherSerializablePlugin.__module__}.{AnotherSerializablePlugin.__qualname__}"
        )

        assert key_a in config
        assert key_b in config
        assert config[key_a] == {"threshold": 0.6, "name": "plugin1"}
        assert config[key_b] == {"multiplier": 5, "enabled": False}

    def test_deserialize_multiple_plugins(self) -> None:
        """Multiple plugins can be deserialized together."""
        key_a = f"{SerializableTestPlugin.__module__}.{SerializableTestPlugin.__qualname__}"
        key_b = (
            f"{AnotherSerializablePlugin.__module__}.{AnotherSerializablePlugin.__qualname__}"
        )

        config = {
            key_a: {"threshold": 0.3, "name": "multi1"},
            key_b: {"multiplier": 10, "enabled": True},
        }

        manager = deserialize_plugin_manager(config)

        # Find both plugins
        plugin_a = [p for p in manager.get_plugins() if isinstance(p, SerializableTestPlugin)]
        plugin_b = [
            p for p in manager.get_plugins() if isinstance(p, AnotherSerializablePlugin)
        ]

        assert len(plugin_a) == 1
        assert len(plugin_b) == 1

        assert plugin_a[0].threshold == 0.3
        assert plugin_a[0].name == "multi1"
        assert plugin_b[0].multiplier == 10
        assert plugin_b[0].enabled is True

