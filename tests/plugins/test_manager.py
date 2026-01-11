"""
Unit tests for plugin manager functionality.

These tests verify the plugin manager's registration, serialization, and utility functions.

For integration tests with actual execution, see tests/evaluation/test_plugins.py.
"""

import pytest

from daglite.plugins.manager import _create_plugin_manager
from daglite.plugins.manager import _get_global_plugin_manager
from daglite.plugins.manager import _initialize_plugin_system
from daglite.plugins.manager import deserialize_plugin_manager
from daglite.plugins.manager import register_plugins
from daglite.plugins.manager import serialize_plugin_manager
from tests.examples.plugins import AnotherSerializablePlugin
from tests.examples.plugins import CounterPlugin
from tests.examples.plugins import NonSerializablePlugin
from tests.examples.plugins import SerializablePlugin


class TestPluginSystem:
    """Test basic plugin system functionality."""

    def test_hooks_initialize_on_import(self) -> None:
        """Hook manager is initialized when daglite is imported."""
        hook_manager = _get_global_plugin_manager()

        assert hook_manager is not None
        assert hook_manager.project_name == "daglite"


class TestHookManagerFunctions:
    """Test hook manager utility functions."""

    def test_initialize_hooks_idempotent(self) -> None:
        """initialize_hooks can be called multiple times safely."""
        # Already called on import, but should be safe to call again
        _initialize_plugin_system()
        _initialize_plugin_system()

        # Manager should still be accessible
        manager = _get_global_plugin_manager()
        assert manager is not None

    def test_get_hook_manager_returns_same_instance(self) -> None:
        """get_hook_manager returns the same global instance."""
        manager1 = _get_global_plugin_manager()
        manager2 = _get_global_plugin_manager()
        assert manager1 is manager2

    def test_register_hooks_validation(self) -> None:
        """register_hooks validates that hooks are instances."""
        # Should raise TypeError when passing class instead of instance
        with pytest.raises(
            TypeError, match="daglite expects plugins to be registered as instances"
        ):
            register_plugins(CounterPlugin)  # Missing ()

    def test_register_hooks_prevents_duplicates(self) -> None:
        """register_hooks doesn't register the same instance twice."""
        counter = CounterPlugin()
        hook_manager = _get_global_plugin_manager()

        # Get initial plugin count
        initial_count = len(hook_manager.get_plugins())

        # Register twice
        register_plugins(counter)
        after_first_count = len(hook_manager.get_plugins())

        register_plugins(counter)  # Should be no-op
        after_second_count = len(hook_manager.get_plugins())

        # Should only increase by 1, not 2
        assert after_first_count == initial_count + 1
        assert after_second_count == after_first_count

        # Cleanup
        hook_manager.unregister(counter)

    def test_create_plugin_manager_with_tracing_enabled(self) -> None:
        """Test plugin manager creation with tracing enabled."""
        from daglite.settings import DagliteSettings
        from daglite.settings import set_global_settings

        # Enable tracing in global settings
        settings = DagliteSettings(enable_plugin_tracing=True)
        set_global_settings(settings)

        # Create new manager
        manager = _create_plugin_manager()

        # Verify tracing is enabled (manager should have trace enabled)
        # Note: We can't directly test if tracing is working without side effects,
        # but we can verify the manager was created successfully
        assert manager is not None
        assert manager.project_name == "daglite"

        # Clean up
        set_global_settings(DagliteSettings())


class TestPluginSerialization:
    """Test plugin manager serialization and deserialization."""

    def test_serialize_empty_plugin_manager(self) -> None:
        """Serializing a plugin manager with no plugins returns empty dict."""
        manager = _create_plugin_manager()
        config = serialize_plugin_manager(manager)
        assert config == {}

    def test_serialize_plugin_manager_with_serializable_plugin(self) -> None:
        """Serializable plugins are included in serialization."""
        manager = _create_plugin_manager()
        plugin = SerializablePlugin(threshold=0.7, name="test")
        manager.register(plugin)

        config = serialize_plugin_manager(manager)

        # Should have one entry with fully qualified class name as key
        assert len(config) == 1
        key = f"{SerializablePlugin.__module__}.{SerializablePlugin.__qualname__}"
        assert key in config
        assert config[key] == {"threshold": 0.7, "name": "test"}

    def test_serialize_plugin_manager_skips_non_serializable(self) -> None:
        """Non-serializable plugins are not included in serialization."""
        manager = _create_plugin_manager()
        manager.register(NonSerializablePlugin())
        manager.register(SerializablePlugin(threshold=0.8, name="included"))

        config = serialize_plugin_manager(manager)

        # Only serializable plugin should be in config
        assert len(config) == 1
        key = f"{SerializablePlugin.__module__}.{SerializablePlugin.__qualname__}"
        assert key in config
        assert config[key] == {"threshold": 0.8, "name": "included"}

    def test_deserialize_plugin_manager_empty(self) -> None:
        """Deserializing empty config returns plugin manager with no plugins."""
        manager = deserialize_plugin_manager({})
        # Should have no user plugins (only hook specs are registered)
        plugins = [
            p
            for p in manager.get_plugins()
            if isinstance(p, (SerializablePlugin, AnotherSerializablePlugin))
        ]
        assert len(plugins) == 0

    def test_deserialize_plugin_manager_with_valid_plugin(self) -> None:
        """Valid plugin configs are deserialized correctly."""
        # Serialize then deserialize
        fqcn = f"{SerializablePlugin.__module__}.{SerializablePlugin.__qualname__}"
        config = {fqcn: {"threshold": 0.95, "name": "deserialized"}}

        manager = deserialize_plugin_manager(config)

        # Find our plugin
        plugins = [p for p in manager.get_plugins() if isinstance(p, SerializablePlugin)]
        assert len(plugins) == 1

        plugin = plugins[0]
        assert plugin.threshold == 0.95
        assert plugin.name == "deserialized"
        # Runtime state should be initialized fresh
        assert plugin.call_count == 0

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Serialization followed by deserialization preserves plugin state."""
        # Create original manager with plugin
        original_manager = _create_plugin_manager()
        original_plugin = SerializablePlugin(threshold=0.75, name="roundtrip")
        original_plugin.call_count = 42  # Runtime state
        original_manager.register(original_plugin)

        # Serialize
        config = serialize_plugin_manager(original_manager)

        # Deserialize
        new_manager = deserialize_plugin_manager(config)

        # Find deserialized plugin
        plugins = [p for p in new_manager.get_plugins() if isinstance(p, SerializablePlugin)]
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
        # NonSerializablePlugin doesn't have to_config/from_config
        fqcn = f"{NonSerializablePlugin.__module__}.{NonSerializablePlugin.__qualname__}"
        config = {fqcn: {}}  # type: ignore

        manager = deserialize_plugin_manager(config)

        # Should log warning about not being serializable
        assert "is not serializable" in caplog.text

        # Should return manager without the plugin
        plugins = [p for p in manager.get_plugins() if isinstance(p, NonSerializablePlugin)]
        assert len(plugins) == 0

    def test_serialize_multiple_plugins(self) -> None:
        """Multiple serializable plugins are all serialized."""
        manager = _create_plugin_manager()
        manager.register(SerializablePlugin(threshold=0.6, name="plugin1"))
        manager.register(AnotherSerializablePlugin(multiplier=5, enabled=False))

        config = serialize_plugin_manager(manager)

        assert len(config) == 2
        key_a = f"{SerializablePlugin.__module__}.{SerializablePlugin.__qualname__}"
        key_b = f"{AnotherSerializablePlugin.__module__}.{AnotherSerializablePlugin.__qualname__}"

        assert key_a in config
        assert key_b in config
        assert config[key_a] == {"threshold": 0.6, "name": "plugin1"}
        assert config[key_b] == {"multiplier": 5, "enabled": False}

    def test_deserialize_multiple_plugins(self) -> None:
        """Multiple plugins can be deserialized together."""
        key_a = f"{SerializablePlugin.__module__}.{SerializablePlugin.__qualname__}"
        key_b = f"{AnotherSerializablePlugin.__module__}.{AnotherSerializablePlugin.__qualname__}"

        config = {
            key_a: {"threshold": 0.3, "name": "multi1"},
            key_b: {"multiplier": 10, "enabled": True},
        }

        manager = deserialize_plugin_manager(config)

        # Find both plugins
        plugin_a = [p for p in manager.get_plugins() if isinstance(p, SerializablePlugin)]
        plugin_b = [p for p in manager.get_plugins() if isinstance(p, AnotherSerializablePlugin)]

        assert len(plugin_a) == 1
        assert len(plugin_b) == 1

        assert plugin_a[0].threshold == 0.3
        assert plugin_a[0].name == "multi1"
        assert plugin_b[0].multiplier == 10
        assert plugin_b[0].enabled is True
