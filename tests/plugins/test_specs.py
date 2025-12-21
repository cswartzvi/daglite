"""
Unit tests for the pluggy hooks system.

These tests verify the plugin system's initialization and registration
mechanisms WITHOUT using evaluate() or evaluate_async(). They focus on:

- Plugin manager initialization
- Hook registration
- Hook validation

For integration tests with actual execution, see tests/evaluation/test_plugins.py.
"""

from daglite.plugins.manager import _get_global_plugin_manager


class TestPluginSystem:
    """Test basic plugin system functionality."""

    def test_hooks_initialize_on_import(self) -> None:
        """Hook manager is initialized when daglite is imported."""
        hook_manager = _get_global_plugin_manager()

        assert hook_manager is not None
        assert hook_manager.project_name == "daglite"
