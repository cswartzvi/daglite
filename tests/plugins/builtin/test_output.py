"""Unit tests for OutputPlugin.

Tests in this file should NOT use evaluate(). Evaluation tests are in tests/evaluation/.
"""

import tempfile
from unittest.mock import Mock

import pytest

from daglite.graph.base import OutputConfig
from daglite.outputs.store import FileOutputStore
from daglite.plugins.builtin.output import OutputPlugin


class TestOutputPluginInit:
    """Tests for OutputPlugin initialization."""

    def test_init_with_none(self):
        """Test initialization with no store."""
        plugin = OutputPlugin(store=None)
        assert plugin.store is None

    def test_init_with_output_store(self):
        """Test initialization with OutputStore instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)
            assert plugin.store is store

    def test_init_with_string_converts_to_file_output_store(self):
        """Test that string path is converted to FileOutputStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = OutputPlugin(store=tmpdir)
            assert isinstance(plugin.store, FileOutputStore)
            assert plugin.store.base_path == tmpdir


class TestOutputPluginAfterNodeExecute:
    """Tests for OutputPlugin.after_node_execute hook."""

    def test_no_output_config_does_nothing(self):
        """Test that empty outputs list is handled gracefully."""
        plugin = OutputPlugin()
        metadata = Mock()
        metadata.name = "test_task"

        # Should not raise even though no store configured
        plugin.after_node_execute(
            metadata=metadata,
            inputs={},
            result=42,
            output_config=(),
            output_extras=[],
            duration=0.1,
            reporter=None,
        )

    def test_output_saved_to_explicit_store(self):
        """Test that output is saved to explicit store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin()

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "123"

            config = OutputConfig(key="result", store=store)
            plugin.after_node_execute(
                metadata=metadata,
                inputs={},
                result=42,
                output_config=(config,),
                output_extras=[{}],
                duration=0.1,
                reporter=None,
            )

            assert store.load("result", int) == 42

    def test_output_saved_to_plugin_default_store(self):
        """Test that output falls back to plugin's default store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "123"

            config = OutputConfig(key="result")
            plugin.after_node_execute(
                metadata=metadata,
                inputs={},
                result=42,
                output_config=(config,),
                output_extras=[{}],
                duration=0.1,
                reporter=None,
            )

            assert store.load("result", int) == 42

    def test_no_store_configured_raises_error(self):
        """Test that missing store raises clear error."""
        plugin = OutputPlugin()

        metadata = Mock()
        metadata.name = "test_task"

        config = OutputConfig(key="result")
        with pytest.raises(ValueError, match="No output store configured"):
            plugin.after_node_execute(
                metadata=metadata,
                inputs={},
                result=42,
                output_config=(config,),
                output_extras=[{}],
                duration=0.1,
                reporter=None,
            )

    def test_key_formatting_with_inputs(self):
        """Test that output keys are formatted with task inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "123"

            config = OutputConfig(key="output_{x}_{y}")
            plugin.after_node_execute(
                metadata=metadata,
                inputs={"x": 5, "y": 10},
                result=15,
                output_config=(config,),
                output_extras=[{}],
                duration=0.1,
                reporter=None,
            )

            assert store.load("output_5_10", int) == 15

    def test_invalid_key_format_raises_error(self):
        """Test that invalid format string raises clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)

            metadata = Mock()
            metadata.name = "test_task"

            config = OutputConfig(key="output_{missing}")
            with pytest.raises(ValueError, match="references parameter"):
                plugin.after_node_execute(
                    metadata=metadata,
                    inputs={"x": 5},
                    result=15,
                    output_config=(config,),
                    output_extras=[{}],
                    duration=0.1,
                    reporter=None,
                )

    def test_extras_with_literal_values(self):
        """Test that literal extra values are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "123"

            # Extras are now already resolved
            config = OutputConfig(key="result")
            plugin.after_node_execute(
                metadata=metadata,
                inputs={},
                result=42,
                output_config=(config,),
                output_extras=[{"version": "1.0", "author": "test"}],
                duration=0.1,
                reporter=None,
            )

    def test_extras_with_task_parameters(self):
        """Test that extras referencing task parameters are resolved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "123"

            # Extras are now already resolved from task parameters
            config = OutputConfig(key="result")
            plugin.after_node_execute(
                metadata=metadata,
                inputs={"version": "v1.0"},
                result=42,
                output_config=(config,),
                output_extras=[{"version": "v1.0"}],  # Already resolved
                duration=0.1,
                reporter=None,
            )

    def test_reporter_receives_notification(self):
        """Test that reporter is notified when output is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            plugin = OutputPlugin(store=store)

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "test-id-123"

            mock_reporter = Mock()

            config = OutputConfig(key="result", name="checkpoint1")
            plugin.after_node_execute(
                metadata=metadata,
                inputs={},
                result=42,
                output_config=(config,),
                output_extras=[{}],
                duration=0.1,
                reporter=mock_reporter,
            )

            # Verify reporter was called
            mock_reporter.report.assert_called_once()
            call_args = mock_reporter.report.call_args
            assert call_args[0][0] == "output_saved"
            event_data = call_args[0][1]
            assert event_data["key"] == "result"
            assert event_data["checkpoint_name"] == "checkpoint1"
            assert event_data["node_id"] == "test-id-123"
            assert event_data["node_name"] == "test_task"

    @pytest.mark.parametrize(
        "config_store,plugin_store,should_succeed",
        [
            (None, None, False),  # No store anywhere
            ("explicit", None, True),  # Only explicit
            (None, "plugin", True),  # Only plugin default
            ("explicit", "plugin", True),  # Both (explicit wins)
        ],
    )
    def test_store_priority(self, config_store, plugin_store, should_succeed):
        """Test store resolution priority."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            stores = {"explicit": FileOutputStore(tmpdir1), "plugin": FileOutputStore(tmpdir2)}

            plugin = OutputPlugin(store=stores.get(plugin_store))

            metadata = Mock()
            metadata.name = "test_task"
            metadata.id = "123"

            if should_succeed:
                config = OutputConfig(key="result", store=stores.get(config_store))
                plugin.after_node_execute(
                    metadata=metadata,
                    inputs={},
                    result=42,
                    output_config=(config,),
                    output_extras=[{}],
                    duration=0.1,
                    reporter=None,
                )

                # Verify saved to correct store
                if config_store:
                    assert stores["explicit"].load("result", int) == 42
                else:
                    assert stores["plugin"].load("result", int) == 42
            else:
                config = OutputConfig(key="result")
                with pytest.raises(ValueError, match="No output store configured"):
                    plugin.after_node_execute(
                        metadata=metadata,
                        inputs={},
                        result=42,
                        output_config=(config,),
                        output_extras=[{}],
                        duration=0.1,
                        reporter=None,
                    )
