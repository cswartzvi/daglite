"""Unit tests for RichLifecycleLoggingPlugin."""

import logging
from unittest.mock import mock_open
from unittest.mock import patch

from daglite_rich.logging import RichLifecycleLoggingPlugin


class TestRichLifecycleLoggingPlugin:
    """Unit tests for RichLifecycleLoggingPlugin."""

    def test_initialization_defaults(self):
        """Test plugin initialization with default parameters."""
        plugin = RichLifecycleLoggingPlugin()

        assert plugin._logger is not None
        # Logger is initialized (level depends on YAML config)

    def test_initialization_with_custom_level(self):
        """Test plugin initialization with custom log level."""
        plugin = RichLifecycleLoggingPlugin(level=logging.DEBUG)

        assert plugin._logger.logger.level == logging.DEBUG

    def test_initialization_with_custom_name(self):
        """Test plugin initialization with custom logger name."""
        plugin = RichLifecycleLoggingPlugin(name="custom.logger")

        assert "custom.logger" in plugin._logger.logger.name

    def test_initialization_with_custom_config(self):
        """Test plugin initialization with custom config dict."""
        custom_config = {
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["console"],
            },
        }

        plugin = RichLifecycleLoggingPlugin(config=custom_config)

        # Config should have been applied
        assert plugin._logger is not None

    def test_loads_rich_yaml_config(self):
        """Test that plugin loads rich-specific logging.yml configuration."""
        # Mock the YAML file
        yaml_content = """
version: 1
formatters:
  rich:
    format: "%(message)s"
handlers:
  rich:
    class: rich.logging.RichHandler
    level: INFO
    formatter: rich
loggers:
  daglite.lifecycle:
    level: INFO
    handlers:
      - rich
    propagate: false
"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                plugin = RichLifecycleLoggingPlugin()

                # Plugin should be initialized with loaded config
                assert plugin._logger is not None

    def test_yaml_config_not_found(self):
        """Test plugin behavior when logging.yml doesn't exist."""
        # When yaml file doesn't exist, it should work with base class default config
        plugin = RichLifecycleLoggingPlugin()

        # Should still initialize successfully
        assert plugin._logger is not None

    def test_serialization(self):
        """Test plugin serialization to/from config."""
        plugin = RichLifecycleLoggingPlugin(level=logging.WARNING)

        # Serialize to config
        config = plugin.to_config()

        # Deserialize from config
        restored_plugin = RichLifecycleLoggingPlugin.from_config(config)

        assert restored_plugin is not None
        assert isinstance(restored_plugin, RichLifecycleLoggingPlugin)

    def test_inherits_from_lifecycle_logging_plugin(self):
        """Test that RichLifecycleLoggingPlugin properly inherits from LifecycleLoggingPlugin."""
        from daglite.plugins.default.logging import LifecycleLoggingPlugin

        plugin = RichLifecycleLoggingPlugin()

        assert isinstance(plugin, LifecycleLoggingPlugin)
        # Should have all the lifecycle logging methods
        assert hasattr(plugin, "before_graph_execute")
        assert hasattr(plugin, "after_graph_execute")
        assert hasattr(plugin, "before_node_execute")
        assert hasattr(plugin, "after_node_execute")
