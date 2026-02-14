import logging
from pathlib import Path
from typing import Any

from typing_extensions import override

from daglite.plugins.builtin.logging import LifecycleLoggingPlugin


class RichLifecycleLoggingPlugin(LifecycleLoggingPlugin):
    """
    Extension of LifecycleLoggingPlugin that adds rich logging capabilities.

    This plugin enhances the default lifecycle logging by integrating with the rich
    library to provide better formatted and more visually appealing log outputs.

    Args:
        name: Optional logger name to use (default: "daglite").
        level: Optional minimum log level to handle on coordinator side (default: INFO).
        config: Optional logging config dict. If not provided, loads rich-specific config.
    """

    def __init__(
        self,
        name: str | None = None,
        level: int = logging.INFO,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name=name, level=level, config=config)

    @property
    @override
    def _config_path(self) -> Path:
        """Path to the rich-specific logging configuration file."""
        return Path(__file__).parent / "logging.json"
