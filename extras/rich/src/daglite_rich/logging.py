import logging
from pathlib import Path
from typing import Any

import yaml
from typing_extensions import override

from daglite.plugins.default.logging import LifecycleLoggingPlugin


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

    @override
    def _load_default_config(self) -> dict[str, Any] | None:
        """Load rich-specific logging configuration from logging.yml."""
        config_path = Path(__file__).parent / "logging.yml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return None
