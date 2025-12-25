"""Default plugins shipped with daglite."""

from daglite.plugins.default.logging import LoggingPlugin
from daglite.plugins.default.logging import get_logger

__all__ = [
    "LoggingPlugin",
    "get_logger",
]
