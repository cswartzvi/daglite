"""Default plugins shipped with daglite."""

from daglite.plugins.default.logging import CentralizedLoggingPlugin
from daglite.plugins.default.logging import get_logger

__all__ = [
    "CentralizedLoggingPlugin",
    "get_logger",
]
