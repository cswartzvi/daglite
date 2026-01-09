"""Default plugins shipped with daglite."""

from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
from daglite.plugins.builtin.logging import get_logger

__all__ = [
    "CentralizedLoggingPlugin",
    "get_logger",
]
