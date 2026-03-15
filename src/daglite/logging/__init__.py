"""Default plugins shipped with daglite."""

from daglite.logging.core import get_logger
from daglite.logging.plugin import CentralizedLoggingPlugin
from daglite.logging.plugin import LifecycleLoggingPlugin

__all__ = [
    "CentralizedLoggingPlugin",
    "LifecycleLoggingPlugin",
    "get_logger",
]
