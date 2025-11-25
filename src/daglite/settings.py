from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from dataclasses import field

_GLOBAL_DAGLITE_SETTINGS: DagliteSettings | None = None
_SETTINGS_LOCK = threading.RLock()


@dataclass(frozen=True)
class DagliteSettings:
    """Configuration settings for daglite."""

    max_backend_threads: int = field(default_factory=lambda: min(32, (os.cpu_count() or 1) + 4))
    """
    Maximum number of threads to be used by the threading backend.

    Defaults to ThreadPoolExecutor's default: min(32, cpu_count + 4).
    """

    max_parallel_processes: int = field(default_factory=lambda: os.cpu_count() or 1)
    """
    Maximum number of parallel processes to be used by the process backend.

    Defaults to the number of CPU cores available.
    """


def get_global_settings() -> DagliteSettings:
    """
    Get the global daglite settings instance (thread-safe).

    If no global settings have been set, returns a default instance.
    """
    with _SETTINGS_LOCK:
        global _GLOBAL_DAGLITE_SETTINGS
        if _GLOBAL_DAGLITE_SETTINGS is None:
            _GLOBAL_DAGLITE_SETTINGS = DagliteSettings()
        return _GLOBAL_DAGLITE_SETTINGS


def set_global_settings(settings: DagliteSettings) -> None:
    """
    Set the global daglite settings instance (thread-safe).

    Warning: If pools have already been created, you must call
    `daglite.backends.local._reset_global_pools()` after changing settings
    to ensure new settings take effect. Best practice is to set settings
    once at application startup before any task execution.

    Args:
        settings (DagliteSettings): Settings to set as global.

    Example:
        >>> from daglite import set_global_settings, DagliteSettings
        >>> from daglite.backends.local import _reset_global_pools
        >>> set_global_settings(DagliteSettings(max_backend_threads=16))
        >>> _reset_global_pools()  # Force pools to be recreated with new settings
    """
    with _SETTINGS_LOCK:
        global _GLOBAL_DAGLITE_SETTINGS
        _GLOBAL_DAGLITE_SETTINGS = settings
