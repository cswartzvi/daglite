from __future__ import annotations

from dataclasses import dataclass

_GLOBAL_DAGLITE_SETTINGS: DagliteSettings | None = None


@dataclass(frozen=True)
class DagliteSettings:
    """Configuration settings for daglite."""

    max_backend_threads: int | None = None
    """
    Maximum number of threads to be used by the the threading backend.

    If None, defaults to ThreadPoolExecutor's default.
    """

    max_parallel_processes: int | None = None
    """
    Maximum number of parallel processes to be used by the process backend.

    If None, defaults to the number of CPU cores available.
    """


def get_global_settings() -> DagliteSettings:
    """
    Get the global daglite settings instance.

    If no global settings have been set, returns a default instance.
    """
    global _GLOBAL_DAGLITE_SETTINGS
    if _GLOBAL_DAGLITE_SETTINGS is None:
        _GLOBAL_DAGLITE_SETTINGS = DagliteSettings()
    return _GLOBAL_DAGLITE_SETTINGS


def set_global_settings(settings: DagliteSettings) -> None:
    """
    Set the global daglite settings instance.

    Args:
        settings (DagliteSettings): Settings to set as global.
    """
    global _GLOBAL_DAGLITE_SETTINGS
    _GLOBAL_DAGLITE_SETTINGS = settings
