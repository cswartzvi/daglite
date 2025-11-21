from __future__ import annotations

from dataclasses import dataclass

_GLOBAL_DAGLITE_SETTINGS: DagLiteSettings | None = None


@dataclass(frozen=True)
class DagLiteSettings:
    """Configuration settings for DAGLite."""

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


def get_global_settings() -> DagLiteSettings:
    """
    Get the global DagLite settings instance.

    If no global settings have been set, returns a default instance.
    """
    global _GLOBAL_DAGLITE_SETTINGS
    if _GLOBAL_DAGLITE_SETTINGS is None:
        _GLOBAL_DAGLITE_SETTINGS = DagLiteSettings()
    return _GLOBAL_DAGLITE_SETTINGS


def set_global_settings(settings: DagLiteSettings) -> None:
    """
    Set the global DagLite settings instance.

    Args:
        settings (DagLiteSettings): Settings to set as global.
    """
    global _GLOBAL_DAGLITE_SETTINGS
    _GLOBAL_DAGLITE_SETTINGS = settings
