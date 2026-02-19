from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from dataclasses import field

from daglite.datasets.store import DatasetStore

_GLOBAL_DAGLITE_SETTINGS: DagliteSettings | None = None
_SETTINGS_LOCK = threading.RLock()


@dataclass(frozen=True)
class DagliteSettings:
    """Configuration settings for daglite."""

    default_backend: str = "inline"
    """Default backend to use for task execution when none is specified."""

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

    max_timeout_workers: int = 4
    """
    Maximum number of worker threads for enforcing task timeouts per backend.

    Each local backend (Inline, threading, multiprocessing) gets its own dedicated
    thread pool of this size for timeout enforcement. Increase this value if you have a
    large number of concurrent tasks with timeouts to avoid delays in timeout enforcement.
    """

    enable_plugin_tracing: bool = field(
        default_factory=lambda: _env_get_bool("DAGLITE_TRACE_HOOKS", False)
    )
    """
    Enable detailed tracing of plugin hook calls.

    When enabled, all plugin hook invocations are logged at DEBUG level, showing hook names,
    arguments, and return values. Useful for debugging plugin behavior but can be verbose.

    Can be set via DAGLITE_TRACE_HOOKS environment variable (1/true/yes to enable).
    """

    enable_graph_optimization: bool = field(
        default_factory=lambda: _env_get_bool("DAGLITE_ENABLE_GRAPH_OPTIMIZATION", True)
    )
    """
    Enable graph optimization passes (e.g., composite node folding).

    When enabled, linear chains of nodes sharing the same backend are folded into composite nodes
    for reduced submission overhead. Disable for debugging or when exact per-node scheduling
    control is needed.

    Can be set via DAGLITE_ENABLE_GRAPH_OPTIMIZATION environment variable (1/true/yes to enable,
    0/false/no to disable). Defaults to True.
    """

    datastore_store: str | DatasetStore = field(
        default_factory=lambda: os.getenv("DAGLITE_DATASTORE_STORE") or "."
    )
    """
    Default dataset store for saving and loading datasets.

    Can be a string path or a DatasetStore instance.
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

    Args:
        settings: Settings to set as global.

    Examples:
        >>> from daglite.settings import set_global_settings, DagliteSettings
        >>> set_global_settings(DagliteSettings(max_backend_threads=16))
    """
    with _SETTINGS_LOCK:
        global _GLOBAL_DAGLITE_SETTINGS
        _GLOBAL_DAGLITE_SETTINGS = settings


def _env_get_bool(var_name: str, default: bool = False) -> bool:
    """Helper to read a boolean environment variable."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")
