"""Daglite: Lightweight Python framework for building eager task pipelines."""

__version__ = "0.8.0"

import daglite.datasets.builtin  # noqa: F401  ensure builtins are registered
from daglite import backends
from daglite import settings
from daglite.eager import eager_task as task
from daglite.parallel import async_map
from daglite.parallel import parallel_map
from daglite.plugins.manager import _initialize_plugin_system
from daglite.session import async_session
from daglite.session import session
from daglite.workflows import workflow

# Initialize hooks system on module import
_initialize_plugin_system()

__all__ = [
    "async_map",
    "async_session",
    "backends",
    "parallel_map",
    "session",
    "settings",
    "task",
    "workflow",
]
