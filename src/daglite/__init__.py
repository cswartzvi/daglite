"""Daglite: Lightweight Python framework for building eager task pipelines."""

__version__ = "0.9.0b6"

import daglite.datasets.builtin  # noqa: F401  ensure builtins are registered
from daglite import backends
from daglite import settings
from daglite.composers import gather_tasks
from daglite.composers import load_dataset
from daglite.composers import map_tasks
from daglite.composers import save_dataset
from daglite.plugins.manager import _initialize_plugin_system
from daglite.session import async_session
from daglite.session import session
from daglite.tasks import task
from daglite.workflows import workflow

# Initialize hooks system on module import
_initialize_plugin_system()

__all__ = [
    "async_session",
    "backends",
    "gather_tasks",
    "load_dataset",
    "map_tasks",
    "save_dataset",
    "session",
    "settings",
    "task",
    "workflow",
]
