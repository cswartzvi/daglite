"""Daglite: Lightweight Python framework for building static DAGs with explicit bindings."""

__version__ = "0.8.0"

import daglite.datasets.builtin  # noqa: F401  ensure builtins are registered
from daglite import backends
from daglite import futures
from daglite import settings
from daglite.futures import load_dataset
from daglite.plugins.manager import _initialize_plugin_system
from daglite.tasks import task
from daglite.workflows import WorkflowResult
from daglite.workflows import workflow

# Initialize hooks system on module import
_initialize_plugin_system()

__all__ = [
    "WorkflowResult",
    "backends",
    "futures",
    "load_dataset",
    "settings",
    "task",
    "workflow",
]
