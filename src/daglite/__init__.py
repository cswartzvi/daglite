"""Daglite: Lightweight Python framework for building static DAGs with explicit bindings."""

__version__ = "0.7.0"

import daglite.datasets.builtin  # noqa: F401  ensure builtins are registered
from daglite import backends
from daglite import futures
from daglite import settings
from daglite.engine import evaluate
from daglite.engine import evaluate_async
from daglite.futures import load_dataset
from daglite.pipelines import load_pipeline
from daglite.pipelines import pipeline
from daglite.plugins.manager import _initialize_plugin_system
from daglite.tasks import task

# Initialize hooks system on module import
_initialize_plugin_system()

__all__ = [
    "backends",
    "evaluate",
    "futures",
    "evaluate_async",
    "load_dataset",
    "load_pipeline",
    "pipeline",
    "settings",
    "task",
]
