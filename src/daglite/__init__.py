"""Daglite: Lightweight Python framework for building static DAGs with explicit bindings."""

__version__ = "0.7.0"

import daglite.datasets.builtin  # noqa: F401  ensure builtins are registered

from . import backends
from . import futures
from . import settings
from .engine import evaluate
from .engine import evaluate_async
from .pipelines import load_pipeline
from .pipelines import pipeline
from .plugins.manager import _initialize_plugin_system
from .tasks import task

# Initialize hooks system on module import
_initialize_plugin_system()

__all__ = [
    "backends",
    "evaluate",
    "futures",
    "evaluate_async",
    "load_pipeline",
    "pipeline",
    "settings",
    "task",
]
