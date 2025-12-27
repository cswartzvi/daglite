"""Daglite: Lightweight Python framework for building static DAGs with explicit bindings."""

__version__ = "0.5.1"

from . import backends
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
    "evaluate_async",
    "load_pipeline",
    "pipeline",
    "settings",
    "task",
]
