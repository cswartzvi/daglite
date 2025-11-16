"""Lightweight Python framework for building static DAGs with explicit bindings."""

__version__ = "0.0.1"

from .engine import evaluate
from .tasks import MapTaskFuture
from .tasks import TaskFuture
from .tasks import task

__all__ = [
    "MapTaskFuture",
    "TaskFuture",
    "evaluate",
    "task",
]
