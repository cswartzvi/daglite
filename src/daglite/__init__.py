"""
Daglite: Lightweight Python framework for building static DAGs with explicit bindings.

Daglite provides a simple, type-safe way to build and execute directed acyclic graphs (DAGs)
of Python functions. Tasks are defined using the @task decorator, composed using fluent
binding operations (.bind, .extend, .zip, .map, .join), and executed with the evaluate() function.

Key Features:
    - Explicit parameter binding (no implicit dependencies)
    - Fan-out/fan-in patterns with .extend() and .zip()
    - Type-safe task composition with generic support
    - Multiple execution backends (local, threading)
    - Async/await support for I/O-bound workloads
    - Lazy evaluation with automatic topological sorting

Basic Usage:
    >>> from daglite import task, evaluate
    >>>
    >>> @task
    >>> def add(x: int, y: int) -> int:
    >>>     return x + y
    >>>
    >>> result = evaluate(add.bind(x=1, y=2))
    >>> print(result)  # 3

For more examples, see the repository's test files.
"""

__version__ = "0.2.0"

from . import backends
from . import futures
from . import settings
from .engine import evaluate
from .pipelines import load_pipeline
from .pipelines import pipeline
from .tasks import task

__all__ = [
    "backends",
    "evaluate",
    "futures",
    "load_pipeline",
    "pipeline",
    "settings",
    "task",
]
