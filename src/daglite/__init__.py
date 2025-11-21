"""
Daglite: Lightweight Python framework for building static DAGs with explicit bindings.

Daglite provides a simple, type-safe way to build and execute directed acyclic graphs (DAGs)
of Python functions. Tasks are defined using the @task decorator, composed using fluent
binding operations (.bind, .extend, .zip, .map, .join), and executed with the evaluate() function.

Key Features:
    - Type-safe task composition with full generic support
    - Explicit parameter binding (no implicit dependencies)
    - Multiple execution backends (local, threading)
    - Async/await support for I/O-bound workloads
    - Fan-out/fan-in patterns with .extend() and .zip()
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

__version__ = "0.0.1"

from .engine import evaluate
from .exceptions import BackendError
from .exceptions import DagliteError
from .exceptions import ExecutionError
from .exceptions import GraphConstructionError
from .exceptions import ParameterError
from .exceptions import TaskConfigurationError
from .tasks import MapTaskFuture
from .tasks import TaskFuture
from .tasks import task

__all__ = [
    # Core API
    "evaluate",
    "task",
    # Task futures
    "MapTaskFuture",
    "TaskFuture",
    # Exceptions
    "BackendError",
    "DagliteError",
    "ExecutionError",
    "GraphConstructionError",
    "ParameterError",
    "TaskConfigurationError",
]
