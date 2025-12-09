"""
Daglite: Lightweight Python framework for building static DAGs with explicit bindings.

Daglite provides a simple, type-safe way to build and execute directed acyclic graphs (DAGs)
of Python functions. Tasks are defined using the @task decorator, composed using fluent
binding operations (.bind, .product, .zip, .map, .join), and executed with evaluate() or
evaluate_async().

Key Features:
    - Explicit parameter binding (no implicit dependencies)
    - Fan-out/fan-in patterns with .product() and .zip()
    - Type-safe task composition with generic support
    - Multiple execution backends (local, threading, multiprocessing)
    - Async execution with sibling parallelism via evaluate_async()
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

Async Usage:
    >>> import asyncio
    >>> from daglite import task, evaluate_async
    >>>
    >>> @task
    >>> def slow_io(x: int) -> int:
    >>> # I/O-bound work
    >>>     return x * 2
    >>>
    >>> async def main():
    >>>     result = await evaluate_async(slow_io.product(x=[1, 2, 3]))
    >>>     print(result)  # [2, 4, 6]
    >>>
    >>> asyncio.run(main())

For more examples, see the repository's test files.
"""

__version__ = "0.4.0"

from . import backends
from . import hooks
from . import settings
from .engine import evaluate
from .engine import evaluate_async
from .pipelines import load_pipeline
from .pipelines import pipeline
from .tasks import task

# Initialize hooks system on module import
hooks.initialize_hooks()

__all__ = [
    "backends",
    "evaluate",
    "evaluate_async",
    "hooks",
    "load_pipeline",
    "pipeline",
    "settings",
    "task",
]
