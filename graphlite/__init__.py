"""graphlite - a tiny task graph and graph reduction framework."""
from __future__ import annotations

from .combinators import conditional, fanout
from .executor import Executor
from .task import Task, task

__all__ = ["Task", "task", "Executor", "fanout", "conditional"]
