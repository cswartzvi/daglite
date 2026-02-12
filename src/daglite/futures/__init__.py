"""Future types for lazy task binding."""

from daglite.futures.base import BaseTaskFuture
from daglite.futures.load_future import DatasetFuture
from daglite.futures.load_future import load_dataset
from daglite.futures.map_future import MapTaskFuture
from daglite.futures.task_future import TaskFuture

__all__ = [
    "BaseTaskFuture",
    "DatasetFuture",
    "MapTaskFuture",
    "TaskFuture",
    "load_dataset",
]
