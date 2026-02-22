"""Future types for lazy task binding."""

from daglite.futures.base import BaseTaskFuture
from daglite.futures.iter_future import IterFuture
from daglite.futures.iter_future import IterMapFuture
from daglite.futures.iter_future import IterReduceFuture
from daglite.futures.load_future import DatasetFuture
from daglite.futures.load_future import load_dataset
from daglite.futures.map_future import MapTaskFuture
from daglite.futures.reduce_future import ReduceFuture
from daglite.futures.task_future import TaskFuture

__all__ = [
    "BaseTaskFuture",
    "DatasetFuture",
    "IterFuture",
    "IterMapFuture",
    "IterReduceFuture",
    "MapTaskFuture",
    "ReduceFuture",
    "TaskFuture",
    "load_dataset",
]
