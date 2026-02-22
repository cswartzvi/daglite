from typing import Any, Callable, Coroutine

from typing_extensions import Literal

Submission = Callable[[], Coroutine[Any, Any, Any]]
"""Backend submission function type, a parameterless function that returns a coroutine."""

MapMode = Literal["product", "zip"]
"""Mapping mode for map nodes, indicating how to combine input sequences."""

ParamKind = Literal["value", "ref", "sequence", "sequence_ref"]
"""Kind of node input parameter, determining how it should be resolved."""

NodeKind = Literal["task", "map", "dataset", "composite_task", "composite_map", "reduce", "iter"]
"""Kind of graph node, determining its execution semantics and required fields."""

ReduceMode = Literal["ordered", "unordered"]
"""Mode for reduce nodes, indicating iteration order or completion order."""
