"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar

from typing_extensions import override

from daglite.backends import Backend
from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.base import GraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TaskNode(GraphNode):
    """Basic function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for this task node."""

    kwargs: Mapping[str, ParamInput]
    """Keyword parameters for the task function."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "task"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return list(self.kwargs.items())

    @override
    def submit(self, resolved_backend: Backend, resolved_inputs: dict[str, Any]) -> Future[Any]:
        return resolved_backend.submit(self.func, **resolved_inputs)


@dataclass(frozen=True)
class MapTaskNode(GraphNode):
    """Map function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for each map iteration."""

    mode: str
    """Mapping mode: 'extend' for Cartesian product, 'zip' for parallel iteration."""

    fixed_kwargs: Mapping[str, ParamInput]
    """Fixed keyword arguments for the mapped function."""

    mapped_kwargs: Mapping[str, ParamInput]
    """Mapped keyword arguments for the mapped function."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "map"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return [*self.fixed_kwargs.items(), *self.mapped_kwargs.items()]

    def _build_map_calls(
        self, fixed: Mapping[str, Any], mapped: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Build the list of function calls for map execution."""
        from itertools import product

        calls: list[dict[str, Any]] = []

        if self.mode == "product":
            items = list(mapped.items())
            names, lists = zip(*items) if items else ([], [])
            for combo in product(*lists):
                kw = dict(fixed)
                for name, val in zip(names, combo):
                    kw[name] = val
                calls.append(kw)
        elif self.mode == "zip":
            lengths = {len(v) for v in mapped.values()}
            if len(lengths) > 1:
                length_details = {name: len(vals) for name, vals in mapped.items()}
                raise ParameterError(
                    f"Map task '{self.name}' with `.zip()` requires all sequences to have the "
                    f"same length. Got mismatched lengths: {length_details}. "
                    f"Consider using `.extend()` if you want a Cartesian product instead."
                )
            n = lengths.pop() if lengths else 0
            for i in range(n):
                kw = dict(fixed)
                for name, vs in mapped.items():
                    kw[name] = vs[i]
                calls.append(kw)
        else:
            raise ExecutionError(
                f"Unknown map mode '{self.mode}'. Expected 'extend' or 'zip'. "
                f"This indicates an internal error in graph construction."
            )

        return calls

    @override
    def submit(
        self, resolved_backend: Backend, resolved_inputs: dict[str, Any]
    ) -> list[Future[Any]]:
        """Submit multiple task executions."""
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}
        calls = self._build_map_calls(fixed, mapped)
        return resolved_backend.submit_many(self.func, calls)
