"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar, override
from uuid import UUID

from daglite.engine import Backend
from daglite.graph.base import GraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput
from daglite.tasks import Task

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TaskNode(GraphNode):
    """
    Task node representation for graph IR.

    Attributes:
        id (UUID): Unique identifier for the task node.
        task (daglite.tasks.Task[Any, T_co]): Task associated with this node.
        params (Mapping[str, ParamInput]): Parameters for the task, as ParamInputs.
        backend (str | Backend | None): Backend name or instance for this node, if any.
    """

    id: UUID
    task: Task[Any, Any]
    params: Mapping[str, ParamInput]
    backend: str | Backend | None

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "task"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return list(self.params.items())

    @override
    def deps(self) -> set[UUID]:
        return {p.ref for p in self.params.values() if p.is_ref and p.ref is not None}

    @override
    def run(self, backend: Backend, values: Mapping[UUID, Any]) -> Any:
        kwargs = {name: p.resolve(values) for name, p in self.params.items()}
        return backend.run_single(self.task.fn, kwargs)


@dataclass(frozen=True)
class MapTaskNode(GraphNode):
    """
    Map task node representation for graph IR.

    Attributes:
        id (UUID): Unique identifier for the map task node.
        task (daglite.tasks.Task[Any, T_co]): Task to be mapped.
        mode (str): Mapping mode, either "extend" or "zip".
        fixed_kwargs (Mapping[str, ParamInput]): Fixed parameters for the task, as ParamInputs.
        mapped_kwargs (Mapping[str, ParamInput]): Mapped parameters for the task., as ParamInputs.
        backend (str | Backend, optional): Backend name or instance for this node, if any.
    """

    id: UUID
    task: Task[Any, Any]
    mode: str
    fixed_kwargs: Mapping[str, ParamInput]
    mapped_kwargs: Mapping[str, ParamInput]
    backend: str | Backend | None

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "map"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return [*self.fixed_kwargs.items(), *self.mapped_kwargs.items()]

    @override
    def deps(self) -> set[UUID]:
        out: set[UUID] = set()
        # Add fixed and mapped kwargs refs
        for p in self.fixed_kwargs.values():
            if p.is_ref and p.ref is not None:
                out.add(p.ref)
        for p in self.mapped_kwargs.values():
            if p.is_ref and p.ref is not None:
                out.add(p.ref)
        return out

    @override
    def run(self, backend: Backend, values: Mapping[UUID, Any]) -> list[Any]:
        fixed = {k: p.resolve(values) for k, p in self.fixed_kwargs.items()}
        mapped = {k: p.resolve_sequence(values) for k, p in self.mapped_kwargs.items()}

        from itertools import product

        calls: list[dict[str, Any]] = []
        if self.mode == "extend":
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
                from daglite.exceptions import ParameterError

                length_details = {name: len(vals) for name, vals in mapped.items()}
                raise ParameterError(
                    f"zip() requires all sequences to have the same length. "
                    f"Got mismatched lengths: {length_details}. "
                    f"Consider using extend() if you want a Cartesian product instead."
                )
            n = lengths.pop() if lengths else 0
            for i in range(n):
                kw = dict(fixed)
                for name, vs in mapped.items():
                    kw[name] = vs[i]
                calls.append(kw)
        else:
            from daglite.exceptions import ExecutionError

            raise ExecutionError(
                f"Unknown map mode '{self.mode}'. Expected 'extend' or 'zip'. "
                f"This indicates an internal error in graph construction."
            )

        return backend.run_many(self.task.fn, calls)
