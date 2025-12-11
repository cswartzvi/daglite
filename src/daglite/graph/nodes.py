"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar

from typing_extensions import override

from daglite.backends import Backend
from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.base import FunctionGraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput
from daglite.graph.utils import materialize_async
from daglite.graph.utils import materialize_sync

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TaskNode(FunctionGraphNode):
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
    def execute(
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> Any:
        hook_manager.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            inputs=resolved_inputs,
            iteration_count=None,
        )

        start_time = time.perf_counter()
        future = resolved_backend.submit(self.func, **resolved_inputs)
        result = future.result()
        result = materialize_sync(result)
        duration = time.perf_counter() - start_time

        hook_manager.hook.after_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            result=result,
            duration=duration,
            iteration_count=None,
        )

        return result

    @override
    async def execute_async(
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> Any:
        hook_manager.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            inputs=resolved_inputs,
            iteration_count=None,
        )

        start_time = time.perf_counter()
        future = resolved_backend.submit(self.func, **resolved_inputs)
        wrapped = asyncio.wrap_future(future)
        result = await wrapped
        result = await materialize_async(result)
        duration = time.perf_counter() - start_time

        hook_manager.hook.after_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            result=result,
            duration=duration,
            iteration_count=None,
        )

        return result


@dataclass(frozen=True)
class MapTaskNode(FunctionGraphNode):
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
    def execute(
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> list[Any]:
        hook_manager.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            inputs=resolved_inputs,
            iteration_count=None,  # Will be set after we know the count
        )

        node_start = time.perf_counter()
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}
        calls = self._build_map_calls(fixed, mapped)
        futures = resolved_backend.submit_many(self.func, calls)

        results = []
        iteration_total = len(futures)

        for idx, future in enumerate(futures):
            hook_manager.hook.before_iteration_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                iteration_index=idx,
                iteration_total=iteration_total,
            )

            iter_start = time.perf_counter()
            iter_result = future.result()
            iter_result = materialize_sync(iter_result)
            iter_duration = time.perf_counter() - iter_start

            hook_manager.hook.after_iteration_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                iteration_index=idx,
                iteration_total=iteration_total,
                result=iter_result,
                duration=iter_duration,
            )

            results.append(iter_result)

        node_duration = time.perf_counter() - node_start

        hook_manager.hook.after_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            result=results,
            duration=node_duration,
            iteration_count=iteration_total,
        )

        return results

    @override
    async def execute_async(  # pragma: no cover
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> list[Any]:
        hook_manager.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            inputs=resolved_inputs,
            iteration_count=None,  # Will be set after we know the count
        )

        node_start = time.perf_counter()
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}
        calls = self._build_map_calls(fixed, mapped)
        futures = resolved_backend.submit_many(self.func, calls)

        results = []
        iteration_total = len(futures)

        for idx, future in enumerate(futures):
            hook_manager.hook.before_iteration_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                iteration_index=idx,
                iteration_total=iteration_total,
            )

            iter_start = time.perf_counter()
            wrapped = asyncio.wrap_future(future)
            iter_result = await wrapped
            iter_result = await materialize_async(iter_result)
            iter_duration = time.perf_counter() - iter_start

            hook_manager.hook.after_iteration_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                iteration_index=idx,
                iteration_total=iteration_total,
                result=iter_result,
                duration=iter_duration,
            )

            results.append(iter_result)

        node_duration = time.perf_counter() - node_start

        hook_manager.hook.after_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            result=results,
            duration=node_duration,
            iteration_count=iteration_total,
        )

        return results
