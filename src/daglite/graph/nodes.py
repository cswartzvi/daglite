"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from uuid import UUID

from typing_extensions import override

from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.base import BaseGraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TaskNode(BaseGraphNode):
    """Basic function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for this task node."""

    kwargs: Mapping[str, ParamInput]
    """Keyword parameters for the task function."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "task"

    @override
    def dependencies(self) -> set[UUID]:
        """Derive dependencies from reference inputs."""
        return {p.ref for p in self.kwargs.values() if p.is_ref and p.ref is not None}

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        """Resolve all keyword parameters."""
        inputs = {}
        for name, param in self.kwargs.items():
            if param.kind in ("sequence", "sequence_ref"):
                inputs[name] = param.resolve_sequence(completed_nodes)
            else:
                inputs[name] = param.resolve(completed_nodes)
        return inputs

    @override
    def run(self, resolved_inputs: dict[str, Any]) -> Any:
        """Execute task function with plugin hooks."""
        from daglite.backends.context import get_plugin_manager
        from daglite.backends.context import get_reporter

        pm = get_plugin_manager()
        _reporter = get_reporter()  # Available for future use

        # Fire before hook
        pm.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=None,  # Backend not available in worker context
            inputs=resolved_inputs,
        )

        start_time = time.time()
        try:
            result = self.func(**resolved_inputs)
            duration = time.time() - start_time

            # Fire after hook
            pm.hook.after_node_execute(
                node_id=self.id,
                node=self,
                backend=None,
                result=result,
                duration=duration,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Fire error hook
            pm.hook.on_node_error(
                node_id=self.id,
                node=self,
                backend=None,
                error=e,
                duration=duration,
            )
            raise

    @override
    async def run_async(self, resolved_inputs: dict[str, Any]) -> Any:
        """Execute task function asynchronously with plugin hooks."""
        from daglite.backends.context import get_plugin_manager
        from daglite.backends.context import get_reporter

        pm = get_plugin_manager()
        _reporter = get_reporter()  # Available for future use

        # Fire before hook
        pm.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=None,
            inputs=resolved_inputs,
        )

        start_time = time.time()
        try:
            result = self.func(**resolved_inputs)
            duration = time.time() - start_time

            # Fire after hook
            pm.hook.after_node_execute(
                node_id=self.id,
                node=self,
                backend=None,
                result=result,
                duration=duration,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Fire error hook
            pm.hook.on_node_error(
                node_id=self.id,
                node=self,
                backend=None,
                error=e,
                duration=duration,
            )
            raise


@dataclass(frozen=True)
class MapTaskNode(BaseGraphNode):
    """Map function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for each map iteration."""

    mode: str
    """Mapping mode: 'extend' for Cartesian product, 'zip' for parallel iteration."""

    fixed_kwargs: Mapping[str, ParamInput]
    """Fixed keyword arguments for the mapped function."""

    mapped_kwargs: Mapping[str, ParamInput]
    """Mapped keyword arguments for the mapped function."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "map"

    @override
    def dependencies(self) -> set[UUID]:
        """Derive dependencies from both fixed and mapped inputs."""
        deps = set()
        for param in self.fixed_kwargs.values():
            if param.is_ref and param.ref is not None:
                deps.add(param.ref)
        for param in self.mapped_kwargs.values():
            if param.is_ref and param.ref is not None:
                deps.add(param.ref)
        return deps

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        """Resolve all fixed and mapped parameters."""
        inputs = {}

        # Resolve fixed kwargs
        for name, param in self.fixed_kwargs.items():
            if param.kind in ("sequence", "sequence_ref"):
                inputs[name] = param.resolve_sequence(completed_nodes)
            else:
                inputs[name] = param.resolve(completed_nodes)

        # Resolve mapped kwargs
        for name, param in self.mapped_kwargs.items():
            if param.kind in ("sequence", "sequence_ref"):
                inputs[name] = param.resolve_sequence(completed_nodes)
            else:
                inputs[name] = param.resolve(completed_nodes)

        return inputs

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
    def run(self, resolved_inputs: dict[str, Any]) -> list[Any]:
        """Execute map iterations with plugin hooks."""
        from daglite.backends.context import get_plugin_manager
        from daglite.backends.context import get_reporter

        pm = get_plugin_manager()
        _reporter = get_reporter()  # Available for future use

        # Split inputs into fixed and mapped
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}

        # Build individual calls
        calls = self._build_map_calls(fixed, mapped)
        iteration_total = len(calls)

        # Fire before hook
        pm.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=None,
            inputs=resolved_inputs,
            iteration_count=iteration_total,
        )

        start_time = time.time()
        results = []

        try:
            for iteration_index, call_kwargs in enumerate(calls):
                # Fire before iteration hook
                pm.hook.before_iteration_execute(
                    node_id=self.id,
                    node=self,
                    backend=None,
                    iteration_index=iteration_index,
                    iteration_total=iteration_total,
                )

                iter_start = time.time()
                try:
                    result = self.func(**call_kwargs)
                    iter_duration = time.time() - iter_start

                    # Fire after iteration hook
                    pm.hook.after_iteration_execute(
                        node_id=self.id,
                        node=self,
                        backend=None,
                        iteration_index=iteration_index,
                        iteration_total=iteration_total,
                        result=result,
                        duration=iter_duration,
                    )

                    results.append(result)

                except Exception:
                    # Iteration-level error - propagate up
                    raise

            duration = time.time() - start_time

            # Fire after hook
            pm.hook.after_node_execute(
                node_id=self.id,
                node=self,
                backend=None,
                result=results,
                duration=duration,
                iteration_count=iteration_total,
            )

            return results

        except Exception as e:
            duration = time.time() - start_time

            # Fire error hook
            pm.hook.on_node_error(
                node_id=self.id,
                node=self,
                backend=None,
                error=e,
                duration=duration,
                iteration_count=iteration_total,
            )
            raise

    @override
    async def run_async(self, resolved_inputs: dict[str, Any]) -> list[Any]:
        """Execute map iterations asynchronously with plugin hooks."""
        from daglite.backends.context import get_plugin_manager
        from daglite.backends.context import get_reporter

        pm = get_plugin_manager()
        _reporter = get_reporter()  # Available for future use

        # Split inputs into fixed and mapped
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}

        # Build individual calls
        calls = self._build_map_calls(fixed, mapped)
        iteration_total = len(calls)

        # Fire before hook
        pm.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=None,
            inputs=resolved_inputs,
            iteration_count=iteration_total,
        )

        start_time = time.time()
        results = []

        try:
            for iteration_index, call_kwargs in enumerate(calls):
                # Fire before iteration hook
                pm.hook.before_iteration_execute(
                    node_id=self.id,
                    node=self,
                    backend=None,
                    iteration_index=iteration_index,
                    iteration_total=iteration_total,
                )

                iter_start = time.time()
                try:
                    result = self.func(**call_kwargs)
                    iter_duration = time.time() - iter_start

                    # Fire after iteration hook
                    pm.hook.after_iteration_execute(
                        node_id=self.id,
                        node=self,
                        backend=None,
                        iteration_index=iteration_index,
                        iteration_total=iteration_total,
                        result=result,
                        duration=iter_duration,
                    )

                    results.append(result)

                except Exception:
                    # Iteration-level error - propagate up
                    raise

            duration = time.time() - start_time

            # Fire after hook
            pm.hook.after_node_execute(
                node_id=self.id,
                node=self,
                backend=None,
                result=results,
                duration=duration,
                iteration_count=iteration_total,
            )

            return results

        except Exception as e:
            duration = time.time() - start_time

            # Fire error hook
            pm.hook.on_node_error(
                node_id=self.id,
                node=self,
                backend=None,
                error=e,
                duration=duration,
                iteration_count=iteration_total,
            )
            raise
