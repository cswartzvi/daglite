"""Composite graph nodes for optimized execution of node chains."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar

from typing_extensions import override

from daglite.backends import Backend
from daglite.graph.base import ChainLink
from daglite.graph.base import CompositeGraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput
from daglite.graph.nodes import MapTaskNode
from daglite.graph.utils import materialize_async
from daglite.graph.utils import materialize_sync

T = TypeVar("T")


@dataclass(frozen=True)
class CompositeTaskNode(CompositeGraphNode):
    """
    Represents a linear chain of TaskNodes executed as a single unit.

    This optimization reduces backend submission overhead by executing
    multiple nodes in sequence within a single backend call.

    The internal chain structure is preserved for:
    - Visualization of grouped execution
    - Hook firing for each internal node
    - Error attribution to specific nodes within the chain
    """

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "task"  # Acts like a task node from Engine's perspective

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        """
        Returns inputs for the composite.

        Includes:
        - All inputs for the first node (to start the chain)
        - External ref parameters from later nodes (dependencies on nodes outside the chain)

        Value parameters for later nodes are baked into their node definitions and
        don't need to be passed through the composite's interface.
        """
        inputs_dict: dict[str, ParamInput] = {}
        chain_node_ids = {link.node.id for link in self.chain}

        # Add first node's inputs
        if self.chain:  # pragma: no branch
            for param_name, param_input in self.chain[0].node.inputs():
                inputs_dict[param_name] = param_input

        # Add external ref params from later nodes (not values)
        for link in self.chain[1:]:
            for param_name, param_input in link.external_params.items():
                # Only include refs to external nodes, not value params
                if param_input.is_ref and param_input.ref not in chain_node_ids:
                    inputs_dict[param_name] = param_input

        return list(inputs_dict.items())

    @override
    def execute(
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> Any:
        """Execute the composite chain synchronously, firing node hooks for each link."""

        def execute_chain(**initial_inputs: Any) -> Any:
            """Execute all nodes in the chain sequentially with hooks."""
            result = initial_inputs
            for link in self.chain:
                # Fire before_node_execute hook for this chain link
                hook_manager.hook.before_node_execute(
                    node_id=link.node.id,
                    node=link.node,
                    backend=resolved_backend,
                    inputs=initial_inputs
                    if link.position == 0
                    else self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    ),
                    iteration_count=None,
                )

                link_start = time.perf_counter()

                if link.position == 0:
                    node_inputs = initial_inputs
                else:
                    node_inputs = self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    )

                result = link.node.func(**node_inputs)
                link_duration = time.perf_counter() - link_start

                # Fire after_node_execute hook for this chain link
                if hook_manager:
                    hook_manager.hook.after_node_execute(
                        node_id=link.node.id,
                        node=link.node,
                        backend=resolved_backend,
                        result=result,
                        duration=link_duration,
                        iteration_count=None,
                    )

            return result

        # Fire before_node_execute for the composite
        hook_manager.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            inputs=resolved_inputs,
            iteration_count=None,
        )

        node_start = time.perf_counter()
        future = resolved_backend.submit(execute_chain, **resolved_inputs)
        result = future.result()
        result = materialize_sync(result)
        node_duration = time.perf_counter() - node_start

        # Fire after_node_execute for the composite
        hook_manager.hook.after_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            result=result,
            duration=node_duration,
            iteration_count=None,
        )

        return result

    @override
    async def execute_async(  # pragma: no cover
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> Any:
        """Execute the composite chain asynchronously, firing node hooks for each link."""

        def execute_chain(**initial_inputs: Any) -> Any:
            """Execute all nodes in the chain sequentially with hooks."""
            result = initial_inputs
            for link in self.chain:
                # Fire before_node_execute hook for this chain link
                if hook_manager:
                    hook_manager.hook.before_node_execute(
                        node_id=link.node.id,
                        node=link.node,
                        backend=resolved_backend,
                        inputs=initial_inputs
                        if link.position == 0
                        else self._build_node_inputs(
                            link=link, flow_value=result, resolved_inputs=resolved_inputs
                        ),
                        iteration_count=None,
                    )

                link_start = time.perf_counter()

                if link.position == 0:
                    node_inputs = initial_inputs
                else:
                    node_inputs = self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    )

                result = link.node.func(**node_inputs)
                link_duration = time.perf_counter() - link_start

                # Fire after_node_execute hook for this chain link
                if hook_manager:
                    hook_manager.hook.after_node_execute(
                        node_id=link.node.id,
                        node=link.node,
                        backend=resolved_backend,
                        result=result,
                        duration=link_duration,
                        iteration_count=None,
                    )

            return result

        # Fire before_node_execute for the composite
        if hook_manager:
            hook_manager.hook.before_node_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                inputs=resolved_inputs,
                iteration_count=None,
            )

        node_start = time.perf_counter()
        future = resolved_backend.submit(execute_chain, **resolved_inputs)
        wrapped = asyncio.wrap_future(future)
        result = await wrapped
        result = await materialize_async(result)
        node_duration = time.perf_counter() - node_start

        # Fire after_node_execute for the composite
        if hook_manager:
            hook_manager.hook.after_node_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                result=result,
                duration=node_duration,
                iteration_count=None,
            )

        return result

    def _build_node_inputs(
        self, link: ChainLink, flow_value: Any, resolved_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Build input dict for a node in the chain."""
        node_inputs: dict[str, Any] = {}

        if link.flow_param:
            node_inputs[link.flow_param] = flow_value

        for param_name, param_input in link.external_params.items():
            if param_input.is_ref:
                # Look up external dependency
                node_inputs[param_name] = resolved_inputs[param_name]
            else:
                # Use baked-in value
                node_inputs[param_name] = param_input.value

        return node_inputs


@dataclass(frozen=True)
class CompositeMapTaskNode(CompositeGraphNode):
    """
    Represents a chain of MapTaskNodes where each iteration executes the full chain.

    Instead of executing map operations level-by-level:
        for x in source: map1(x)
        for y in map1_results: map2(y)
        for z in map2_results: map3(z)

    We execute per-iteration:
        for x in source: map3(map2(map1(x)))

    This reduces submissions from (iterations × chain_length) to (iterations),
    improves cache locality, and reduces memory for intermediate results.
    """

    source_map: MapTaskNode
    """The initial map operation that defines the iteration space."""

    chain: tuple[ChainLink, ...]
    """
    Ordered sequence of map operations to apply per iteration.

    First link corresponds to source_map, subsequent links are .map() operations.
    """

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "map"  # Acts like a map node from Engine's perspective

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        """
        Returns inputs for the composite map.

        Includes:
        - Mapped parameters from source_map
        - Fixed parameters from source_map
        - All external fixed parameters from chain
        """
        inputs_dict: dict[str, ParamInput] = {}

        for name, param in self.source_map.inputs():
            inputs_dict[name] = param

        for link in self.chain[1:]:
            inputs_dict.update(link.external_params)

        return list(inputs_dict.items())

    @override
    def execute(
        self,
        resolved_backend: Backend,
        resolved_inputs: dict[str, Any],
        hook_manager: Any,
    ) -> list[Any]:
        """Execute composite map synchronously, firing iteration and node hooks."""

        def execute_iteration_with_hooks(**iteration_inputs: Any) -> Any:
            """Execute full chain for one iteration with node hooks."""
            result = iteration_inputs

            for link in self.chain:
                # Fire before_node_execute hook for this chain link
                hook_manager.hook.before_node_execute(
                    node_id=link.node.id,
                    node=link.node,
                    backend=resolved_backend,
                    inputs=iteration_inputs
                    if link.position == 0
                    else self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    ),
                    iteration_count=None,
                )

                link_start = time.perf_counter()

                if link.position == 0:
                    node_inputs = iteration_inputs
                else:
                    node_inputs = self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    )

                result = link.node.func(**node_inputs)
                link_duration = time.perf_counter() - link_start

                # Fire after_node_execute hook for this chain link
                if hook_manager:
                    hook_manager.hook.after_node_execute(
                        node_id=link.node.id,
                        node=link.node,
                        backend=resolved_backend,
                        result=result,
                        duration=link_duration,
                        iteration_count=None,
                    )

            return result

        # Fire before_node_execute for the composite
        hook_manager.hook.before_node_execute(
            node_id=self.id,
            node=self,
            backend=resolved_backend,
            inputs=resolved_inputs,
            iteration_count=None,
        )

        node_start = time.perf_counter()

        # Build calls from source map configuration
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.source_map.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.source_map.mapped_kwargs}
        calls = self.source_map._build_map_calls(fixed, mapped)
        futures = resolved_backend.submit_many(execute_iteration_with_hooks, calls)

        results = []
        iteration_total = len(futures)

        for idx, future in enumerate(futures):
            # Fire before_iteration_execute hook
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

            # Fire after_iteration_execute hook
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

        # Fire after_node_execute for the composite
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
        hook_manager: Any | None = None,
    ) -> list[Any]:
        """Execute composite map asynchronously, firing iteration and node hooks."""

        def execute_iteration_with_hooks(**iteration_inputs: Any) -> Any:
            """Execute full chain for one iteration with node hooks."""
            result = iteration_inputs

            for link in self.chain:
                # Fire before_node_execute hook for this chain link
                if hook_manager:
                    hook_manager.hook.before_node_execute(
                        node_id=link.node.id,
                        node=link.node,
                        backend=resolved_backend,
                        inputs=iteration_inputs
                        if link.position == 0
                        else self._build_node_inputs(
                            link=link, flow_value=result, resolved_inputs=resolved_inputs
                        ),
                        iteration_count=None,
                    )

                link_start = time.perf_counter()

                if link.position == 0:
                    node_inputs = iteration_inputs
                else:
                    node_inputs = self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    )

                result = link.node.func(**node_inputs)
                link_duration = time.perf_counter() - link_start

                # Fire after_node_execute hook for this chain link
                if hook_manager:
                    hook_manager.hook.after_node_execute(
                        node_id=link.node.id,
                        node=link.node,
                        backend=resolved_backend,
                        result=result,
                        duration=link_duration,
                        iteration_count=None,
                    )

            return result

        # Fire before_node_execute for the composite
        if hook_manager:
            hook_manager.hook.before_node_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                inputs=resolved_inputs,
                iteration_count=None,
            )

        node_start = time.perf_counter()

        # Build calls from source map configuration
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.source_map.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.source_map.mapped_kwargs}
        calls = self.source_map._build_map_calls(fixed, mapped)
        futures = resolved_backend.submit_many(execute_iteration_with_hooks, calls)

        results = []
        iteration_total = len(futures)

        for idx, future in enumerate(futures):
            # Fire before_iteration_execute hook
            if hook_manager:
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

            # Fire after_iteration_execute hook
            if hook_manager:
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

        # Fire after_node_execute for the composite
        if hook_manager:
            hook_manager.hook.after_node_execute(
                node_id=self.id,
                node=self,
                backend=resolved_backend,
                result=results,
                duration=node_duration,
                iteration_count=iteration_total,
            )

        return results

    def _build_node_inputs(
        self, link: ChainLink, flow_value: Any, resolved_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Build input dict for a map function in the chain."""
        node_inputs: dict[str, Any] = {}

        if link.flow_param:
            node_inputs[link.flow_param] = flow_value
        else:
            # Fallback: use first parameter
            sig = inspect.signature(link.node.func)
            first_param = next(iter(sig.parameters.keys()))
            node_inputs[first_param] = flow_value

        for param_name, param_input in link.external_params.items():
            if param_input.is_ref:
                node_inputs[param_name] = resolved_inputs[param_name]
            else:
                node_inputs[param_name] = param_input.value

        return node_inputs
