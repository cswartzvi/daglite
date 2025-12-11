"""Composite graph nodes for optimized execution of node chains."""

from __future__ import annotations

import inspect
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar

from typing_extensions import override

from daglite.backends import Backend
from daglite.graph.base import GraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode

T = TypeVar("T")


@dataclass(frozen=True)
class ChainLink:
    """
    Represents one node in a composite chain with its connection metadata.

    This structure enables visualization tools to show the internal
    structure of composite nodes.
    """

    node: TaskNode | MapTaskNode
    """The actual node in the chain."""

    position: int
    """Position in the chain (0-indexed)."""

    flow_param: str | None
    """Name of the parameter that receives the flowed value from the previous node."""

    external_params: dict[str, ParamInput]
    """Parameters from outside the chain (literals, fixed, or external futures)."""


@dataclass(frozen=True)
class CompositeTaskNode(GraphNode):
    """
    Represents a linear chain of TaskNodes executed as a single unit.

    This optimization reduces backend submission overhead by executing
    multiple nodes in sequence within a single backend call.

    The internal chain structure is preserved for:
    - Visualization of grouped execution
    - Hook firing for each internal node
    - Error attribution to specific nodes within the chain
    """

    chain: tuple[ChainLink, ...]
    """Ordered sequence of nodes forming the chain."""

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
        if self.chain:
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
    def submit(self, resolved_backend: Backend, resolved_inputs: dict[str, Any]) -> Future[Any]:
        def execute_chain(**initial_inputs: Any) -> Any:
            """Execute all nodes in the chain sequentially."""
            result = initial_inputs
            for link in self.chain:
                if link.position == 0:
                    node_inputs = initial_inputs  # First node uses initial inputs directly
                else:
                    node_inputs = self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    )
                result = link.node.func(**node_inputs)
            return result

        return resolved_backend.submit(execute_chain, **resolved_inputs)

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
class CompositeMapTaskNode(GraphNode):
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
    def submit(
        self, resolved_backend: Backend, resolved_inputs: dict[str, Any]
    ) -> list[Future[Any]]:
        """
        Execute iterations where each applies the full chain.

        Returns list of Futures, one per iteration.
        """

        def execute_iteration(**iteration_inputs: Any) -> Any:
            """Execute full chain for one iteration."""
            result = iteration_inputs

            for link in self.chain:
                if link.position == 0:
                    node_inputs = iteration_inputs
                else:
                    node_inputs = self._build_node_inputs(
                        link=link, flow_value=result, resolved_inputs=resolved_inputs
                    )

                result = link.node.func(**node_inputs)

            return result

        # Build calls from source map configuration
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.source_map.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.source_map.mapped_kwargs}
        calls = self.source_map._build_map_calls(fixed, mapped)

        return resolved_backend.submit_many(execute_iteration, calls)

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
