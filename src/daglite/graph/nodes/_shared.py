"""Helper functions for graph nodes."""

from typing import Any, Mapping
from uuid import UUID

from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import NodeOutputConfig


def collect_dependencies(
    kwargs: Mapping[str, NodeInput], output_configs: tuple[NodeOutputConfig, ...] | None = None
) -> set[UUID]:
    """
    Collects upstream node dependencies from the provided parameters and outputs.

    Args:
        kwargs: Keyword arguments to inspect for node input dependencies.
        output_configs: Optional tuple of `NodeOutput` instances to inspect for additional
            dependencies.

    Returns:
        A set of node IDs of upstream dependencies for graph construction.
    """
    dependencies: set[UUID] = set()

    # Collect dependencies from input parameters
    for value in kwargs.values():
        if value.reference is not None:
            dependencies.add(value.reference)

    # Collect dependencies from output configurations
    for config in output_configs if output_configs else []:
        for param in config.dependencies.values():
            if param.reference is not None:
                dependencies.add(param.reference)
    return dependencies


def resolve_inputs(
    kwargs: Mapping[str, NodeInput], completed_nodes: Mapping[UUID, Any]
) -> dict[str, Any]:
    """
    Resolves a mapping of parameter names to `NodeInput` instances into concrete values.

    Args:
        kwargs: Mapping of parameter names to `NodeInput` instances to resolve.
        completed_nodes: Mapping from node IDs to their computed values, used for resolving
            reference inputs.

    Returns:
        A dictionary mapping parameter names to their resolved concrete values.
    """
    return {name: pm.resolve(completed_nodes) for name, pm in kwargs.items()}


def resolve_output_parameters(
    output_configs: tuple[NodeOutputConfig, ...], completed_nodes: Mapping[UUID, Any]
) -> list[dict[str, Any]]:
    """
    Resolves parameters for the given output configurations based on completed node results.

    Args:
        output_configs: Tuple of `NodeOutputConfig` that will have their parameters resolved.
        completed_nodes: Mapping of completed node IDs to their execution results.

    Returns:
        A list of dictionaries containing resolved parameter values for each output configuration.
    """
    outputs: list[dict[str, Any]] = []
    for config in output_configs:
        values: dict[str, Any] = {}
        for name, param in config.dependencies.items():
            values[name] = param.resolve(completed_nodes)
        outputs.append(values)
    return outputs
