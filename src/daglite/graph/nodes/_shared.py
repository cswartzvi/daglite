"""Helper functions for graph nodes."""

from dataclasses import replace
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


def remap_node_changes(
    id_mapping: Mapping[UUID, UUID],
    output_configs: tuple[NodeOutputConfig, ...],
    **kwargs_fields: Mapping[str, NodeInput],
) -> dict[str, Any]:
    """
    Computes changed-field kwargs for use with `dataclasses.replace()`.

    Remaps one or more `NodeInput` mappings and `output_configs` using *id_mapping*, collecting
    only the fields that actually changed.  Pass the result directly to `replace(node, **changes)`
    — if the dict is empty, nothing changed and `replace()` is unnecessary.

    If nodes need more granular control (e.g. remapping some but not all `NodeInput` fields, or
    remapping other fields beyond `NodeInput` and `output_configs`), they can call the underlying
    helper functions `remap_node_inputs()` and `remap_output_configs()` directly.

    Args:
        id_mapping: Mapping from old node IDs to new node IDs.
        output_configs: The node's `output_configs` tuple to remap.
        **kwargs_fields: Named `NodeInput` mappings to remap, keyed by their dataclass
            field name (e.g. `kwargs=self.kwargs`, or `fixed_kwargs=self.fixed_kwargs`,
            `mapped_kwargs=self.mapped_kwargs`).

    Returns:
        Dict of changed fields suitable for passing to `replace(node, **changes)`.
        Empty if nothing changed.
    """
    changes: dict[str, Any] = {}
    for field_name, kwargs in kwargs_fields.items():
        new_kwargs = remap_node_inputs(kwargs, id_mapping)
        if new_kwargs is not kwargs:
            changes[field_name] = new_kwargs
    new_oc = remap_output_configs(output_configs, id_mapping)
    if new_oc is not None:
        changes["output_configs"] = new_oc
    return changes


def remap_node_inputs(
    kwargs: Mapping[str, NodeInput], id_mapping: Mapping[UUID, UUID]
) -> Mapping[str, NodeInput]:
    """
    Remaps `NodeInput` references using the provided ID mapping.

    Args:
        kwargs: Input mapping to remap.
        id_mapping: Mapping from old node IDs to new node IDs.

    Returns:
        A new mapping with updated references, or the original if unchanged.
    """
    changed = False
    new_kwargs: dict[str, NodeInput] = {}

    for name, param in kwargs.items():
        if param.reference is not None and param.reference in id_mapping:
            new_param = NodeInput(
                _kind=param._kind,
                value=param.value,
                reference=id_mapping[param.reference],
            )
            new_kwargs[name] = new_param
            changed = True
        else:
            new_kwargs[name] = param

    return new_kwargs if changed else kwargs


def remap_output_configs(
    output_configs: tuple[NodeOutputConfig, ...], id_mapping: Mapping[UUID, UUID]
) -> tuple[NodeOutputConfig, ...] | None:
    """
    Remaps `NodeInput` references inside output-config dependencies.

    Returns `None` if no reference was remapped (callers can use this sentinel to skip a
    `replace()` call).

    Args:
        output_configs: The output configurations to remap.
        id_mapping: Mapping from old node IDs to new node IDs.

    Returns:
        A new tuple of output configs with updated references, or ``None`` if unchanged.
    """
    changed = False
    new_configs: list[NodeOutputConfig] = []
    for config in output_configs:
        new_deps = remap_node_inputs(config.dependencies, id_mapping)
        if new_deps is not config.dependencies:
            new_configs.append(replace(config, dependencies=new_deps))
            changed = True
        else:
            new_configs.append(config)
    return tuple(new_configs) if changed else None
