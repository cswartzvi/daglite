"""Helper functions for constructing graph IR components from task future definitions."""

from typing import Any, Mapping

from daglite._validation import check_key_placeholders
from daglite.futures.base import BaseTaskFuture
from daglite.futures.base import FutureOutput
from daglite.graph.base import OutputConfig
from daglite.graph.base import ParamInput


def build_graph_parameters(kwargs: Mapping[str, Any]) -> dict[str, ParamInput]:
    """
    Builds graph IR parameters from the provided kwargs.

    Note converts task futures to reference parameters, and passes through concrete values as-is.

    Args:
        kwargs: Keyword arguments to resolve into graph parameters.

    Returns:
        A dictionary mapping parameter names to ParamInput instances for graph construction.
    """
    resolved: dict[str, Any] = {}
    for name, value in kwargs.items():
        if isinstance(value, BaseTaskFuture):
            resolved[name] = ParamInput.from_ref(value.id)
        else:
            resolved[name] = ParamInput.from_value(value)
    return resolved


def build_output_configs(
    future_outputs: tuple[FutureOutput, ...], placeholders: set[str]
) -> tuple[OutputConfig, ...]:
    """
    Builds graph IR output configurations from the provided future outputs.

    Args:
        future_outputs: Tuple of FutureOutput instances from the TaskFuture.
        placeholders: Base set of available placeholder names for validating output keys.
            This typically includes root-level parameters and any outputs from upstream nodes.

    Returns:
        Tuple of OutputConfig instances for graph construction.
    """

    output_configs: list[OutputConfig] = []
    for future_output in future_outputs:
        placeholders |= future_output.extras.keys()
        check_key_placeholders(future_output.key, placeholders)

        dependencies = build_graph_parameters(future_output.extras)
        output_config = OutputConfig(
            key=future_output.key,
            name=future_output.name,
            format=future_output.format,
            store=future_output.store,
            dependencies=dependencies,
            options=future_output.options or {},
        )
        output_configs.append(output_config)

    return tuple(output_configs)
