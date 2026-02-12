"""Helper functions for constructing graph IR components from task future definitions."""

from typing import Any, Mapping

from daglite._validation import check_key_placeholders
from daglite.futures.base import BaseTaskFuture
from daglite.futures.base import OutputFuture
from daglite.graph.base import InputParam
from daglite.graph.base import OutputConfig
from daglite.graph.builder import GraphBuilder


def collect_dependencies(
    kwargs: Mapping[str, Any], outputs: tuple[OutputFuture, ...] | None = None
) -> list[GraphBuilder]:
    """
    Collects graph dependencies from the provided kwargs and future outputs.

    Args:
        kwargs: Keyword arguments to inspect for task future dependencies.
        outputs: Optional tuple of FutureOutput instances to inspect for additional dependencies.

    Returns:
        A list of GraphBuilder instances representing the dependencies for a task future.
    """

    deps: list[GraphBuilder] = []

    for value in kwargs.values():
        if isinstance(value, BaseTaskFuture):
            deps.append(value)

    outputs = outputs if outputs else tuple()
    for future_output in outputs:
        for value in future_output.extras.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)
    return deps


def build_parameters(kwargs: Mapping[str, Any]) -> dict[str, InputParam]:
    """
    Builds graph IR parameters for task futures from the provided kwargs.

    Note converts task futures to reference parameters, and passes through concrete values as-is.

    Args:
        kwargs: Keyword arguments to resolve into graph parameters.

    Returns:
        A dictionary mapping parameter names to InputParam instances for graph construction.
    """
    params: dict[str, Any] = {}
    for name, value in kwargs.items():
        if isinstance(value, BaseTaskFuture):
            params[name] = InputParam.from_ref(value.id)
        else:
            params[name] = InputParam.from_value(value)
    return params


def build_map_parameters(kwargs: Mapping[str, Any]) -> dict[str, InputParam]:
    """
    Builds graph IR parameters for mapped task futures from the provided kwargs.

    Note converts task futures to reference parameters, and passes through concrete values as-is.

    Args:
        kwargs: Keyword arguments to resolve into graph parameters.

    Returns:
        A dictionary mapping parameter names to InputParam instances for graph construction.
    """
    from daglite.futures.map_future import MapTaskFuture
    from daglite.futures.task_future import BaseTaskFuture

    params: dict[str, InputParam] = {}
    for name, seq in kwargs.items():
        if isinstance(seq, MapTaskFuture):
            params[name] = InputParam.from_sequence_ref(seq.id)
        elif isinstance(seq, BaseTaskFuture):
            params[name] = InputParam.from_sequence_ref(seq.id)
        else:
            params[name] = InputParam.from_sequence(seq)

    return params


def build_output_configs(
    future_outputs: tuple[OutputFuture, ...], base_placeholders: set[str]
) -> tuple[OutputConfig, ...]:
    """
    Builds graph IR output configurations from the provided future outputs.

    Args:
        future_outputs: Tuple of FutureOutput instances from the TaskFuture.
        base_placeholders: Base set of available placeholder names for validating output keys.
            This typically includes root-level parameters and any outputs from upstream nodes.

    Returns:
        Tuple of OutputConfig instances for graph construction.
    """

    output_configs: list[OutputConfig] = []
    for future_output in future_outputs:
        placeholders = base_placeholders | future_output.extras.keys()
        check_key_placeholders(future_output.key, placeholders)

        dependencies = build_parameters(future_output.extras)
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
