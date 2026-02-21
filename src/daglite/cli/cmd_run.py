"""CLI ``run`` command for executing pipelines."""

from __future__ import annotations

import asyncio
import inspect
import types
import typing
import warnings
from typing import Any, Union, get_args, get_origin

import click

from daglite.cli._shared import parse_param_value
from daglite.engine import evaluate
from daglite.engine import evaluate_async
from daglite.pipelines import load_pipeline
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings


@click.command()
@click.argument("pipeline")
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Pipeline parameter in format 'name=value'. Can be specified multiple times.",
)
@click.option(
    "--backend",
    "-b",
    default="inline",
    help="Backend to use for execution (e.g., 'inline', 'threading').",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Enable sibling parallelism via async evaluation.",
)
@click.option(
    "--settings",
    "-s",
    multiple=True,
    help="Setting override in format 'name=value'. Can be specified multiple times.",
)
def run(
    pipeline: str,
    param: tuple[str, ...],
    backend: str,
    parallel: bool,
    settings: tuple[str, ...],
) -> None:
    r"""
    Run a daglite pipeline.

    PIPELINE should be a dotted path to a pipeline function decorated with @pipeline,
    e.g., 'myproject.pipelines.my_pipeline'.

    Examples:
    \b
    # Run a simple pipeline
    daglite run myproject.pipelines.simple_pipeline

    \b
    # Run with parameters
    daglite run myproject.pipelines.data_pipeline --param input_file=data.csv
    --param num_workers=4

    \b
    # Run with custom backend and settings
    daglite run myproject.pipelines.parallel_pipeline --backend threading
    --settings max_backend_threads=16
    """
    # Load the pipeline
    try:
        pipeline_obj = load_pipeline(pipeline)
    except (ValueError, ModuleNotFoundError, AttributeError, TypeError) as e:
        raise click.ClickException(str(e)) from e

    params: dict[str, Any] = {}
    typed_params = pipeline_obj.get_typed_params()

    # Warn if passing params to an untyped pipeline
    if param and not pipeline_obj.has_typed_params():
        warnings.warn(
            f"Pipeline '{pipeline_obj.name}' has untyped parameters. "
            "Parameter values will be passed as strings. "
            "Consider adding type annotations for automatic type conversion.",
            UserWarning,
            stacklevel=2,
        )

    for p in param:
        if "=" not in p:
            raise click.BadParameter(f"Invalid parameter format: '{p}'. Expected 'name=value'")

        param_name, param_value = p.split("=", 1)

        if param_name not in typed_params:
            raise click.BadParameter(
                f"Unknown parameter: '{param_name}'. "
                f"Available parameters: {list(typed_params.keys())}"
            )

        params[param_name] = parse_param_value(param_value, typed_params[param_name])

    # Check for missing required parameters
    sig = pipeline_obj.signature
    missing_params = []
    for param_name, param_info in sig.parameters.items():
        if param_info.default == inspect.Parameter.empty and param_name not in params:
            missing_params.append(param_name)

    if missing_params:
        raise click.BadParameter(
            f"Missing required parameters: {missing_params}. "
            f"Use --param name=value to provide them."
        )

    # Build settings dict: start with --backend, then layer --settings overrides
    settings_dict: dict[str, Any] = {"default_backend": backend}

    for s in settings:
        if "=" not in s:
            raise click.BadParameter(f"Invalid setting format: '{s}'. Expected 'name=value'")

        setting_name, setting_value = s.split("=", 1)

        # Validate setting name
        fields = DagliteSettings.__dataclass_fields__
        if setting_name not in fields:
            raise click.BadParameter(
                f"Unknown setting: '{setting_name}'. Available settings: {', '.join(fields)}"
            )

        # Parse setting value using field type introspection
        type_hints = typing.get_type_hints(DagliteSettings)
        field_type = type_hints[setting_name]
        # For Union types (e.g. str | DatasetStore), use the first concrete type
        if get_origin(field_type) is Union or isinstance(field_type, types.UnionType):
            field_type = get_args(field_type)[0]
        try:
            settings_dict[setting_name] = parse_param_value(setting_value, field_type)
        except ValueError as e:
            raise click.BadParameter(
                f"Invalid value for setting '{setting_name}': '{setting_value}'. {e}"
            ) from e

    # Apply settings globally for this run
    set_global_settings(DagliteSettings(**settings_dict))

    # Call the pipeline to get the NodeBuilder
    try:
        graph = pipeline_obj(**params)
    except Exception as e:  # pragma: no cover
        raise click.ClickException(f"Error calling pipeline: {e}") from e

    # Execute the graph
    click.echo(f"Running pipeline: {pipeline_obj.name}")
    if params:
        click.echo(f"Parameters: {params}")
    click.echo(f"Backend: {backend}")
    if parallel:
        click.echo("Parallel execution: enabled")

    try:
        if parallel:
            # Async evaluation enables sibling parallelism
            result = asyncio.run(evaluate_async(graph))
        else:
            result = evaluate(graph)
        click.echo("\nPipeline completed successfully!")
        click.echo(f"Result: {result}")
    except Exception as e:
        raise click.ClickException(f"Pipeline execution failed: {e}") from e
