"""CLI ``run`` command for executing workflows."""

from __future__ import annotations

import asyncio
import inspect
import types
import typing
import warnings
from typing import Any, Union, get_args, get_origin

import click

from daglite.cli._shared import parse_param_value
from daglite.engine import evaluate_workflow
from daglite.engine import evaluate_workflow_async
from daglite.pipelines import load_pipeline
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings


@click.command()
@click.argument("workflow")
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Workflow parameter in format 'name=value'. Can be specified multiple times.",
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
    workflow: str,
    param: tuple[str, ...],
    backend: str,
    parallel: bool,
    settings: tuple[str, ...],
) -> None:
    r"""
    Run a daglite workflow.

    WORKFLOW should be a dotted path to a function decorated with @workflow,
    e.g., 'myproject.workflows.my_workflow'.

    Examples:
    \b
    # Run a simple workflow
    daglite run myproject.workflows.simple_workflow

    \b
    # Run with parameters
    daglite run myproject.workflows.data_workflow --param input_file=data.csv
    --param num_workers=4

    \b
    # Run with custom backend and settings
    daglite run myproject.workflows.parallel_workflow --backend threading
    --settings max_backend_threads=16
    """
    # Load the workflow
    try:
        workflow_obj = load_pipeline(workflow)
    except (ValueError, ModuleNotFoundError, AttributeError, TypeError) as e:
        raise click.ClickException(str(e)) from e

    params: dict[str, Any] = {}
    typed_params = workflow_obj.get_typed_params()

    # Warn if passing params to an untyped workflow
    if param and not workflow_obj.has_typed_params():
        warnings.warn(
            f"Workflow '{workflow_obj.name}' has untyped parameters. "
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
    sig = workflow_obj.signature
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

    # Call the workflow to get the futures
    try:
        raw = workflow_obj(**params)
        futures = workflow_obj._collect_futures(raw)
    except Exception as e:  # pragma: no cover
        raise click.ClickException(f"Error calling workflow: {e}") from e

    # Execute the graph
    click.echo(f"Running workflow: {workflow_obj.name}")
    if params:
        click.echo(f"Parameters: {params}")
    click.echo(f"Backend: {backend}")
    if parallel:
        click.echo("Parallel execution: enabled")

    try:
        if parallel:
            result = asyncio.run(evaluate_workflow_async(futures))
        else:
            result = evaluate_workflow(futures)

        click.echo("\nWorkflow completed successfully!")

        # Display results — single sink: plain "Result: value";
        # multi-sink: labelled "Results:\n  name: value"
        names = list(result.keys())
        if len(names) == 1 and len(result.all(names[0])) == 1:
            click.echo(f"Result: {result.all(names[0])[0]}")
        else:
            click.echo("Results:")
            for name in result.keys():
                values = result.all(name)
                if len(values) == 1:
                    click.echo(f"  {name}: {values[0]}")
                else:
                    for i, value in enumerate(values):
                        click.echo(f"  {name}[{i}]: {value}")

    except Exception as e:
        raise click.ClickException(f"Workflow execution failed: {e}") from e
