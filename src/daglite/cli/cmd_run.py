"""CLI ``run`` command for executing workflows."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from daglite.cli._shared import parse_settings_overrides
from daglite.cli._shared import parse_workflow_params
from daglite.cli._shared import setup_cli_plugins
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings
from daglite.workflows import load_workflow


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

    WORKFLOW should be a dotted path to a workflow function decorated with @workflow,
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
    try:
        workflow_obj = load_workflow(workflow)
    except (ValueError, ModuleNotFoundError, AttributeError, TypeError) as e:
        raise click.ClickException(str(e)) from e

    setup_cli_plugins()
    params = parse_workflow_params(workflow_obj, param)

    # Apply settings overrides
    settings_dict: dict[str, Any] = {"default_backend": backend}
    settings_dict.update(parse_settings_overrides(settings))
    set_global_settings(DagliteSettings(**settings_dict))

    # Execute the workflow
    try:
        if parallel:
            asyncio.run(workflow_obj.run_async(**params))  # type: ignore[call-arg]
        else:
            workflow_obj.run(**params)  # type: ignore[call-arg]
    except Exception as e:
        raise click.ClickException(f"Workflow execution failed: {e}") from e
