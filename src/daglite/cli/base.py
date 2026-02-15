from __future__ import annotations

import click

import daglite
from daglite.cli.cmd_run import run


@click.group(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120})
@click.version_option(version=daglite.__version__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Daglite - Lightweight Python framework for building static DAGs."""


cli.add_command(run)
