"""Daglite CLI - Command-line interface for daglite."""

from __future__ import annotations

import click

from daglite import __version__
from daglite.cli.cmds import parse_param_value
from daglite.cli.cmds import run

__all__ = ["cli", "run", "parse_param_value"]


@click.group(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120})
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Daglite - Lightweight Python framework for building static DAGs."""


cli.add_command(run)


if __name__ == "__main__":
    cli()
