"""CLI ``list`` command for discovering workflows in a module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click

from daglite.workflows import Workflow


def _find_workflows_in_module(module_path: str) -> list[tuple[str, Workflow]]:
    """Import *module_path* and return ``(dotted_path, workflow)`` for every Workflow found."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:  # pragma: no cover
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise click.ClickException(f"Cannot import module '{module_path}': {e}") from e

    found = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, Workflow):
            found.append((f"{module_path}.{attr_name}", obj))
    return found


@click.command("list")
@click.argument("modules", nargs=-1, required=True, metavar="MODULE [MODULE...]")
def list_workflows(modules: tuple[str, ...]) -> None:
    r"""
    List workflows found in the given module(s).

    MODULE should be a dotted importable path, e.g. 'myproject.workflows'.
    Multiple modules can be supplied at once.

    Examples:
    \b
    # List all workflows in a module
    daglite list myproject.workflows

    \b
    # List workflows from multiple modules
    daglite list myproject.workflows myproject.other_workflows
    """
    found: list[tuple[str, Workflow]] = []
    for module_path in modules:
        found.extend(_find_workflows_in_module(module_path))

    if not found:
        click.echo("No workflows found.")
        return

    max_path_len = max(len(path) for path, _ in found)
    col_width = max(max_path_len, 40)

    for dotted_path, wf in found:
        first_line = next(
            (line.strip() for line in wf.description.splitlines() if line.strip()), ""
        )
        if len(first_line) > 72:
            first_line = first_line[:69] + "..."
        if first_line:
            click.echo(f"{dotted_path:<{col_width}}  {first_line}")
        else:
            click.echo(dotted_path)
