from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from daglite.futures.base import BaseTaskFuture
from daglite.workflows import Workflow, workflow

Dag = BaseTaskFuture
"""
Type alias for workflow return annotations.

Workflows are entry points — the ``Dag`` alias provides a readable annotation
for functions that build and return a task graph:

    @workflow
    def compute(x: int, y: int) -> Dag[int]:
        return add(x=x, y=y)

    @workflow
    def sweep(n: int) -> Dag[int]:
        return double.map(x=list(range(n)))
"""

# ---------------------------------------------------------------------------
# Backwards-compatibility aliases — @pipeline / Pipeline are deprecated in
# favour of @workflow / Workflow.  Both names resolve to the same object.
# ---------------------------------------------------------------------------
pipeline = workflow
Pipeline = Workflow


def load_pipeline(pipeline_path: str) -> Workflow[Any]:
    """
    Load a workflow from a module path.

    Accepts objects decorated with either ``@workflow`` or the deprecated
    ``@pipeline`` alias.

    Args:
        pipeline_path: Dotted path to the workflow (e.g. 'mymodule.my_workflow').

    Returns:
        The loaded Workflow instance.

    Raises:
        ValueError: If the path is invalid.
        ModuleNotFoundError: If the module cannot be found.
        AttributeError: If the attribute does not exist in the module.
        TypeError: If the loaded object is not a Workflow instance.
    """
    if "." not in pipeline_path:
        raise ValueError(
            f"Invalid workflow path: '{pipeline_path}'. Expected format: 'module.workflow_name'"
        )

    module_path, attr_name = pipeline_path.rsplit(".", 1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:  # pragma: no cover
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)

    if not hasattr(module, attr_name):
        raise AttributeError(f"Workflow '{attr_name}' not found in module '{module_path}'")

    obj = getattr(module, attr_name)

    if not isinstance(obj, Workflow):
        raise TypeError(
            f"'{pipeline_path}' is not a Workflow. "
            f"Did you forget to use the @workflow decorator?"
        )

    return obj
