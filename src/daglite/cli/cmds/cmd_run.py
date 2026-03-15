from typing import Any

from daglite.cli._shared import build_workflow_app
from daglite.cli._shared import get_workflow
from daglite.cli._shared import normalize_tokens


def run_workflow(target: str, workflow_args: list[str]) -> Any:
    """
    Runs a workflow target with the given argument tokens.

    Args:
        target: The workflow import path in the format 'module:workflow' or 'module.workflow'.
        workflow_args: List of argument tokens to pass to the workflow.

    Returns:
        The return value of the workflow function.
    """
    workflow_obj = get_workflow(target)
    workflow_app = build_workflow_app(workflow_obj, target, show_meta_flags=True)
    return workflow_app(normalize_tokens(workflow_args))


# NOTE: The following function is used strictly for the `daglite run <target> --help` command,
#       which is a special case that does not require any arguments to be passed to the workflow


def run_help(target: str):
    """Run a workflow target with arguments.

    Targets can be specified as:
      - module.sub_module:workflow_func
      - module.sub_module.workflow_func
      - path/to/module.py:workflow_func

    Usage:
      - daglite run module:workflow --param1 value1 --param2 value2
      - daglite run path/to/module.py:workflow --x 3 --y 4
    """  # noqa: D213
    pass  # pragma: no cover
