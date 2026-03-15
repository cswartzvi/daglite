"""Utility functions for CLI parameter parsing."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
import sys
import types
import typing
from dataclasses import dataclass
from functools import lru_cache
from functools import wraps
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from cyclopts import App

from daglite.workflows import AsyncWorkflow
from daglite.workflows import BaseWorkflow
from daglite.workflows import SyncWorkflow
from daglite.workflows import Workflow

# region Internal API

HELP_FLAGS = ["-h", "--help"]


def setup_cli_plugins() -> None:
    """
    Auto-register output plugins for CLI runs.

    Prefers daglite-rich (progress bars + rich logging) when installed;
    falls back to the builtin LifecycleLoggingPlugin.  Skips registration
    if the user has already registered a compatible plugin.
    """
    from daglite.plugins.manager import has_plugin
    from daglite.plugins.manager import register_plugins

    try:
        from daglite_rich.logging import RichLifecycleLoggingPlugin

        from daglite.logging.plugin import LifecycleLoggingPlugin

        if not has_plugin(LifecycleLoggingPlugin):
            register_plugins(RichLifecycleLoggingPlugin())
    except ImportError:  # pragma: no cover – only reached when daglite-rich is not installed
        from daglite.logging.plugin import LifecycleLoggingPlugin

        if not has_plugin(LifecycleLoggingPlugin):
            register_plugins(LifecycleLoggingPlugin())


def normalize_tokens(tokens: list[str] | tuple[str, ...] | None) -> list[str]:
    """
    Normalizes a list or tuple of tokens into a list of strings.

    Args:
        tokens: A list or tuple of tokens, or None.

    Returns:
        A list of strings. If `tokens` is None, returns an empty list.
    """
    if tokens is None:
        return []
    return [str(token) for token in tokens]


def parse_settings_overrides(settings: tuple[str, ...]) -> dict[str, Any]:
    """
    Parse `--settings` override strings into a validated dict.

    Each entry must be in `name=value` format and must correspond to a known field on
    `DagliteSettings`. Values are coerced to the field's declared type.

    Args:
        settings: Tuple of raw `name=value` strings supplied on the CLI.

    Returns:
        Dictionary of validated setting overrides ready to merge into a
        `DagliteSettings` constructor call.

    Raises:
        click.BadParameter: For malformed entries, unknown names, or bad values.
    """
    from daglite.settings import DagliteSettings

    overrides: dict[str, Any] = {}

    for s in settings:
        if "=" not in s:
            raise ValueError(f"Invalid setting format: {s!r}. Expected 'name=value'.")

        setting_name, setting_value = s.split("=", 1)

        fields = DagliteSettings.__dataclass_fields__
        if setting_name not in fields:
            raise ValueError(
                f"Unknown setting: {setting_name!r}. Available settings: {', '.join(fields)}"
            )

        type_hints = typing.get_type_hints(DagliteSettings)
        field_type = type_hints[setting_name]

        # For Union types (e.g. str | DatasetStore), use the first concrete type
        if get_origin(field_type) is Union or isinstance(field_type, types.UnionType):
            field_type = get_args(field_type)[0]
        try:
            overrides[setting_name] = _parse_param_value(setting_value, field_type)
        except ValueError as e:
            raise ValueError(
                f"Invalid value for setting {setting_name!r}: {setting_value!r}. {e}"
            ) from e

    return overrides


def discover_workflows(root_modname: str) -> list[tuple[str, _WorkflowMeta]]:
    """
    Discover all workflows under a given root module.

    Args:
        root_modname: A dotted module name **or** a filesystem path
            (e.g. ``tests/examples/workflows.py``) to search for workflows.

    Returns:
        A list of tuples containing the dotted path to each workflow and its metadata.
    """
    root_modname = _filepath_to_module(root_modname)
    _ensure_cwd_on_path()

    results: list[tuple[str, _WorkflowMeta]] = []

    for modname in _iter_modules_under(root_modname):
        mod = importlib.import_module(modname)
        for name, obj in inspect.getmembers(mod):
            if isinstance(obj, BaseWorkflow):
                path = f"{modname}:{name}"
                metadata = _WorkflowMeta(name=obj.name, description=obj.description)
                results.append((path, metadata))

    results.sort(key=lambda item: item[0])
    return results


def get_workflow(target: str) -> Workflow:
    """
    Resolves a workflow import path and return the corresponding workflow object.

    Args:
        target: The workflow import path in the format 'module:workflow' or 'module.workflow'.

    Returns:
        The resolved workflow object.
    """
    workflow = _resolve_import_path(target)
    if not isinstance(workflow, (SyncWorkflow, AsyncWorkflow)):
        raise TypeError(
            f"{target!r} resolved to {type(workflow).__name__}, but it's not a @workflow function."
        )
    return workflow


def validate_workflow_args(target: str, workflow_args: list[str]) -> inspect.BoundArguments:
    """
    Validates workflow arguments against the workflow's signature.

    Args:
        target: The workflow import path in the format 'module:workflow' or 'module.workflow'.
        workflow_args: List of argument tokens to validate.

    Returns:
        An `inspect.BoundArguments` object containing the validated arguments.
    """
    workflow_obj = get_workflow(target)
    wf_app = build_workflow_app(workflow_obj, target, show_meta_flags=False)

    _command, bound, ignored = wf_app.parse_args(
        normalize_tokens(workflow_args),
        print_error=False,
        exit_on_error=False,
    )
    if ignored:
        raise ValueError(f"Unrecognized arguments: {ignored}")
    return bound


def build_workflow_app(workflow: Workflow, target: str, *, show_meta_flags: bool) -> App:
    """Build a temporary Cyclopts app for one concrete workflow."""
    wf_app = App(
        name=workflow.name,
        usage=f"{_cli_name()} run {target}",
        help_flags=["-h", "--help"] if show_meta_flags else [],
        version_flags=["--version"] if show_meta_flags else [],
    )

    @wf_app.default
    @wraps(workflow.func)
    def _default(*args: Any, **kwargs: Any) -> None:
        setup_cli_plugins()
        if isinstance(workflow, AsyncWorkflow):
            asyncio.run(workflow(*args, **kwargs))
            return
        workflow(*args, **kwargs)

    return wf_app


def print_run_error(exception: Exception) -> None:
    """Print a user-friendly error message for a failed workflow run."""
    prefix = "Error: "
    print(f"{prefix}{exception}")
    print(" " * len(prefix) + "Tip: use 'daglite describe <target>' to see workflow parameters.")


# region Internal helpers


@dataclass(frozen=True)
class _WorkflowMeta:
    """Metadata rendered by discovery/list output."""

    name: str
    description: str


@lru_cache(maxsize=256)
def _resolve_import_path(spec: str) -> Any:
    spec = _filepath_to_module(spec)
    mod_name, attr = _split_target(spec)

    _ensure_cwd_on_path()

    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise AttributeError(f"Module {mod_name!r} has no attribute {attr!r}.") from e


def _ensure_cwd_on_path() -> None:
    """Ensure the current working directory is on *sys.path*."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _filepath_to_module(spec: str) -> str:
    """
    Convert a filesystem path to a dotted module path when applicable.

    Handles both plain module roots (``tests/examples/workflows.py``) and
    target specs with a colon (``tests/examples/workflows.py:my_wf``).
    If *spec* doesn't look like a filesystem path it is returned unchanged.
    """
    # Split off an optional ":attr" suffix before processing the path part.
    attr_suffix = ""
    if ":" in spec:
        path_part, attr_suffix = spec.split(":", 1)
    else:
        path_part = spec

    # Heuristic: treat it as a filepath if it contains a path separator or
    # ends with ".py".
    if "/" not in path_part and "\\" not in path_part and not path_part.endswith(".py"):
        return spec  # already a dotted module path

    path_part = path_part.rstrip("/").rstrip("\\")
    if path_part.endswith(".py"):
        path_part = path_part[:-3]
    # Strip leading "./"
    if path_part.startswith("./") or path_part.startswith(".\\"):
        path_part = path_part[2:]

    module_path = path_part.replace("/", ".").replace("\\", ".")
    if attr_suffix:
        return f"{module_path}:{attr_suffix}"
    return module_path


def _iter_modules_under(root_modname: str):
    root = importlib.import_module(root_modname)
    yield root.__name__

    if not hasattr(root, "__path__"):
        return

    for module_info in pkgutil.walk_packages(root.__path__, prefix=root.__name__ + "."):
        yield module_info.name


def _parse_param_value(value: str, param_type: type | None) -> Any:
    """
    Parse a parameter value string into the appropriate type.

    Args:
        value: String value to parse.
        param_type: Target type to convert to, or None for string.

    Returns:
        Parsed value in the appropriate type.

    Raises:
        ValueError: If the value cannot be converted to the target type.
    """
    if param_type is None:
        return value

    # Handle common types
    if param_type is int:
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{value}' to int") from e
    elif param_type is float:
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{value}' to float") from e
    elif param_type is bool:
        return value.lower() in ("true", "1", "yes", "y")
    elif param_type is str:
        return value
    else:
        # For other types, try direct conversion or return as string
        try:
            return param_type(value)
        except Exception:
            return value


def _resolved_signature(workflow: Workflow) -> inspect.Signature:
    signature = workflow.signature
    try:
        hints = typing.get_type_hints(workflow.func)
    except Exception:
        hints = {}

    params = [
        parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
        for parameter in signature.parameters.values()
    ]
    return_annotation = hints.get("return", signature.return_annotation)
    return signature.replace(parameters=params, return_annotation=return_annotation)


def _split_target(spec: str) -> tuple[str, str]:
    if ":" in spec:
        mod_name, attr = spec.split(":", 1)
    else:
        if "." not in spec:
            raise ValueError("Expected 'module:workflow' or 'module.workflow'.")
        mod_name, attr = spec.rsplit(".", 1)

    if not mod_name or not attr:
        raise ValueError("Expected 'module:workflow' or 'module.workflow'.")

    return mod_name, attr


def _cli_name() -> str:
    argv0 = Path(sys.argv[0]).name
    if argv0 == "__main__.py" and __package__:
        return f"py -m {__package__}"
    return argv0


def _is_async_workflow(workflow: Workflow) -> bool:
    """Check if a workflow is an instance of AsyncWorkflow."""
    from daglite.workflows import AsyncWorkflow

    return isinstance(workflow, AsyncWorkflow)
