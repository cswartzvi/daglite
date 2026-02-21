"""Utility functions for CLI parameter parsing."""

import inspect
import types
import typing
import warnings
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import click

if TYPE_CHECKING:
    from daglite.workflows import Workflow


def parse_param_value(value: str, param_type: type | None) -> Any:
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


def parse_workflow_params(workflow_obj: "Workflow[Any]", param: tuple[str, ...]) -> dict[str, Any]:
    """
    Parse and validate `--param` strings against a workflow's signature.

    Emits a `UserWarning` when parameters are passed to a workflow with no type
    annotations, since values will be treated as plain strings.

    Args:
        workflow_obj: The loaded `Workflow` whose signature is used for validation
            and type coercion.
        param: Tuple of raw `name=value` strings supplied on the CLI.

    Returns:
        Dictionary of validated, type-coerced parameter values.

    Raises:
        click.BadParameter: For malformed entries, unknown names, or missing
            required parameters.
    """
    params: dict[str, Any] = {}
    typed_params = workflow_obj.get_typed_params()

    # Warn if passing params to an untyped workflow
    if param and not workflow_obj.has_typed_params():
        warnings.warn(
            f"Workflow '{workflow_obj.name}' has untyped parameters. "
            "Parameter values will be passed as strings. "
            "Consider adding type annotations for automatic type conversion.",
            UserWarning,
            stacklevel=3,
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

        try:
            params[param_name] = parse_param_value(param_value, typed_params[param_name])
        except ValueError as e:
            raise click.BadParameter(
                f"Invalid value for parameter '{param_name}': '{param_value}'. {e}"
            ) from e

    # Check for missing required parameters
    missing_params = [
        name
        for name, info in workflow_obj.signature.parameters.items()
        if info.default is inspect.Parameter.empty and name not in params
    ]
    if missing_params:
        raise click.BadParameter(
            f"Missing required parameters: {missing_params}. "
            f"Use --param name=value to provide them."
        )

    return params


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
            raise click.BadParameter(f"Invalid setting format: '{s}'. Expected 'name=value'")

        setting_name, setting_value = s.split("=", 1)

        fields = DagliteSettings.__dataclass_fields__
        if setting_name not in fields:
            raise click.BadParameter(
                f"Unknown setting: '{setting_name}'. Available settings: {', '.join(fields)}"
            )

        type_hints = typing.get_type_hints(DagliteSettings)
        field_type = type_hints[setting_name]

        # For Union types (e.g. str | DatasetStore), use the first concrete type
        if get_origin(field_type) is Union or isinstance(field_type, types.UnionType):
            field_type = get_args(field_type)[0]
        try:
            overrides[setting_name] = parse_param_value(setting_value, field_type)
        except ValueError as e:
            raise click.BadParameter(
                f"Invalid value for setting '{setting_name}': '{setting_value}'. {e}"
            ) from e

    return overrides
