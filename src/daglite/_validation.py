"""Internal shared validation and template resolution functions."""

import string
from typing import Any


def check_key_template(key: str) -> None:
    """Validates a key template string for well-formed {placeholder} syntax."""
    try:
        parsed = list(string.Formatter().parse(key))
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid key template '{key}': {e}. Key templates use {{name}} placeholders for "
            f"parameter substitution."
        ) from e

    for _, field_name, _, _ in parsed:
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError(
                f"Invalid key template '{key}': empty placeholder '{{}}' is not allowed. "
                f"Use named placeholders like '{{param_name}}' instead."
            )


def check_key_placeholders(key: str, allowed: set[str]) -> None:
    """
    Checks that all placeholders in a key template string are present in the available variable set.

    Args:
        key: Template string containing {placeholders} to validate.
        allowed: Set of names that are allowed to be used as placeholders in the key template.

    Raises:
        ValueError: If any placeholder in the key template is not in the allowed names set.
    """
    placeholders = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(key)
        if field_name is not None
    }
    unknown = placeholders - allowed
    if unknown:
        raise ValueError(
            f"Key template '{key}' references {unknown} which won't be available "
            f"at runtime. Available placeholders: {sorted(allowed)}. "
        )


def has_placeholders(template: str) -> bool:
    """
    Return `True` if *template* contains at least one ``{name}`` placeholder.

    A cheap check used to skip formatting when no substitution is needed.
    """
    return any(field_name is not None for _, field_name, _, _ in string.Formatter().parse(template))


def resolve_template(template: str, bound_args: dict[str, Any]) -> str:
    """
    Resolve ``{placeholder}`` references in *template* using *bound_args*.

    Only placeholders that appear in *bound_args* are substituted; literal text
    and unknown placeholders are left unchanged (via `str.format_map` with a
    default-dict wrapper). This is intentionally lenient so that partial
    resolution is possible.

    Args:
        template: A string potentially containing ``{name}`` placeholders.
        bound_args: Mapping of parameter names to their runtime values.

    Returns:
        The formatted string with all resolvable placeholders substituted.

    Examples:
        >>> resolve_template("output_{split}.csv", {"split": "train"})
        'output_train.csv'
        >>> resolve_template("no_placeholders", {"x": 1})
        'no_placeholders'
    """
    if not has_placeholders(template):
        return template
    return template.format_map(_DefaultDict(bound_args))


class _DefaultDict(dict[str, Any]):
    """
    Dict subclass that returns ``{key}`` for missing keys.

    Used by `resolve_template` so that unresolvable placeholders survive
    formatting instead of raising `KeyError`.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
