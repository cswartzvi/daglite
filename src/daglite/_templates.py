"""Functions for parsing and resolving string templates with `{placeholder}` syntax."""

import string
from typing import Any

_formatter = string.Formatter()


def parse_template(template: str) -> frozenset[str]:
    """
    Parse *template*, validate syntax, and return the set of placeholder names.

    Raises `ValueError` for malformed templates (e.g. un-matched braces or empty `{}` placeholders).

    Returns:
        A (possibly empty) frozenset of placeholder names found in *template*.
    """
    try:
        parsed = list(_formatter.parse(template))
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid key template '{template}': {e}. Key templates use {{name}} placeholders "
            f"for parameter substitution."
        ) from e

    names: set[str] = set()
    for _, field_name, _, _ in parsed:
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError(
                f"Invalid key template '{template}': empty placeholder '{{}}' is not allowed. "
                f"Use named placeholders like '{{param_name}}' instead."
            )
        names.add(field_name)
    return frozenset(names)


def resolve_template(template: str, bound_args: dict[str, Any]) -> str:
    """
    Resolve `{placeholder}` references in *template* using *bound_args*.

    Only placeholders that appear in *bound_args* are substituted; literal text and unknown
    placeholders are left unchanged (via `str.format_map` with a default-dict wrapper). This is
    intentionally lenient so that partial resolution is possible.

    Args:
        template: A string potentially containing `{name}` placeholders.
        bound_args: Mapping of parameter names to their runtime values.

    Returns:
        The formatted string with all resolvable placeholders substituted.

    Examples:
        >>> resolve_template("output_{split}.csv", {"split": "train"})
        'output_train.csv'
        >>> resolve_template("no_placeholders", {"x": 1})
        'no_placeholders'
    """
    return template.format_map(_DefaultDict(bound_args))


class _DefaultDict(dict[str, Any]):
    """
    Dict subclass that returns `{key}` for missing keys.

    Used by `resolve_template` so that unresolvable placeholders survive
    formatting instead of raising `KeyError`.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
