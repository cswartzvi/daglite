"""Internal shared validation functions."""

import string


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
