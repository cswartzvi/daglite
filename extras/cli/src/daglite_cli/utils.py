from typing import Any


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
