"""Default cache hashing strategy."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable

from daglite.serialization import default_registry


def default_cache_hash(func: Callable, bound_args: dict[str, Any]) -> str:
    """
    Generate cache key from function source and parameter values.

    Uses smart hashing strategies from SerializationRegistry to avoid
    performance issues with large objects (e.g., numpy arrays, dataframes).

    Args:
        func: The function being cached
        bound_args: Bound parameter values

    Returns:
        SHA256 hex digest string

    Examples:
        >>> def add(x: int, y: int) -> int:
        ...     return x + y
        >>> hash1 = default_cache_hash(add, {"x": 1, "y": 2})
        >>> hash2 = default_cache_hash(add, {"x": 1, "y": 2})
        >>> hash1 == hash2
        True
        >>> hash3 = default_cache_hash(add, {"x": 1, "y": 3})
        >>> hash1 == hash3
        False
    """
    h = hashlib.sha256()

    # Hash function source
    try:
        source = inspect.getsource(func)
        h.update(source.encode())
    except (OSError, TypeError):  # pragma: no cover
        h.update(func.__qualname__.encode())

    # Hash each parameter using registry's strategies; bound_args can be either a dict or
    # BoundArguments object
    items = bound_args.arguments.items() if hasattr(bound_args, "arguments") else bound_args.items()  # type: ignore
    for name, value in sorted(items):
        param_hash = default_registry.hash_value(value)
        h.update(f"{name}={param_hash}".encode())

    return h.hexdigest()


__all__ = ["default_cache_hash"]
